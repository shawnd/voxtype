//! Parakeet streaming transcriber via `parakeet-rs::ParakeetUnified`
//!
//! Wraps the cache-aware streaming pipeline so voxtype can emit live partial
//! transcripts during recording and a final transcript on hotkey release.
//!
//! Selected when `[parakeet] streaming = true`. Otherwise the batch
//! `ParakeetTranscriber` is used.
//!
//! # Architecture
//!
//! - The transcriber holds a [`ParakeetUnifiedHandle`] (shared model weights)
//!   and a `Mutex<ParakeetUnified>` for batch-mode fallback. The same handle
//!   spawns fresh streaming instances per [`start_stream`] call.
//! - For batch [`Transcriber::transcribe`] the `Mutex<ParakeetUnified>` is
//!   reset between calls so no streaming cache leaks between recordings.
//! - For [`StreamingTranscriber::start_stream`] a fresh `ParakeetUnified` is
//!   constructed from the handle and moved into the spawned drive task.
//!
//! # Partial / Final policy
//!
//! Partial and Final text are **deltas**, never cumulative transcripts.
//! After speech followed by a configurable pause, flush the remaining tail,
//! emit Final, reset all decoder caches, and advance the segment ID. Capture
//! stays armed until EOF or cancellation. Energy is measured in 20ms frames,
//! using audio duration rather than inference speed or wall-clock time.

use super::parakeet::{build_execution_config, resolve_model_path};
use super::streaming::{StreamHandle, StreamingEvent, StreamingTranscriber};
use super::{TimedSegment, Transcriber};
use crate::config::ParakeetConfig;
use crate::error::TranscribeError;
use parakeet_rs::{ParakeetUnified, ParakeetUnifiedHandle, UnifiedStreamingConfig};
use std::sync::Mutex;
use tokio::sync::{mpsc, oneshot};

fn validate_endpoint_config(config: &ParakeetConfig) -> Result<(), TranscribeError> {
    if !config.streaming_pause_secs.is_finite() || config.streaming_pause_secs < 0.0 {
        return Err(TranscribeError::InitFailed(
            "[parakeet] streaming_pause_secs must be finite and >= 0 (0 disables pauses)".into(),
        ));
    }
    if !config.streaming_silence_threshold.is_finite()
        || !(0.0..=1.0).contains(&config.streaming_silence_threshold)
        || config.streaming_silence_threshold == 0.0
    {
        return Err(TranscribeError::InitFailed(
            "[parakeet] streaming_silence_threshold must be finite and in (0, 1]".into(),
        ));
    }
    Ok(())
}

// Keep endpoint/event policy testable without loading model weights.
trait SegmentDecoder {
    fn chunk(&mut self, audio: &[f32]) -> Result<String, TranscribeError>;
    fn flush_tail(&mut self) -> Result<String, TranscribeError>;
    fn reset_context(&mut self);
}

impl SegmentDecoder for ParakeetUnified {
    fn chunk(&mut self, audio: &[f32]) -> Result<String, TranscribeError> {
        self.transcribe_chunk(audio).map_err(|e| {
            TranscribeError::InferenceFailed(format!("Parakeet transcribe_chunk failed: {e}"))
        })
    }

    fn flush_tail(&mut self) -> Result<String, TranscribeError> {
        self.flush()
            .map_err(|e| TranscribeError::InferenceFailed(format!("Parakeet flush failed: {e}")))
    }

    fn reset_context(&mut self) {
        self.reset();
    }
}

struct SegmentStream<D> {
    decoder: D,
    frame: Vec<f32>,
    pause_samples: usize,
    threshold: f32,
    quiet_samples: usize,
    speech_seen: bool,
    segment_id: u64,
    segment_has_text: bool,
    output_ends_with_space: bool,
    output_seen: bool,
}

impl<D: SegmentDecoder> SegmentStream<D> {
    fn new(decoder: D, pause_secs: f32, threshold: f32) -> Self {
        Self {
            decoder,
            frame: Vec::with_capacity(320),
            pause_samples: (pause_secs as f64 * 16000.0).ceil() as usize,
            threshold,
            quiet_samples: 0,
            speech_seen: false,
            segment_id: 0,
            segment_has_text: false,
            output_ends_with_space: false,
            output_seen: false,
        }
    }

    fn prepare_delta(&mut self, mut text: String) -> String {
        if text.is_empty() {
            return text;
        }
        // A reset tokenizer may omit the leading space it would have emitted
        // with prior context. Insert it only when the next segment has text,
        // so a trailing pause never types an extra character.
        if !self.segment_has_text
            && self.output_seen
            && !self.output_ends_with_space
            && !text.starts_with(char::is_whitespace)
            && !text.starts_with(['.', ',', '!', '?', ':', ';', ')', ']'])
        {
            text.insert(0, ' ');
        }
        self.segment_has_text = true;
        self.output_seen = true;
        self.output_ends_with_space = text.ends_with(char::is_whitespace);
        text
    }

    fn push(
        &mut self,
        mut audio: &[f32],
        emit: &mut impl FnMut(StreamingEvent),
    ) -> Result<(), TranscribeError> {
        while !audio.is_empty() {
            let count = (320 - self.frame.len()).min(audio.len());
            self.frame.extend_from_slice(&audio[..count]);
            audio = &audio[count..];
            if self.frame.len() == 320 {
                self.process_frame(emit)?;
            }
        }
        Ok(())
    }

    fn process_frame(
        &mut self,
        emit: &mut impl FnMut(StreamingEvent),
    ) -> Result<(), TranscribeError> {
        let energy = self.frame.iter().map(|s| s * s).sum::<f32>() / self.frame.len() as f32;
        if energy >= self.threshold * self.threshold {
            self.speech_seen = true;
            self.quiet_samples = 0;
        } else if self.speech_seen {
            self.quiet_samples = self.quiet_samples.saturating_add(self.frame.len());
        }
        let delta = self.decoder.chunk(&self.frame)?;
        self.frame.clear();
        let text = self.prepare_delta(delta);
        if !text.is_empty() {
            emit(StreamingEvent::Partial {
                text,
                segment_id: self.segment_id,
            });
        }
        if self.pause_samples > 0 && self.speech_seen && self.quiet_samples >= self.pause_samples {
            self.finalize(emit)?;
            self.decoder.reset_context();
            self.segment_id += 1;
            self.segment_has_text = false;
            self.speech_seen = false;
            self.quiet_samples = 0;
        }
        Ok(())
    }

    fn finalize(&mut self, emit: &mut impl FnMut(StreamingEvent)) -> Result<(), TranscribeError> {
        let tail = self.decoder.flush_tail()?;
        let text = self.prepare_delta(tail);
        // Even an empty tail is a boundary: partials still need committing.
        emit(StreamingEvent::Final {
            text,
            segment_id: self.segment_id,
        });
        Ok(())
    }

    fn finish(&mut self, emit: &mut impl FnMut(StreamingEvent)) -> Result<(), TranscribeError> {
        if !self.frame.is_empty() {
            self.process_frame(emit)?;
        }
        self.finalize(emit)
    }
}

fn drive_stream<D: SegmentDecoder>(
    mut segment: SegmentStream<D>,
    mut samples_rx: mpsc::Receiver<Vec<f32>>,
    events_tx: mpsc::Sender<StreamingEvent>,
    mut cancel_rx: oneshot::Receiver<()>,
) {
    let runtime = tokio::runtime::Handle::current();
    let mut cancelled = false;
    let mut emit = |event| {
        let _ = runtime.block_on(events_tx.send(event));
    };
    loop {
        // Cancellation must wake a receiver even if no audio arrives.
        let chunk = runtime.block_on(async {
            tokio::select! {
                biased;
                _ = &mut cancel_rx => { cancelled = true; None }
                chunk = samples_rx.recv() => chunk,
            }
        });
        let Some(chunk) = chunk else { break };
        if let Err(err) = segment.push(&chunk, &mut emit) {
            emit(StreamingEvent::Error(err));
            emit(StreamingEvent::Ended);
            return;
        }
    }
    // On cancel discard untyped audio; the daemon rewinds typed deltas.
    if !cancelled {
        if let Err(err) = segment.finish(&mut emit) {
            emit(StreamingEvent::Error(err));
        }
    }
    emit(StreamingEvent::Ended);
}

/// Streaming-capable Parakeet transcriber backed by `ParakeetUnified`.
pub struct ParakeetStreamingTranscriber {
    /// Shared model weights. Cloned cheaply for each streaming session.
    handle: ParakeetUnifiedHandle,
    /// Streaming-config snapshot used both for batch (so the handle's
    /// `from_shared_with_streaming_config` succeeds) and for spawning each
    /// streaming task.
    streaming_config: UnifiedStreamingConfig,
    /// Reusable batch instance for `Transcriber::transcribe`. Reset between
    /// calls so no inter-call state leaks.
    batch: Mutex<ParakeetUnified>,
    pause_secs: f32,
    silence_threshold: f32,
}

impl ParakeetStreamingTranscriber {
    pub fn new(config: &ParakeetConfig) -> Result<Self, TranscribeError> {
        validate_endpoint_config(config)?;
        // Streaming-capability gate. The cache-aware streaming pipeline
        // needs both `tokenizer.model` at load time and a decoder graph
        // sized for the streaming inference loop. Models in the registry
        // that don't meet both (e.g. istupakov's TDT-v3) hit either a
        // tokenizer-not-found at load or an ONNX Gather shape error at
        // first chunk — neither is a useful diagnostic. Fail fast here
        // with a message that names the model and the known-good
        // alternative. Unknown model names (custom user directories)
        // fall through with a warning since we can't validate them.
        if crate::setup::model::is_known_parakeet_model(&config.model)
            && !crate::setup::model::is_streaming_compatible_parakeet(&config.model)
        {
            return Err(TranscribeError::InitFailed(format!(
                "Parakeet streaming is enabled but model `{}` does not support \
                 cache-aware streaming.\n\n\
                 Fix one of:\n\n  \
                 - Disable streaming in config.toml under [parakeet]:\n\n      \
                 streaming = false\n\n  \
                 - Or switch to the streaming-compatible model:\n\n      \
                 voxtype setup model {streaming_model}\n\n    \
                 and set [parakeet] model = \"{streaming_model}\" in config.toml.",
                config.model,
                streaming_model = crate::setup::model::DEFAULT_PARAKEET_STREAMING_MODEL,
            )));
        }
        if !crate::setup::model::is_known_parakeet_model(&config.model) {
            tracing::warn!(
                model = %config.model,
                "Parakeet streaming requested with a model not in the registry; \
                 cannot validate streaming compatibility ahead of time. If load \
                 fails below, switch [parakeet] model to {} or set streaming = false.",
                crate::setup::model::DEFAULT_PARAKEET_STREAMING_MODEL
            );
        }

        let model_path = resolve_model_path(&config.model)?;

        tracing::info!(
            "Loading Parakeet streaming (ParakeetUnified) model from {:?}",
            model_path
        );
        let start = std::time::Instant::now();

        let exec_config = build_execution_config();
        let handle = ParakeetUnifiedHandle::load(&model_path, exec_config).map_err(|e| {
            TranscribeError::InitFailed(format!(
                "Parakeet streaming (ParakeetUnified) init failed: {}\n\n\
                Streaming requires a TDT v3 model directory containing tokenizer.model.\n\
                If you're using a TDT v2 directory, switch to TDT v3 or set\n\
                [parakeet] streaming = false to use the batch pipeline.",
                e
            ))
        })?;

        let streaming_config = UnifiedStreamingConfig {
            chunk_secs: config.streaming_chunk_secs,
            left_context_secs: config.streaming_left_context_secs,
            right_context_secs: config.streaming_right_context_secs,
        }
        .validate()
        .map_err(|e| {
            TranscribeError::InitFailed(format!(
                "Invalid Parakeet streaming config: {}. \
                Check streaming_chunk_secs, streaming_left_context_secs, \
                streaming_right_context_secs in [parakeet].",
                e
            ))
        })?;

        let batch = ParakeetUnified::from_shared_with_streaming_config(&handle, streaming_config)
            .map_err(|e| {
            TranscribeError::InitFailed(format!(
                "Failed to spawn batch ParakeetUnified instance: {}",
                e
            ))
        })?;

        tracing::info!(
            "Parakeet streaming model loaded in {:.2}s (chunk={:.2}s, \
            left={:.2}s, right={:.2}s)",
            start.elapsed().as_secs_f32(),
            streaming_config.chunk_secs,
            streaming_config.left_context_secs,
            streaming_config.right_context_secs,
        );

        Ok(Self {
            handle,
            streaming_config,
            batch: Mutex::new(batch),
            pause_secs: config.streaming_pause_secs,
            silence_threshold: config.streaming_silence_threshold,
        })
    }
}

impl Transcriber for ParakeetStreamingTranscriber {
    fn transcribe(&self, samples: &[f32]) -> Result<String, TranscribeError> {
        if samples.is_empty() {
            return Err(TranscribeError::AudioFormat(
                "Empty audio buffer".to_string(),
            ));
        }

        let mut batch = self.batch.lock().map_err(|e| {
            TranscribeError::InferenceFailed(format!("Failed to lock ParakeetUnified mutex: {}", e))
        })?;

        // Reset between batch calls so accumulated state from a prior
        // streaming or batch invocation doesn't bleed in.
        batch.reset();

        let result = batch
            .transcribe_audio(samples.to_vec(), 16000, 1)
            .map_err(|e| {
                TranscribeError::InferenceFailed(format!("ParakeetUnified inference failed: {}", e))
            })?;

        Ok(result.trim().to_string())
    }

    fn transcribe_timed(&self, _samples: &[f32]) -> Result<Vec<TimedSegment>, TranscribeError> {
        // ParakeetUnified does not expose timed segments in the same shape as
        // the batch ParakeetTranscriber. Users who need timed segments should
        // run with [parakeet] streaming = false.
        Err(TranscribeError::InferenceFailed(
            "transcribe_timed is not supported in streaming mode. \
            Set [parakeet] streaming = false for timed segments."
                .to_string(),
        ))
    }

    fn as_streaming(&self) -> Option<&dyn StreamingTranscriber> {
        Some(self)
    }
}

impl StreamingTranscriber for ParakeetStreamingTranscriber {
    fn start_stream(
        &self,
        samples_rx: mpsc::Receiver<Vec<f32>>,
    ) -> Result<StreamHandle, TranscribeError> {
        // Spawn a fresh ParakeetUnified instance for this session.
        let unified =
            ParakeetUnified::from_shared_with_streaming_config(&self.handle, self.streaming_config)
                .map_err(|e| {
                    TranscribeError::InitFailed(format!(
                        "Failed to spawn streaming ParakeetUnified instance: {}",
                        e
                    ))
                })?;

        let (events_tx, events_rx) = mpsc::channel::<StreamingEvent>(64);
        let (cancel_tx, cancel_rx) = oneshot::channel::<()>();
        let segment = SegmentStream::new(unified, self.pause_secs, self.silence_threshold);

        let task = tokio::task::spawn_blocking(move || -> Result<(), TranscribeError> {
            drive_stream(segment, samples_rx, events_tx, cancel_rx);
            Ok(())
        });

        // Map the spawn_blocking JoinHandle to the trait's expected
        // JoinHandle<Result<(), TranscribeError>> shape.
        let task = tokio::spawn(async move {
            match task.await {
                Ok(r) => r,
                Err(join_err) => Err(TranscribeError::InferenceFailed(format!(
                    "Parakeet streaming task panicked: {}",
                    join_err
                ))),
            }
        });

        Ok(StreamHandle {
            events: events_rx,
            cancel: cancel_tx,
            task,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Uses the user's already-downloaded model; never captures a microphone
    /// or sends text to the desktop. Run explicitly with --ignored.
    #[test]
    #[ignore = "requires downloaded parakeet-unified-en-0.6b and ONNX runtime"]
    fn real_model_pause_resume_smoke() {
        let config = ParakeetConfig {
            model: "parakeet-unified-en-0.6b".into(),
            streaming: true,
            ..Default::default()
        };
        let transcriber = ParakeetStreamingTranscriber::new(&config).unwrap();
        let decoder = ParakeetUnified::from_shared_with_streaming_config(
            &transcriber.handle,
            transcriber.streaming_config,
        )
        .unwrap();
        let mut stream = SegmentStream::new(decoder, 1.0, 0.002);
        let mut wav = hound::WavReader::open("tests/fixtures/vad/speech_hello.wav").unwrap();
        assert_eq!(wav.spec().sample_rate, 16000);
        assert_eq!(wav.spec().channels, 1);
        let audio: Vec<f32> = wav
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect();
        let mut events = Vec::new();
        for _ in 0..2 {
            for chunk in audio.chunks(997) {
                stream.push(chunk, &mut |e| events.push(e)).unwrap();
            }
            stream
                .push(&vec![0.0; 24000], &mut |e| events.push(e))
                .unwrap();
        }
        stream.finish(&mut |e| events.push(e)).unwrap();
        let transcript = text(&events);
        eprintln!("Pause/resume transcript: {transcript:?}");
        assert!(stream.segment_id >= 2);
        for id in [0, 1] {
            assert!(events.iter().any(|e| matches!(e,
                StreamingEvent::Partial { text, segment_id } | StreamingEvent::Final { text, segment_id }
                    if *segment_id == id && !text.trim().is_empty()
            )));
        }
        assert!(!transcript.trim().is_empty());
    }

    #[derive(Default)]
    struct FakeDecoder {
        resets: usize,
        samples: usize,
        tail: String,
        fail_flush: bool,
    }

    impl SegmentDecoder for FakeDecoder {
        fn chunk(&mut self, audio: &[f32]) -> Result<String, TranscribeError> {
            self.samples += audio.len();
            Ok(if audio.iter().any(|s| *s > 0.002) {
                "go"
            } else {
                ""
            }
            .into())
        }
        fn flush_tail(&mut self) -> Result<String, TranscribeError> {
            if self.fail_flush {
                return Err(TranscribeError::InferenceFailed(
                    "test flush failure".into(),
                ));
            }
            Ok(self.tail.clone())
        }
        fn reset_context(&mut self) {
            self.resets += 1;
        }
    }

    fn stream() -> SegmentStream<FakeDecoder> {
        SegmentStream::new(FakeDecoder::default(), 1.0, 0.002)
    }

    #[tokio::test]
    async fn cancellation_wakes_idle_audio_receiver_and_does_not_flush_tail() {
        for drop_sender in [false, true] {
            let (audio_tx, audio_rx) = mpsc::channel(4);
            let (events_tx, mut events_rx) = mpsc::channel(64);
            let (cancel_tx, cancel_rx) = oneshot::channel();
            let mut segment = stream();
            segment.decoder.tail = "must not type".into();
            let task = tokio::task::spawn_blocking(move || {
                drive_stream(segment, audio_rx, events_tx, cancel_rx)
            });
            audio_tx.send(vec![0.003; 320]).await.unwrap();
            assert!(matches!(
                events_rx.recv().await,
                Some(StreamingEvent::Partial { .. })
            ));
            if drop_sender {
                drop(cancel_tx);
            } else {
                cancel_tx.send(()).unwrap();
            }
            let event = tokio::time::timeout(std::time::Duration::from_secs(2), events_rx.recv())
                .await
                .unwrap();
            assert!(matches!(event, Some(StreamingEvent::Ended)));
            task.await.unwrap();
            assert!(events_rx.recv().await.is_none());
            drop(audio_tx);
        }
    }

    #[tokio::test]
    async fn graceful_eof_flushes_tail_then_ends() {
        let (audio_tx, audio_rx) = mpsc::channel(4);
        let (events_tx, mut events_rx) = mpsc::channel(64);
        let (_cancel_tx, cancel_rx) = oneshot::channel();
        let mut segment = stream();
        segment.decoder.tail = "!".into();
        audio_tx.send(vec![0.003; 320]).await.unwrap();
        drop(audio_tx);
        tokio::task::spawn_blocking(move || drive_stream(segment, audio_rx, events_tx, cancel_rx))
            .await
            .unwrap();
        assert!(
            matches!(events_rx.recv().await, Some(StreamingEvent::Partial { text, .. }) if text == "go")
        );
        assert!(
            matches!(events_rx.recv().await, Some(StreamingEvent::Final { text, .. }) if text == "!")
        );
        assert!(matches!(
            events_rx.recv().await,
            Some(StreamingEvent::Ended)
        ));
        assert!(events_rx.recv().await.is_none());
    }

    fn text(events: &[StreamingEvent]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                StreamingEvent::Partial { text, .. } | StreamingEvent::Final { text, .. } => {
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn pause_resets_once_then_resumes_with_new_id_and_spacing() {
        let mut stream = stream();
        stream.decoder.tail = "!".into();
        let mut events = Vec::new();
        let mut emit = |e| events.push(e);
        stream.push(&vec![0.0; 32000], &mut emit).unwrap();
        assert_eq!(stream.decoder.resets, 0);
        stream.push(&[0.003; 320], &mut emit).unwrap();
        stream.push(&vec![0.0; 15680], &mut emit).unwrap();
        assert_eq!(stream.decoder.resets, 0);
        stream.push(&[0.0; 320], &mut emit).unwrap();
        assert_eq!(stream.decoder.resets, 1);
        stream.push(&vec![0.0; 48000], &mut emit).unwrap();
        assert_eq!(stream.decoder.resets, 1);
        stream.push(&[0.003; 320], &mut emit).unwrap();
        stream.push(&vec![0.0; 16000], &mut emit).unwrap();
        assert_eq!(stream.decoder.resets, 2);
        assert_eq!(text(&events), "go! go!");
        let finals: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                StreamingEvent::Final { segment_id, .. } => Some(*segment_id),
                _ => None,
            })
            .collect();
        assert_eq!(finals, [0, 1]);
    }

    #[test]
    fn quiet_endings_restart_pause_timer_and_identical_deltas_are_preserved() {
        let mut stream = stream();
        let mut events = Vec::new();
        let mut emit = |e| events.push(e);
        stream.push(&[0.003; 320], &mut emit).unwrap();
        stream.push(&vec![0.0; 12800], &mut emit).unwrap();
        stream.push(&[0.0021; 320], &mut emit).unwrap();
        stream.push(&vec![0.0; 12800], &mut emit).unwrap();
        assert_eq!(stream.decoder.resets, 0);
        stream.push(&[0.0; 3200], &mut emit).unwrap();
        assert_eq!(stream.decoder.resets, 1);
        assert_eq!(text(&events), "gogo");
    }

    #[test]
    fn transport_chunk_boundaries_do_not_change_endpoints_or_drop_audio() {
        let mut audio = vec![0.003; 320];
        audio.extend(vec![0.0; 16000]);
        audio.extend(vec![0.003; 113]);
        let mut a = stream();
        let mut b = stream();
        let mut ea = Vec::new();
        let mut eb = Vec::new();
        a.push(&audio, &mut |e| ea.push(e)).unwrap();
        for part in audio.chunks(137) {
            b.push(part, &mut |e| eb.push(e)).unwrap();
        }
        a.finish(&mut |e| ea.push(e)).unwrap();
        b.finish(&mut |e| eb.push(e)).unwrap();
        assert_eq!(a.decoder.samples, audio.len());
        assert_eq!(b.decoder.samples, audio.len());
        assert_eq!(text(&ea), "go go");
        assert_eq!(text(&ea), text(&eb));
        assert_eq!(a.decoder.resets, b.decoder.resets);
    }

    #[test]
    fn zero_pause_disables_reset_and_flush_error_does_not_replay_partials() {
        let mut stream = SegmentStream::new(FakeDecoder::default(), 0.0, 0.002);
        let mut events = Vec::new();
        stream.push(&[0.003; 640], &mut |e| events.push(e)).unwrap();
        stream
            .push(&vec![0.0; 32000], &mut |e| events.push(e))
            .unwrap();
        assert_eq!(stream.decoder.resets, 0);
        stream.decoder.fail_flush = true;
        assert!(stream.finish(&mut |e| events.push(e)).is_err());
        assert_eq!(text(&events), "gogo");
    }

    #[test]
    fn endpoint_config_defaults_and_validation() {
        let config: ParakeetConfig = toml::from_str("model = 'custom'").unwrap();
        assert_eq!(config.streaming_pause_secs, 1.0);
        assert_eq!(config.streaming_silence_threshold, 0.002);
        assert!(validate_endpoint_config(&config).is_ok());
        for pause in [-1.0, f32::NAN, f32::INFINITY] {
            let invalid = ParakeetConfig {
                streaming_pause_secs: pause,
                ..config.clone()
            };
            assert!(validate_endpoint_config(&invalid).is_err());
        }
        for threshold in [0.0, -0.1, 1.1, f32::NAN, f32::INFINITY] {
            let invalid = ParakeetConfig {
                streaming_silence_threshold: threshold,
                ..config.clone()
            };
            assert!(validate_endpoint_config(&invalid).is_err());
        }
    }

    /// Smoke test that the type and trait signatures compile. Real model-
    /// driven tests require ONNX runtime + a downloaded model and live in
    /// the integration test suite gated behind `--features parakeet`.
    #[test]
    fn streaming_config_validation_rejects_zero_chunk() {
        let cfg = UnifiedStreamingConfig {
            chunk_secs: 0.0,
            left_context_secs: 1.0,
            right_context_secs: 0.5,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn streaming_config_validation_accepts_defaults() {
        let cfg = UnifiedStreamingConfig::default();
        assert!(cfg.validate().is_ok());
    }

    /// Regression for #442: enabling streaming on a known non-streaming model
    /// (e.g. `parakeet-tdt-0.6b-v3`) must fail early in `new()` with a
    /// message that names the model and points the user at the
    /// streaming-compatible alternative, instead of falling through to an
    /// opaque ONNX Gather shape error at the first audio chunk.
    #[test]
    fn new_rejects_streaming_on_known_incompatible_model() {
        let cfg = ParakeetConfig {
            model: "parakeet-tdt-0.6b-v3".to_string(),
            streaming: true,
            ..ParakeetConfig::default()
        };
        // Can't use `.expect_err`: ParakeetStreamingTranscriber wraps
        // parakeet-rs types that don't derive Debug.
        let err = match ParakeetStreamingTranscriber::new(&cfg) {
            Ok(_) => panic!("streaming on a non-streaming model should error before model load"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("parakeet-tdt-0.6b-v3"),
            "error must name the configured model: {msg}"
        );
        assert!(
            msg.contains(crate::setup::model::DEFAULT_PARAKEET_STREAMING_MODEL),
            "error must name the streaming-compatible alternative: {msg}"
        );
        assert!(
            msg.contains("streaming = false") || msg.contains("voxtype setup model"),
            "error must offer at least one concrete fix: {msg}"
        );
    }
}
