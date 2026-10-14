//! Parakeet engine configuration.

use serde::{Deserialize, Serialize};

use super::super::default_on_demand_loading;

/// Parakeet model architecture type
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ParakeetModelType {
    /// CTC (Connectionist Temporal Classification) - faster, character-level output
    Ctc,
    /// TDT (Token-Duration-Transducer) - recommended, proper punctuation and word boundaries
    #[default]
    Tdt,
}

/// Parakeet speech-to-text configuration (ONNX-based, alternative to Whisper)
/// Requires: cargo build --features parakeet
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ParakeetConfig {
    /// Path to model directory containing ONNX model files
    /// For TDT: encoder-model.onnx, decoder_joint-model.onnx, vocab.txt
    /// For CTC: model.onnx, tokenizer.json
    pub model: String,

    /// Model architecture type: "tdt" (default, recommended) or "ctc"
    /// Auto-detected from model directory structure if not specified
    #[serde(default)]
    pub model_type: Option<ParakeetModelType>,

    /// Load model on-demand when recording starts (true) or keep loaded (false)
    #[serde(default = "default_on_demand_loading")]
    pub on_demand_loading: bool,

    /// Use the cache-aware streaming pipeline (parakeet-rs `ParakeetUnified`)
    /// instead of the batch CTC/TDT models. When true, voxtype emits live
    /// text deltas during recording and flushes the tail on pauses or stop.
    /// Requires a streaming-capable model directory
    /// (TDT v3 family with `tokenizer.model`).
    /// Default: false (batch pipeline, identical to pre-streaming behavior).
    #[serde(default)]
    pub streaming: bool,

    /// Streaming chunk length in seconds. Smaller = lower latency, more
    /// inference overhead. Maps to `UnifiedStreamingConfig::chunk_secs`.
    #[serde(default = "default_streaming_chunk_secs")]
    pub streaming_chunk_secs: f32,

    /// Streaming left context in seconds. Maps to
    /// `UnifiedStreamingConfig::left_context_secs`.
    #[serde(default = "default_streaming_left_context_secs")]
    pub streaming_left_context_secs: f32,

    /// Streaming right context in seconds. Maps to
    /// `UnifiedStreamingConfig::right_context_secs`.
    #[serde(default = "default_streaming_right_context_secs")]
    pub streaming_right_context_secs: f32,

    /// Finalize and reset streaming decoder context after this much quiet
    /// audio following speech. Zero disables pause segmentation.
    #[serde(default = "default_streaming_pause_secs")]
    pub streaming_pause_secs: f32,

    /// RMS amplitude below which a 20ms frame counts as quiet (16kHz PCM).
    /// Lower this for quiet microphones; raise it for background noise.
    #[serde(default = "default_streaming_silence_threshold")]
    pub streaming_silence_threshold: f32,
}

fn default_streaming_pause_secs() -> f32 {
    1.0
}

fn default_streaming_silence_threshold() -> f32 {
    0.002
}

fn default_streaming_chunk_secs() -> f32 {
    0.32
}

fn default_streaming_left_context_secs() -> f32 {
    5.6
}

fn default_streaming_right_context_secs() -> f32 {
    0.32
}

impl Default for ParakeetConfig {
    fn default() -> Self {
        Self {
            model: "parakeet-tdt-0.6b-v3".to_string(),
            model_type: None, // Auto-detect
            on_demand_loading: false,
            streaming: false,
            streaming_chunk_secs: default_streaming_chunk_secs(),
            streaming_left_context_secs: default_streaming_left_context_secs(),
            streaming_right_context_secs: default_streaming_right_context_secs(),
            streaming_pause_secs: default_streaming_pause_secs(),
            streaming_silence_threshold: default_streaming_silence_threshold(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn test_parse_parakeet_model_type_tdt() {
        let toml_str = r#"
            engine = "parakeet"

            [hotkey]
            key = "SCROLLLOCK"

            [audio]
            device = "default"
            sample_rate = 16000
            max_duration_secs = 60

            [whisper]
            model = "base.en"
            language = "en"

            [output]
            mode = "type"

            [parakeet]
            model = "parakeet-tdt-0.6b-v3"
            model_type = "tdt"
        "#;

        let config: Config = toml::from_str(toml_str).unwrap();
        let parakeet = config.parakeet.unwrap();
        assert_eq!(parakeet.model, "parakeet-tdt-0.6b-v3");
        assert_eq!(parakeet.model_type, Some(ParakeetModelType::Tdt));
    }

    #[test]
    fn test_parse_parakeet_model_type_ctc() {
        let toml_str = r#"
            engine = "parakeet"

            [hotkey]
            key = "SCROLLLOCK"

            [audio]
            device = "default"
            sample_rate = 16000
            max_duration_secs = 60

            [whisper]
            model = "base.en"
            language = "en"

            [output]
            mode = "type"

            [parakeet]
            model = "parakeet-ctc-0.6b"
            model_type = "ctc"
        "#;

        let config: Config = toml::from_str(toml_str).unwrap();
        let parakeet = config.parakeet.unwrap();
        assert_eq!(parakeet.model, "parakeet-ctc-0.6b");
        assert_eq!(parakeet.model_type, Some(ParakeetModelType::Ctc));
    }

    #[test]
    fn test_parakeet_model_type_defaults_to_none_for_auto_detection() {
        let toml_str = r#"
            engine = "parakeet"

            [hotkey]
            key = "SCROLLLOCK"

            [audio]
            device = "default"
            sample_rate = 16000
            max_duration_secs = 60

            [whisper]
            model = "base.en"
            language = "en"

            [output]
            mode = "type"

            [parakeet]
            model = "parakeet-tdt-0.6b-v3"
        "#;

        let config: Config = toml::from_str(toml_str).unwrap();
        let parakeet = config.parakeet.unwrap();
        // model_type should be None (will be auto-detected at runtime)
        assert!(parakeet.model_type.is_none());
    }

    #[test]
    fn test_parakeet_config_default() {
        let config = ParakeetConfig::default();
        assert_eq!(config.model, "parakeet-tdt-0.6b-v3");
        assert!(config.model_type.is_none());
        assert!(!config.on_demand_loading);
    }

    #[cfg(feature = "parakeet")]
    #[test]
    fn streaming_defaults_validate_with_parakeet_api() {
        // Both construction paths must work when users only enable streaming.
        let parsed: ParakeetConfig =
            toml::from_str("model = \"parakeet-unified-en-0.6b\"\nstreaming = true").unwrap();
        for config in [ParakeetConfig::default(), parsed] {
            let validated = parakeet_rs::UnifiedStreamingConfig {
                chunk_secs: config.streaming_chunk_secs,
                left_context_secs: config.streaming_left_context_secs,
                right_context_secs: config.streaming_right_context_secs,
            }
            .validate()
            .expect("streaming defaults must satisfy the decoder's mel-frame alignment");
            assert_eq!(validated.chunk_secs, 0.32);
            assert_eq!(validated.left_context_secs, 5.6);
            assert_eq!(validated.right_context_secs, 0.32);
        }
    }

    #[test]
    fn test_parakeet_model_type_enum_default() {
        // ParakeetModelType defaults to Tdt
        assert_eq!(ParakeetModelType::default(), ParakeetModelType::Tdt);
    }
}
