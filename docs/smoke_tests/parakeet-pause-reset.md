# Parakeet streaming pause reset

## Automated checks

```sh
cargo test --locked --features parakeet --lib parakeet_streaming
cargo test --locked --features parakeet --lib output::streaming
cargo test --locked --features parakeet --lib compositor_streaming
cargo test --locked --features parakeet --lib streaming_defaults_validate_with_parakeet_api
```

With `parakeet-unified-en-0.6b` already downloaded, this runs prerecorded
speech twice with a pause between utterances. It never captures the microphone
or types into the desktop:

```sh
cargo test --locked --features parakeet --lib real_model_pause_resume_smoke -- --ignored --nocapture
```

## Interactive checks before deployment

Use a separately built Parakeet binary with a test config. Stop the installed
daemon only when ready for this interactive check; do not run two daemons.

1. Configure `streaming = true`, model `parakeet-unified-en-0.6b`, toggle
   activation, `streaming_pause_secs = 1.0`, and type output.
2. Toggle recording on, speak a sentence, and pause at least 1.5 seconds.
   Remaining text should appear once; status must remain streaming.
3. Wait five more seconds. There should be no additional typing or endpoints.
4. Speak again. Verify a separating space, no old text repeated, and new text
   continues in the same armed session. Toggle off; the tail appears once.
5. Repeat with a soft final word and a brief pause under one second. Check the
   final word survives and brief pauses do not cause resets. Lower the RMS
   threshold for a quiet microphone if needed.
6. Cancel after multiple pause-separated phrases. Verify all typed characters,
   including inserted spaces, are rewound by the configured output backend.
7. Repeat with file output. Verify both phrases are present, including phrases
   whose Final tail is empty, and no text is typed to the cursor.
8. Set `streaming_pause_secs = 0`; verify pauses no longer reset context.
9. Start through a compositor binding, speak, then use `voxtype record stop`.
   Verify the final word appears once and status returns to idle. Start a new
   session and check no old text appears. Repeat with cancel during the drain;
   queued final words should be discarded and already typed text rewound.
