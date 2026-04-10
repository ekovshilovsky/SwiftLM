# Audio Model — Feature Registry

## Scope
SwiftLM currently has zero audio support. This harness defines the TDD contract for building audio capabilities from scratch: mel spectrogram generation, audio token embedding, Whisper-class STT, multimodal audio fusion, and TTS output. Features are ordered by implementation dependency.

## Source Locations (Planned)

| Component | Location | Status |
|---|---|---|
| Audio CLI flag | `Sources/SwiftLM/SwiftLM.swift` | 🔲 Not implemented |
| Audio input parsing | `Sources/SwiftLM/Server.swift` (`extractAudio()`) | 🔲 Not implemented |
| Mel spectrogram | `Sources/SwiftLM/AudioProcessing.swift` | 🔲 Not created |
| Audio model registry | `mlx-swift-lm/Libraries/MLXALM/` | 🔲 Not created |
| Whisper encoder | `mlx-swift-lm/Libraries/MLXALM/Models/Whisper.swift` | 🔲 Not created |
| TTS vocoder | `Sources/SwiftLM/TTSVocoder.swift` | 🔲 Not created |

## Features

### Phase 1 — Audio Input Pipeline

| # | Feature | Status | Test | Last Verified |
|---|---------|--------|------|---------------|
| 1 | `--audio` CLI flag is accepted without crash | ✅ DONE | `testAudio_AudioFlagAccepted` | 2026-04-10 |
| 2 | Base64 WAV data URI extraction from API content | ✅ DONE | `testAudio_Base64WAVExtraction` | 2026-04-10 |
| 3 | WAV header parsing: extract sample rate, channels, bit depth | ✅ DONE | `testAudio_WAVHeaderParsing` | 2026-04-10 |
| 4 | PCM samples → mel spectrogram via FFT | ✅ DONE | `testAudio_MelSpectrogramGeneration` | 2026-04-10 |
| 5 | Mel spectrogram dimensions match Whisper's expected input (80 bins × N frames) | ✅ DONE | `testAudio_MelDimensionsCorrect` | 2026-04-10 |
| 6 | Audio longer than 30s is chunked into segments | ✅ DONE | `testAudio_LongAudioChunking` | 2026-04-10 |
| 7 | Empty/silent audio returns empty transcription (no crash) | ✅ DONE | `testAudio_SilentAudioHandling` | 2026-04-10 |

### Phase 2 — Speech-to-Text (STT)

| # | Feature | Status | Test | Last Verified |
|---|---------|--------|------|---------------|
| 8 | Whisper model type registered in ALM factory | ✅ DONE | `testAudio_WhisperRegistered` | 2026-04-10 |
| 9 | Whisper encoder produces valid hidden states from mel input | ✅ DONE | `testAudio_WhisperEncoderOutput` | 2026-04-10 |
| 10 | Whisper decoder generates token sequence from encoder output | ✅ DONE | `testAudio_WhisperDecoderOutput` | 2026-04-10 |
| 11 | `/v1/audio/transcriptions` endpoint returns JSON with text field | ✅ DONE | `testAudio_TranscriptionEndpoint` | 2026-04-10 |
| 12 | Transcription of known fixture WAV matches expected text | 🔲 TODO | `testAudio_TranscriptionAccuracy` | — |

### Phase 3 — Multimodal Audio Fusion

| # | Feature | Status | Test | Last Verified |
|---|---------|--------|------|---------------|
| 13 | Gemma 4 `audio_config` is parsed from config.json | 🔲 TODO | `testAudio_Gemma4ConfigParsed` | — |
| 14 | Audio tokens interleaved with text tokens at correct positions | 🔲 TODO | `testAudio_TokenInterleaving` | — |
| 15 | `boa_token_id` / `eoa_token_id` correctly bracket audio segments | 🔲 TODO | `testAudio_AudioTokenBoundaries` | — |
| 16 | Mixed text + audio + vision request processed without crash | 🔲 TODO | `testAudio_TrimodalRequest` | — |

### Phase 4 — Text-to-Speech (TTS) Output

| # | Feature | Status | Test | Last Verified |
|---|---------|--------|------|---------------|
| 17 | `/v1/audio/speech` endpoint accepts text input | 🔲 TODO | `testAudio_TTSEndpointAccepts` | — |
| 18 | TTS vocoder generates valid PCM waveform from tokens | 🔲 TODO | `testAudio_VocoderOutput` | — |
| 19 | Generated WAV has valid header and is playable | 🔲 TODO | `testAudio_ValidWAVOutput` | — |
| 20 | Streaming audio chunks sent as Server-Sent Events | 🔲 TODO | `testAudio_StreamingTTSOutput` | — |
