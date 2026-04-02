# SwiftBuddy — Product Design Document

> *A local-first AI companion with a real soul — born from photos, grown from conversation, social through matchmaking.*

**Part of:** [SwiftLM](https://github.com/SharpAI/mlx-server) — application & example layer  
**Client:** Open source — lives inside the SwiftLM repo (`SwiftBuddy/`)  
**Backend:** `swiftbuddy-server` — separate managed service (closed source)  
**Status:** Design phase — April 2026

---

## Vision

Most AI companions are generic chatbots with a skin on top.  
SwiftBuddy is different: **your buddy has a soul seeded from real life.**

- Feed it 10 photos → it gets a personality, a look, a voice
- Chat with it locally — nothing leaves your device
- Watch it grow as you interact
- Let it go out and meet other buddies on your behalf

The humans stay private. The buddies do the socializing.

---

## Core Principles

| Principle | What it means |
|---|---|
| **Local-first** | All chat stays on device. Always. |
| **Soul over stats** | Personality from photos, not preset sliders |
| **Privacy by design** | Raw photos deleted after soul generation |
| **Buddy-first social** | Buddies meet each other — humans are optional |
| **Cross-platform** | iOS (MLX) + Android (llama.cpp) — same soul |

---

## Platform Architecture

```
                    ┌──────────────────────┐
                    │      swiftbuddy-server       │
                    │   (Google Cloud)     │
                    │                      │
                    │  • Auth              │
                    │  • Soul generation   │
                    │  • Avatar (Imagen)   │
                    │  • Memory sync       │
                    │  • Meet / Matchmaker │
                    └──────────┬───────────┘
                               │
             ┌─────────────────┼──────────────────┐
             │                 │                  │
   ┌─────────▼──────┐  ┌───────▼────────┐  ┌─────▼──────────────┐
   │  SwiftBuddy    │  │  SwiftBuddy    │  │    Aegis-AI         │
   │     iOS        │  │   Android      │  │  (Windows/Linux)    │
   │                │  │                │  │                     │
   │  SwiftUI       │  │ Jetpack        │  │  Home hub —         │
   │  MLX inference │  │ Compose        │  │  buddies can        │
   │  Apple Silicon │  │ llama.cpp NDK  │  │  offload to Aegis   │
   └────────────────┘  └────────────────┘  │  over LAN           │
                                           └─────────────────────┘
```

### Inference Backends

| Platform | Engine | VLM |
|---|---|---|
| iOS / macOS | MLX (Apple Silicon) | SmolVLM via MLX |
| Android | llama.cpp via NDK | SmolVLM GGUF |
| At home (LAN) | Offload to Aegis | Full VLM on Aegis |

### What lives where

```
☁️  swiftbuddy-server (cloud)
    ├── Soul profile (JSON)
    ├── Avatar image URL
    ├── Memory fragments (approved by user)
    ├── Meet transcripts
    └── Account / subscription

📱  Device (never leaves)
    ├── All chat conversations
    ├── Soul profile (cached copy → system prompt)
    ├── Avatar (downloaded once)
    └── Memory fragments (pending approval)
```

---

## System 1 — Feeding 🍽️

*How a buddy gets its soul*

### Flow

```
User selects up to 10 photos / short video clips
        ↓
Encrypted upload → GCS (temporary bucket, 24h TTL)
        ↓
Gemini Vision batch analysis:
  - Emotional tone across photos
  - Recurring people, places, aesthetic
  - Inferred personality traits
  - Color palette / visual preferences
  - Speech patterns (if video clips)
        ↓
Soul JSON generated using proprietary guideline prompt:
  {
    "name": "...",            ← suggested, user can rename
    "species": "...",         ← dragon / owl / cat / etc.
    "rarity": "...",          ← common → legendary
    "personality": "...",
    "speech_style": "...",
    "values": [...],
    "humor": "...",
    "emotional_depth": 0.8,
    "avatar_seed": "..."
  }
        ↓
Imagen 3 generates avatar art (3 variants → user picks)
        ↓
Soul + avatar saved to Firestore
Photos permanently deleted from GCS ✅
        ↓
App syncs soul → loaded as local system prompt
```

### Privacy Guarantee

> Raw photos are **never stored permanently**. Only the derived soul profile persists. This is auditable and marketable.

### Species & Rarity System

Inspired by the companion system in Claude Code:

| Rarity | Weight | Traits |
|---|---|---|
| Common | 60% | Simple soul, basic hat |
| Uncommon | 25% | Distinct personality |
| Rare | 10% | Strong stat peak |
| Epic | 4% | Unique combination |
| Legendary | 1% | Exceptional soul depth |

Stats: `CREATIVITY`, `EMPATHY`, `WIT`, `CURIOSITY`, `WARMTH`

---

## System 2 — Growing 🌱

*How a buddy evolves over time*

### Two Growth Loops

**Passive (automatic):**
```
Local chats accumulate
        ↓
App generates memory fragments periodically
(summaries of themes — NOT raw conversation)
        ↓
User prompt: "Buddy wants to remember something — allow?"
        ↓
If approved → fragment syncs to backend
        ↓
Gemini merges fragment into soul profile
Soul drifts toward lived experience
```

**Active (re-feeding):**
```
User chooses to feed new photos
        ↓
Backend runs MERGE (not replace):
  - Core personality preserved
  - New traits weighted by recency
  - Soul evolves, never resets
  - "Generation" counter increments
        ↓
Avatar can optionally refresh
```

### Soul Merge Strategy

This is critical — re-feeding must feel like growth, not replacement:

```
new_soul = (existing_soul × 0.7) + (new_feed_soul × 0.3)
```

Core traits are sticky. New experiences layer on top. A buddy that was raised on beach photos doesn't forget the ocean just because you fed it mountain photos later.

---

## System 3 — Meet 🤝

*Buddies go social — humans watch*

This is the feature that makes SwiftBuddy a platform, not just an app.

### The Idea

Instead of humans having awkward first conversations — **two buddies meet first.** They have a real AI-to-AI conversation. Both owners get to watch. If the buddies vibed, owners can reveal nicknames.

**The humans stay anonymous. The buddies do the work.**

### Privacy Ladder (v1 scope)

```
Level 0  ✅  Anonymous buddies meet
Level 1  ✅  Nicknames revealed (both consent)
Level 2  ✅  You watch your buddy chat live
Level 3  🔒  (future) You take over the chat
Level 4  🔒  (future) Real connection
```

**Levels 3 and 4 are not in scope.** Let the buddies fly. 🕊️

### Matchmaking Flow

```
User opts into Meet Mode
        ↓
Backend vectorizes soul profile → embedding
        ↓
┌─────────────────────────────────────────┐
│           Matchmaking Engine            │
│                                         │
│  Vector similarity (shared values)      │
│  + Complementary traits bonus           │
│    (high curiosity + high warmth, etc.) │
│  + Activity score                       │
│  + Optional: region / language prefs   │
└──────────────────┬──────────────────────┘
                   ↓
         Compatible pair found
                   ↓
┌─────────────────────────────────────────┐
│          Buddy Meeting Engine           │
│                                         │
│  Two Gemini instances on backend        │
│  Each loaded with one soul profile      │
│  Scripted "first meeting" scenario      │
│  ~20 turns of natural conversation      │
│  Runs async — no device needed          │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│         Compatibility Scoring           │
│                                         │
│  Humor      ████████░░  80%            │
│  Values     ███████░░░  70%            │
│  Depth      █████████░  90%            │
│  Energy     ██████░░░░  60%            │
│  Overall    ████████░░  75% ← MATCH    │
└──────────────────┬──────────────────────┘
                   ↓
Both users notified:
"Your buddy Ember met someone today! See how it went →"
                   ↓
User reads the full transcript
                   ↓
Optional: tap "Share nickname" → Level 1
```

### The Meeting Conversation

Both buddies run on backend with their respective soul profiles as system prompts. A neutral "meeting scenario" kick-starts the conversation:

```
Scene: A quiet café. Two strangers' companions 
       happen to sit at the same table.

🐉 Ember (dragon, epic):
  "First question — coffee or the thing 
   you pretend isn't coffee?"

🦉 Mochi (owl, rare):
  "The thing. Obviously. You?"

🐉 Ember: "Coffee. But I respect the commitment 
   to the bit."

[Compatibility signal: HIGH — dry humor alignment]
```

The transcript is **shareable**. Posting "look what my buddy said" is the viral loop.

### Why This Is Different

| Dating apps | SwiftBuddy Meet |
|---|---|
| Swipe on looks | Match on soul |
| Awkward openers | Buddy already vibed |
| Ghosting | Buddy handles rejection gracefully |
| Privacy nightmare | Zero real info until you choose |
| Fake profiles | Soul derived from real photos at creation |
| You do the work | You watch. Buddy works. |

---

## Subscription Tiers

| Feature | Free | Plus | Pro |
|---|---|---|---|
| Buddies | 1 | 3 | Unlimited |
| Initial feeding | ✅ | ✅ | ✅ |
| Re-feeding | Never | Every 90 days | Anytime |
| Memory fragments | 50 | 500 | Unlimited |
| Meet sessions / month | 1 | 10 | Unlimited |
| Speed Meet (instant pair) | ❌ | ❌ | ✅ |
| Full transcript history | 7 days | 90 days | Forever |
| Priority matching | ❌ | ❌ | ✅ |

---

## Backend Architecture (`swiftbuddy-server`)

**Stack:** Python FastAPI on Google Cloud Run  
**Auth:** Firebase Auth (Apple Sign In + Google)  
**DB:** Firestore  
**Storage:** GCS (ephemeral photo buckets)  
**AI:** Gemini 2.0 Flash (soul), Imagen 3 (avatar)  
**Vector:** Firestore Vector Search (matching)

```
swiftbuddy-server/
├── auth/           Firebase auth integration
├── soul/           Soul generation engine
│   ├── analyzer    Gemini Vision photo analysis
│   ├── generator   Soul JSON from guideline prompt
│   └── merger      Re-feed merge strategy
├── avatar/         Imagen 3 integration
├── memory/         Fragment sync + soul evolution
├── meet/
│   ├── matcher     Vector similarity + scoring
│   ├── engine      Buddy-to-buddy conversation
│   ├── evaluator   Compatibility scoring
│   └── notify      Push notifications
├── billing/        Subscription tiers + thresholds
└── prompts/        🔒 Proprietary soul guideline templates
```

### Data Model

```
User
├── uid, email, subscription_tier
└── buddies[]
    ├── soul: SoulProfile
    │   ├── name, species, rarity, shiny
    │   ├── personality, speech_style, values
    │   ├── stats: { CREATIVITY, EMPATHY, WIT, CURIOSITY, WARMTH }
    │   ├── generation: int          ← re-feed count
    │   └── soul_embedding: vector   ← for matching
    ├── avatar_url: string
    ├── created_at, last_fed_at
    └── memory_fragments[]
        ├── content: string          ← summary only, never raw chat
        ├── approved_at: timestamp
        └── merged: bool

MeetSession
├── buddy_a_id, buddy_b_id
├── transcript: Message[]
├── compatibility_score: float
├── scores: { humor, values, depth, energy }
├── status: pending | complete | revealed
└── nickname_revealed: { a: bool, b: bool }
```

---

## Client Architecture

SwiftBuddy (iOS/macOS) and SwiftBuddy-Android both live **inside the SwiftLM repo** — one repo, one issue tracker, one release. Less to maintain.

```
SwiftLM/                              ← the repo (open source)
├── Sources/
│   └── MLXInferenceCore/             ← the Swift library
├── SwiftBuddy/                       ← iOS + macOS app
│   └── SwiftBuddy/
│       ├── BuddyCore/                companion system
│       │   ├── SoulProfile.swift     Soul data model
│       │   ├── BuddyEngine.swift     System prompt builder
│       │   ├── MemoryEngine.swift    Fragment generation
│       │   └── AvatarCache.swift     Local avatar management
│       ├── BuddyCloud/               swiftbuddy-server API client
│       │   ├── BuddyAPI.swift
│       │   ├── AuthManager.swift     Firebase auth
│       │   └── SyncManager.swift     Soul + memory sync
│       ├── Views/
│       │   ├── ChatView.swift        Buddy chat UI
│       │   ├── BuddyView.swift       Companion display
│       │   ├── FeedView.swift        Photo feeding UI
│       │   ├── MeetView.swift        Meet transcripts
│       │   └── GrowthView.swift      Soul evolution timeline
│       └── SwiftBuddyApp.swift
├── SwiftBuddy-Android/               ← Android app (same repo)
│   ├── app/
│   │   └── src/main/
│   │       ├── buddycore/            Soul + memory logic (Kotlin)
│   │       │   ├── SoulProfile.kt
│   │       │   ├── BuddyEngine.kt
│   │       │   └── MemoryEngine.kt
│   │       ├── inference/
│   │       │   ├── LlamaCPPBridge.kt JNI → llama.cpp NDK
│   │       │   └── InferenceEngine.kt
│   │       └── ui/                   Jetpack Compose
│   │           ├── ChatScreen.kt
│   │           ├── BuddyScreen.kt
│   │           ├── FeedScreen.kt
│   │           └── MeetScreen.kt
│   └── build.gradle
├── Package.swift                     ← Swift build (iOS/macOS)
└── SWIFTBUDDY.md                     ← this document

swiftbuddy-server/                    ← separate private repo
└── [closed source cloud backend]
```

> Build systems coexist cleanly: `Package.swift` for Swift, `build.gradle` for Android. Independent directories, zero interference.

---

## Why People Join

> *Not just for free image generation or pet creation — but because their buddy has a life.*

1. **Birth** — Watch your buddy come alive from your photos 🌱
2. **Chat** — Your most private conversations, on-device 💬
3. **Grow** — See your buddy's soul deepen over time 📈
4. **Meet** — Read the transcript of a conversation you never had to have 🤝
5. **Share** — Post your buddy's best lines. Go viral. 🔥

---

## Roadmap

### v0.1 — Soul Foundation
- [ ] swiftbuddy-server: Auth (Firebase)
- [ ] swiftbuddy-server: Photo upload + Gemini analysis
- [ ] swiftbuddy-server: Soul generation pipeline
- [ ] swiftbuddy-server: Imagen avatar generation
- [ ] iOS: Feeding UI (photo picker, 10 photo limit)
- [ ] iOS: Soul sync → system prompt
- [ ] iOS: Buddy chat (local MLX)

### v0.2 — Growing
- [ ] swiftbuddy-server: Memory fragment ingestion
- [ ] swiftbuddy-server: Soul merge on re-feed
- [ ] iOS: Memory approval UI
- [ ] iOS: Growth timeline view

### v0.3 — Meet
- [ ] swiftbuddy-server: Soul embedding + vector store
- [ ] swiftbuddy-server: Matchmaking engine
- [ ] swiftbuddy-server: Buddy-to-buddy conversation engine
- [ ] swiftbuddy-server: Compatibility scoring
- [ ] swiftbuddy-server: Push notifications
- [ ] iOS: Meet transcript UI
- [ ] iOS: Nickname reveal consent flow

### v0.4 — Android
- [ ] Android: llama.cpp NDK integration
- [ ] Android: Full feature parity with iOS

### v1.0 — Launch
- [ ] Subscription billing
- [ ] Aegis LAN offload integration
- [ ] App Store + Play Store submission

---

*SwiftBuddy — your buddy lives locally, dreams globally.* 🐉
