# TurboQuant test suite notes

## Upstream regression tests

`Integration/UpstreamRegressionTests.swift` guards the behaviors that
upstream SwiftLM depends on: CLI flag surface, standard model
loading path, the three HTTP endpoints (`/health`, `/v1/models`,
`/v1/chat/completions`), and the prompt KV cache.

Two tests run on every `swift test` invocation:
- `testExistingCLIFlagsStillAccepted` spawns `SwiftLM --help` and
  asserts each upstream-documented flag is still in the output.
- `testStandardModelLoadingStillWorks` spawns
  `SwiftLM --model <synthesized dir> --info` against a fabricated
  minimal `config.json` and asserts the partition plan renders. No
  weights required.

Two tests opt in via an environment variable:
- `testExistingAPIEndpointsUnchanged` and
  `testExistingKVCacheStillWorks` launch a real server subprocess
  against a local model and validate the OpenAI-compatible
  response shapes. To enable them:

  ```bash
  swift build --product SwiftLM
  SWIFTLM_TEST_MODEL=/path/to/local/mlx-model swift test \
    --filter UpstreamRegressionTests
  ```

  Without `SWIFTLM_TEST_MODEL` set, those two tests XCTSkip with a
  message pointing to this README. The default contributor
  workflow stays hermetic.

Both groups also require `SwiftLM` to be built (`swift build --product
SwiftLM`); `swift test` does not build executable targets
automatically and the tests XCTSkip cleanly if the binary is
missing.



## KeychainClusterKeyStore — tested via mocked `SecItemClient`

`KeychainClusterKeyStore` wraps macOS's data-protection Keychain
through `Security.framework`'s `SecItem*` C API. The data-protection
Keychain requires a `keychain-access-groups` entitlement on the
calling process; Apple treats that entitlement as *restricted*, so
the ad-hoc signature that `swift test` produces on the xctest bundle
is not permitted to carry it. Apple Mobile File Integrity enforces
this at load time — attempts to inject the entitlement via linker
tricks (`-sectcreate __TEXT __entitlements`) plus an ad-hoc
`codesign` are refused with AMFI error `-424 "The file is adhoc
signed but contains restricted entitlements"`.

Rather than ship a parallel Xcode project and signing apparatus
purely to run a handful of tests against a live Keychain, the test
suite injects a `SecItemClient` protocol boundary and substitutes a
recording in-memory fake. Every branch of the store's save / load /
delete logic is exercised — query attributes, access-group
presence, `kSecUseDataProtectionKeychain` opt-in, accessibility
constants, partial-record handling, and unexpected-status error
propagation are all asserted against the recorded call log.

What the mock *does not* verify is Apple's own SecItem semantics,
which are Apple's test coverage. If you want to verify the live
path against your personal Developer ID on a dev machine, use the
recipe below.

## Running KeychainClusterKeyStore tests against a live Keychain

This is entirely optional — CI and standard contributor workflow
run the mock-based tests and do not need a signed binary.

1. Add a copy of the tests that call the default
   `KeychainClusterKeyStore(service:accessGroup:)` initializer (no
   injected `secItemClient`), so the real `SystemSecItemClient` is
   used.
2. Create a local `entitlements.plist`:
   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
     "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
   <plist version="1.0">
   <dict>
     <key>keychain-access-groups</key>
     <array>
       <string>$(AppIdentifierPrefix)com.turboquant.cluster</string>
     </array>
   </dict>
   </plist>
   ```
3. After `swift build --build-tests`, codesign the xctest bundle
   with your Developer ID:
   ```bash
   codesign --force \
     --sign "Developer ID Application: Your Name (TEAMID)" \
     --entitlements entitlements.plist \
     --options runtime \
     .build/arm64-apple-macosx/debug/SwiftLMPackageTests.xctest
   ```
4. Run the test bundle via `xcrun xctest` (not `swift test`, which
   re-links and re-signs):
   ```bash
   xcrun xctest .build/arm64-apple-macosx/debug/SwiftLMPackageTests.xctest
   ```

Apple's "Personal Team" free-tier signing does not carry the
restricted entitlement on macOS either, so this path specifically
requires a paid Developer account. The mock-based test suite stays
definitive for contributors without one.
