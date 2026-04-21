# TurboQuant test suite notes

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
