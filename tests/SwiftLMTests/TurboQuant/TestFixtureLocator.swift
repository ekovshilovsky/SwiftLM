// Resolves filesystem paths to large local test fixtures (TurboQuant-
// converted models, multi-gigabyte safetensors checkpoints) that cannot
// reasonably be checked into the repository. Each fixture's path is
// sourced from a dedicated environment variable so the tests stay
// portable across machines and CI environments — no hardcoded
// per-developer paths in the suite. Tests that depend on a fixture
// skip cleanly when its environment variable is unset.
//
// Convention: each fixture has an accessor that returns the URL or
// `nil`, and a `require...()` accessor that returns the URL or throws
// `XCTSkip`. Most tests want the require-form so they can write
// `let fixtureRoot = try TurboQuantTestFixtures.requireQwenCoder3B()`
// at the top of a test method and let the failure path skip the test
// without further branching.
//
// The fixture-specific accessor naming (e.g. `requireQwenCoder3B`) is
// deliberate first-iteration scaffolding while only one converted
// fixture is in active use. As additional fixtures land, this enum
// gains parallel accessors and the test base classes generalise to
// loop over every available fixture; the model code itself is already
// architecture-driven from `config.json` and does not need changes
// when a new fixture is added.

import Foundation
import XCTest

enum TurboQuantTestFixtures {

    /// Local path to the TurboQuant-converted Qwen2.5-Coder-3B-TQ8
    /// fixture, sourced from the `TQ_FIXTURE_DIR` environment
    /// variable. Returns `nil` when the variable is unset or empty so
    /// callers can skip cleanly.
    static var qwenCoder3B: URL? {
        let env = ProcessInfo.processInfo.environment["TQ_FIXTURE_DIR"]
        guard let path = env, !path.isEmpty else { return nil }
        return URL(fileURLWithPath: path)
    }

    /// Returns the Qwen2.5-Coder-3B-TQ8 fixture root URL after
    /// verifying the `tq_shard_metadata.json` sidecar is present, or
    /// throws `XCTSkip` with a message guiding the caller to set
    /// `TQ_FIXTURE_DIR` if the fixture is unavailable.
    static func requireQwenCoder3B() throws -> URL {
        guard let root = qwenCoder3B else {
            throw XCTSkip(
                "TQ_FIXTURE_DIR is not set. Export it pointing at a " +
                "TurboQuant-converted Qwen2.5-Coder-3B-TQ8 directory to " +
                "run TurboQuant fixture-dependent tests."
            )
        }
        let sidecar = root.appendingPathComponent("tq_shard_metadata.json")
        if !FileManager.default.fileExists(atPath: sidecar.path) {
            throw XCTSkip(
                "TurboQuant fixture not present at \(root.path). " +
                "Verify TQ_FIXTURE_DIR points at a directory containing " +
                "tq_shard_metadata.json."
            )
        }
        return root
    }
}
