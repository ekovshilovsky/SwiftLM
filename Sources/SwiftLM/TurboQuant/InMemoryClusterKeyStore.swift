// Volatile cluster key store. Exists so tests and development flows
// can exercise code paths that depend on a ClusterKeyStore without
// touching the filesystem or the Keychain. State lives in one instance
// and evaporates when the instance is released.
//
// The entire type is wrapped in `#if DEBUG` so it does not ship in
// release binaries. Production code paths have no legitimate reason to
// accept a volatile store — an accidental reference from production
// code should fail to compile in release rather than silently run
// against an in-memory fake. Tests compile in debug by default, so
// this imposes no test-side cost.

#if DEBUG

import Foundation

/// In-process cluster record storage. Thread-safe via a single lock;
/// the serialization is coarse-grained but correct, and the store is
/// not a hot path (saves happen at cluster-join time, loads at process
/// start). Debug-only — not present in release builds.
public final class InMemoryClusterKeyStore: ClusterKeyStore, @unchecked Sendable {
    private let lock = NSLock()
    private var record: ClusterRecord?

    public init(initial: ClusterRecord? = nil) {
        self.record = initial
    }

    public func load() throws -> ClusterRecord? {
        lock.lock(); defer { lock.unlock() }
        return record
    }

    public func save(_ record: ClusterRecord) throws {
        lock.lock(); defer { lock.unlock() }
        self.record = record
    }

    public func delete() throws {
        lock.lock(); defer { lock.unlock() }
        self.record = nil
    }
}

#endif // DEBUG
