// File-backed cluster key store. The default v1 implementation:
// always-works, no entitlements, no signing, no Keychain. Persists the
// cluster record under `~/Library/Application Support/SwiftLM/cluster/`
// as two files:
//
//   cluster.id   16 bytes, 0644 — public cluster UUID, doubles as the
//                Argon2id salt and is broadcast in Bonjour TXT
//   cluster.key  32 bytes, 0600 — master key (secret)
//
// The parent directory is created at mode 0700 so even if individual
// files are temporarily written with a looser umask-influenced mode
// (during the atomic-write window) the directory itself denies access
// to other local users.
//
// Writes are atomic (tmp file + rename on the same filesystem) and
// followed by an explicit chmod so the final mode is not filtered by
// the process umask. Loads return nil when either file is absent —
// partial state (one file missing) is treated as "no record" so the
// caller re-runs the cluster-join flow rather than using a half-record.

import Foundation

public struct FileClusterKeyStore: ClusterKeyStore {
    public let directory: URL

    /// POSIX modes applied after each write.
    private let idMode: mode_t = 0o644
    private let keyMode: mode_t = 0o600
    private let dirMode: mode_t = 0o700

    public init(directory: URL) {
        self.directory = directory
    }

    /// Default production location under the current user's Application
    /// Support directory.
    public static func `default`() throws -> FileClusterKeyStore {
        let fm = FileManager.default
        let urls = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)
        guard let support = urls.first else {
            throw ClusterKeyStoreError.ioError(
                "Application Support directory unavailable")
        }
        return FileClusterKeyStore(
            directory: support
                .appendingPathComponent("SwiftLM", isDirectory: true)
                .appendingPathComponent("cluster", isDirectory: true)
        )
    }

    private var idPath: URL { directory.appendingPathComponent("cluster.id") }
    private var keyPath: URL { directory.appendingPathComponent("cluster.key") }

    public func load() throws -> ClusterRecord? {
        let fm = FileManager.default
        guard fm.fileExists(atPath: idPath.path),
              fm.fileExists(atPath: keyPath.path) else {
            return nil
        }
        let id = try Data(contentsOf: idPath)
        let key = try Data(contentsOf: keyPath)
        return ClusterRecord(clusterId: id, key: key)
    }

    public func save(_ record: ClusterRecord) throws {
        try ensureDirectory()
        try atomicWrite(data: record.clusterId, to: idPath, mode: idMode)
        try atomicWrite(data: record.key, to: keyPath, mode: keyMode)
    }

    public func delete() throws {
        let fm = FileManager.default
        for url in [idPath, keyPath] where fm.fileExists(atPath: url.path) {
            try fm.removeItem(at: url)
        }
    }

    /// Create or re-tighten the parent directory so its mode is 0700
    /// regardless of the ambient umask at creation time.
    private func ensureDirectory() throws {
        let fm = FileManager.default
        if !fm.fileExists(atPath: directory.path) {
            try fm.createDirectory(
                at: directory,
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: NSNumber(value: dirMode)]
            )
        }
        // Always re-chmod: createDirectory's attributes are advisory on
        // some filesystems, and a pre-existing directory might have been
        // created with a looser mode.
        if chmod(directory.path, dirMode) != 0 {
            throw ClusterKeyStoreError.ioError(
                "chmod(\(directory.path), 0700) failed: \(String(cString: strerror(errno)))"
            )
        }
    }

    /// Atomic write backed by `Data.write(options: .atomic)` which uses
    /// a tmp-file-in-same-dir plus rename, followed by explicit chmod to
    /// override the process umask. The tmp file's brief window with a
    /// umask-filtered mode is not a concern because the parent directory
    /// is 0700 (other local users cannot enter it).
    private func atomicWrite(data: Data, to url: URL, mode: mode_t) throws {
        try data.write(to: url, options: [.atomic])
        if chmod(url.path, mode) != 0 {
            throw ClusterKeyStoreError.ioError(
                "chmod(\(url.path), \(String(mode, radix: 8))) failed: " +
                "\(String(cString: strerror(errno)))"
            )
        }
    }
}
