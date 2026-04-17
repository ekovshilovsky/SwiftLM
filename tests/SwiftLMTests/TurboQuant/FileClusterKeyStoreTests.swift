// FileClusterKeyStore tests. Each test uses a fresh temp directory so
// nothing the suite does touches the user's real Application Support
// tree. Coverage includes the round-trip, permission enforcement on
// both files and the parent directory, partial-state handling, and
// parent-directory auto-creation.

import XCTest
import TurboQuantKit

final class FileClusterKeyStoreTests: XCTestCase {
    private var tempRoot: URL!
    private var store: FileClusterKeyStore!

    override func setUpWithError() throws {
        try super.setUpWithError()
        // Unique per-test so parallel invocations can't collide.
        tempRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("FileClusterKeyStoreTests-\(UUID().uuidString)",
                                    isDirectory: true)
        store = FileClusterKeyStore(directory: tempRoot)
    }

    override func tearDownWithError() throws {
        if let tempRoot, FileManager.default.fileExists(atPath: tempRoot.path) {
            try FileManager.default.removeItem(at: tempRoot)
        }
        try super.tearDownWithError()
    }

    private func sampleRecord() -> ClusterRecord {
        ClusterRecord(
            clusterId: Data(repeating: 0x5A, count: 16),
            key: Data(repeating: 0xC3, count: 32)
        )
    }

    private func posixMode(_ path: String) throws -> mode_t {
        let attrs = try FileManager.default.attributesOfItem(atPath: path)
        let number = try XCTUnwrap(attrs[.posixPermissions] as? NSNumber)
        return mode_t(number.uint16Value)
    }

    // MARK: - Round-trip

    func testSaveAndLoadRoundTrip() throws {
        let record = sampleRecord()
        try store.save(record)

        let loaded = try XCTUnwrap(try store.load())
        XCTAssertEqual(loaded, record)
    }

    func testLoadReturnsNilBeforeAnyWrite() throws {
        XCTAssertNil(try store.load())
    }

    func testSaveOverwritesPriorRecord() throws {
        try store.save(ClusterRecord(
            clusterId: Data(repeating: 0x11, count: 16),
            key: Data(repeating: 0xAA, count: 32)))
        try store.save(ClusterRecord(
            clusterId: Data(repeating: 0x22, count: 16),
            key: Data(repeating: 0xBB, count: 32)))

        let loaded = try XCTUnwrap(try store.load())
        XCTAssertEqual(loaded.clusterId, Data(repeating: 0x22, count: 16))
        XCTAssertEqual(loaded.key, Data(repeating: 0xBB, count: 32))
    }

    // MARK: - Delete

    func testDeleteRemovesBothFiles() throws {
        try store.save(sampleRecord())
        try store.delete()

        XCTAssertNil(try store.load())
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: tempRoot.appendingPathComponent("cluster.id").path))
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: tempRoot.appendingPathComponent("cluster.key").path))
    }

    func testDeleteWithoutPriorSaveIsNotAnError() throws {
        // Idempotent cleanup: callers rely on delete() to be safe even
        // when nothing is there.
        XCTAssertNoThrow(try store.delete())
    }

    // MARK: - Permissions

    func testFilePermissionsAfterSave() throws {
        try store.save(sampleRecord())

        let idMode = try posixMode(tempRoot.appendingPathComponent("cluster.id").path)
        let keyMode = try posixMode(tempRoot.appendingPathComponent("cluster.key").path)

        // cluster.id is public by design (it's broadcast in Bonjour TXT
        // and used as the Argon2id salt), so owner-read/write + group
        // and other read is acceptable.
        XCTAssertEqual(idMode & 0o777, 0o644,
                       "cluster.id must end up at 0644 regardless of umask")

        // cluster.key holds secret material; only the owner may read it.
        XCTAssertEqual(keyMode & 0o777, 0o600,
                       "cluster.key must end up at 0600 regardless of umask")
    }

    func testParentDirectoryPermissions() throws {
        try store.save(sampleRecord())
        let dirMode = try posixMode(tempRoot.path)
        // Directory is 0700: even if a file inside briefly has a looser
        // mode during the atomic-write window, the directory denies
        // traversal to anyone other than the owner.
        XCTAssertEqual(dirMode & 0o777, 0o700,
                       "cluster directory must be 0700 regardless of umask")
    }

    func testParentDirectoryIsCreatedIfMissing() throws {
        // Nested path beneath a non-existent ancestor — save must
        // create the whole chain.
        let nested = tempRoot.appendingPathComponent("deep/nested/dir",
                                                     isDirectory: true)
        let nestedStore = FileClusterKeyStore(directory: nested)
        try nestedStore.save(sampleRecord())

        XCTAssertTrue(FileManager.default.fileExists(atPath: nested.path))
        let loaded = try XCTUnwrap(try nestedStore.load())
        XCTAssertEqual(loaded, sampleRecord())
    }

    func testExistingDirectoryWithLoosePermsIsTightenedOnSave() throws {
        // Create the target dir in advance with a loose mode, then save
        // and confirm the store retightens it to 0700.
        try FileManager.default.createDirectory(
            at: tempRoot,
            withIntermediateDirectories: true,
            attributes: [.posixPermissions: NSNumber(value: UInt16(0o755))]
        )
        XCTAssertEqual(try posixMode(tempRoot.path) & 0o777, 0o755)

        try store.save(sampleRecord())
        XCTAssertEqual(try posixMode(tempRoot.path) & 0o777, 0o700)
    }

    // MARK: - Partial state

    func testPartialStateIsTreatedAsNoRecord() throws {
        // Simulate a crash between writing cluster.id and cluster.key:
        // id is present on disk, key is not. load() must return nil
        // (not a half-record) so the caller re-runs the full join flow.
        try store.save(sampleRecord())
        try FileManager.default.removeItem(
            at: tempRoot.appendingPathComponent("cluster.key"))

        XCTAssertNil(try store.load())
    }
}
