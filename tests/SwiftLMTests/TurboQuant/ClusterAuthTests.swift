// ClusterAuth unit tests. The known-answer test (KAT) is the primary guard
// against a corrupted or mis-patched vendored Argon2 source: if the KAT
// passes, the implementation is correct by definition. The remaining tests
// cover HKDF domain separation and deterministic derivation.

import XCTest
import CryptoKit
import TurboQuantKit

final class ClusterAuthTests: XCTestCase {

    // MARK: - Argon2id known-answer tests

    // The expected digests below are NOT reproduced from any runtime — they
    // are transcribed verbatim from hardcoded hex literals in the
    // phc-winner-argon2 reference project's own test suite (src/test.c) at
    // pinned upstream revision f57e61e19229. Those values were cross-
    // validated against RFC 9106 and against independent Argon2 impls
    // during the Password Hash Competition peer review.
    //
    // This is what makes the check a genuine correctness test: if our
    // Swift bridge produces the same bytes, the vendored C computes the
    // same function as every conformant Argon2 implementation on the
    // planet. If someone tampers with the vendored source, these literals
    // will not shift to accommodate it.
    //
    // Note: hashtest() at src/test.c:47 passes `1 << m` to argon2_hash,
    // so the reference file's `m=16` parameter = 65536 KiB here.

    func testArgon2idKAT_password_somesalt_t2_m64MiB() {
        // Source: src/test.c line 233-236 of phc-winner-argon2 @ f57e61e19229
        //   hashtest(version, 2, 16, 1, "password", "somesalt",
        //            "09316115d5cf24ed5a15a31a3ba326e5cf32edc24702987c02b6566f61913cf7",
        //            ..., Argon2_id);
        let digest = ClusterAuth.argon2idRaw(
            passphrase: "password",
            salt: Data("somesalt".utf8),
            iterations: 2,
            memoryKiB: 65_536,  // 1 << 16
            parallelism: 1,
            outputLength: 32
        )
        let expected: [UInt8] = [
            0x09, 0x31, 0x61, 0x15, 0xd5, 0xcf, 0x24, 0xed,
            0x5a, 0x15, 0xa3, 0x1a, 0x3b, 0xa3, 0x26, 0xe5,
            0xcf, 0x32, 0xed, 0xc2, 0x47, 0x02, 0x98, 0x7c,
            0x02, 0xb6, 0x56, 0x6f, 0x61, 0x91, 0x3c, 0xf7,
        ]
        XCTAssertEqual(Array(digest), expected,
                       "Argon2id KAT mismatch — vendored source may be corrupted")
    }

    func testArgon2idKAT_differentPassword_distinctOutput() {
        // Source: src/test.c line 257-260 of phc-winner-argon2 @ f57e61e19229
        //   hashtest(version, 2, 16, 1, "differentpassword", "somesalt",
        //            "0b84d652cf6b0c4beaef0dfe278ba6a80df6696281d7e0d2891b817d8c458fde",
        //            ..., Argon2_id);
        let digest = ClusterAuth.argon2idRaw(
            passphrase: "differentpassword",
            salt: Data("somesalt".utf8),
            iterations: 2,
            memoryKiB: 65_536,
            parallelism: 1,
            outputLength: 32
        )
        let expected: [UInt8] = [
            0x0b, 0x84, 0xd6, 0x52, 0xcf, 0x6b, 0x0c, 0x4b,
            0xea, 0xef, 0x0d, 0xfe, 0x27, 0x8b, 0xa6, 0xa8,
            0x0d, 0xf6, 0x69, 0x62, 0x81, 0xd7, 0xe0, 0xd2,
            0x89, 0x1b, 0x81, 0x7d, 0x8c, 0x45, 0x8f, 0xde,
        ]
        XCTAssertEqual(Array(digest), expected,
                       "Argon2id KAT mismatch on second vector")
    }

    func testArgon2idKAT_differentSalt_distinctOutput() {
        // Source: src/test.c line 261-264 of phc-winner-argon2 @ f57e61e19229
        //   hashtest(version, 2, 16, 1, "password", "diffsalt",
        //            "bdf32b05ccc42eb15d58fd19b1f856b113da1e9a5874fdcc544308565aa8141c",
        //            ..., Argon2_id);
        let digest = ClusterAuth.argon2idRaw(
            passphrase: "password",
            salt: Data("diffsalt".utf8),
            iterations: 2,
            memoryKiB: 65_536,
            parallelism: 1,
            outputLength: 32
        )
        let expected: [UInt8] = [
            0xbd, 0xf3, 0x2b, 0x05, 0xcc, 0xc4, 0x2e, 0xb1,
            0x5d, 0x58, 0xfd, 0x19, 0xb1, 0xf8, 0x56, 0xb1,
            0x13, 0xda, 0x1e, 0x9a, 0x58, 0x74, 0xfd, 0xcc,
            0x54, 0x43, 0x08, 0x56, 0x5a, 0xa8, 0x14, 0x1c,
        ]
        XCTAssertEqual(Array(digest), expected,
                       "Argon2id KAT mismatch on third vector")
    }

    // MARK: - Master key derivation

    func testMasterKeyDeterministic() {
        let salt = Data(repeating: 0x42, count: 16)
        let key1 = ClusterAuth.deriveMasterKey(passphrase: "correct horse battery staple",
                                               salt: salt)
        let key2 = ClusterAuth.deriveMasterKey(passphrase: "correct horse battery staple",
                                               salt: salt)
        XCTAssertEqual(key1, key2)
        XCTAssertEqual(key1.count, 32)
    }

    func testMasterKeyDiffersByPassphrase() {
        let salt = Data(repeating: 0x42, count: 16)
        let k1 = ClusterAuth.deriveMasterKey(passphrase: "apple", salt: salt)
        let k2 = ClusterAuth.deriveMasterKey(passphrase: "orange", salt: salt)
        XCTAssertNotEqual(k1, k2)
    }

    func testMasterKeyDiffersBySalt() {
        let saltA = Data(repeating: 0xAA, count: 16)
        let saltB = Data(repeating: 0xBB, count: 16)
        let k1 = ClusterAuth.deriveMasterKey(passphrase: "same", salt: saltA)
        let k2 = ClusterAuth.deriveMasterKey(passphrase: "same", salt: saltB)
        XCTAssertNotEqual(k1, k2)
    }

    // MARK: - HKDF domain separation

    func testSubkeyDomainSeparation() {
        let master = Data(repeating: 0x11, count: 32)
        let auth = ClusterAuth.deriveSubkey(master: master, info: "tq-handshake-auth")
        let mac  = ClusterAuth.deriveSubkey(master: master, info: "tq-heartbeat-mac")
        let snap = ClusterAuth.deriveSubkey(master: master, info: "tq-snapshot-enc")

        XCTAssertNotEqual(auth.withUnsafeBytes { Data($0) },
                          mac.withUnsafeBytes  { Data($0) })
        XCTAssertNotEqual(auth.withUnsafeBytes { Data($0) },
                          snap.withUnsafeBytes { Data($0) })
        XCTAssertNotEqual(mac.withUnsafeBytes  { Data($0) },
                          snap.withUnsafeBytes { Data($0) })
    }

    func testSubkeyReproducibleAcrossCalls() {
        let master = Data(repeating: 0x11, count: 32)
        let a = ClusterAuth.deriveSubkey(master: master, info: "tq-handshake-auth")
        let b = ClusterAuth.deriveSubkey(master: master, info: "tq-handshake-auth")
        XCTAssertEqual(a.withUnsafeBytes { Data($0) },
                       b.withUnsafeBytes { Data($0) })
    }
}
