// Cluster authentication. Two-stage key derivation: Argon2id hardens the
// user-supplied passphrase against offline brute-force, then HKDF derives
// purpose-specific subkeys (handshake auth, heartbeat MAC, snapshot
// encryption) with cryptographic domain separation.

import CArgon2
import CryptoKit
import Foundation

public enum ClusterAuth {

    // MARK: - Argon2id parameters (RFC 9106 §4 interactive profile)

    // Recommended by RFC 9106 for interactive logins on commodity hardware.
    // Derivation takes >100ms on M-series silicon, which is long enough to
    // deter offline attack but short enough to remain tolerable for the
    // once-per-session cluster join flow.
    public static let defaultIterations:  UInt32 = 3
    public static let defaultMemoryKiB:   UInt32 = 65_536   // 64 MiB
    public static let defaultParallelism: UInt32 = 1
    public static let masterKeyLength:    Int    = 32        // 256 bits

    // MARK: - Low-level Argon2id wrapper

    /// Run Argon2id with caller-supplied parameters. Exposed for testing
    /// against known-answer vectors; production callers should use
    /// deriveMasterKey() to apply the recommended parameters consistently.
    public static func argon2idRaw(passphrase: String,
                                   salt: Data,
                                   iterations: UInt32,
                                   memoryKiB: UInt32,
                                   parallelism: UInt32,
                                   outputLength: Int) -> Data {
        let pwdBytes = Array(passphrase.utf8)
        var output = [UInt8](repeating: 0, count: outputLength)

        let rc: Int32 = pwdBytes.withUnsafeBufferPointer { pwdBuf in
            salt.withUnsafeBytes { (saltBuf: UnsafeRawBufferPointer) -> Int32 in
                output.withUnsafeMutableBufferPointer { outBuf in
                    argon2id_hash_raw(
                        iterations,
                        memoryKiB,
                        parallelism,
                        pwdBuf.baseAddress,
                        pwdBuf.count,
                        saltBuf.baseAddress,
                        saltBuf.count,
                        outBuf.baseAddress,
                        outBuf.count
                    )
                }
            }
        }
        precondition(rc == ARGON2_OK.rawValue,
                     "argon2id_hash_raw failed with code \(rc)")
        return Data(output)
    }

    // MARK: - High-level derivation API

    /// Derive a 256-bit master key from a user passphrase and salt using
    /// Argon2id with the RFC 9106 interactive-profile parameters.
    public static func deriveMasterKey(passphrase: String, salt: Data) -> Data {
        return argon2idRaw(passphrase: passphrase,
                           salt: salt,
                           iterations: defaultIterations,
                           memoryKiB: defaultMemoryKiB,
                           parallelism: defaultParallelism,
                           outputLength: masterKeyLength)
    }

    /// Derive a purpose-specific subkey from the master using HKDF-SHA256.
    /// The `info` parameter provides cryptographic domain separation: subkeys
    /// derived with different info strings are uncorrelated even when the
    /// same master key is used. Convention: info strings use the "tq-*"
    /// namespace so cluster keys cannot collide with unrelated protocols.
    public static func deriveSubkey(master: Data,
                                    info: String,
                                    length: Int = 32) -> SymmetricKey {
        return HKDF<SHA256>.deriveKey(
            inputKeyMaterial: SymmetricKey(data: master),
            info: Data(info.utf8),
            outputByteCount: length
        )
    }
}
