// Tests for the passphrase resolver. Each test injects fakes for the
// IO-touching dependencies (interactive prompt, file read, command run)
// so the full decision tree is exercised without spawning subprocesses,
// reading real files, or attaching to a TTY.

import XCTest
@testable import TurboQuantKit

final class PassphraseResolverTests: XCTestCase {

    // MARK: - Source selection

    /// Direct --passphrase value passes through verbatim — no trimming
    /// because operators may intentionally use leading/trailing spaces.
    func testDirectSourceReturnsValueVerbatim() throws {
        let resolved = try resolvePassphrase(
            direct: "correct horse battery staple",
            file: nil,
            command: nil,
            isStdinTTY: false,
            readInteractive: { XCTFail("should not prompt"); return "" },
            readFile: { _ in XCTFail("should not read file"); return "" },
            runCommand: { _ in XCTFail("should not run command"); return (0, "", "") }
        )
        XCTAssertEqual(resolved, "correct horse battery staple")
    }

    /// File source reads via the injected helper and trims one trailing
    /// newline — the common `echo "secret" > file` workflow.
    func testFileSourceTrimsTrailingNewline() throws {
        let url = URL(fileURLWithPath: "/tmp/fake-passphrase-file")
        let resolved = try resolvePassphrase(
            direct: nil,
            file: url,
            command: nil,
            isStdinTTY: false,
            readInteractive: { XCTFail("should not prompt"); return "" },
            readFile: { passedURL in
                XCTAssertEqual(passedURL, url)
                return "correct-horse-battery\n"
            },
            runCommand: { _ in XCTFail("should not run command"); return (0, "", "") }
        )
        XCTAssertEqual(resolved, "correct-horse-battery")
    }

    /// File source without a trailing newline is preserved as-is so a
    /// caller producing content via `printf` instead of `echo` does not
    /// have its passphrase silently mangled.
    func testFileSourceWithoutNewlinePreserved() throws {
        let resolved = try resolvePassphrase(
            direct: nil,
            file: URL(fileURLWithPath: "/tmp/whatever"),
            command: nil,
            isStdinTTY: false,
            readInteractive: { XCTFail("should not prompt"); return "" },
            readFile: { _ in "correct-horse-battery" },
            runCommand: { _ in XCTFail("should not run command"); return (0, "", "") }
        )
        XCTAssertEqual(resolved, "correct-horse-battery")
    }

    /// Command source captures stdout and trims one trailing newline.
    /// The status code is checked separately by `testCommandFailsWhenStatusNonZero`.
    func testCommandSourceTrimsTrailingNewlineFromStdout() throws {
        let resolved = try resolvePassphrase(
            direct: nil,
            file: nil,
            command: "echo correct-horse-battery",
            isStdinTTY: false,
            readInteractive: { XCTFail("should not prompt"); return "" },
            readFile: { _ in XCTFail("should not read file"); return "" },
            runCommand: { cmd in
                XCTAssertEqual(cmd, "echo correct-horse-battery")
                return (0, "correct-horse-battery\n", "")
            }
        )
        XCTAssertEqual(resolved, "correct-horse-battery")
    }

    /// Interactive source is used as the fallback when no flag is given
    /// and stdin is a TTY. The injected closure stands in for
    /// `readpassphrase(3)` in production.
    func testInteractiveFallbackUsedWhenStdinIsTTY() throws {
        let resolved = try resolvePassphrase(
            direct: nil,
            file: nil,
            command: nil,
            isStdinTTY: true,
            readInteractive: { "correct-horse-battery" },
            readFile: { _ in XCTFail("should not read file"); return "" },
            runCommand: { _ in XCTFail("should not run command"); return (0, "", "") }
        )
        XCTAssertEqual(resolved, "correct-horse-battery")
    }

    // MARK: - Failure modes

    /// Specifying any two of the three flag-driven sources is a fatal
    /// error. The case-list documents every pairwise combination so a
    /// future fourth source addition is forced to extend coverage.
    func testMultipleSourcesIsFatal() throws {
        let pairs: [(String?, URL?, String?, [String])] = [
            ("x", URL(fileURLWithPath: "/tmp/a"), nil,
                ["--passphrase", "--passphrase-file"]),
            ("x", nil, "echo y",
                ["--passphrase", "--passphrase-command"]),
            (nil, URL(fileURLWithPath: "/tmp/a"), "echo y",
                ["--passphrase-file", "--passphrase-command"]),
            ("x", URL(fileURLWithPath: "/tmp/a"), "echo y",
                ["--passphrase", "--passphrase-file", "--passphrase-command"]),
        ]
        for (direct, file, command, expectedNames) in pairs {
            XCTAssertThrowsError(
                try resolvePassphrase(
                    direct: direct,
                    file: file,
                    command: command,
                    isStdinTTY: true,
                    readInteractive: { "ignored" },
                    readFile: { _ in "ignored" },
                    runCommand: { _ in (0, "ignored", "") }
                )
            ) { err in
                guard case .multipleSourcesSpecified(let names) =
                        err as? PassphraseResolutionError else {
                    XCTFail("expected multipleSourcesSpecified, got \(err)")
                    return
                }
                XCTAssertEqual(names, expectedNames)
            }
        }
    }

    /// No source plus non-TTY stdin is the canonical "automation forgot
    /// to pass a flag" misconfiguration. The error name is the signal
    /// to the operator.
    func testNonInteractiveAndNoSourceThrows() {
        XCTAssertThrowsError(
            try resolvePassphrase(
                direct: nil, file: nil, command: nil,
                isStdinTTY: false,
                readInteractive: { XCTFail("should not prompt"); return "" },
                readFile: { _ in XCTFail("should not read"); return "" },
                runCommand: { _ in XCTFail("should not run"); return (0, "", "") }
            )
        ) { err in
            XCTAssertEqual(
                err as? PassphraseResolutionError,
                .nonInteractiveAndNoSource
            )
        }
    }

    /// File read errors are wrapped with the URL so the operator can
    /// see exactly which path failed without grepping through the
    /// stack trace.
    func testFileReadFailureIsWrapped() {
        struct StubError: Error {}
        let url = URL(fileURLWithPath: "/does/not/exist")
        XCTAssertThrowsError(
            try resolvePassphrase(
                direct: nil, file: url, command: nil,
                isStdinTTY: false,
                readInteractive: { XCTFail("nope"); return "" },
                readFile: { _ in throw StubError() },
                runCommand: { _ in XCTFail("nope"); return (0, "", "") }
            )
        ) { err in
            guard case .fileReadFailed(let failedURL, _) =
                    err as? PassphraseResolutionError else {
                XCTFail("expected fileReadFailed, got \(err)")
                return
            }
            XCTAssertEqual(failedURL, url)
        }
    }

    /// A non-zero command exit is reported with status and stderr so
    /// operators do not have to re-run the command themselves to see
    /// why it failed.
    func testCommandFailsWhenStatusNonZero() {
        XCTAssertThrowsError(
            try resolvePassphrase(
                direct: nil, file: nil, command: "false",
                isStdinTTY: false,
                readInteractive: { XCTFail("nope"); return "" },
                readFile: { _ in XCTFail("nope"); return "" },
                runCommand: { _ in (42, "", "command not found: foo") }
            )
        ) { err in
            guard case .commandFailed(let status, let stderr) =
                    err as? PassphraseResolutionError else {
                XCTFail("expected commandFailed, got \(err)")
                return
            }
            XCTAssertEqual(status, 42)
            XCTAssertEqual(stderr, "command not found: foo")
        }
    }

    /// An empty file or empty command output is rejected with a
    /// dedicated error so the operator is not misled into thinking the
    /// passphrase was too short.
    func testEmptyPassphraseRejected() {
        XCTAssertThrowsError(
            try resolvePassphrase(
                direct: nil,
                file: URL(fileURLWithPath: "/tmp/empty"),
                command: nil,
                isStdinTTY: false,
                readInteractive: { XCTFail("nope"); return "" },
                readFile: { _ in "\n" },
                runCommand: { _ in XCTFail("nope"); return (0, "", "") }
            )
        ) { err in
            guard case .emptyPassphrase = err as? PassphraseResolutionError else {
                XCTFail("expected emptyPassphrase, got \(err)")
                return
            }
        }
    }

    /// Below the 8-character floor is a dedicated error so the operator
    /// sees the exact threshold rather than a generic "invalid input."
    func testPassphraseBelowMinimumLengthRejected() {
        XCTAssertThrowsError(
            try resolvePassphrase(
                direct: "short",
                file: nil, command: nil,
                isStdinTTY: false,
                readInteractive: { XCTFail("nope"); return "" },
                readFile: { _ in XCTFail("nope"); return "" },
                runCommand: { _ in XCTFail("nope"); return (0, "", "") }
            )
        ) { err in
            guard case .tooShort(let length, let minimum) =
                    err as? PassphraseResolutionError else {
                XCTFail("expected tooShort, got \(err)")
                return
            }
            XCTAssertEqual(length, 5)
            XCTAssertEqual(minimum, minimumPassphraseLength)
        }
    }

    /// The exact-minimum length is accepted so the threshold is
    /// inclusive, matching the documented behavior.
    func testExactlyMinimumLengthAccepted() throws {
        let exactly8 = String(repeating: "a", count: minimumPassphraseLength)
        let resolved = try resolvePassphrase(
            direct: exactly8,
            file: nil, command: nil,
            isStdinTTY: false,
            readInteractive: { XCTFail("nope"); return "" },
            readFile: { _ in XCTFail("nope"); return "" },
            runCommand: { _ in XCTFail("nope"); return (0, "", "") }
        )
        XCTAssertEqual(resolved, exactly8)
    }
}
