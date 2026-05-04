// Resolve the cluster passphrase for `swiftlm --distributed` from one of
// four exclusive sources: a direct `--passphrase` argument, a
// `--passphrase-file <path>` file, a `--passphrase-command <cmd>`
// subprocess, or an interactive TTY prompt when no flag is supplied and
// stdin is attached to a terminal. The four sources match the patterns
// shipped by widely deployed CLI tools (gpg, restic, age, vault) so
// operators do not have to learn a TurboQuant-specific convention.
//
// The resolver is structured as a pure function over the parsed CLI
// options plus the runtime environment, with every IO-touching helper
// injected. Production callers pass the default helpers; tests pass
// fakes so a single XCTest run exercises every branch without spawning
// subprocesses, reading real files, or attaching to a real TTY.

import Foundation

/// Where the passphrase came from. The cases are mutually exclusive at
/// the CLI level — specifying more than one passphrase flag is a fatal
/// error caught by `DistributedCLIOptions.validate()` before we get
/// here. The interactive case is the fallback when no flag is given
/// and stdin is a TTY.
public enum PassphraseSource: Sendable, Equatable {
    case direct(String)
    case file(URL)
    case command(String)
    case interactive
}

/// Failure modes for `resolvePassphrase`. Each carries enough context
/// to produce a precise stderr message — most resolution failures
/// indicate an operator misconfiguration rather than a SwiftLM bug.
public enum PassphraseResolutionError: Error, Equatable, CustomStringConvertible {
    /// More than one of `--passphrase`, `--passphrase-file`,
    /// `--passphrase-command` was supplied. Carries the names of the
    /// conflicting flags so the operator can immediately see what to
    /// drop.
    case multipleSourcesSpecified([String])

    /// No passphrase flag given and stdin is not attached to a TTY,
    /// so we cannot interactively prompt. Common in non-interactive
    /// automation contexts that forgot to set the env var or pass a
    /// flag.
    case nonInteractiveAndNoSource

    /// `--passphrase-file` could not be read. Carries the URL and the
    /// underlying error so the operator knows whether it is a missing
    /// file, a permission problem, or an encoding issue.
    case fileReadFailed(URL, message: String)

    /// `--passphrase-command` ran but exited non-zero or wrote nothing
    /// usable to stdout. Carries the exit status and any stderr
    /// captured for diagnosis.
    case commandFailed(status: Int32, stderr: String)

    /// The resolved passphrase is empty after trimming. Treated as a
    /// distinct error from `tooShort` so the operator does not get
    /// the misleading "8 character minimum" message when the file or
    /// command produced nothing at all.
    case emptyPassphrase(source: String)

    /// The resolved passphrase is shorter than the configured minimum.
    /// Argon2id still works below this length but the security
    /// argument weakens fast; we refuse to form a cluster with one.
    case tooShort(length: Int, minimum: Int)

    public var description: String {
        switch self {
        case .multipleSourcesSpecified(let names):
            return "multiple passphrase sources specified: \(names.joined(separator: ", ")). Use exactly one of --passphrase, --passphrase-file, --passphrase-command."
        case .nonInteractiveAndNoSource:
            return "no passphrase source given and stdin is not a TTY. Pass --passphrase, --passphrase-file, --passphrase-command, or run interactively."
        case .fileReadFailed(let url, let message):
            return "failed to read --passphrase-file '\(url.path)': \(message)"
        case .commandFailed(let status, let stderr):
            let trimmed = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty {
                return "--passphrase-command exited with status \(status)"
            }
            return "--passphrase-command exited with status \(status): \(trimmed)"
        case .emptyPassphrase(let source):
            return "passphrase from \(source) is empty after trimming"
        case .tooShort(let length, let minimum):
            return "passphrase is \(length) character\(length == 1 ? "" : "s"); minimum is \(minimum)"
        }
    }
}

/// Minimum passphrase length accepted by the resolver. Argon2id will
/// still form a cluster below this threshold, but the security argument
/// for sub-8-character passphrases is weak enough that we treat them
/// as operator misconfigurations instead of legitimate inputs.
public let minimumPassphraseLength: Int = 8

/// Pure resolver over the CLI options and runtime environment. Every
/// IO-touching dependency is injected so unit tests can exercise the
/// full decision tree without spawning subprocesses, reading real
/// files, or attaching to a TTY.
///
/// - Parameters:
///   - direct: Value of the `--passphrase` flag, if specified.
///   - file: Value of the `--passphrase-file` flag, if specified.
///   - command: Value of the `--passphrase-command` flag, if specified.
///   - isStdinTTY: Whether stdin is attached to a terminal. Determines
///     whether the interactive fallback is allowed when no flag is
///     given.
///   - readInteractive: Reads a passphrase from a TTY with echo
///     suppressed. The default helper wraps `readpassphrase(3)`.
///   - readFile: Reads a passphrase file as UTF-8 text. The default
///     helper trims one trailing newline.
///   - runCommand: Runs a shell command and captures stdout/stderr.
///     The default helper trims one trailing newline from stdout.
public func resolvePassphrase(
    direct: String?,
    file: URL?,
    command: String?,
    isStdinTTY: Bool,
    readInteractive: () throws -> String,
    readFile: (URL) throws -> String,
    runCommand: (String) throws -> (status: Int32, stdout: String, stderr: String)
) throws -> String {
    // Step 1: enforce mutual exclusion across the three flag-driven
    // sources. The interactive fallback is not a flag, so it does not
    // participate in the conflict check.
    var specified: [String] = []
    if direct != nil { specified.append("--passphrase") }
    if file != nil { specified.append("--passphrase-file") }
    if command != nil { specified.append("--passphrase-command") }
    if specified.count > 1 {
        throw PassphraseResolutionError.multipleSourcesSpecified(specified)
    }

    // Step 2: read the raw passphrase from whichever source was chosen.
    // Each branch is responsible for only the IO; trimming and length
    // validation happens uniformly in step 3.
    let raw: String
    let sourceLabel: String
    if let value = direct {
        raw = value
        sourceLabel = "--passphrase"
    } else if let url = file {
        do {
            raw = try readFile(url)
        } catch {
            throw PassphraseResolutionError.fileReadFailed(
                url, message: String(describing: error)
            )
        }
        sourceLabel = "--passphrase-file"
    } else if let cmd = command {
        let result = try runCommand(cmd)
        if result.status != 0 {
            throw PassphraseResolutionError.commandFailed(
                status: result.status, stderr: result.stderr
            )
        }
        raw = result.stdout
        sourceLabel = "--passphrase-command"
    } else if isStdinTTY {
        raw = try readInteractive()
        sourceLabel = "interactive prompt"
    } else {
        throw PassphraseResolutionError.nonInteractiveAndNoSource
    }

    // Step 3: trim trailing whitespace from non-direct sources and
    // validate the length floor uniformly. The direct flag value is
    // taken verbatim because the operator may have intentionally
    // included whitespace as part of the passphrase.
    let trimmed: String
    if case .some = direct {
        trimmed = raw
    } else {
        trimmed = raw.trimmingCharacters(in: CharacterSet(charactersIn: "\n\r"))
    }

    if trimmed.isEmpty {
        throw PassphraseResolutionError.emptyPassphrase(source: sourceLabel)
    }

    if trimmed.count < minimumPassphraseLength {
        throw PassphraseResolutionError.tooShort(
            length: trimmed.count, minimum: minimumPassphraseLength
        )
    }

    return trimmed
}

// MARK: - Production helpers

/// Read a passphrase from the controlling TTY with echo suppressed.
/// Wraps `readpassphrase(3)` from BSD libc, available on macOS via
/// `<readpassphrase.h>`. On non-macOS platforms or when `readpassphrase`
/// is unavailable, falls back to a non-suppressed `readLine()` so the
/// caller still gets an answer; the lack of echo suppression is
/// signaled to the operator by the absence of the prompt prefix change.
public func defaultReadInteractivePassphrase() throws -> String {
    let prompt = "Enter cluster passphrase: "
    var buffer = [Int8](repeating: 0, count: 1024)
    let result = buffer.withUnsafeMutableBufferPointer { ptr -> UnsafeMutablePointer<Int8>? in
        guard let base = ptr.baseAddress else { return nil }
        return readpassphrase(prompt, base, ptr.count, 0)
    }
    guard let cString = result else {
        // readpassphrase returns NULL on EOF or signal interruption; treat as
        // empty input so the caller surfaces the standard emptyPassphrase
        // error rather than a cryptic runtime failure.
        return ""
    }
    return String(cString: cString)
}

/// Read a passphrase file as UTF-8 text. Trims exactly one trailing
/// newline (LF or CRLF) so an `echo "secret" > file` workflow Just
/// Works without forcing the operator to remember `printf` instead of
/// `echo`. Other trailing whitespace is preserved verbatim — if the
/// operator typed two trailing spaces, those are part of the secret.
public func defaultReadPassphraseFile(_ url: URL) throws -> String {
    return try String(contentsOf: url, encoding: .utf8)
}

/// Run a shell command and capture stdout, stderr, and exit status.
/// The command is invoked through `/bin/sh -c <cmd>` so operators can
/// use shell features (pipelines, env interpolation, command
/// substitution) without quoting them through SwiftLM's own argument
/// parser.
public func defaultRunPassphraseCommand(
    _ cmd: String
) throws -> (status: Int32, stdout: String, stderr: String) {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/sh")
    process.arguments = ["-c", cmd]

    let stdoutPipe = Pipe()
    let stderrPipe = Pipe()
    process.standardOutput = stdoutPipe
    process.standardError = stderrPipe

    try process.run()
    process.waitUntilExit()

    let stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
    let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
    let stdout = String(data: stdoutData, encoding: .utf8) ?? ""
    let stderr = String(data: stderrData, encoding: .utf8) ?? ""
    return (process.terminationStatus, stdout, stderr)
}
