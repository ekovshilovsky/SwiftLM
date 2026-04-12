import XCTest
@testable import SwiftLM

final class MemoryCalculatorTests: XCTestCase {

    func testQwen35_27B_TQ8_at1MContext() {
        let budget = MemoryCalculator.calculate(
            modelParameterCount: 27_000_000_000,
            bitsPerWeight: 8,
            contextLength: 1_000_000,
            numLayers: 28,
            numHeads: 8,
            headDim: 128,
            kvBits: 4,  // ~3.5 effective
            decodeWindowTokens: 131_072
        )
        // 27B * 1 byte = ~27 GB weights
        XCTAssertGreaterThan(budget.weightMemoryBytes, 25_000_000_000)
        XCTAssertLessThan(budget.weightMemoryBytes, 30_000_000_000)

        // Total should be roughly 50 GB for this config on 128GB machine
        XCTAssertTrue(budget.fits, "Qwen3.5-27B TQ8 at 1M context should fit on 128GB")
    }

    func testExceedsMemoryRefuses() {
        let budget = MemoryCalculator.calculate(
            modelParameterCount: 671_000_000_000,  // DeepSeek-V3 671B
            bitsPerWeight: 8,
            contextLength: 1_000_000,
            numLayers: 60,
            numHeads: 128,
            headDim: 128,
            kvBits: 4,
            decodeWindowTokens: 131_072
        )
        // 671B at 8-bit = ~671 GB weights alone
        XCTAssertFalse(budget.fits, "671B model should not fit on single node")
    }

    func testBudgetDescriptionFormatted() {
        let budget = MemoryCalculator.calculate(
            modelParameterCount: 27_000_000_000,
            bitsPerWeight: 8,
            contextLength: 8192,
            numLayers: 28,
            numHeads: 8,
            headDim: 128,
            kvBits: 4,
            decodeWindowTokens: 8192
        )
        XCTAssertTrue(budget.description.contains("GB"))
        XCTAssertTrue(budget.description.contains("Status:"))
    }
}
