// BonjourDiscovery unit tests. Exercises the DiscoveryInfo TXT record
// encode/decode round-trip plus the model+version compatibility check
// used to filter incompatible peers during cluster formation.

import XCTest
import TurboQuantKit

final class BonjourDiscoveryTests: XCTestCase {

    // MARK: - TXT Record

    func testTxtRecordEncoding() {
        let info = DiscoveryInfo(
            model: "ekovshilovsky/Qwen2.5-32B-TQ8",
            memoryGB: 128,
            status: .discovering,
            cluster: nil,
            version: "0.1.0",
            rdma: .unsupported
        )
        let txt = info.toTxtRecord()
        XCTAssertEqual(txt["model"], "ekovshilovsky/Qwen2.5-32B-TQ8")
        XCTAssertEqual(txt["memory_gb"], "128")
        XCTAssertEqual(txt["status"], "discovering")
        XCTAssertEqual(txt["rdma"], "unsupported")
        XCTAssertNil(txt["cluster"], "cluster key must be omitted when not in a cluster")
    }

    func testTxtRecordDecoding() {
        let txt: [String: String] = [
            "model": "ekovshilovsky/Qwen2.5-32B-TQ8",
            "memory_gb": "64",
            "status": "coordinator",
            "cluster": "my-home-lab",
            "version": "0.1.0",
            "rdma": "available"
        ]
        let info = DiscoveryInfo(fromTxtRecord: txt)
        XCTAssertNotNil(info)
        XCTAssertEqual(info?.model, "ekovshilovsky/Qwen2.5-32B-TQ8")
        XCTAssertEqual(info?.memoryGB, 64)
        XCTAssertEqual(info?.status, .coordinator)
        XCTAssertEqual(info?.cluster, "my-home-lab")
        XCTAssertEqual(info?.rdma, .available)
    }

    func testModelMismatchFiltered() {
        let myModel = "ekovshilovsky/Qwen2.5-32B-TQ8"
        let peerTxt: [String: String] = [
            "model": "ekovshilovsky/Qwen3.5-27B-TQ8",
            "memory_gb": "32",
            "status": "discovering",
            "version": "0.1.0",
            "rdma": "unsupported"
        ]
        let peer = DiscoveryInfo(fromTxtRecord: peerTxt)
        XCTAssertNotNil(peer)
        XCTAssertFalse(peer!.isCompatible(with: myModel, version: "0.1.0"))
    }
}
