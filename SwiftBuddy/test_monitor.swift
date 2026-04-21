import Foundation
import Metal

var info = task_vm_info_data_t()
var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)

let result = withUnsafeMutablePointer(to: &info) { ptr in
    ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
    }
}
print(info.phys_footprint)

var size = mach_msg_type_number_t(MemoryLayout<host_cpu_load_info_data_t>.size / MemoryLayout<integer_t>.size)
var cpuLoadInfo = host_cpu_load_info()
let hostPort = mach_host_self()

let cpuResult = withUnsafeMutablePointer(to: &cpuLoadInfo) {
    $0.withMemoryRebound(to: integer_t.self, capacity: Int(size)) {
        host_statistics64(hostPort, HOST_CPU_LOAD_INFO, $0, &size)
    }
}

print(cpuLoadInfo.cpu_ticks.0)
let device = MTLCreateSystemDefaultDevice()
print(device?.currentAllocatedSize ?? 0)

