// Prints the thermal state macOS reports for this machine: 0 nominal, 1 fair, 2 serious,
// 3 critical. scripts/auto-brake.sh builds it once and reads it at each tick, because
// `pmset -g therm` records nothing on Apple silicon until the machine is already throttled.
import Foundation

print(ProcessInfo.processInfo.thermalState.rawValue)
