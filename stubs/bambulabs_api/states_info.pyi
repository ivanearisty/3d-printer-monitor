from enum import Enum

class GcodeState(Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PREPARE = "PREPARE"
    PAUSE = "PAUSE"
    FINISH = "FINISH"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"
