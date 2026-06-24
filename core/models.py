from dataclasses import dataclass, field
from enum import Enum


class Role(str, Enum):
    VIRGO    = "virgo"
    PISCES   = "pisces"
    CRITIC   = "critic"
    SEER     = "seer"
    ARCHIVIST = "archivist"
    REVIEWER = "reviewer"


# Each role maps to a specific model
ROLE_MODELS: dict[Role, str] = {
    Role.VIRGO:     "qwen3:8b",
    Role.PISCES:    "qwen3:8b",
    Role.CRITIC:    "llama3.2:3b",
    Role.SEER:      "qwen2.5:3b",
    Role.ARCHIVIST: "gemma4:latest",
    Role.REVIEWER:  "phi4-mini",
}

# Generous timeouts per model size (seconds)
ROLE_TIMEOUTS: dict[Role, float] = {
    Role.VIRGO:     270.0,
    Role.PISCES:    270.0,
    Role.CRITIC:    120.0,
    Role.SEER:      90.0,
    Role.ARCHIVIST: 720.0,
    Role.REVIEWER:  120.0,
}

ROLE_NUM_PREDICT: dict[Role, int] = {
    Role.VIRGO:     1024,   # analytical, should be concise
    Role.PISCES:    2048,   # expansive writer, needs room
    Role.CRITIC:    1024,   # lists only
    Role.SEER:      1024,   # tiny model, lists only
    Role.ARCHIVIST: 4096,   # synthesis — needs room for full prose output
    Role.REVIEWER:  1024,   # JSON output
}

ROLE_KEEP_ALIVE: dict[Role, str] = {
    Role.VIRGO:     "5m",
    Role.PISCES:    "5m",
    Role.CRITIC:    "5m",
    Role.SEER:      "5m",
    Role.REVIEWER:  "5m",
    Role.ARCHIVIST: "15m",   # synthesis is the longest call; keep it resident
}

@dataclass
class RoleResponse:
    role: Role
    model: str
    content: str
    duration_ms: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.content)

    def __repr__(self) -> str:
        status = "OK" if self.ok else f"ERR: {self.error}"
        return f"RoleResponse({self.role.value} | {self.model} | {self.duration_ms}ms | {status})"
