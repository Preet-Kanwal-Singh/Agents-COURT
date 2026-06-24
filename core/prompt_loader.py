from pathlib import Path

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_VALID_ROLES = ["virgo", "pisces", "critic", "seer", "archivist", "reviewer"]


def get_system_prompt(role: str) -> str:
    """
    Load the system prompt for a given role from its dedicated file
    in prompts/. Each file is tuned independently for the model
    assigned to that role — no markdown parsing, no section
    extraction, no shared-section stitching.
    """
    role = role.lower()
    if role not in _VALID_ROLES:
        raise ValueError(f"Unknown role '{role}'. Valid roles: {_VALID_ROLES}")

    path = PROMPTS_DIR / f"{role}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found for role '{role}': {path}")

    return path.read_text(encoding="utf-8").strip()


def list_available_roles() -> list[str]:
    return list(_VALID_ROLES)
