"""Light-weight, JSON-backed persistence layer used by *VoiceGuard*.

This replaces the original SQLAlchemy / MySQL implementation so the complete
application can run completely offline with **zero external services**.

The public interface intentionally mirrors a tiny subset of SQLAlchemy that is
referenced in *main.py* so that no changes are required there:

    from db import SessionLocal, init_db, User

* ``init_db()``           – Ensure the *users.json* file exists.
* ``SessionLocal()``      – Return a session-like object exposing ``add``,
                            ``commit`` and ``query`` (only the parts the GUI
                            uses).
* ``User``                – A small ``dataclass`` holding user attributes and
                            a helper ``full_name`` method.

Internally all users are stored in a ``users.json`` file in the project root
in the following format:

    {
        "users": [
            {
                "id": 1,
                "first_name": "Ada",
                "last_name": "Lovelace",
                "date_of_birth": "1815-12-10",
                "embedding": "[... JSON list of floats ...]"
            },
            ...
        ]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import date
from pathlib import Path
from typing import Any, List, Sequence


# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

USERS_FILE = Path("users.json")


def _read_users_file() -> List[dict[str, Any]]:
    if USERS_FILE.exists():
        try:
            with USERS_FILE.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                users = data.get("users", [])
            else:
                # Legacy format: plain list stored directly
                users = data
        except (json.JSONDecodeError, OSError):
            users = []
    else:
        users = []
    return users


def _write_users_file(users: Sequence[dict[str, Any]]) -> None:
    USERS_FILE.write_text(json.dumps({"users": users}, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Dataclass representing a single user
# ---------------------------------------------------------------------------


@dataclass
class User:
    first_name: str
    last_name: str
    date_of_birth: date
    embedding: str  # JSON string holding list[float]
    id: int | None = field(default=None)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def full_name(self) -> str:  # noqa: D401 – simple property style
        return f"{self.first_name} {self.last_name}"

    # Support comparison when searching identical objects
    def __eq__(self, other: object) -> bool:  # noqa: D401 – not docstring
        if not isinstance(other, User):
            return NotImplemented
        return (
            self.id == other.id
            and self.first_name == other.first_name
            and self.last_name == other.last_name
            and self.date_of_birth == other.date_of_birth
            and self.embedding == other.embedding
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def init_db() -> None:
    """Create an **empty** users file on first run.

    The *VoiceGuard* GUI only calls this once when the application starts.
    """

    if not USERS_FILE.exists():
        _write_users_file([])


# A deliberately *tiny* ORM-like façade – just enough to satisfy main.py


class _Query:
    """Extremely small subset of the SQLAlchemy query API used in the UI."""

    def __init__(self, data: List[User]):
        self._data = data

    def all(self) -> List[User]:  # noqa: D401 – not docstring
        return self._data


class _Session:
    """Session object mirroring the three methods used by the GUI."""

    def __init__(self):
        self._new: List[User] = []

    # ------------------------------------------------------------------
    # SQLAlchemy-like *public* methods
    # ------------------------------------------------------------------

    def add(self, user: User) -> None:  # noqa: D401 – not docstring
        if not isinstance(user, User):
            raise TypeError("Only User instances can be added to the session.")
        self._new.append(user)

    def commit(self) -> None:  # noqa: D401 – not docstring
        if not self._new:
            return

        # Load current users, determine next id
        current_raw = _read_users_file()
        current_users = [
            _dict_to_user(d) for d in current_raw
        ]

        max_id = max([u.id for u in current_users if u.id is not None] or [0])

        # Assign IDs and merge
        for idx, u in enumerate(self._new, start=1):
            if u.id is None:
                u.id = max_id + idx
        combined_users = current_users + self._new

        # Persist to disk
        _write_users_file([_user_to_dict(u) for u in combined_users])

        # Clear pending list
        self._new.clear()

    def query(self, model):  # noqa: D401 – not docstring
        if model is not User:
            raise TypeError("Only User model is supported in this lightweight DB layer.")

        users = [_dict_to_user(d) for d in _read_users_file()]
        return _Query(users)


def SessionLocal() -> _Session:  # noqa: D401 – simple wrapper
    """Factory matching the original SQLAlchemy signature."""

    return _Session()


# ---------------------------------------------------------------------------
# Internal conversion helpers
# ---------------------------------------------------------------------------


def _user_to_dict(u: User) -> dict[str, Any]:  # noqa: D401 – not docstring
    data = asdict(u)
    data["date_of_birth"] = u.date_of_birth.isoformat()
    return data


def _dict_to_user(d: dict[str, Any]) -> User:  # noqa: D401 – not docstring
    return User(
        id=d.get("id"),
        first_name=d["first_name"],
        last_name=d["last_name"],
        date_of_birth=date.fromisoformat(d["date_of_birth"]),
        embedding=d["embedding"],
    )


__all__ = [
    "User",
    "SessionLocal",
    "init_db",
    "delete_user",
]


# ---------------------------------------------------------------------------
# Additional helper – delete user by ID
# ---------------------------------------------------------------------------


def delete_user(user_id: int) -> bool:  # noqa: D401 – simple helper
    """Remove a user from the *users.json* store.

    Parameters
    ----------
    user_id:
        The primary key of the user to remove.

    Returns
    -------
    bool
        ``True`` if a user was removed, ``False`` otherwise.
    """

    if not isinstance(user_id, int):
        raise TypeError("user_id must be an integer")

    users_raw = _read_users_file()
    initial_len = len(users_raw)

    users_filtered = [u for u in users_raw if u.get("id") != user_id]

    if len(users_filtered) == initial_len:
        # Nothing removed – id not found
        return False

    _write_users_file(users_filtered)
    return True

