"""Compatibility wrapper for the canonical postgres_orchestration_db module."""
from __future__ import annotations

from postgres_orchestration_db import (  # noqa: F401
    LocalOrchestrationDb,
    LocalOrchestrationDbConfig,
    PostgresOrchestrationDb,
    PostgresOrchestrationDbConfig,
)
