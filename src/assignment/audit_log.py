"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input and its monotonic start time keyed by request ID."""
        key = request_id or f"{user_id}:{len(self.logs) + len(self._open)}"
        self._open[key] = {
            "user_id": user_id,
            "input": text,
            "started_at": utc_now_iso(),
            "started_monotonic": time.perf_counter(),
        }
        return key

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store output, layer decision, and measured latency."""
        key = request_id or user_id
        entry = self._open.pop(key, None)
        if entry is None:
            entry = {
                "user_id": user_id,
                "input": None,
                "started_at": utc_now_iso(),
                "started_monotonic": time.perf_counter(),
            }
        elapsed_ms = round((time.perf_counter() - entry.pop("started_monotonic")) * 1000, 3)
        entry.update({
            "request_id": request_id or key,
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "completed_at": utc_now_iso(),
            "latency_ms": elapsed_ms,
        })
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
