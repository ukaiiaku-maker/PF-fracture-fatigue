"""Durable reservation ledger shared by Ast analytical and 1-D calls."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time


class SubmissionLedger:
    def __init__(self, directory: Path, analytical_limit=128, spatial_limit=3,
                 wall_limit_s=900):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.journal = self.directory / "submissions.jsonl"
        self.receipts = self.directory / "receipts.jsonl"
        self.limits = {"reduced": analytical_limit, "spatial": spatial_limit}
        self.wall_limit_s = wall_limit_s

    @staticmethod
    def _read(path):
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    @staticmethod
    def _append(path, row):
        with path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def submissions(self):
        return self._read(self.journal)

    def completions(self):
        return self._read(self.receipts)

    def started_unix_s(self):
        entries = self.submissions()
        return entries[0]["submitted_unix_s"] if entries else None

    def remaining(self, level):
        return max(0, self.limits[level] - sum(r["level"] == level
                                               for r in self.submissions()))

    def reserve(self, level, key, stage):
        if level not in self.limits:
            raise ValueError("unknown solver level")
        entries = self.submissions()
        if sum(r["level"] == level for r in entries) >= self.limits[level]:
            raise RuntimeError(f"{level} submission budget exhausted")
        now = time.time()
        if entries and now-entries[0]["submitted_unix_s"] >= self.wall_limit_s:
            raise RuntimeError("forward wall-time budget exhausted")
        evaluation_id = f"astv2-{len(entries)+1:04d}"
        row = {"evaluation_id": evaluation_id, "level": level, "key": key,
               "stage": stage, "submitted_unix_s": now}
        self._append(self.journal, row)
        return row

    def complete(self, evaluation_id, outcome):
        submitted = {r["evaluation_id"]: r for r in self.submissions()}
        if evaluation_id not in submitted:
            raise RuntimeError("completion without reservation")
        if evaluation_id in {r["evaluation_id"] for r in self.completions()}:
            raise RuntimeError("duplicate completion")
        self._append(self.receipts, {"evaluation_id":evaluation_id,
                    "completed_unix_s":time.time(),"outcome":outcome})

    def audit(self):
        submitted = self.submissions()
        completed = {r["evaluation_id"] for r in self.completions()}
        return {"submitted_reduced":sum(r["level"]=="reduced" for r in submitted),
                "submitted_spatial":sum(r["level"]=="spatial" for r in submitted),
                "completed":len(completed),
                "unresolved_ids":[r["evaluation_id"] for r in submitted
                                  if r["evaluation_id"] not in completed],
                "first_submission_unix_s":self.started_unix_s(),
                "wall_limit_s":self.wall_limit_s,"limits":self.limits}
