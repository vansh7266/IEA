from __future__ import annotations

import json
import uuid
from collections import Counter
from pathlib import Path
from typing import Iterable

from .models import FoldedContext, SessionState, TurnRecord


class ContextFolder:
    def fold(self, turns: Iterable[TurnRecord]) -> FoldedContext:
        turns = list(turns)
        if not turns:
            return FoldedContext(
                summary="No prior edits. Start from the source image and preserve non-target regions by default.",
                active_entities=[],
                constraints=[
                    "Preserve regions outside the inferred bounding box unless the edit is explicitly global.",
                    "Use preview generation before locking the edit box.",
                ],
                turn_count=0,
            )

        target_counts = Counter()
        verb_counts = Counter()
        rewards = []
        for turn in turns:
            rewards.append(turn.reward)
            for edit in turn.parsed_edits:
                target_counts[edit.target] += 1
                verb_counts[edit.verb] += 1

        entities = [target for target, _ in target_counts.most_common(6)]
        primary_verbs = ", ".join(verb for verb, _ in verb_counts.most_common(3))
        reward_summary = 0.0 if not rewards else sum(rewards) / len(rewards)
        summary = (
            f"{len(turns)} prior edit turn(s). "
            f"Most frequent targets: {', '.join(entities) if entities else 'none'}. "
            f"Most common edit verbs: {primary_verbs or 'none'}. "
            f"Average quality reward so far: {reward_summary:.2f}."
        )
        constraints = [
            "Preserve regions outside the inferred bounding box unless the edit is explicitly global.",
            "Reuse prior entities and style cues when the user continues a session.",
            "Prefer lower-risk local edits before broad restyling steps.",
        ]
        return FoldedContext(
            summary=summary,
            active_entities=entities,
            constraints=constraints,
            turn_count=len(turns),
        )


class SessionStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"

    def load(self, session_id: str) -> SessionState:
        path = self._path(session_id)
        if not path.exists():
            return SessionState(session_id=session_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return SessionState.from_dict(payload)

    def create(self) -> SessionState:
        return SessionState(session_id=uuid.uuid4().hex[:12])

    def load_or_create(self, session_id: str | None) -> SessionState:
        if session_id:
            return self.load(session_id)
        return self.create()

    def save(self, state: SessionState) -> None:
        path = self._path(state.session_id)
        path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
