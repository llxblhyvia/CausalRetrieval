from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence


@dataclass
class ContactEvent:
    triggered: bool
    step: int | None = None
    ncon: int = 0
    filtered: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


class ContactDetector:
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        contact_cfg = cfg.get("contact", {})
        self.use_filtered_contacts = bool(contact_cfg.get("use_filtered_contacts", True))
        self.gripper_patterns = tuple(str(x).lower() for x in contact_cfg.get("gripper_name_patterns", []))
        self.target_patterns = tuple(str(x).lower() for x in contact_cfg.get("target_name_patterns", []))
        target_name = cfg.get("env", {}).get("target_object_name")
        if target_name:
            self.target_patterns += (str(target_name).lower(),)
        self.min_contact_steps = int(contact_cfg.get("min_contact_steps", 1))
        self.fired = False
        self._consecutive = 0

    def reset(self) -> None:
        self.fired = False
        self._consecutive = 0

    def check(self, env: Any, step: int) -> ContactEvent:
        if self.fired:
            return ContactEvent(False)
        sim = getattr(env, "sim", getattr(getattr(env, "env", None), "sim", None))
        data = getattr(sim, "data", None)
        ncon = int(getattr(data, "ncon", 0) or 0)
        matched, details = self._filtered_match(sim, data, ncon)
        raw_contact = ncon > 0
        is_contact = matched if self.use_filtered_contacts and details.get("filter_available") else raw_contact
        self._consecutive = self._consecutive + 1 if is_contact else 0
        if self._consecutive >= self.min_contact_steps:
            self.fired = True
            return ContactEvent(True, step=step, ncon=ncon, filtered=matched, details=details)
        return ContactEvent(False, ncon=ncon, filtered=matched, details=details)

    def _filtered_match(self, sim: Any, data: Any, ncon: int) -> tuple[bool, Dict[str, Any]]:
        details: Dict[str, Any] = {"filter_available": False, "pairs": []}
        if sim is None or data is None or ncon <= 0:
            return False, details
        model = getattr(sim, "model", None)
        contacts = getattr(data, "contact", None)
        if model is None or contacts is None:
            return False, details
        details["filter_available"] = True
        for idx in range(ncon):
            contact = contacts[idx]
            geom1 = self._geom_name(model, int(getattr(contact, "geom1", -1)))
            geom2 = self._geom_name(model, int(getattr(contact, "geom2", -1)))
            pair = (geom1, geom2)
            details["pairs"].append(pair)
            if self._pair_matches(pair):
                return True, details
        return False, details

    def _geom_name(self, model: Any, geom_id: int) -> str:
        if geom_id < 0:
            return ""
        if hasattr(model, "geom_id2name"):
            return str(model.geom_id2name(geom_id) or "")
        if hasattr(model, "id2name"):
            try:
                return str(model.id2name(geom_id, "geom") or "")
            except TypeError:
                return str(model.id2name("geom", geom_id) or "")
        return str(geom_id)

    def _pair_matches(self, pair: Sequence[str]) -> bool:
        a, b = (str(pair[0]).lower(), str(pair[1]).lower())
        has_gripper = self._matches_any(a, self.gripper_patterns) or self._matches_any(b, self.gripper_patterns)
        if not self.target_patterns:
            return has_gripper
        has_target = self._matches_any(a, self.target_patterns) or self._matches_any(b, self.target_patterns)
        return has_gripper and has_target

    @staticmethod
    def _matches_any(name: str, patterns: Sequence[str]) -> bool:
        return any(pattern and pattern in name for pattern in patterns)
