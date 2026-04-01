from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from typing import Any
import json


@dataclass(frozen=True)
class OrganPack:
    id: str
    display_name: str
    annotation_taxonomy: str
    description: str
    default_study_context: str
    supported_input_layout: str
    workflow_defaults: dict[str, Any]
    artifact_contract: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "annotation_taxonomy": self.annotation_taxonomy,
            "description": self.description,
            "default_study_context": self.default_study_context,
            "supported_input_layout": self.supported_input_layout,
            "workflow_defaults": self.workflow_defaults,
            "artifact_contract": self.artifact_contract,
        }


def _load_pack_payload(pack_id: str) -> dict[str, Any]:
    data_path = files("spatho.organ_packs").joinpath("data", f"{pack_id}.json")
    with data_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_pack(pack_id: str) -> OrganPack:
    payload = _load_pack_payload(pack_id)
    return OrganPack(
        id=str(payload["id"]),
        display_name=str(payload["display_name"]),
        annotation_taxonomy=str(payload["annotation_taxonomy"]),
        description=str(payload["description"]),
        default_study_context=str(payload["default_study_context"]),
        supported_input_layout=str(payload["supported_input_layout"]),
        workflow_defaults=dict(payload["workflow_defaults"]),
        artifact_contract=dict(payload["artifact_contract"]),
    )


def list_organ_packs() -> list[OrganPack]:
    data_dir = files("spatho.organ_packs").joinpath("data")
    packs: list[OrganPack] = []
    for resource in sorted(data_dir.iterdir(), key=lambda item: item.name):
        if resource.name.endswith(".json"):
            packs.append(_build_pack(resource.name[:-5]))
    return packs


def get_organ_pack(pack_id: str) -> OrganPack:
    normalized = pack_id.strip().lower()
    for pack in list_organ_packs():
        if pack.id == normalized:
            return pack
    available = ", ".join(pack.id for pack in list_organ_packs())
    raise ValueError(f"Unsupported organ pack '{pack_id}'. Available packs: {available}")
