"""Registro de recomendaciones de K'INAL (decisión 0.4: A y B).

Cada recomendación emitida queda reconstruible (arquitectura §11_OUTPUTS):
- SQLite operativo: `backend/data/kinal_recommendations.db` (consultable).
- Espejo CSV versionable: `KINAL-main/11_OUTPUTS/recommendations/recommendations_log.csv`.

Campos alineados con la arquitectura §7.2 (`11_OUTPUTS/recommendations/`):
confidence / uncertainty_band / assumptions / data_vintage / model_version /
baseline_option / human_action / override_reason.
"""
from __future__ import annotations

import csv
import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from typing import Any

from services.paths import BACKEND_DIR, RECOMMENDATIONS_DIR

DB_PATH = BACKEND_DIR / "data" / "kinal_recommendations.db"
CSV_PATH = RECOMMENDATIONS_DIR / "recommendations_log.csv"

CSV_FIELDS = [
    "recommendation_id",
    "date",
    "decision_type",
    "parcel_id",
    "option_recommended",
    "confidence",
    "uncertainty_band",
    "assumptions",
    "data_vintage",
    "model_version",
    "baseline_option",
    "human_action",
    "override_reason",
]

_lock = threading.Lock()


def _ensure_storage() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendations (
                recommendation_id TEXT PRIMARY KEY,
                date TEXT NOT NULL,
                decision_type TEXT NOT NULL,
                parcel_id TEXT,
                option_recommended TEXT,
                confidence TEXT,
                uncertainty_band TEXT,
                assumptions TEXT,
                data_vintage TEXT,
                model_version TEXT,
                baseline_option TEXT,
                human_action TEXT,
                override_reason TEXT,
                payload_json TEXT
            )
            """
        )
    RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(CSV_FIELDS)


def log_recommendation(
    decision_type: str,
    parcel_id: str,
    option_recommended: str,
    confidence: str,
    uncertainty_band: str,
    assumptions: list[str],
    data_vintage: str,
    model_version: str,
    payload: dict[str, Any],
    baseline_option: str = "calendario tradicional / siembra en mayo (línea base v1)",
) -> str:
    """Registra una recomendación y devuelve su ID (REC-YYYYMMDD-XXXXXXXX)."""
    _ensure_storage()
    rid = f"REC-{datetime.now(UTC).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"
    row = {
        "recommendation_id": rid,
        "date": datetime.now(UTC).isoformat(),
        "decision_type": decision_type,
        "parcel_id": parcel_id,
        "option_recommended": option_recommended,
        "confidence": confidence,
        "uncertainty_band": uncertainty_band,
        "assumptions": json.dumps(assumptions, ensure_ascii=False),
        "data_vintage": data_vintage,
        "model_version": model_version,
        "baseline_option": baseline_option,
        "human_action": "",
        "override_reason": "",
    }
    with _lock:
        with sqlite3.connect(DB_PATH) as con:
            con.execute(
                """
                INSERT INTO recommendations
                    (recommendation_id, date, decision_type, parcel_id,
                     option_recommended, confidence, uncertainty_band,
                     assumptions, data_vintage, model_version,
                     baseline_option, human_action, override_reason, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rid, row["date"], decision_type, parcel_id,
                    option_recommended, confidence, uncertainty_band,
                    row["assumptions"], data_vintage, model_version,
                    baseline_option, "", "", json.dumps(payload, ensure_ascii=False),
                ),
            )
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=CSV_FIELDS).writerow(row)
    return rid


def register_override(recommendation_id: str, human_action: str, override_reason: str) -> bool:
    """Registra lo que el decisor hizo finalmente (3.3-A: anónimo).

    El CSV no se sobrescribe: se agrega una fila nueva con el mismo ID
    (mismo estilo de asientos que DECISION_LOG).
    """
    _ensure_storage()
    with _lock:
        with sqlite3.connect(DB_PATH) as con:
            cur = con.execute(
                "SELECT payload_json FROM recommendations WHERE recommendation_id = ?",
                (recommendation_id,),
            )
            row = cur.fetchone()
            if row is None:
                return False
            con.execute(
                """
                UPDATE recommendations
                SET human_action = ?, override_reason = ?
                WHERE recommendation_id = ?
                """,
                (human_action, override_reason, recommendation_id),
            )
            con.execute(
                "SELECT decision_type, parcel_id, option_recommended, confidence, "
                "uncertainty_band, assumptions, data_vintage, model_version, "
                "baseline_option, date FROM recommendations WHERE recommendation_id = ?",
                (recommendation_id,),
            )
            original = con.execute(
                "SELECT * FROM recommendations WHERE recommendation_id = ?",
                (recommendation_id,),
            ).fetchone()
            columns = [d[0] for d in con.execute("SELECT * FROM recommendations LIMIT 1").description]
            original_map = dict(zip(columns, original, strict=True))
        csv_row = {
            "recommendation_id": recommendation_id,
            "date": datetime.now(UTC).isoformat(),
            "decision_type": original_map.get("decision_type", ""),
            "parcel_id": original_map.get("parcel_id", ""),
            "option_recommended": original_map.get("option_recommended", ""),
            "confidence": original_map.get("confidence", ""),
            "uncertainty_band": original_map.get("uncertainty_band", ""),
            "assumptions": original_map.get("assumptions", ""),
            "data_vintage": original_map.get("data_vintage", ""),
            "model_version": original_map.get("model_version", ""),
            "baseline_option": original_map.get("baseline_option", ""),
            "human_action": human_action,
            "override_reason": override_reason,
        }
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=CSV_FIELDS).writerow(csv_row)
    return True
