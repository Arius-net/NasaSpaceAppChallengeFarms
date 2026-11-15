"""Resolución robusta de rutas del repositorio KINAL-main (decisión 0.1-A).

Orden de resolución:
1. Variable de entorno `KINAL_MAIN_DIR` (override explícito).
2. `kinal-app/KINAL-main` (ubicación actual del proyecto).
3. `hackaton/KINAL-main` (ubicación histórica, un nivel arriba).
4. `backend/KINAL-main` (empaquetado).
"""
from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _resolve_kinal_main_dir() -> Path:
    override = os.getenv("KINAL_MAIN_DIR", "").strip()
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    try:
        candidates.append(BACKEND_DIR.parent / "KINAL-main")
        candidates.append(BACKEND_DIR.parents[1] / "KINAL-main")
    except IndexError:
        pass
    candidates.append(BACKEND_DIR / "KINAL-main")
    candidates.append(Path.cwd() / "KINAL-main")
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate.resolve()
        except OSError:
            continue
    return BACKEND_DIR.resolve()


KINAL_MAIN_DIR = _resolve_kinal_main_dir()

# Artefactos versionados del entregable D01 (solo lectura)
GEOJSON_FRAILESCA = KINAL_MAIN_DIR / "05_DATA" / "reference" / "frailesca_municipios.geojson"
CSV_POWER_FRAILESCA = (
    KINAL_MAIN_DIR / "04_RESEARCH" / "D01_inventory" / "artifacts" / "nasa_power_frailesca.csv"
)
RECOMMENDATIONS_DIR = KINAL_MAIN_DIR / "11_OUTPUTS" / "recommendations"

FRAILESCA_BBOX = {
    "lat_min": -90.0, "lat_max": 90.0,
    "lon_min": -180.0, "lon_max": 180.0,
}
