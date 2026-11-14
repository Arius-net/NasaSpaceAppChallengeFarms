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

BACKEND_DIR = Path(__file__).resolve().parents[1]


def _resolve_kinal_main_dir() -> Path:
    override = os.getenv("KINAL_MAIN_DIR", "").strip()
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    candidates += [
        BACKEND_DIR.parent / "KINAL-main",
        BACKEND_DIR.parents[1] / "KINAL-main",
        BACKEND_DIR / "KINAL-main",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    # Ubicación esperada por defecto (para mensajes de error claros)
    return (BACKEND_DIR.parent / "KINAL-main").resolve()


KINAL_MAIN_DIR = _resolve_kinal_main_dir()

# Artefactos versionados del entregable D01 (solo lectura)
GEOJSON_FRAILESCA = KINAL_MAIN_DIR / "05_DATA" / "reference" / "frailesca_municipios.geojson"
CSV_POWER_FRAILESCA = (
    KINAL_MAIN_DIR / "04_RESEARCH" / "D01_inventory" / "artifacts" / "nasa_power_frailesca.csv"
)
RECOMMENDATIONS_DIR = KINAL_MAIN_DIR / "11_OUTPUTS" / "recommendations"

# Bbox del núcleo municipal Frailesca (hallazgo F-05 de D01, evidencia EV-0005):
# delimita el ámbito del piloto validado y el uso del CSV de respaldo D01.
FRAILESCA_BBOX = {
    "lat_min": 15.6645, "lat_max": 16.5858,
    "lon_min": -93.7710, "lon_max": -92.2955,
}
