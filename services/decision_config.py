"""Carga centralizada de `config/decision_thresholds.yaml` (única fuente de verdad).

Todos los consumidores (motor de reglas, prompt del LLM, conversión GWET del
servicio NASA) leen los umbrales desde aquí. Nunca copiar números a mano.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

BACKEND_DIR = Path(__file__).resolve().parent.parent
THRESHOLDS_PATH = BACKEND_DIR / "config" / "decision_thresholds.yaml"

_DEFAULT_THRESHOLDS: dict[str, Any] = {
    "version": "0.4.0",
    "validated": True,
    "confianza_max": "ALTA",
    "humedad_suelo": {"escala": "fracción de saturación 0-1 (escala nativa POWER/GLDAS)"},
    "siembra": {
        "sembrar": {"superficie_min": 0.47, "radicular_min": 0.362},
        "falso_inicio": {
            "superficie_min": 0.47,
            "superficie_min_banda": 0.44,
            "radicular_max": 0.362,
            "radicular_transicion_min": 0.332,
        },
        "suelo_seco": {},
    },
    "alerta_sequia": {"estres_min": 0.35, "precaucion_min": 0.362},
    "gdd_maiz": {
        "temperatura_base_c": 10.0,
        "ciclo_completo_gdd_c_aprox": 1530,
        "floracion_gdd_c_aprox": 840,
    },
    "viabilidad": {
        "gdd_min_maiz_c": 1200,
        "dias_helada_max_ciclo": 15,
        "precip_anual_min_temporal_mm": 400,
        "modis_clases": {
            "prohibidas": [0, 13, 15, 16, 17],
            "permitidas": [12, 14],
            "marginales": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
    },
}


@lru_cache(maxsize=1)
def load_thresholds() -> dict[str, Any]:
    """Lee y valida mínimamente el YAML de umbrales (cacheado)."""
    if not THRESHOLDS_PATH.exists():
        return dict(_DEFAULT_THRESHOLDS)
    with open(THRESHOLDS_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or dict(_DEFAULT_THRESHOLDS)

    for clave in ("siembra", "alerta_sequia", "humedad_suelo"):
        if clave not in cfg:
            cfg[clave] = _DEFAULT_THRESHOLDS[clave]
    return cfg


def reload_thresholds() -> dict[str, Any]:
    """Fuerza recarga (útil tras editar el YAML sin reiniciar el servidor)."""
    load_thresholds.cache_clear()
    return load_thresholds()


def resumen_umbrales(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resumen legible de los umbrales activos (para respuestas y logs)."""
    cfg = cfg or load_thresholds()
    return {
        "version": cfg.get("version"),
        "validated": bool(cfg.get("validated", False)),
        "confianza_max": cfg.get("confianza_max"),
        "superficie_min": cfg["siembra"]["sembrar"]["superficie_min"],
        "radicular_min": cfg["siembra"]["sembrar"]["radicular_min"],
        "estres_min": cfg["alerta_sequia"]["estres_min"],
        "precaucion_min": cfg["alerta_sequia"]["precaucion_min"],
    }
