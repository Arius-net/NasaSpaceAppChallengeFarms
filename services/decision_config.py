"""Carga centralizada de `config/decision_thresholds.yaml` (única fuente de verdad).

Todos los consumidores (motor de reglas, prompt del LLM, conversión GWET del
servicio NASA) leen los umbrales desde aquí. Nunca copiar números a mano.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

BACKEND_DIR = Path(__file__).resolve().parents[1]
THRESHOLDS_PATH = BACKEND_DIR / "config" / "decision_thresholds.yaml"


@lru_cache(maxsize=1)
def load_thresholds() -> dict[str, Any]:
    """Lee y valida mínimamente el YAML de umbrales (cacheado)."""
    if not THRESHOLDS_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró la configuración de umbrales: {THRESHOLDS_PATH}"
        )
    with open(THRESHOLDS_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    for clave in ("siembra", "alerta_sequia", "humedad_suelo"):
        if clave not in cfg:
            raise KeyError(f"decision_thresholds.yaml sin la sección '{clave}'")
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
