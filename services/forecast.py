"""Servicio de pronóstico K'INAL (Fase 3): GWETROOT a 7 días con banda de incertidumbre.

- Modelo: XGBoost cuantílico (0.10/0.50/0.90) entrenado con datos multianuales
  del punto piloto (ver 12_SCRIPTS/ml/train_forecast_model.py, D-0005/EV-0009).
- Honestidad (decisión 3.1): si las métricas de test indican que el modelo NO
  supera a la persistencia, la respuesta lo declara y aplica MODO DEGRADADO
  (se entrega el estado actual como referencia, con leyenda visible).
- Reglas: los umbrales se leen de config/decision_thresholds.yaml (fuente única).
"""
from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from services.decision_config import load_thresholds
from services.decision_rules import cap_confianza_para
from services.nasa_api import get_nasa_weather_data, get_recent_power_dataframe

BACKEND_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BACKEND_DIR / "models"
BUNDLE_PATH = MODEL_DIR / "gwetroot_t7.joblib"
META_PATH = MODEL_DIR / "forecast_metadata.json"

_lock = threading.Lock()


@lru_cache(maxsize=1)
def _cargar_modelo() -> dict[str, Any] | None:
    """Carga el bundle de modelos cuantílicos (cacheado)."""
    with _lock:
        if not BUNDLE_PATH.exists():
            return None
        return joblib.load(BUNDLE_PATH)


@lru_cache(maxsize=1)
def cargar_metadata() -> dict[str, Any] | None:
    if not META_PATH.exists():
        return None
    with open(META_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def recargar_modelo() -> bool:
    """Fuerza recarga (tras reentrenar) — usado por /api/config/reload y el scheduler."""
    _cargar_modelo.cache_clear()
    cargar_metadata.cache_clear()
    return _cargar_modelo() is not None


def modelo_disponible() -> bool:
    return _cargar_modelo() is not None


def _preparar_serie_completa(df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    if df.empty:
        weather = get_nasa_weather_data(lat=lat, lon=lon)
        serie = weather.get("serie_diaria", [])
        if serie:
            df = pd.DataFrame(
                [
                    {
                        "PRECTOTCORR": float(d.get("precipitacion_mm", 0.0)),
                        "T2M_MAX": float(d.get("temp_max_c", 25.0)),
                        "T2M_MIN": float(d.get("temp_min_c", 15.0)),
                        "T2M": float(d.get("temp_media_c", 20.0)),
                        "RH2M": float(d.get("humedad_relativa_pct", 65.0)),
                    }
                    for d in serie
                ],
                index=pd.to_datetime([d.get("fecha") for d in serie], errors="coerce"),
            )
    df = df.copy()
    defaults = {
        "PRECTOTCORR": 0.0,
        "T2M_MAX": 25.0,
        "T2M_MIN": 15.0,
        "T2M": 20.0,
        "RH2M": 65.0,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").ffill().bfill().fillna(val)

    rain_7d = df["PRECTOTCORR"].rolling(7, min_periods=1).sum()
    rh_7d = df["RH2M"].rolling(7, min_periods=1).mean()
    gwettop_est = np.clip(0.22 + (rain_7d * 0.0078) + ((rh_7d - 55.0) * 0.0033), 0.02, 1.0)
    gwetroot_est = np.clip(0.27 + (rain_7d * 0.0040) + ((rh_7d - 55.0) * 0.0022), 0.02, 1.0)

    if "GWETTOP" not in df.columns:
        df["GWETTOP"] = gwettop_est
    else:
        df["GWETTOP"] = pd.to_numeric(df["GWETTOP"], errors="coerce").ffill().bfill().fillna(gwettop_est)

    if "GWETROOT" not in df.columns:
        df["GWETROOT"] = gwetroot_est
    else:
        df["GWETROOT"] = pd.to_numeric(df["GWETROOT"], errors="coerce").ffill().bfill().fillna(gwetroot_est)

    return df


def _construir_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["Rain_Roll3"] = df["PRECTOTCORR"].rolling(3, min_periods=1).sum()
    df["Rain_Roll7"] = df["PRECTOTCORR"].rolling(7, min_periods=1).sum()
    df["Temp_Roll3"] = df["T2M"].rolling(3, min_periods=1).mean()
    df["RootWet_Lag1"] = df["GWETROOT"].shift(1).bfill().fillna(df["GWETROOT"])
    df["RootWet_Lag3"] = df["GWETROOT"].shift(3).bfill().fillna(df["GWETROOT"])
    return df.dropna(subset=features).tail(1)


def get_forecast(lat: float, lon: float, timeout_seconds: float = 8.0) -> dict[str, Any]:
    """Pronóstico de humedad radicular a 7 días con banda 10-90% y degradación honesta."""
    bundle = _cargar_modelo()
    metadata = cargar_metadata()
    cfg = load_thresholds()

    if bundle is None or metadata is None:
        return {
            "status": "indisponible",
            "motivo": "Modelo no entrenado aún (ejecutar 12_SCRIPTS/ml/train_forecast_model.py).",
            "degradacion": {
                "nivel": "sin_modelo",
                "leyenda": "Pronóstico no disponible: se usa el estado actual como referencia.",
            },
        }

    features = list(bundle["features"])
    raw_df = get_recent_power_dataframe(lat, lon, days_back=35, timeout_seconds=timeout_seconds)
    df = _preparar_serie_completa(raw_df, lat, lon)
    fila = _construir_features(df, features)

    X = fila[features]
    p10 = float(np.clip(bundle["cuantiles"]["q10"].predict(X)[0], 0.0, 1.0))
    p50 = float(np.clip(bundle["cuantiles"]["q50"].predict(X)[0], 0.0, 1.0))
    p90 = float(np.clip(bundle["cuantiles"]["q90"].predict(X)[0], 0.0, 1.0))

    gwetroot_actual = float(df["GWETROOT"].dropna().tail(5).mean())
    gwettop_actual = float(df["GWETTOP"].dropna().tail(5).mean()) if "GWETTOP" in df.columns else None
    fecha_dato = str(df.dropna(subset=["PRECTOTCORR"]).index.max().date()) if df["PRECTOTCORR"].notna().any() else None

    alerta = cfg["alerta_sequia"]
    estres_min = float(alerta["estres_min"])
    precaucion_min = float(alerta["precaucion_min"])

    def clasificar(valor: float) -> str:
        if valor < estres_min:
            return "ALERTA: Estrés Hídrico"
        if valor < precaucion_min:
            return "PRECAUCIÓN: Bajando reserva"
        return "ÓPTIMO"

    supera_rmse = bool(metadata.get("supera_persistencia_rmse"))
    supera_costo = bool(metadata.get("supera_persistencia_costo"))
    m = metadata.get("metricas_test", {})
    fecha_validez = (datetime.now(UTC) + timedelta(days=7)).strftime("%Y-%m-%d")

    if supera_rmse and supera_costo:
        nivel_degradacion = "operativo"
        leyenda = (
            "Pronóstico del modelo ML (XGBoost cuantílico): en el test cronológico superó a la "
            f"persistencia en RMSE ({m.get('rmse_modelo')} vs {m.get('rmse_persistencia')}) y en "
            f"costo de decisión ({m.get('costo_decision_modelo')} vs {m.get('costo_decision_persistencia')}); "
            f"banda 10-90% con cobertura empírica {m.get('cobertura_banda_10_90', 0):.0%}."
        )
    elif supera_rmse or supera_costo:
        nivel_degradacion = "operativo_con_salvedades"
        leyenda = (
            "Pronóstico operativo con SALVEDAD honesta: "
            f"RMSE {'supera' if supera_rmse else 'NO supera'} la persistencia "
            f"({m.get('rmse_modelo')} vs {m.get('rmse_persistencia')}) pero el costo de decisión "
            f"{'supera' if supera_costo else 'NO supera'} la persistencia "
            f"({m.get('costo_decision_modelo')} vs {m.get('costo_decision_persistencia')}). "
            "Interpretación: más precisión global, pero más señales de siembra falsas cerca del "
            "umbral — use la banda 10-90% y trate el valor central (p50) como orientativo."
        )
    else:
        nivel_degradacion = "modelo_sin_valor_agregado"
        leyenda = (
            "MODO DEGRADADO: en la evaluación honesta el modelo NO superó a la persistencia "
            "(regla DECISION_SPEC: no se presenta como mejora). Se muestra el estado actual "
            "como referencia y el modelo solo a título informativo."
        )

    cap = cap_confianza_para(lat, lon)
    payload = {
        "status": "success",
        "timestamp_pronostico": datetime.now(UTC).isoformat(),
        "coordenadas": {"latitud": lat, "longitud": lon},
        "fecha_dato": fecha_dato,
        "fecha_validez_pronostico": fecha_validez,
        "horizonte_dias": 7,
        "estado_actual": {
            "gwetroot": round(gwetroot_actual, 3),
            "gwettop": round(gwettop_actual, 3) if gwettop_actual is not None else None,
            "clasificacion": clasificar(gwetroot_actual),
            "unidad": "fracción de saturación (0-1)",
        },
        "pronostico": {
            "gwetroot_p10": round(p10, 3),
            "gwetroot_p50": round(p50, 3),
            "gwetroot_p90": round(p90, 3),
            "clasificacion_escenario_central": clasificar(p50),
            "riesgo_anticipado": {
                "banda_inferior_bajo_estres": bool(p10 < estres_min),
                "nota": (
                    "Aun en el escenario húmedo (p10) la reserva caería bajo el umbral de estrés "
                    f"({estres_min}) — anticipa ALERTA con los umbrales validados."
                    if p10 < estres_min
                    else "La banda 10-90% no anticipa cruce del umbral de estrés en 7 días."
                ),
            },
        },
        "degradacion": {
            "nivel": nivel_degradacion,
            "leyenda": leyenda,
            "supera_persistencia_rmse": supera_rmse,
            "supera_persistencia_costo": supera_costo,
        },
        "confianza_max": cap,
        "modelo": {
            "tipo": "XGBoost cuantílico 0.10/0.50/0.90",
            "entrenado_hasta": metadata.get("datos", {}).get("fin"),
            "metricas_test": metadata.get("metricas_test"),
            "escala": "fracción de saturación 0-1 (nativa, EV-0009)",
        },
        "supuestos": [
            "Pronóstico basado en la serie reciente de NASA POWER (punto piloto entrenado en "
            "Villaflores); fuera del piloto es extrapolación.",
            "La banda 10-90% refleja la incertidumbre del modelo, no la de las observaciones "
            "futuras (GLDAS ~0.5°×0.625°).",
            "Los umbrales usados provienen de decision_thresholds.yaml (EV-0009).",
        ],
    }
    return payload
