"""K'INAL - Backend API (FastAPI) | PASO 5 + FASE 0: API con umbrales unificados y trazabilidad."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services.ai_advisor import generate_agronomic_recommendation
from services.decision_config import load_thresholds, reload_thresholds, resumen_umbrales
from services.farm_history import get_farm_context_for_llm
from services.forecast import cargar_metadata, get_forecast, modelo_disponible, recargar_modelo
from services.geo_viability import evaluar_viabilidad, mensaje_no_viable
from services.land_cover import precargar_puntos
from services.paths import GEOJSON_FRAILESCA
from services.recommendation_log import log_recommendation, register_override
from services.nasa_api import get_nasa_weather_data

logger = logging.getLogger("uvicorn.error")

load_dotenv()

FRAILESCA_GEOJSON_PATH = GEOJSON_FRAILESCA

app = FastAPI(
    title="K'INAL Decision-Support API",
    description=(
        "API Backend de K'INAL para extracción agroclimática NASA e IA Agronómica "
        "(NASA Space Apps Challenge). Umbrales validados multianualmente (EV-0009) "
        "en escala nativa, preflight de viabilidad geográfica-agroclimática "
        "(rechazo determinista en agua/hielo/Polo antes del motor y del LLM), "
        "pronóstico GWETROOT a 7 días con banda de incertidumbre y degradación "
        "honesta, registro de recomendaciones y overrides del decisor."
    ),
    version="0.9.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CoordinateRequest(BaseModel):
    lat: float = Field(16.2333, ge=-90.0, le=90.0, description="Latitud de la parcela")
    lon: float = Field(-93.2667, ge=-180.0, le=180.0, description="Longitud de la parcela")
    farm_id: str | None = Field(None, description="ID de la parcela en Supabase (opcional)")


@app.get("/api/forecast", tags=["Forecast"])
def get_forecast_endpoint(
    lat: float = Query(
        16.2333,
        ge=-90.0,
        le=90.0,
        description="Latitud de la parcela",
    ),
    lon: float = Query(
        -93.2667,
        ge=-180.0,
        le=180.0,
        description="Longitud de la parcela",
    ),
) -> dict[str, Any]:
    """Pronóstico de GWETROOT a 7 días (banda 10-90%) con leyenda de degradación honesta (decisión 3.1)."""
    viab = evaluar_viabilidad(lat, lon)
    resultado = get_forecast(lat, lon)
    resultado["viabilidad"] = viab
    return resultado


class OverrideRequest(BaseModel):
    human_action: str = Field(
        ...,
        description="Lo que el decisor finalmente hará: SEMBRAR | ESPERAR | NO_SEMBRAR | OTRA",
    )
    motivo: str = Field("", description="Motivo del override (opcional, anónimo)")


@app.get("/api/health", tags=["Health"])
def health_check() -> dict[str, Any]:
    """Estado del servidor + resumen de umbrales y del modelo de pronóstico."""
    try:
        umbrales = resumen_umbrales(load_thresholds())
    except Exception as exc:  # noqa: BLE001
        umbrales = {"error": str(exc)}
    meta = cargar_metadata() or {}
    return {
        "status": "ok",
        "service": "K'INAL Backend API",
        "step": "FASE 0-4 - Umbrales validados (EV-0009) + preflight viabilidad + pronóstico a 7 días + trazabilidad",
        "timestamp": datetime.now(UTC).isoformat(),
        "umbrales": umbrales,
        "forecast": {
            "modelo_disponible": modelo_disponible(),
            "supera_persistencia_rmse": meta.get("supera_persistencia_rmse"),
            "supera_persistencia_costo": meta.get("supera_persistencia_costo"),
            "entrenado_hasta": (meta.get("datos") or {}).get("fin"),
        },
        "scheduler": {"activo": _scheduler is not None and _scheduler.running},
    }


@app.post("/api/config/reload", tags=["Health"])
def reload_config() -> dict[str, Any]:
    """Recarga umbrales y modelo de pronóstico sin reiniciar el servidor (para editar en demo)."""
    cfg = reload_thresholds()
    modelo_ok = recargar_modelo()
    return {"status": "ok", "umbrales": resumen_umbrales(cfg), "modelo_forecast_recargado": modelo_ok}


@app.get("/api/frailesca-geojson", tags=["Geospatial Reference"])
def get_frailesca_geojson() -> dict[str, Any]:
    """Sirve en modo solo lectura los polígonos municipales de La Frailesca (05_DATA/reference)."""
    if not FRAILESCA_GEOJSON_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "GeoJSON de La Frailesca no encontrado en "
                f"{FRAILESCA_GEOJSON_PATH} (configura KINAL_MAIN_DIR si está en otra ruta)"
            ),
        )
    with open(FRAILESCA_GEOJSON_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


@app.get("/api/weather", tags=["NASA Weather"])
def get_weather(
    lat: float = Query(
        16.2333,
        ge=-90.0,
        le=90.0,
        description="Latitud de la parcela (por defecto: Villaflores, La Frailesca, Chiapas)",
    ),
    lon: float = Query(
        -93.2667,
        ge=-180.0,
        le=180.0,
        description="Longitud de la parcela (por defecto: Villaflores, La Frailesca, Chiapas)",
    ),
) -> dict[str, Any]:
    """Consulta datos agroclimáticos de NASA POWER para (lat, lon) con fallback rápido."""
    return get_nasa_weather_data(lat=lat, lon=lon)


def _recomendar(
    lat: float, lon: float, farm_context: str | None = None
) -> dict[str, Any]:
    """Genera la recomendación y la registra (SQLite + espejo CSV 11_OUTPUTS)."""
    viab = evaluar_viabilidad(lat, lon)
    weather_data = get_nasa_weather_data(lat=lat, lon=lon)
    forecast_data = get_forecast(lat=lat, lon=lon)

    if viab.get("veredicto") == "NO_VIABLE":
        rec = mensaje_no_viable(viab, lat, lon)
        serie = weather_data.get("serie_diaria", []) or []
        rec["data_vintage"] = {
            "fecha_dato": serie[-1].get("fecha") if serie else None,
            "fuente": weather_data.get("fuente"),
            "modo_extraccion": weather_data.get("modo_extraccion"),
            "timestamp_consulta": weather_data.get("timestamp_consulta"),
        }
        rec["umbrales_aplicados"] = resumen_umbrales(load_thresholds())
        result: dict[str, Any] = {
            "status": "success",
            "timestamp_recomendacion": datetime.now(UTC).isoformat(),
            "motor_ia": {
                "proveedor": "Preflight de viabilidad (determinista, sin LLM)",
                "nota": "NO_VIABLE resuelto por composición/régimen de la zona; no se consultó LLM.",
            },
            "coordenadas": weather_data.get("coordenadas", {"latitud": lat, "longitud": lon}),
            "recomendacion_agronomica": rec,
            "viabilidad": viab,
            "datos_meteorologicos": {
                "fuente": weather_data.get("fuente"),
                "modo_extraccion": weather_data.get("modo_extraccion"),
                "resumen": weather_data.get("resumen_meteorologico"),
                "serie_diaria": weather_data.get("serie_diaria"),
            },
            "pronostico": forecast_data.get("pronostico"),
            "estado_actual": forecast_data.get("estado_actual"),
            "forecast": forecast_data,
        }
        try:
            rid = log_recommendation(
                decision_type="viabilidad_terreno",
                parcel_id=f"({lat:.4f}, {lon:.4f})",
                option_recommended=str(rec.get("decision_siembra", "")),
                confidence=str(rec.get("nivel_confianza", "")),
                uncertainty_band="gate estructural de zona (no depende del día)",
                assumptions=list(rec.get("supuestos", [])),
                data_vintage=json.dumps(
                    {"fuente": viab.get("fuente"), "timestamp": viab.get("timestamp")},
                    ensure_ascii=False,
                ),
                model_version="preflight_geografico_v1 (NE 50m + POWER climatology)",
                payload=rec,
                baseline_option="no aplica (zona no cultivable)",
            )
            result["recommendation_id"] = rid
        except Exception as exc:  # noqa: BLE001
            result["recommendation_id"] = None
            result["motor_ia"]["nota"] = f"{result['motor_ia']['nota']} | log: {exc}"
        return result

    result = generate_agronomic_recommendation(
        weather_data=weather_data, farm_context=farm_context
    )
    rec = result.get("recomendacion_agronomica", {})
    if viab.get("veredicto") == "CONDICIONAL_RIEGO":
        rec.setdefault("supuestos", []).append(
            f"PREFLIGHT HÍDRICO: {viab.get('motivo', 'Precipitación anual insuficiente para temporal sin riego.')}"
        )
        rec["viabilidad_condicion"] = "CONDICIONAL_RIEGO"
    result["viabilidad"] = viab
    result["pronostico"] = forecast_data.get("pronostico")
    result["estado_actual"] = forecast_data.get("estado_actual")
    result["forecast"] = forecast_data
    try:
        rid = log_recommendation(
            decision_type="fecha_siembra",
            parcel_id=f"({lat:.4f}, {lon:.4f})",
            option_recommended=str(rec.get("decision_siembra", "")),
            confidence=str(rec.get("nivel_confianza", "")),
            uncertainty_band=str((rec.get("incertidumbre") or {}).get("banda", "")),
            assumptions=list(rec.get("supuestos", [])),
            data_vintage=json.dumps(rec.get("data_vintage", {}), ensure_ascii=False),
            model_version=str(result.get("motor_ia", {}).get("proveedor", "")),
            payload=rec,
        )
        result["recommendation_id"] = rid
    except Exception as exc:  # noqa: BLE001
        result["recommendation_id"] = None
        result["motor_ia"]["nota"] = f"{result.get('motor_ia', {}).get('nota') or ''} | log: {exc}"
    return result


@app.get("/api/recommendation", tags=["AI Agronomic Advisor"])
def get_recommendation(
    lat: float = Query(
        16.2333,
        ge=-90.0,
        le=90.0,
        description="Latitud de la parcela (por defecto: Villaflores, La Frailesca, Chiapas)",
    ),
    lon: float = Query(
        -93.2667,
        ge=-180.0,
        le=180.0,
        description="Longitud de la parcela (por defecto: Villaflores, La Frailesca, Chiapas)",
    ),
) -> dict[str, Any]:
    """Orquesta la consulta climática a la NASA y genera la recomendación agronómica con IA (GET)."""
    return _recomendar(lat, lon)


@app.post("/api/recommendation", tags=["AI Agronomic Advisor"])
async def post_recommendation(body: CoordinateRequest) -> dict[str, Any]:
    """Orquesta la consulta climática a la NASA y genera la recomendación agronómica con IA (POST)."""
    farm_context = ""
    if body.farm_id:
        try:
            ctx = await get_farm_context_for_llm(body.farm_id)
            if ctx and not ctx.startswith("CONTEXTO DE PARCELA"):
                farm_context = ctx
        except Exception:  # noqa: BLE001
            farm_context = ""
    return _recomendar(body.lat, body.lon, farm_context=farm_context or None)


@app.post("/api/recommendations/{recommendation_id}/override", tags=["AI Agronomic Advisor"])
def post_override(recommendation_id: str, body: OverrideRequest) -> dict[str, Any]:
    """Registra lo que el decisor finalmente hizo (override anónimo, 3.3-A).

    Materia prima para la evaluación de la decisión (arquitectura §11_OUTPUTS).
    """
    ok = register_override(recommendation_id, body.human_action, body.motivo)
    if not ok:
        raise HTTPException(
            status_code=404,
            detail=f"Recomendación '{recommendation_id}' no encontrada en el registro",
        )
    return {
        "status": "ok",
        "recommendation_id": recommendation_id,
        "human_action": body.human_action,
        "override_reason": body.motivo,
        "timestamp": datetime.now(UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# Scheduler semanal (decisión 3.4): pronóstico del piloto registrado en el log
# ---------------------------------------------------------------------------
_scheduler: Any = None
PILOTO_LAT, PILOTO_LON = 16.2333, -93.2667


def _tarea_semanal_pronostico() -> None:
    """Pronóstico semanal del punto piloto, registrado como recomendación trazable."""
    try:
        pronostico = get_forecast(PILOTO_LAT, PILOTO_LON)
        p = pronostico.get("pronostico") or {}
        rid = log_recommendation(
            decision_type="pronostico_semanal",
            parcel_id=f"({PILOTO_LAT:.4f}, {PILOTO_LON:.4f})",
            option_recommended=str(p.get("clasificacion_escenario_central", "")),
            confidence=str(pronostico.get("confianza_max", "")),
            uncertainty_band=f"p10={p.get('gwetroot_p10')} / p90={p.get('gwetroot_p90')}",
            assumptions=list(pronostico.get("supuestos", [])),
            data_vintage=json.dumps({"fecha_dato": pronostico.get("fecha_dato")}, ensure_ascii=False),
            model_version="XGBoost cuantílico GWETROOT(t+7)",
            payload={
                "status": pronostico.get("status"),
                "fecha_validez": pronostico.get("fecha_validez_pronostico"),
                "estado_actual": pronostico.get("estado_actual"),
                "pronostico": p,
                "degradacion": pronostico.get("degradacion"),
            },
            baseline_option="estado actual (persistencia)",
        )
        logger.info("Pronóstico semanal registrado: %s", rid)
    except Exception as exc:  # noqa: BLE001 — un fallo del job no debe tumbar el API
        logger.warning("Job semanal de pronóstico falló: %s", exc)


@app.on_event("startup")
def _arrancar_scheduler() -> None:
    """Arranca APScheduler con el job semanal (decisión 3.4: APScheduler real)."""
    global _scheduler
    try:
        from apscheduler.schedulers.background import BackgroundScheduler

        _scheduler = BackgroundScheduler(timezone="UTC")
        _scheduler.add_job(
            _tarea_semanal_pronostico,
            trigger="cron",
            day_of_week="mon",
            hour=7,
            minute=0,
            id="pronostico_semanal_piloto",
            replace_existing=True,
        )
        _scheduler.start()
        logger.info("APScheduler activo: job 'pronostico_semanal_piloto' (lunes 07:00 UTC)")
    except Exception as exc:  # noqa: BLE001
        logger.warning("APScheduler no disponible: %s", exc)
    # Precarga MODIS (Fase B de viabilidad): cache de uso de suelo para el demo
    try:
        precargar_puntos(
            [
                {"lat": 16.2333, "lon": -93.2667},  # piloto Villaflores (Frailesca)
                {"lat": 20.5235, "lon": -100.8157},  # Bajío
                {"lat": 27.4828, "lon": -109.9304},  # Norte
                {"lat": 21.1425, "lon": -88.1514},   # Golfo
                {"lat": 20.6015, "lon": -89.0122},   # Península
                {"lat": 23.0, "lon": 12.0},          # Sahara (escenario composición)
            ],
            max_polls=600,
        )
        logger.info("Precarga MODIS MCD12Q1 lanzada en hilo demonio (6 puntos)")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Precarga MODIS no lanzada: %s", exc)


@app.on_event("shutdown")
def _detener_scheduler() -> None:
    if _scheduler is not None and getattr(_scheduler, "running", False):
        _scheduler.shutdown(wait=False)
