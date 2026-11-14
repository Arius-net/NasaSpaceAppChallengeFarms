"""K'INAL - Backend API (FastAPI) | PASO 3-5: API Completa con GeoJSON + NASA + IA."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services.ai_advisor import generate_agronomic_recommendation
from services.nasa_api import get_nasa_weather_data

load_dotenv()

BACKEND_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BACKEND_DIR.parents[1]
FRAILESCA_GEOJSON_PATH = (
    WORKSPACE_ROOT
    / "KINAL-main"
    / "05_DATA"
    / "reference"
    / "frailesca_municipios.geojson"
)

app = FastAPI(
    title="K'INAL Decision-Support API",
    description="API Backend de K'INAL para extracción agroclimática NASA e IA Agronómica (NASA Space Apps Challenge).",
    version="0.5.0",
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


@app.get("/api/health", tags=["Health"])
def health_check() -> dict[str, Any]:
    """Endpoint básico de prueba para verificar que el servidor FastAPI está activo."""
    return {
        "status": "ok",
        "service": "K'INAL Backend API",
        "step": "PASO 5 - Geovisualización y Dashboard Integrados",
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.get("/api/frailesca-geojson", tags=["Geospatial Reference"])
def get_frailesca_geojson() -> dict[str, Any]:
    """Sirve en modo solo lectura los polígonos municipales de La Frailesca (05_DATA/reference)."""
    if not FRAILESCA_GEOJSON_PATH.exists():
        raise HTTPException(status_code=404, detail="GeoJSON de La Frailesca no encontrado")
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
    weather_data = get_nasa_weather_data(lat=lat, lon=lon)
    return generate_agronomic_recommendation(weather_data=weather_data)


@app.post("/api/recommendation", tags=["AI Agronomic Advisor"])
def post_recommendation(body: CoordinateRequest) -> dict[str, Any]:
    """Orquesta la consulta climática a la NASA y genera la recomendación agronómica con IA (POST)."""
    weather_data = get_nasa_weather_data(lat=body.lat, lon=body.lon)
    return generate_agronomic_recommendation(weather_data=weather_data)
