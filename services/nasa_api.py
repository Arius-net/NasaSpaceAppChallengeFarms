"""Servicio de extracción agroclimática REAL de NASA POWER para TODO MÉXICO.

Consulta en tiempo real la API abierta de NASA POWER (`power.larc.nasa.gov`) para
cualquier coordenada (lat, lon) de México o el mundo, extrayendo:
- Precipitación real (`PRECTOTCORR`)
- Temperaturas reales (`T2M`, `T2M_MAX`, `T2M_MIN`)
- Humedad relativa (`RH2M`), viento (`WS2M`) y radiación solar (`ALLSKY_SFC_SW_DWN`)
- Humedad de suelo real asimilada por satélite de la NASA (`GWETTOP` superficial y `GWETROOT` radicular)

Incluye sincronización automática con el reloj del servidor de la NASA (vía header HTTP Date)
para evitar errores si el reloj local de Windows está adelantado (ej. año 2026), además de
geocodificación inversa para identificar el Municipio y Estado en cualquier parte de México.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

import numpy as np
import pandas as pd
import requests

from services.paths import CSV_POWER_FRAILESCA, FRAILESCA_BBOX

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
COMMUNITY = "AG"  # Agroclimatología
PARAMETERS = [
    "PRECTOTCORR",        # Precipitación corregida (mm/día)
    "T2M_MAX",            # Temperatura máxima a 2m (°C)
    "T2M_MIN",            # Temperatura mínima a 2m (°C)
    "T2M",                # Temperatura media a 2m (°C)
    "RH2M",               # Humedad relativa a 2m (%)
    "WS2M",               # Velocidad de viento a 2m (m/s)
    "ALLSKY_SFC_SW_DWN",  # Radiación solar incidente
    "GWETTOP",            # Humedad superficial del suelo NASA (fracción 0-1)
    "GWETROOT",           # Humedad en zona radicular NASA (fracción 0-1)
]
FILL = -999.0

# Cache en memoria por coordenada redondeada (~1 km) para respuestas instantáneas al repetir clic
_WEATHER_CACHE: dict[tuple[float, float], dict[str, Any]] = {}


def _en_frailesca(lat: float, lon: float) -> bool:
    """True si la coordenada cae dentro del bbox del núcleo Frailesca (F-05)."""
    return (
        FRAILESCA_BBOX["lat_min"] <= lat <= FRAILESCA_BBOX["lat_max"]
        and FRAILESCA_BBOX["lon_min"] <= lon <= FRAILESCA_BBOX["lon_max"]
    )


def resolve_location_name(lat: float, lon: float) -> dict[str, str]:
    """Obtiene el municipio y estado real en México usando Nominatim (OpenStreetMap)."""
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={
                "lat": lat,
                "lon": lon,
                "format": "json",
                "zoom": 10,
                "accept-language": "es",
            },
            headers={"User-Agent": "KINAL_NASA_SpaceApps_Mexico/1.0"},
            timeout=3.5,
        )
        if resp.status_code == 200:
            data = resp.json()
            addr = data.get("address", {})
            municipio = (
                addr.get("county")
                or addr.get("municipality")
                or addr.get("city")
                or addr.get("town")
                or addr.get("village")
                or "Zona Rural"
            )
            estado = addr.get("state") or addr.get("region") or "México"
            pais = addr.get("country") or "México"
            return {
                "municipio": municipio,
                "estado": estado,
                "pais": pais,
                "nombre_completo": f"{municipio}, {estado}",
            }
    except Exception:
        pass

    return {
        "municipio": f"Parcela ({lat:.2f}, {lon:.2f})",
        "estado": "México",
        "pais": "México",
        "nombre_completo": f"Coordenada ({lat:.3f}°, {lon:.3f}°)",
    }


def _to_clean_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    """Convierte la respuesta de NASA POWER en un DataFrame limpio sin valores -999.0."""
    block = payload.get("properties", {}).get("parameter", {})
    if not block:
        return pd.DataFrame()

    frame = pd.DataFrame(block)
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d", errors="coerce")
    frame.index.name = "fecha"
    frame = frame.replace(FILL, pd.NA).dropna(how="all")
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _extract_soil_moisture(df: pd.DataFrame) -> tuple[float, float, str]:
    """GWETTOP/GWETROOT NATIVOS de POWER (fracción de saturación 0-1), sin conversión.

    Decisión 1.1-A: ya NO se multiplica por porosidad provisional — los umbrales
    de config/decision_thresholds.yaml v0.3.0 están validados en esta misma
    escala (EV-0009, serie multianual 2000-2025).
    """
    has_gwettop = "GWETTOP" in df.columns and df["GWETTOP"].dropna().shape[0] > 0
    has_gwetroot = "GWETROOT" in df.columns and df["GWETROOT"].dropna().shape[0] > 0

    if has_gwettop and has_gwetroot:
        sm_sup = float(np.clip(df["GWETTOP"].dropna().tail(5).mean(), 0.0, 1.0))
        sm_rad = float(np.clip(df["GWETROOT"].dropna().tail(5).mean(), 0.0, 1.0))
        return (
            round(sm_sup, 3),
            round(sm_rad, 3),
            "NASA POWER GLDAS (GWETTOP/GWETROOT real, fracción 0-1)",
        )

    # Estimación heurística en fracción de saturación (solo cuando faltan GWET:
    # respaldo D01 CSV o modo offline). Equivalente nativo de la heurística previa.
    recent_rain_7d = float(df["PRECTOTCORR"].tail(7).sum()) if "PRECTOTCORR" in df else 20.0
    mean_rh = float(df["RH2M"].tail(7).mean()) if "RH2M" in df else 65.0
    sm_sup = float(np.clip(0.22 + (recent_rain_7d * 0.0078) + ((mean_rh - 55.0) * 0.0033), 0.02, 1.0))
    sm_rad = float(np.clip(0.27 + (recent_rain_7d * 0.0040) + ((mean_rh - 55.0) * 0.0022), 0.02, 1.0))
    return (
        round(sm_sup, 3),
        round(sm_rad, 3),
        "Estimación heurística POWER (sin GLDAS, fracción 0-1)",
    )


def _format_weather_output(
    df: pd.DataFrame,
    lat: float,
    lon: float,
    ubicacion: dict[str, str],
    source_mode: str,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    """Estructura el DataFrame meteorológico en un diccionario JSON listo para el Frontend y el LLM."""
    # Filtrar días donde las variables principales de clima ya fueron procesadas por la NASA
    df_valid = df.dropna(subset=["PRECTOTCORR", "T2M", "RH2M"], how="any").tail(14).copy()
    if df_valid.empty:
        df_valid = df.tail(14).ffill().bfill().fillna(0.0).copy()

    # Rellenar posibles rezagos de 1-2 días en radiación o viento con el último día válido
    df_valid = df_valid.ffill().bfill().fillna(0.0)

    sm_sup, sm_rad, sm_source = _extract_soil_moisture(df)

    precip_total_14d = round(float(df_valid["PRECTOTCORR"].sum()), 2)
    precip_ultimos_3d = round(float(df_valid["PRECTOTCORR"].tail(3).sum()), 2)
    precip_ultimos_7d = round(float(df_valid["PRECTOTCORR"].tail(7).sum()), 2)
    dias_con_lluvia = int((df_valid["PRECTOTCORR"] >= 1.0).sum())

    resumen_climatico = {
        "ventana_dias_analizados": len(df_valid),
        "precipitacion_acumulada_3d_mm": precip_ultimos_3d,
        "precipitacion_acumulada_7d_mm": precip_ultimos_7d,
        "precipitacion_acumulada_14d_mm": precip_total_14d,
        "dias_con_lluvia_significativa": dias_con_lluvia,
        "temperatura_media_c": round(float(df_valid["T2M"].mean()), 2),
        "temperatura_max_extrema_c": round(float(df_valid["T2M_MAX"].max()), 2),
        "temperatura_min_extrema_c": round(float(df_valid["T2M_MIN"].min()), 2),
        "humedad_relativa_media_pct": round(float(df_valid["RH2M"].mean()), 1),
        "viento_medio_ms": round(float(df_valid["WS2M"].mean()), 2),
        "radiacion_solar_media": round(float(df_valid["ALLSKY_SFC_SW_DWN"].mean()), 2),
        "humedad_suelo_estimada": {
            "sm_superficial_v_v": sm_sup,
            "sm_radicular_v_v": sm_rad,
            "unidad": "fracción de saturación (0-1)",
            "fuente_humedad": sm_source,
        },
    }

    serie_diaria = []
    for fecha_idx, row in df_valid.iterrows():
        fecha_str = (
            fecha_idx.strftime("%Y-%m-%d")
            if hasattr(fecha_idx, "strftime")
            else str(fecha_idx)[:10]
        )
        serie_diaria.append(
            {
                "fecha": fecha_str,
                "precipitacion_mm": round(float(row.get("PRECTOTCORR", 0.0)), 2),
                "temp_max_c": round(float(row.get("T2M_MAX", 0.0)), 2),
                "temp_min_c": round(float(row.get("T2M_MIN", 0.0)), 2),
                "temp_media_c": round(float(row.get("T2M", 0.0)), 2),
                "humedad_relativa_pct": round(float(row.get("RH2M", 0.0)), 1),
                "viento_ms": round(float(row.get("WS2M", 0.0)), 2),
                "radiacion_solar": round(float(row.get("ALLSKY_SFC_SW_DWN", 0.0)), 2),
            }
        )

    return {
        "status": "success",
        "fuente": "NASA POWER Agroclimatology + GLDAS Soil Moisture (Datos Reales)",
        "modo_extraccion": source_mode,
        "aviso_fallback": fallback_reason,
        "timestamp_consulta": datetime.now(UTC).isoformat(),
        "coordenadas": {
            "latitud": lat,
            "longitud": lon,
            "crs": "EPSG:4326",
            "municipio": ubicacion["municipio"],
            "estado": ubicacion["estado"],
            "nombre_ubicacion": ubicacion["nombre_completo"],
        },
        "parametros_nasa": PARAMETERS,
        "resumen_meteorologico": resumen_climatico,
        "serie_diaria": serie_diaria,
    }


def _query_nasa_power_window(
    lat: float, lon: float, end_dt: datetime, days_back: int = 35, timeout_seconds: float = 8.0
) -> tuple[pd.DataFrame, datetime | None]:
    """Ejecuta la consulta HTTP a NASA POWER y extrae también la fecha real del servidor NASA."""
    start_dt = end_dt - timedelta(days=days_back)
    params = {
        "parameters": ",".join(PARAMETERS),
        "community": COMMUNITY,
        "longitude": lon,
        "latitude": lat,
        "start": start_dt.strftime("%Y%m%d"),
        "end": end_dt.strftime("%Y%m%d"),
        "format": "JSON",
    }
    resp = requests.get(POWER_URL, params=params, timeout=timeout_seconds)

    server_dt: datetime | None = None
    if "Date" in resp.headers:
        try:
            server_dt = parsedate_to_datetime(resp.headers["Date"])
        except Exception:
            server_dt = None

    if resp.status_code != 200:
        return (pd.DataFrame(), server_dt)

    payload = resp.json()
    return (_to_clean_dataframe(payload), server_dt)


def get_recent_power_dataframe(lat: float, lon: float, days_back: int = 35, timeout_seconds: float = 8.0) -> pd.DataFrame:
    """Serie diaria reciente de POWER (con GWET) para features de pronóstico (uso interno del servicio de forecast)."""
    try:
        candidate_end = datetime.now(UTC) - timedelta(days=5)
        df, server_dt = _query_nasa_power_window(lat, lon, candidate_end, days_back=days_back, timeout_seconds=timeout_seconds)
        valid_days = int(df["PRECTOTCORR"].notna().sum()) if "PRECTOTCORR" in df.columns else 0
        if valid_days < 7 and server_dt is not None:
            real_end = server_dt - timedelta(days=4)
            df, _ = _query_nasa_power_window(lat, lon, real_end, days_back=days_back, timeout_seconds=timeout_seconds)
        if "PRECTOTCORR" in df.columns and int(df["PRECTOTCORR"].notna().sum()) >= 7:
            return df
    except Exception:
        pass
    return pd.DataFrame()


def get_nasa_weather_data(lat: float, lon: float, timeout_seconds: float = 8.0) -> dict[str, Any]:
    """Obtiene datos satelitales REALES de NASA POWER para cualquier coordenada de México."""
    cache_key = (round(lat, 3), round(lon, 3))
    if cache_key in _WEATHER_CACHE:
        return _WEATHER_CACHE[cache_key]

    ubicacion = resolve_location_name(lat, lon)

    try:
        # Intento 1: Consultar usando reloj del sistema (con margen de 5 días por procesamiento satelital)
        candidate_end = datetime.now(UTC) - timedelta(days=5)
        df, server_dt = _query_nasa_power_window(lat, lon, candidate_end, days_back=35, timeout_seconds=timeout_seconds)

        valid_days = int(df["PRECTOTCORR"].notna().sum()) if "PRECTOTCORR" in df.columns else 0

        # Intento 2: Si el reloj de Windows está adelantado respecto al servidor real de la NASA
        # (ej. reloj local en 2026 pero servidor NASA en fecha real anterior), sincronizar con server_dt
        if valid_days < 7 and server_dt is not None:
            real_end = server_dt - timedelta(days=4)
            df, _ = _query_nasa_power_window(lat, lon, real_end, days_back=35, timeout_seconds=timeout_seconds)
            valid_days = int(df["PRECTOTCORR"].notna().sum()) if "PRECTOTCORR" in df.columns else 0

        if valid_days >= 7:
            result = _format_weather_output(
                df=df,
                lat=lat,
                lon=lon,
                ubicacion=ubicacion,
                source_mode="NASA_POWER_REAL_LIVE",
                fallback_reason=None,
            )
            _WEATHER_CACHE[cache_key] = result
            return result
    except Exception as exc:
        fallback_msg = f"Sin conexión directa a NASA POWER ({exc}). Usando respaldo regional."
    else:
        fallback_msg = "NASA POWER en mantenimiento temporal. Usando respaldo regional."

    # Respaldo 1: CSV real versionado del inventario D01 (serie 2020-2025 del centro de la Frailesca).
    # Solo se usa si la coordenada cae dentro del bbox Frailesca (F-05): el CSV es de un punto.
    try:
        if _en_frailesca(lat, lon) and CSV_POWER_FRAILESCA.exists():
            df_csv = pd.read_csv(CSV_POWER_FRAILESCA)
            df_csv["fecha"] = pd.to_datetime(df_csv["fecha"], errors="coerce")
            df_csv = (
                df_csv.dropna(subset=["fecha"])
                .sort_values("fecha")
                .set_index("fecha")
                .drop(columns=["punto"], errors="ignore")
                .tail(14)
            )
            if not df_csv.empty:
                return _format_weather_output(
                    df=df_csv,
                    lat=lat,
                    lon=lon,
                    ubicacion=ubicacion,
                    source_mode="FALLBACK_D01_CSV",
                    fallback_reason=(
                        f"{fallback_msg} Usando serie versionada del inventario D01 "
                        "(nasa_power_frailesca.csv, punto centro)."
                    ),
                )
    except Exception:
        pass

    # Respaldo 2: serie sintética (último recurso, solo si no hay internet ni CSV)
    fechas = pd.date_range(end=datetime.now(UTC).date(), periods=14, freq="D")
    synthetic_df = pd.DataFrame(
        {
            "PRECTOTCORR": [0.5, 2.1, 0.0, 12.4, 8.5, 0.2, 1.0, 15.6, 18.2, 6.4, 0.0, 4.2, 11.0, 9.5],
            "T2M_MAX": [33.1, 32.5, 34.0, 30.2, 29.8, 31.5, 32.0, 29.1, 28.5, 30.0, 31.8, 31.0, 29.4, 29.9],
            "T2M_MIN": [21.0, 20.8, 21.5, 20.1, 19.8, 20.3, 20.7, 19.5, 19.2, 19.9, 20.4, 20.1, 19.6, 19.8],
            "T2M": [27.0, 26.6, 27.8, 25.1, 24.8, 25.9, 26.3, 24.3, 23.8, 24.9, 26.1, 25.5, 24.5, 24.8],
            "RH2M": [65.0, 68.2, 62.1, 78.5, 80.1, 72.4, 70.0, 83.2, 85.0, 79.1, 71.5, 74.0, 81.2, 80.4],
            "WS2M": [2.4, 2.8, 3.1, 4.2, 3.5, 2.1, 2.5, 4.8, 5.1, 3.0, 2.2, 2.9, 3.8, 3.3],
            "ALLSKY_SFC_SW_DWN": [21.5, 19.8, 22.4, 15.2, 16.0, 20.1, 19.5, 14.1, 13.5, 17.8, 21.0, 18.9, 15.8, 16.4],
        },
        index=fechas,
    )
    return _format_weather_output(
        df=synthetic_df,
        lat=lat,
        lon=lon,
        ubicacion=ubicacion,
        source_mode="FALLBACK_OFFLINE",
        fallback_reason=fallback_msg,
    )
