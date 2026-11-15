"""Preflight de viabilidad geográfica-agroclimática de K'INAL (Fase A del plan).

Responde ANTES del motor de reglas y del LLM a la pregunta que la humedad de
suelo no puede responder: ¿es posible sembrar aquí en absoluto?

Gates en orden (composición y régimen, no solo agua en el suelo):
1. TIERRA-AGUA: máscara Natural Earth offline (ne_50m_land.geojson, EV-0010).
   Mar / lagos / hielo marino -> NO_VIABLE (hecho físico, confianza ALTA).
2. RÉGIMEN TÉRMICO: GDD anual base 10 °C y heladas estimadas, derivados de la
   climatología mensual de POWER (1 llamada, EV-0011). Si el maíz no completa
   su ciclo térmico -> NO_VIABLE (umbral hipótesis, confianza MEDIA).
3. RÉGIMEN HÍDRICO: precipitación anual insuficiente para temporal -> 
   CONDICIONAL_RIEGO (no descarta: el Norte/Bajío riegan; continúa con aviso).

Fase B (pendiente): clase de uso de suelo MODIS MCD12Q1 (urbano/agua/hielo de
continente) vía AppEEARS.
"""
from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

import requests

from services.decision_config import load_thresholds
from services.land_cover import obtener_clase_uso_suelo
from services.paths import KINAL_MAIN_DIR

LAND_GEOJSON = KINAL_MAIN_DIR / "05_DATA" / "reference" / "ne_50m_land.geojson"
CLIMATOLOGY_URL = "https://power.larc.nasa.gov/api/temporal/climatology/point"

# Días por mes (feb ≈ 28.25 años-promedio) para GDD desde medias mensuales
_DIAS_MES = (31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_MESES = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
_SENTINEL = -900.0  # POWER marca meses sin datos con -999/-999.9


def _valor(mes_dict: dict, mes: str) -> float | None:
    """Valor mensual válido o None si falta (excluye sentinels y la clave ANN)."""
    try:
        val = float(mes_dict.get(mes, _SENTINEL - 1))
    except (TypeError, ValueError):
        return None
    return None if val < _SENTINEL else val

_VIAB_CACHE: dict[tuple[float, float], dict[str, Any]] = {}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Gate 1: máscara tierra-agua (ray-casting puro, sin dependencias)
# ---------------------------------------------------------------------------
def _punto_en_anillo(lon: float, lat: float, anillo: list) -> bool:
    dentro = False
    n = len(anillo)
    j = n - 1
    for i in range(n):
        xi, yi = anillo[i][0], anillo[i][1]
        xj, yj = anillo[j][0], anillo[j][1]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi):
            dentro = not dentro
        j = i
    return dentro


@lru_cache(maxsize=1)
def _cargar_mascara() -> list[dict] | None:
    """Carga los polígonos de tierra una sola vez; None si falta el archivo."""
    with _lock:
        if not LAND_GEOJSON.exists():
            return None
        try:
            data = json.loads(LAND_GEOJSON.read_text(encoding="utf-8"))
            return data.get("features", [])
        except Exception:
            return None


def es_tierra(lat: float, lon: float) -> bool | None:
    """True=tierra, False=agua/hielo marino, None=no se puede evaluar (sin máscara)."""
    features = _cargar_mascara()
    if features is None:
        return None
    for feat in features:
        geom = feat.get("geometry") or {}
        if geom.get("type") == "Polygon":
            if _punto_en_anillo(lon, lat, geom["coordinates"][0]):
                return True
        elif geom.get("type") == "MultiPolygon":
            for poly in geom["coordinates"]:
                if _punto_en_anillo(lon, lat, poly[0]):
                    return True
    return False


# ---------------------------------------------------------------------------
# Gate 2-3: climatología POWER (1 llamada -> GDD, heladas, precipitación)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _climatologia_power(lat: float, lon: float) -> dict[str, Any] | None:
    """Medias climáticas multi-anuales por mes (POWER climatology, sin fechas)."""
    try:
        resp = requests.get(
            CLIMATOLOGY_URL,
            params={
                "parameters": "T2M,T2M_MIN,PRECTOTCORR",
                "community": "AG",
                "longitude": lon,
                "latitude": lat,
                "format": "JSON",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        block = resp.json().get("properties", {}).get("parameter", {})
        t2m = block.get("T2M") or {}
        t2m_min = block.get("T2M_MIN") or {}
        prec = block.get("PRECTOTCORR") or {}
        if not any(_valor(t2m, m) is not None for m in _MESES):
            return None

        # GDD anual base 10 °C: suma mensual de max(0, T2M_media - 10) x dias
        gdd_anual = 0.0
        for mes, dias in zip(_MESES, _DIAS_MES, strict=True):
            t = _valor(t2m, mes)
            if t is not None:
                gdd_anual += max(0.0, t - 10.0) * dias

        # Heladas estimadas: meses con T2M_MIN media < 0 °C (cuenta días del mes).
        # Se separa (a) total anual [informativo] y (b) solapamiento con la
        # ventana de temporal abr-oct [gate]: el maíz de temporal no cruza el
        # invierno, por lo que heladas de diciembre-febrero no bloquean el ciclo.
        dias_helada = 0.0
        dias_helada_ventana = 0.0
        meses_helada: list[str] = []
        for i, (mes, dias) in enumerate(zip(_MESES, _DIAS_MES, strict=True)):
            tmin = _valor(t2m_min, mes)
            if tmin is not None and tmin < 0.0:
                dias_helada += dias
                meses_helada.append(mes)
                if 3 <= i <= 9:  # abr-oct
                    dias_helada_ventana += dias

        # PRECTOTCORR climatology es TASA mm/dia por mes -> total = tasa x dias
        precip_anual = 0.0
        precip_ciclo = 0.0
        for i, (mes, dias) in enumerate(zip(_MESES, _DIAS_MES, strict=True)):
            p = _valor(prec, mes)
            if p is not None:
                precip_anual += p * dias
                if 3 <= i <= 9:  # abr-oct (ventana de temporal)
                    precip_ciclo += p * dias

        t_validos = [t for m in _MESES if (t := _valor(t2m, m)) is not None]
        cobertura = sum(1 for m in _MESES if _valor(t2m, m) is not None)

        return {
            "gdd_anual_base10_c": round(gdd_anual, 1),
            "dias_helada_estimados": round(dias_helada, 0),
            "dias_helada_ventana_temporal": round(dias_helada_ventana, 0),
            "meses_helada_invernal": meses_helada,
            "precip_anual_mm": round(precip_anual, 0),
            "precip_ciclo_abr_oct_mm": round(precip_ciclo, 0),
            "t2m_media_anual_c": round(sum(t_validos) / max(1, len(t_validos)), 1),
            "meses_con_datos": cobertura,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Veredicto compuesto
# ---------------------------------------------------------------------------
def evaluar_viabilidad(lat: float, lon: float) -> dict[str, Any]:
    """Preflight completo. Nunca lanza: ante fallos de capa degrada con aviso."""
    clave = (round(lat, 3), round(lon, 3))
    with _lock:
        if clave in _VIAB_CACHE:
            return _VIAB_CACHE[clave]

    cfg = load_thresholds().get("viabilidad", {})
    gdd_min = float(cfg.get("gdd_min_maiz_c", 1200))
    heladas_max = float(cfg.get("dias_helada_max_ciclo", 15))
    precip_min = float(cfg.get("precip_anual_min_temporal_mm", 400))
    modis_cfg = cfg.get("modis_clases", {})
    modis_prohibidas = set(modis_cfg.get("prohibidas", [0, 13, 15, 16, 17]))
    modis_permitidas = set(modis_cfg.get("permitidas", [12, 14]))
    modis_marginales = set(modis_cfg.get("marginales", []))

    ev: dict[str, Any] = {}
    avisos: list[str] = []
    veredicto: dict[str, Any] | None = None

    # Gate 1: tierra-agua (físico, no depende de hipótesis)
    tierra = es_tierra(lat, lon)
    if tierra is None:
        avisos.append("Máscara tierra-agua no disponible (ne_50m_land.geojson ausente): gate geográfico omitido.")
    else:
        ev["es_tierra"] = tierra

    if tierra is False:
        veredicto = {
            "veredicto": "NO_VIABLE",
            "nivel": "geografico",
            "motivo": "La coordenada cae sobre agua o hielo marino (máscara Natural Earth): no existe suelo cultivable.",
            "confianza": "ALTA",
        }
    else:
        # Gate 2: régimen térmico
        clim = _climatologia_power(lat, lon)
        if clim is None:
            avisos.append("Climatología POWER no disponible: gates térmico/hídrico omitidos en esta consulta.")
        else:
            ev.update(clim)
            if clim.get("meses_con_datos", 12) < 12:
                avisos.append(
                    f"Climatología POWER incompleta ({clim['meses_con_datos']}/12 meses con datos): "
                    "los gates térmico/hídrico se evaluaron solo con los meses disponibles."
                )
            if clim["gdd_anual_base10_c"] < gdd_min:
                veredicto = {
                    "veredicto": "NO_VIABLE",
                    "nivel": "termico",
                    "motivo": (
                        f"Régimen térmico incompatible con maíz: GDD anual {clim['gdd_anual_base10_c']:.0f} °C "
                        f"< requisito de ciclo corto (~{gdd_min:.0f} °C, base 10 °C), con "
                        f"{clim['dias_helada_estimados']:.0f} días de helada estimados. "
                        "El maíz no completa su ciclo en esta zona."
                    ),
                    "confianza": "MEDIA",
                }
            elif clim["dias_helada_ventana_temporal"] > heladas_max:
                veredicto = {
                    "veredicto": "NO_VIABLE",
                    "nivel": "termico",
                    "motivo": (
                        f"Régimen de heladas incompatible con maíz de temporal: "
                        f"{clim['dias_helada_ventana_temporal']:.0f} días de helada estimados DENTRO de "
                        f"la ventana de temporal (abr-oct) > tolerancia (~{heladas_max:.0f}); "
                        f"meses con helada invernal: {', '.join(clim['meses_helada_invernal']) or 'ninguno'}."
                    ),
                    "confianza": "MEDIA",
                }
            else:
                # Gate 2.5 (Fase B, D-0008): composición de la zona vía MODIS
                # MCD12Q1 LC_Type1 (AppEEARS). Responde "por la composición de
                # la zona no se puede plantar" (urbano/hielo/desnudo/agua).
                detenido = False
                lc = obtener_clase_uso_suelo(lat, lon)
                if lc is None:
                    avisos.append(
                        "Uso de suelo MODIS (AppEEARS) no disponible en esta consulta: "
                        "gate de composición omitido; se evaluó solo régimen climático."
                    )
                else:
                    ev["modis_clase_lc_type1"] = lc["clase"]
                    ev["modis_clase_nombre"] = lc["nombre"]
                    ev["modis_anio"] = lc.get("anio")
                    if lc["clase"] in modis_prohibidas:
                        veredicto = {
                            "veredicto": "NO_VIABLE",
                            "nivel": "composicion",
                            "motivo": (
                                f"Por la composición de la zona no se puede plantar: uso de suelo "
                                f"clasificado como {lc['nombre']} (clase {lc['clase']}, MODIS "
                                f"MCD12Q1 {lc.get('anio') or 'último disponible'})."
                            ),
                            "confianza": "MEDIA",
                        }
                        detenido = True
                    elif lc["clase"] not in modis_permitidas and lc["clase"] in modis_marginales:
                        avisos.append(
                            f"Uso de suelo MODIS = {lc['nombre']} (clase {lc['clase']}): sembrar maíz "
                            "requiere habilitación de la parcela (no es tierra de cultivo activa)."
                        )
                if not detenido:
                    # Gate 3: régimen hídrico (continúa con aviso, no rechaza)
                    if clim["precip_anual_mm"] < precip_min:
                        veredicto = {
                            "veredicto": "CONDICIONAL_RIEGO",
                            "nivel": "hidrico",
                            "motivo": (
                                f"Precipitación anual ({clim['precip_anual_mm']:.0f} mm) insuficiente "
                                f"para maíz de temporal sin riego (~{precip_min:.0f} mm): viable "
                                "SOLO con riego. Evaluando humedad y decisión con esa condición."
                            ),
                            "confianza": "MEDIA",
                        }
                    else:
                        veredicto = {
                            "veredicto": "VIABLE",
                            "nivel": None,
                            "motivo": "Zona viable para maíz de temporal según máscara geográfica, uso de suelo y régimen climático.",
                            "confianza": "ALTA",
                        }
    if veredicto is None:
        veredicto = {
            "veredicto": "VIABLE",
            "nivel": None,
            "motivo": "Preflight sin datos completos; no se detectaron impedimentos.",
            "confianza": "MEDIA",
        }

    resultado = {
        **veredicto,
        "evidencia": ev,
        "avisos": avisos,
        "umbrales_usados": {"gdd_min_maiz_c": gdd_min, "dias_helada_max_ciclo": heladas_max, "precip_anual_min_temporal_mm": precip_min},
        "fuente": (
            "máscara Natural Earth 50m (EV-0010) + POWER climatology (EV-0011)"
            + (" + MODIS MCD12Q1.061 LC_Type1 vía AppEEARS (EV-0012)" if "modis_clase_lc_type1" in ev else "")
            + "; umbrales en decision_thresholds.yaml (hipótesis)"
        ),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    with _lock:
        _VIAB_CACHE[clave] = resultado
    return resultado


def mensaje_no_viable(veredicto: dict[str, Any], lat: float, lon: float) -> dict[str, Any]:
    """Recomendación de rechazo DETERMINISTA (no consulta LLM): decisión honesta desde el inicio."""
    ev = veredicto.get("evidencia", {})
    evidencia_txt = ", ".join(f"{k}: {v}" for k, v in ev.items()) or "sin evidencia numérica disponible"
    return {
        "decision_siembra": "NO VIABLE PARA SIEMBRA",
        "nivel_confianza": veredicto.get("confianza", "ALTA"),
        "semaforo_color": "alert",
        "diagnostico_climatico": (
            f"Preflight de viabilidad ({veredicto.get('nivel', 'geografico')}): {veredicto['motivo']}"
        ),
        "recomendacion_rotacion_cultivos": (
            "No aplica rotación de maíz en esta coordenada. Consultar cultivos/insumos compatibles "
            "con la zona (p. ej., acuacultura/pastoreo en agua; especies de clima frío si hay "
            "régimen térmico limitado) fuera del alcance de este módulo."
        ),
        "gestion_suelo_y_agua": (
            "Sin manejo de suelo aplicable: el impedimento es de composición/régimen, no de "
            "humedad puntual. No invertir semilla ni jornales aquí."
        ),
        "alerta_riesgo_climatico": (
            f"IMPEDIMENTO ESTRUCTURAL: {veredicto['motivo']} Evidencia: {evidencia_txt}."
        ),
        "mensaje_productor": (
            "Los satélites y la geografía indican que en este punto no es posible sembrar maíz "
            "de temporal: " + veredicto["motivo"]
        ),
        "supuestos": [
            "El rechazo proviene del preflight geográfico-agroclimático (composición de la zona), "
            "no del nivel de humedad del día.",
            "Umbrales térmicos/hídricos de viabilidad son hipótesis documentadas en "
            "decision_thresholds.yaml (pendientes de validación con la línea base D01).",
            *veredicto.get("avisos", []),
        ],
        "validacion_umbrales": "VIABLE: hecho geográfico (ALTA) / térmico: umbrales hipótesis (MEDIA)",
        "viabilidad": veredicto,
    }
