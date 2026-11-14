"""Servicio de Asesoría Agronómica con IA (LLM / RAG) para K'INAL — Cobertura Nacional México.

Recibe el JSON meteorológico real de NASA POWER (`services/nasa_api.py`) para cualquier
municipio y estado de México, y genera recomendaciones adaptadas a la región agroclimática
(Norte árido/semiárido, Bajío templado, Trópico húmedo/subhúmedo, Altiplano, etc.).
"""
from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from typing import Any

import requests

SYSTEM_PROMPT = """Eres el Ingeniero Agrónomo Principal del sistema K'INAL (NASA Space Apps Challenge), especialista en agroclimatología, adaptación al cambio climático, rotación de cultivos y manejo de suelos en TODO MÉXICO (desde zonas áridas del Norte como Sonora/Sinaloa/Chihuahua, pasando por El Bajío y Altiplano, hasta el Trópico húmedo y subhúmedo de Chiapas, Veracruz y Yucatán).

Tu objetivo es analizar los datos meteorológicos y de humedad del suelo REALES provenientes de NASA POWER y GLDAS/SMAP para el municipio, estado y coordenadas indicadas, y emitir una recomendación práctica, clara y trazable para el agricultor de esa región.

REGLAS CIENTÍFICAS DE REFERENCIA (K'INAL):
1. Ventana de Siembra:
   - SEMBRAR: Humedad superficial >= 0.25 cm³/cm³ Y humedad radicular >= 0.20 cm³/cm³ con condiciones térmicas e hídricas favorables.
   - ESPERAR (Falso Inicio / Poca Reserva): Humedad superficial >= 0.25 cm³/cm³ pero humedad radicular < 0.20 cm³/cm³.
   - ESPERAR (Suelo Seco / Requiere Riego): Humedad superficial < 0.25 cm³/cm³.
2. Estrés Hídrico:
   - Si la humedad radicular cae por debajo de 0.15 cm³/cm³, activar ALERTA de estrés hídrico o necesidad de riego de auxilio.

DEBES RESPONDER ÚNICAMENTE CON UN OBJETO JSON VÁLIDO con la siguiente estructura exacta:
{
  "decision_siembra": "SEMBRAR | ESPERAR (Falso Inicio) | ESPERAR (Suelo Seco)",
  "nivel_confianza": "ALTA | MEDIA | BAJA",
  "semaforo_color": "signal | warn | alert",
  "diagnostico_climatico": "Resumen breve (1-2 oraciones) del estado actual en ese municipio y estado.",
  "recomendacion_rotacion_cultivos": "Consejo específico de rotación/asociación de cultivos adaptado a ese Estado de México y a la humedad/temperatura actual.",
  "gestion_suelo_y_agua": "Acciones concretas de manejo de suelo y agua para esa zona.",
  "alerta_riesgo_climatico": "Estado del riesgo de sequía, estrés térmico o exceso de humedad.",
  "mensaje_productor": "Recomendación directa en lenguaje sencillo y claro para el agricultor."
}
"""


def _build_user_prompt(weather_data: dict[str, Any]) -> str:
    coords = weather_data.get("coordenadas", {})
    resumen = weather_data.get("resumen_meteorologico", {})
    serie_reciente = weather_data.get("serie_diaria", [])[-7:]

    return (
        f"Analiza los siguientes datos satelitales REALES de la NASA para la parcela ubicada en "
        f"{coords.get('nombre_ubicacion', 'México')} "
        f"(Latitud {coords.get('latitud')}, Longitud {coords.get('longitud')}):\n\n"
        f"RESUMEN METEOROLÓGICO (Últimos 14 días):\n"
        f"{json.dumps(resumen, indent=2, ensure_ascii=False)}\n\n"
        f"SERIE DIARIA RECIENTE (Últimos 7 días):\n"
        f"{json.dumps(serie_reciente, indent=2, ensure_ascii=False)}\n\n"
        f"Genera la recomendación agronómica adaptada a {coords.get('nombre_ubicacion', 'esta región')} en el formato JSON solicitado."
    )


def _extract_json_block(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)


def _call_gemini_api(user_prompt: str, api_key: str) -> dict[str, Any]:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "responseMimeType": "application/json",
        },
    }
    resp = requests.post(url, json=payload, timeout=12)
    resp.raise_for_status()
    data = resp.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return _extract_json_block(text)


def _call_openai_api(user_prompt: str, api_key: str) -> dict[str, Any]:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=12)
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    return _extract_json_block(text)


def _classify_mexican_agro_region(lat: float, lon: float, estado: str) -> dict[str, str]:
    """Determina la vocación agrícola regional de México según latitud/estado."""
    estado_low = estado.lower()
    if lat >= 23.5 or any(k in estado_low for k in ["sonora", "sinaloa", "chihuahua", "baja", "coahuila", "nuevo león", "tamaulipas", "durango"]):
        return {
            "zona": "Norte / Noroeste Agroindustrial (Árido - Semiárido)",
            "cultivos_optimos": "maíz de alto rendimiento, frijol azufrado/pinto, garbanzo, sorgo forrajero o trigo cristalino",
            "cultivos_sequia": "sorgo, cártamo, garbanzo de humedad residual o ajonjolí",
        }
    if 19.0 <= lat < 23.5 and any(k in estado_low for k in ["guanajuato", "jalisco", "michoacán", "querétaro", "aguascalientes", "zacatecas", "méxico", "puebla", "tlaxcala", "hidalgo"]):
        return {
            "zona": "El Bajío y Altiplano Central (Templado Subhúmedo)",
            "cultivos_optimos": "maíz grano en rotación con frijol, cebada, haba, avena forrajera o brócoli/hortalizas",
            "cultivos_sequia": "frijol de ciclo corto, sorgo, avena forrajera o amaranto",
        }
    return {
        "zona": "Sur-Sureste Tropical y Subtropical",
        "cultivos_optimos": "maíz de temporal en sistema Milpa intercalado con frijol, calabaza o leguminosas de cobertura (canavalia/mucuna)",
        "cultivos_sequia": "sorgo, ajonjolí, caupí (frijol pelón) o híbridos de maíz tolerantes a canícula",
    }


def _generate_rule_based_advisor_fallback(weather_data: dict[str, Any]) -> dict[str, Any]:
    """Motor Agronómico Nacional basado en datos reales NASA POWER + vocación regional de México."""
    coords = weather_data.get("coordenadas", {})
    lat = float(coords.get("latitud", 16.2333))
    lon = float(coords.get("longitud", -93.2667))
    nombre_ubicacion = coords.get("nombre_ubicacion", f"({lat:.3f}, {lon:.3f})")
    estado = coords.get("estado", "México")

    resumen = weather_data.get("resumen_meteorologico", {})
    suelo = resumen.get("humedad_suelo_estimada", {})

    sm_sup = float(suelo.get("sm_superficial_v_v", 0.25))
    sm_rad = float(suelo.get("sm_radicular_v_v", 0.20))
    precip_7d = float(resumen.get("precipitacion_acumulada_7d_mm", 0.0))
    precip_14d = float(resumen.get("precipitacion_acumulada_14d_mm", 0.0))
    temp_media = float(resumen.get("temperatura_media_c", 24.0))
    temp_max = float(resumen.get("temperatura_max_extrema_c", 30.0))
    rh_media = float(resumen.get("humedad_relativa_media_pct", 65.0))

    perfil_regional = _classify_mexican_agro_region(lat, lon, estado)

    if sm_sup >= 0.25 and sm_rad >= 0.20:
        decision = "SEMBRAR"
        confianza = "ALTA"
        semaforo = "signal"
        rotacion = (
            f"Región {perfil_regional['zona']} ({nombre_ubicacion}): Condiciones de humedad óptimas "
            f"para establecer {perfil_regional['cultivos_optimos']}. Se recomienda intercalar o rotar "
            "con leguminosas para fijar nitrógeno y romper ciclos de plagas."
        )
        suelo_agua = (
            f"Con {precip_14d:.1f} mm acumulados en 14 días y humedad radicular de {sm_rad:.2f} cm³/cm³, "
            "mantenga labranza de conservación con al menos 30% de rastrojo en superficie para retener "
            "esta reserva hídrica."
        )
        alerta = (
            f"ÓPTIMO en {nombre_ubicacion}: Humedad radicular ({sm_rad:.2f} cm³/cm³) y superficial "
            f"({sm_sup:.2f} cm³/cm³) en niveles seguros para germinación y desarrollo."
        )
        msg_productor = (
            f"En {nombre_ubicacion} la tierra tiene buena humedad arriba ({sm_sup:.2f}) y en la raíz "
            f"({sm_rad:.2f}), con {precip_7d:.1f} mm de lluvia reciente. Es buen momento para sembrar "
            "aprovechando la humedad acumulada."
        )
    elif (sm_sup >= 0.22 and sm_rad >= 0.17) or (sm_sup >= 0.25 and sm_rad < 0.20):
        decision = "ESPERAR (Falso Inicio / Poca Reserva)"
        confianza = "MEDIA"
        semaforo = "warn"
        rotacion = (
            f"Región {perfil_regional['zona']} ({nombre_ubicacion}): Humedad en transición. "
            f"Si siembra en temporal, priorice variedades de ciclo corto o rote hacia "
            f"{perfil_regional['cultivos_sequia']}."
        )
        suelo_agua = (
            "Evite voltear el suelo en exceso para no perder humedad superficial por evaporación "
            f"(temperatura máxima reciente: {temp_max:.1f} °C). Implemente bordería o captación de lluvia."
        )
        alerta = (
            f"PRECAUCIÓN en {nombre_ubicacion}: Humedad superficial en {sm_sup:.2f} cm³/cm³ y radicular "
            f"en {sm_rad:.2f} cm³/cm³. Reserva justa ante una eventual pausa de lluvias."
        )
        msg_productor = (
            f"En {nombre_ubicacion} hay algo de humedad en el suelo ({precip_7d:.1f} mm en 7 días), "
            "pero la reserva en la raíz aún es moderada. Si no cuenta con riego de auxilio, espere "
            "a que se consoliden las lluvias."
        )
    else:
        decision = "ESPERAR (Suelo Seco)"
        confianza = "ALTA"
        semaforo = "alert"
        rotacion = (
            f"Región {perfil_regional['zona']} ({nombre_ubicacion}): Suelo con déficit para temporal "
            f"estricto. Si dispone de riego tecnificado o humedad residual, considere rotar con "
            f"{perfil_regional['cultivos_sequia']} de baja demanda hídrica."
        )
        suelo_agua = (
            f"Déficit hídrico en superficie ({sm_sup:.2f} cm³/cm³). Conserve la cobertura de rastrojo "
            f"(mulch) para proteger el suelo de la radiación y evaporación ({temp_max:.1f} °C máx); "
            "no siembre en secano sin riego de asiento."
        )
        alerta = (
            f"ALERTA DE SEQUÍA / SUELO SECO en {nombre_ubicacion}: Humedad superficial ({sm_sup:.2f}) "
            f"y radicular ({sm_rad:.2f} cm³/cm³) por debajo del umbral de germinación (0.25)."
        )
        msg_productor = (
            f"En {nombre_ubicacion} los satélites de la NASA reportan poca humedad en el suelo "
            f"({precip_14d:.1f} mm de lluvia en 14 días y humedad superficial de {sm_sup:.2f}). "
            "No arriesgue la semilla de temporal ahora; cuide el rastrojo o aplique riego de auxilio."
        )

    return {
        "decision_siembra": decision,
        "nivel_confianza": confianza,
        "semaforo_color": semaforo,
        "diagnostico_climatico": (
            f"{nombre_ubicacion} ({lat:.4f}°, {lon:.4f}°): {precip_14d:.1f} mm de lluvia en 14 días "
            f"({precip_7d:.1f} mm en 7 días), temp. media {temp_media:.1f} °C (máx {temp_max:.1f} °C) "
            f"y humedad relativa de {rh_media:.1f}%."
        ),
        "recomendacion_rotacion_cultivos": rotacion,
        "gestion_suelo_y_agua": suelo_agua,
        "alerta_riesgo_climatico": alerta,
        "mensaje_productor": msg_productor,
    }


def generate_agronomic_recommendation(weather_data: dict[str, Any]) -> dict[str, Any]:
    """Orquesta la generación del consejo agronómico usando LLM o Motor Nacional K'INAL."""
    user_prompt = _build_user_prompt(weather_data)

    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    llm_provider = "K'INAL AgroEngine Nacional + NASA GLDAS"
    llm_error: str | None = None
    recommendation_payload: dict[str, Any] | None = None

    if gemini_key:
        try:
            recommendation_payload = _call_gemini_api(user_prompt, gemini_key)
            llm_provider = "Google Gemini 2.0 Flash"
        except Exception as exc:
            llm_error = f"Gemini fallback: {exc}"

    if recommendation_payload is None and openai_key:
        try:
            recommendation_payload = _call_openai_api(user_prompt, openai_key)
            llm_provider = "OpenAI GPT-4o-mini"
        except Exception as exc:
            llm_error = f"OpenAI fallback: {exc}"

    if recommendation_payload is None:
        recommendation_payload = _generate_rule_based_advisor_fallback(weather_data)

    return {
        "status": "success",
        "timestamp_recomendacion": datetime.now(UTC).isoformat(),
        "motor_ia": {
            "proveedor": llm_provider,
            "nota": llm_error,
        },
        "coordenadas": weather_data.get("coordenadas", {}),
        "recomendacion_agronomica": recommendation_payload,
        "datos_meteorologicos": {
            "fuente": weather_data.get("fuente"),
            "modo_extraccion": weather_data.get("modo_extraccion"),
            "resumen": weather_data.get("resumen_meteorologico"),
            "serie_diaria": weather_data.get("serie_diaria"),
        },
    }
