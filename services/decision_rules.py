"""Motor de reglas de decisión de K'INAL (Fase 0: reglas unificadas).

Un solo lugar define la lógica de decisión — el fallback determinista y el
prompt del LLM derivan de aquí, por lo que NO pueden divergir (corrección
del problema 1.2: reglas duplicadas con valores distintos).

Fuente de los números: `config/decision_thresholds.yaml` (decisión 0.2-B).
Estado: umbrales marcados como hipótesis de trabajo (Q-01/D03 abierta);
la validación multianual (notebook 05) los reemplazará con evidencia.
"""
from __future__ import annotations

from typing import Any

from services.decision_config import load_thresholds
from services.paths import FRAILESCA_BBOX

DECISION_SEMBRAR = "SEMBRAR"
DECISION_FALSO_INICIO = "ESPERAR (Falso Inicio / Poca Reserva)"
DECISION_SUELO_SECO = "ESPERAR (Suelo Seco)"

ALERTA_ESTRES = "ALERTA: Estrés Hídrico"
ALERTA_PRECAUCION = "PRECAUCIÓN: Bajando reserva"
ALERTA_OPTIMO = "ÓPTIMO"


def clasificar_siembra(sm_superficial: float, sm_radicular: float) -> dict[str, str]:
    """Clasifica la decisión de siembra con las reglas unificadas del YAML."""
    cfg = load_thresholds()
    sembrar = cfg["siembra"]["sembrar"]
    falso = cfg["siembra"]["falso_inicio"]

    if (
        sm_superficial >= float(sembrar["superficie_min"])
        and sm_radicular >= float(sembrar["radicular_min"])
    ):
        decision = DECISION_SEMBRAR
    elif (
        sm_superficial >= float(falso["superficie_min"]) and sm_radicular < float(sembrar["radicular_min"])
    ) or (
        sm_superficial >= float(falso["superficie_min_banda"])
        and sm_radicular >= float(falso["radicular_transicion_min"])
    ):
        decision = DECISION_FALSO_INICIO
    else:
        decision = DECISION_SUELO_SECO

    return {"decision": decision, "confianza_bruta": _confianza_por_regla(decision)}


def clasificar_alerta_sequia(sm_radicular: float) -> str:
    """Clasifica el estado de estrés hídrico con la humedad radicular."""
    cfg = load_thresholds()
    alerta = cfg["alerta_sequia"]
    if sm_radicular < float(alerta["estres_min"]):
        return ALERTA_ESTRES
    if sm_radicular < float(alerta["precaucion_min"]):
        return ALERTA_PRECAUCION
    return ALERTA_OPTIMO


def _confianza_por_regla(decision: str) -> str:
    """Confianza 'bruta' del motor de reglas (el caller la limita si no hay validación)."""
    if decision in (DECISION_SEMBRAR, DECISION_SUELO_SECO):
        return "ALTA"
    return "MEDIA"


def cap_confianza_para(lat: float, lon: float) -> str:
    cfg = load_thresholds()
    if not cfg.get("validated", False):
        return str(cfg.get("confianza_max", "MEDIA"))
    if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
        return str(cfg.get("confianza_max", "ALTA"))
    return "MEDIA"


def texto_reglas_para_prompt() -> str:
    """Renderiza las reglas del YAML en texto para inyectarlas al prompt del LLM.

    Garantiza que el LLM aplique EXACTAMENTE los mismos umbrales que el
    motor determinista (corrección del problema 1.2).
    """
    cfg = load_thresholds()
    sembrar = cfg["siembra"]["sembrar"]
    falso = cfg["siembra"]["falso_inicio"]
    alerta = cfg["alerta_sequia"]
    if cfg.get("validated", False):
        estado = (
            "VALIDADOS con evidencia (EV-0009, serie multianual 2000-2025 del piloto "
            "La Frailesca; en otras regiones son extrapolación)"
        )
    else:
        estado = "AÚN NO VALIDADOS (hipótesis de trabajo; Q-01/D03 abierta)"

    return (
        "REGLAS CIENTÍFICAS DE REFERENCIA (decision_thresholds.yaml "
        f"v{cfg.get('version')} — umbrales {estado}):\n"
        "Escala de humedad: fracción de saturación 0-1 (GWETTOP/GWETROOT nativos de POWER/GLDAS).\n"
        "1. Decisión de Siembra (evaluar en orden exacto):\n"
        f"   a. SEMBRAR: humedad superficial >= {sembrar['superficie_min']} "
        f"Y humedad radicular >= {sembrar['radicular_min']}.\n"
        "   b. ESPERAR (Falso Inicio / Poca Reserva): "
        f"(superficial >= {falso['superficie_min']} Y radicular < {sembrar['radicular_min']}) "
        f"O (superficial >= {falso['superficie_min_banda']} Y radicular >= {falso['radicular_transicion_min']}).\n"
        "   c. ESPERAR (Suelo Seco / Requiere Riego): cualquier otro caso.\n"
        "2. Estrés Hídrico (alerta_riesgo_climatico):\n"
        f"   - radicular < {alerta['estres_min']}: ALERTA de estrés hídrico / riego de auxilio.\n"
        f"   - radicular < {alerta['precaucion_min']}: PRECAUCIÓN, reserva bajo el nivel sembrable.\n"
        f"   - radicular >= {alerta['precaucion_min']}: ÓPTIMO.\n"
        "3. Nivel de confianza: 'ALTA' solo dentro del piloto validado (La Frailesca); "
        "en cualquier otra región de México usa 'MEDIA' como máximo.\n"
    )


def supuestos_config() -> list[str]:
    """Supuestos declarados que acompañan a cualquier recomendación (DECISION_SPEC §3)."""
    cfg = load_thresholds()
    supuestos = []
    if cfg.get("validated", False):
        supuestos.append(
            "Umbrales validados con serie multianual NASA POWER 2000-2025 "
            "(27 inicios de lluvias, criterio Stern-Cooper; evidencia EV-0009)."
        )
        supuestos.append(
            "La validación corresponde al piloto La Frailesca (Villaflores); "
            "su uso en otras regiones es extrapolación pendiente de validación regional."
        )
    else:
        supuestos.append(
            "Los umbrales de decisión aún NO están validados con serie multianual "
            "(Q-01/D03 abierta); son hipótesis de trabajo bajo revisión."
        )
    supuestos.append(
        "La humedad de suelo proviene de NASA POWER/GLDAS con resolución gruesa "
        "(~0.5°×0.625°, celda ≈50 km): describe el entorno regional, no la parcela individual."
    )
    supuestos.append(
        "GWETTOP/GWETROOT se usan en su escala nativa (fracción de saturación 0-1), "
        "sin conversión a cm³/cm³ (decisión 1.1-A)."
    )
    supuestos.append(
        "K'INAL apoya la decisión; la decisión final es de la persona y queda "
        "registrada en el log de recomendaciones."
    )
    return supuestos
