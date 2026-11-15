"""Servicio de recuperación de historial agronómico de parcelas para el motor RAG.

Consulta en Supabase (PostgreSQL) la ficha técnica de una parcela (`farms`) y su
historial de ciclos agrícolas (`crop_history`), estructurando un bloque de texto
listo para inyectarse en el prompt de `services/ai_advisor.py`.
"""
from __future__ import annotations

import asyncio
from typing import Any

from db.supabase_client import get_db


def _fetch_farm_and_history_sync(farm_id: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Ejecuta las consultas a las tablas `farms` y `crop_history` en Supabase."""
    client = get_db()

    farm_resp = (
        client.table("farms")
        .select("id, user_id, name, location_lat, location_lon, area_hectares, soil_type, created_at")
        .eq("id", farm_id)
        .limit(1)
        .execute()
    )
    farm_rows = farm_resp.data or []
    if not farm_rows:
        return None, []

    farm_data = farm_rows[0]

    history_resp = (
        client.table("crop_history")
        .select("id, farm_id, cycle_year, crop_type, planting_date, harvest_date, yield_notes, created_at")
        .eq("farm_id", farm_id)
        .order("cycle_year", desc=True)
        .execute()
    )
    history_rows = history_resp.data or []
    return farm_data, history_rows


def _format_farm_context(farm: dict[str, Any], history: list[dict[str, Any]]) -> str:
    """Da formato legible y estructurado al contexto de la parcela para el LLM."""
    nombre = farm.get("name") or "Parcela sin nombre"
    lat = farm.get("location_lat")
    lon = farm.get("location_lon")
    coords_txt = (
        f"Latitud {float(lat):.4f}, Longitud {float(lon):.4f}"
        if lat is not None and lon is not None
        else "No especificadas"
    )
    area = farm.get("area_hectares")
    area_txt = f"{area} ha" if area is not None else "No especificada"
    suelo = farm.get("soil_type") or "No especificado"

    lines = [
        "=== CONTEXTO DE PARCELA E HISTORIAL AGRONÓMICO (RAG - SUPABASE) ===",
        f"ID de Parcela: {farm.get('id')}",
        f"Nombre de la Parcela: {nombre}",
        f"Coordenadas: {coords_txt}",
        f"Superficie: {area_txt}",
        f"Tipo de Suelo: {suelo}",
        "",
        "HISTORIAL DE CICLOS AGRÍCOLAS:",
    ]

    if not history:
        lines.append("- Sin registros previos de cultivos para esta parcela.")
    else:
        for item in history:
            ciclo = item.get("cycle_year", "S/F")
            cultivo = item.get("crop_type") or "maíz de temporal"
            siembra = item.get("planting_date") or "No registrada"
            cosecha = item.get("harvest_date") or "No registrada"
            notas = item.get("yield_notes") or "Sin observaciones de rendimiento"
            lines.append(
                f"- Ciclo {ciclo} | Cultivo: {cultivo} | "
                f"Siembra: {siembra} | Cosecha: {cosecha} | "
                f"Notas/Rendimiento: {notas}"
            )

    return "\n".join(lines)


async def get_farm_context_for_llm(farm_id: str) -> str:
    """Consulta asíncronamente en Supabase los datos e historial de la parcela para el LLM.

    Args:
        farm_id: UUID de la parcela en la tabla `farms`.

    Returns:
        String formateado con la ficha de la parcela y su historial de cultivos,
        o un mensaje descriptivo si la parcela no existe o Supabase no está disponible.
    """
    clean_id = (farm_id or "").strip()
    if not clean_id:
        return "CONTEXTO DE PARCELA: No se proporcionó un ID de parcela válido."

    try:
        farm, history = await asyncio.to_thread(_fetch_farm_and_history_sync, clean_id)
    except Exception as exc:  # noqa: BLE001 — degradación elegante si Supabase no está configurado
        return (
            f"CONTEXTO DE PARCELA ({clean_id}): No fue posible recuperar el historial "
            f"desde Supabase ({exc})."
        )

    if farm is None:
        return f"CONTEXTO DE PARCELA ({clean_id}): Parcela no encontrada en la base de datos."

    return _format_farm_context(farm, history)
