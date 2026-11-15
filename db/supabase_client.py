"""Cliente de conexión a Supabase (PostgreSQL) para K'INAL.

Implementa un patrón Singleton seguro en hilos y provee la función de
inyección de dependencias `get_db()` para FastAPI y servicios de dominio.
"""
from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from supabase import Client

load_dotenv()

_lock = threading.Lock()
_supabase_client: Client | None = None


class SupabaseConfigError(RuntimeError):
    """Error lanzado cuando faltan las credenciales o la librería de Supabase."""


def get_supabase_credentials() -> tuple[str, str]:
    """Obtiene y valida SUPABASE_URL y SUPABASE_KEY desde el entorno."""
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()
    if not url or not key:
        raise SupabaseConfigError(
            "Credenciales de Supabase no configuradas. Defina SUPABASE_URL y SUPABASE_KEY en .env"
        )
    return url, key


def get_db() -> Client:
    """Retorna la instancia Singleton del cliente de Supabase (inyección de dependencias)."""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    with _lock:
        if _supabase_client is not None:
            return _supabase_client

        url, key = get_supabase_credentials()
        try:
            from supabase import create_client
        except ImportError as exc:
            raise SupabaseConfigError(
                "El paquete 'supabase' (supabase-py) no está instalado. "
                "Instálelo con: pip install supabase"
            ) from exc

        _supabase_client = create_client(url, key)
        return _supabase_client


def reset_db_client() -> None:
    """Reinicia el cliente Singleton (útil para pruebas o recarga de configuración)."""
    global _supabase_client
    with _lock:
        _supabase_client = None
