"""Fase B del preflight de viabilidad: uso de suelo MODIS MCD12Q1 via AppEEARS.

Puerta de COMPOSICION de la zona entre el gate termico y el hidrico:
¿que hay en el terreno segun la clasificacion satelital? Tierras de cultivo
pasan sin friccion; vegetacion natural continua con aviso (requiere
habilitacion); urbano/nieve-hielo/terreno desnudo/agua -> NO_VIABLE.

Implementacion: AppEEARS API point task (1 punto, producto MCD12Q1.061,
capa LC_Type1) — evita descargar granulos de 500MB. Token Earthdata desde
.env (EARTHDATA_USERNAME / EARTHDATA_PASSWORD).

Robustez ante la cola de AppEEARS (las point tasks pueden quedar en
"pending" minutos u horas si el servicio esta saturado):
1. Cache en disco por punto (05_DATA/reference/appeears_lc_cache).
2. COSECHA: antes de crear una tarea nueva se busca entre las tareas del
   usuario una ya terminada para el mismo punto (las tareas "pending" de
   consultas anteriores se pueden recolectar cuando la cola avance).
3. PRECARGA: precargar_puntos() corre en hilo demonio al arrancar y llena
   el cache de disco con polls largos para que en el demo la respuesta sea
   instantanea.
4. Consultas en vivo: poll acotado (~20 s) y degradacion honesta — si no
   hay dato, el gate se omite con aviso. Nunca lanza.
"""
from __future__ import annotations

import csv
import io
import json
import threading
import time
from datetime import UTC, datetime
from typing import Any

import requests

from services.paths import KINAL_MAIN_DIR

APPEARS_BASE = "https://appeears.earthdatacloud.nasa.gov/api"
CACHE_DIR = KINAL_MAIN_DIR / "05_DATA" / "reference" / "appeears_lc_cache"
TASK_NAME = "kinal_modis_lc_type1"
POLL_SEG = 2.0

# Etiquetas IGBP de LC_Type1 (MCD12Q1.061) para mensajes legibles
NOMBRES_CLASE = {
    0: "cuerpo de agua",
    1: "bosque siempreverde de coniferas",
    2: "bosque siempreverde de latifoliadas",
    3: "bosque caducifolio de coniferas",
    4: "bosque caducifolio de latifoliadas",
    5: "bosque mixto",
    6: "matorral cerrado",
    7: "matorral abierto",
    8: "sabana leñosa",
    9: "sabana",
    10: "pastizal",
    11: "humedal permanente",
    12: "tierras de cultivo",
    13: "urbano y construido",
    14: "mosaico cultivo/vegetacion natural",
    15: "nieve y hielo",
    16: "terreno desnudo",
    17: "cuerpo de agua",
}

_lock = threading.Lock()
_ultimo_fallo: float = 0.0  # backoff para NO crear mas tareas tras un fallo
_REINTENTO_SEG = 300.0
_tareas_enviadas: dict[tuple[float, float], str] = {}  # (lat, lon) -> task_id


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def os_env(nombre: str) -> str | None:
    import os

    return os.getenv(nombre)


def _credenciales() -> tuple[str, str] | None:
    usuario = os_env("EARTHDATA_USERNAME")
    clave = os_env("EARTHDATA_PASSWORD")
    if not usuario or not clave:
        return None
    return usuario, clave


def _cache_path(lat: float, lon: float):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"lc_{lat}_{lon}.json"


def _leer_cache(lat: float, lon: float) -> dict[str, Any] | None:
    try:
        p = _cache_path(lat, lon)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "clase" in data:
                return data
    except Exception:
        pass
    return None


def _guardar_cache(lat: float, lon: float, resultado: dict[str, Any]) -> None:
    try:
        _cache_path(lat, lon).write_text(
            json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# AppEEARS: login, extraccion, tarea, cosecha
# ---------------------------------------------------------------------------
def _login(sesion: requests.Session) -> dict[str, str] | None:
    """Login con HTTP Basic auth y cuerpo vacio -> headers Bearer."""
    cred = _credenciales()
    if cred is None:
        return None
    try:
        r = sesion.post(f"{APPEARS_BASE}/login", auth=cred, timeout=15)
        if r.status_code != 200:
            return None
        token = r.json().get("token")
        return {"Authorization": f"Bearer {token}"} if token else None
    except Exception:
        return None


def _extraer_valor_csv(bytes_csv: bytes) -> dict[str, Any] | None:
    """Lee el CSV de punto de AppEEARS y toma el valor del anio mas reciente."""
    texto = bytes_csv.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(texto))
    filas = list(reader)
    mejor: dict[str, str] | None = None
    for fila in filas:
        if fila.get("value") in (None, ""):
            continue
        if mejor is None or str(fila.get("date", "")) > str(mejor.get("date", "")):
            mejor = fila
    if mejor is None:
        return None
    try:
        clase = int(float(str(mejor.get("value", "")).strip()))
    except ValueError:
        return None
    return {
        "clase": clase,
        "nombre": NOMBRES_CLASE.get(clase, f"clase {clase}"),
        "anio": str(mejor.get("date", ""))[:4] or None,
    }


def _descargar_bundle(sesion: requests.Session, encabezados: dict, task_id: str) -> dict[str, Any] | None:
    try:
        rb = sesion.get(f"{APPEARS_BASE}/bundle/{task_id}", headers=encabezados, timeout=15)
        if rb.status_code != 200:
            return None
        destino = next(
            (a for a in rb.json().get("files", []) if str(a.get("file_name", "")).endswith("-results.csv")),
            None,
        )
        if destino is None:
            return None
        ra = sesion.get(
            f"{APPEARS_BASE}/bundle/{task_id}/{destino['file_id']}", headers=encabezados, timeout=60
        )
        if ra.status_code != 200:
            return None
        valor = _extraer_valor_csv(ra.content)
        if valor is None:
            return None
        return {**valor, "fuente": "MODIS MCD12Q1.061 LC_Type1 (AppEEARS)"}
    except Exception:
        return None


def _cosechar_tareas(sesion: requests.Session, encabezados: dict, lat_c: float, lon_c: float) -> dict[str, Any] | None:
    """Busca una tarea kinal_modis_lc_type1 YA TERMINADA para este punto y extrae su CSV."""
    try:
        rl = sesion.get(f"{APPEARS_BASE}/task", headers=encabezados, timeout=15)
        if rl.status_code != 200:
            return None
        for t in rl.json():
            if str(t.get("task_name", "")) != TASK_NAME or t.get("status") != "done":
                continue
            coords = ((t.get("params") or {}).get("coordinates") or [])
            if not coords:
                continue
            try:
                if (
                    round(float(coords[0]["latitude"]), 3) != lat_c
                    or round(float(coords[0]["longitude"]), 3) != lon_c
                ):
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            valor = _descargar_bundle(sesion, encabezados, str(t["task_id"]))
            if valor is not None:
                return valor
    except Exception:
        return None
    return None


def _enviar_y_esperar(
    sesion: requests.Session, encabezados: dict, lat_c: float, lon_c: float, max_polls: int
) -> dict[str, Any] | None:
    """Crea la point task y hace poll acotado. Devuelve None si no termina a tiempo."""
    global _ultimo_fallo
    with _lock:
        if _ultimo_fallo and (time.time() - _ultimo_fallo) < _REINTENTO_SEG:
            return None
    anio_fin = datetime.now(UTC).year - 2  # MCD12Q1 llega ~2 anios atras
    anio_ini = anio_fin - 2
    tarea = {
        "task_type": "point",
        "task_name": TASK_NAME,
        "params": {
            "coordinates": [{"latitude": lat_c, "longitude": lon_c, "id": "0"}],
            "dates": [{"startDate": "01-01", "endDate": "12-31", "recurring": True, "yearRange": [anio_ini, anio_fin]}],
            "layers": [{"product": "MCD12Q1.061", "layer": "LC_Type1"}],
            "output": {"projection": "geographic", "format": {"type": "csv"}},
        },
    }
    try:
        r_task = sesion.post(f"{APPEARS_BASE}/task", headers=encabezados, json=tarea, timeout=30)
        if r_task.status_code not in (200, 202):
            with _lock:
                _ultimo_fallo = time.time()
            return None
        task_id = str(r_task.json().get("task_id") or "")
        if not task_id:
            with _lock:
                _ultimo_fallo = time.time()
            return None
        with _lock:
            _tareas_enviadas[(lat_c, lon_c)] = task_id
        for _ in range(max_polls):
            time.sleep(POLL_SEG)
            r_st = sesion.get(f"{APPEARS_BASE}/task/{task_id}", headers=encabezados, timeout=15)
            if r_st.status_code != 200:
                break
            estado = r_st.json().get("status")
            if estado == "done":
                return _descargar_bundle(sesion, encabezados, task_id)
            if estado in ("error", "cancelled"):
                with _lock:
                    _tareas_enviadas.pop((lat_c, lon_c), None)
                    _ultimo_fallo = time.time()
                return None
        return None  # pendiente: la tarea queda en la cuenta para cosecha futura
    except Exception:
        with _lock:
            _ultimo_fallo = time.time()
        return None


def obtener_clase_uso_suelo(
    lat: float, lon: float, max_polls: int = 10, esperar_en_vuelo: bool = False
) -> dict[str, Any] | None:
    """Clase de uso de suelo MODIS LC_Type1 del punto. None si no disponible.

    Orden: cache en disco -> cosecha de tareas ya terminadas -> tarea en
    vuelo (solo si esperar_en_vuelo) -> nueva point task con poll acotado.
    Nunca lanza.
    """
    lat_c, lon_c = round(float(lat), 3), round(float(lon), 3)
    en_cache = _leer_cache(lat_c, lon_c)
    if en_cache is not None:
        return en_cache
    sesion = requests.Session()
    encabezados = _login(sesion)
    if encabezados is None:
        return None
    # 1) cosecha: una tarea anterior para el mismo punto puede estar ya lista
    cosechado = _cosechar_tareas(sesion, encabezados, lat_c, lon_c)
    if cosechado is not None:
        cosechado["timestamp_consulta"] = datetime.now(UTC).isoformat()
        _guardar_cache(lat_c, lon_c, cosechado)
        return cosechado
    # 2) si ya hay una tarea en vuelo para este punto, no duplicar
    with _lock:
        tid_en_vuelo = _tareas_enviadas.get((lat_c, lon_c))
    if tid_en_vuelo is not None:
        if not esperar_en_vuelo:
            return None
        try:
            for _ in range(max_polls):
                time.sleep(POLL_SEG)
                r_st = sesion.get(
                    f"{APPEARS_BASE}/task/{tid_en_vuelo}", headers=encabezados, timeout=15
                )
                if r_st.status_code != 200:
                    return None
                estado = r_st.json().get("status")
                if estado == "done":
                    resultado = _descargar_bundle(sesion, encabezados, tid_en_vuelo)
                    if resultado is not None:
                        resultado["timestamp_consulta"] = datetime.now(UTC).isoformat()
                        _guardar_cache(lat_c, lon_c, resultado)
                        return resultado
                    return None
                if estado in ("error", "cancelled"):
                    with _lock:
                        _tareas_enviadas.pop((lat_c, lon_c), None)
                    return None
        except Exception:
            return None
        return None
    # 3) nueva tarea con poll acotado
    resultado = _enviar_y_esperar(sesion, encabezados, lat_c, lon_c, max_polls=max_polls)
    if resultado is None:
        return None
    resultado["timestamp_consulta"] = datetime.now(UTC).isoformat()
    _guardar_cache(lat_c, lon_c, resultado)
    return resultado


def precargar_puntos(puntos: list[dict[str, float]], max_polls: int = 600) -> threading.Thread:
    """Llena el cache de disco en hilo demonio (polls largos: hasta ~20 min/punto).

    Se llama en el startup del API con los puntos del demo; si la cola de
    AppEEARS esta saturada las tareas quedan en la cuenta y se cosechan en
    consultas/arranques posteriores.
    """

    def _trabajo() -> None:
        for p in puntos:
            lat_c, lon_c = round(float(p["lat"]), 3), round(float(p["lon"]), 3)
            if _leer_cache(lat_c, lon_c) is not None:
                continue
            resultado = obtener_clase_uso_suelo(
                lat_c, lon_c, max_polls=max_polls, esperar_en_vuelo=True
            )
            if resultado is None:
                continue  # silencioso: el gate se degradara con aviso si hace falta

    hilo = threading.Thread(target=_trabajo, daemon=True, name="kinal_precarga_modis")
    hilo.start()
    return hilo
