"""Test puntual de autenticación NASA Earthdata (usa variables del .env, no imprime secretos)."""
import os
import sys

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

user = os.getenv("EARTHDATA_USERNAME", "")
pwd = os.getenv("EARTHDATA_PASSWORD", "")
if not user or not pwd:
    print("FALTAN CREDENCIALES en .env")
    sys.exit(2)

print(f"Usuario: {user} (password cargado, {len(pwd)} caracteres)")

import earthaccess  # noqa: E402

try:
    auth = earthaccess.login(strategy="environment")
    if auth.authenticated:
        print("LOGIN_OK: earthaccess autenticado correctamente.")
        sys.exit(0)
    print("LOGIN_FALLO: earthaccess no autenticó (credenciales rechazadas).")
    sys.exit(1)
except Exception as exc:  # noqa: BLE001
    print(f"LOGIN_FALLO: {type(exc).__name__}: {exc}")
    sys.exit(1)
