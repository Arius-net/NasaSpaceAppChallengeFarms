# K'INAL Backend

Backend de K'INAL construido con FastAPI para consultar datos agroclimáticos
de NASA POWER y generar recomendaciones agronómicas.

## Conectar con el frontend

El backend escucha en `http://localhost:8000` y el frontend Next.js en
`http://localhost:3000`. El frontend obtiene la URL del backend desde
`NEXT_PUBLIC_API_URL`.

### 1. Iniciar el backend

Desde esta carpeta:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Configurar e iniciar el frontend

En otra terminal, desde `frontend`:

```powershell
Copy-Item .env.example .env.local
# .env.local debe contener:
# NEXT_PUBLIC_API_URL=http://localhost:8000
npm install
npm run dev
```

Abre `http://localhost:3000`. No uses una URL terminada en `/api`, porque las
rutas del frontend agregan ese prefijo.

### Consumo desde el frontend

```ts
const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const response = await fetch(
  `${apiUrl}/api/recommendation?lat=16.2333&lon=-93.2667`
);
const recommendation = await response.json();
```

Rutas principales:

- `GET /api/health` — comprueba que el backend está disponible.
- `GET /api/recommendation?lat=...&lon=...` — obtiene una recomendación.
- `GET /api/weather?lat=...&lon=...` — consulta datos agroclimáticos.
- `GET /api/forecast?lat=...&lon=...` — obtiene el pronóstico.
- `GET /api/frailesca-geojson` — carga los polígonos del mapa.

La API ya permite solicitudes desde `localhost:3000` mediante CORS. La
documentación interactiva está disponible en `http://localhost:8000/docs`.
