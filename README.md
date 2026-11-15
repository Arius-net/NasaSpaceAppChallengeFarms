# K'INAL Backend

Backend de K'INAL construido con FastAPI para consultar datos agroclimáticos
de NASA POWER y generar recomendaciones agronómicas.

## Inicio

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn main:app --reload
```

La API estará disponible en `http://127.0.0.1:8000` y su documentación en
`http://127.0.0.1:8000/docs`. 