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

## Modelo nacional

La adquisición y el entrenamiento nacional viven en `KINAL-main/12_SCRIPTS`.
Desde la carpeta `kinal-app` se puede ejecutar una prueba pequeña:

```powershell
python KINAL-main\12_SCRIPTS\acquisition\fetch_national_power.py --resolution 0.5 --start-year 2000 --end-year 2025 --max-points 2
python KINAL-main\12_SCRIPTS\ml\train_national_forecast_model.py --limit-files 2
```

Para descargar toda la cuadrícula nacional se omite `--max-points`. La descarga
es reanudable y guarda un manifiesto en `KINAL-main/05_DATA/raw/nasa_power_national`.
Cuando exista `backend/models/gwetroot_t7_national.joblib`, la API lo preferirá
automáticamente sobre el modelo del piloto.
