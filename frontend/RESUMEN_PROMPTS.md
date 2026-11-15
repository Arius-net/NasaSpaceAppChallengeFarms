# Resumen de prompts destacables

## 1. Corrección del error de `lucide-react`

**Solicitud:** Corregir el error de compilación `Module not found: Can't resolve 'lucide-react'`.

**Acciones realizadas:**

- Se verificó que `lucide-react` ya estaba declarado en `package.json` y `package-lock.json`.
- Se detectó que el paquete no estaba instalado físicamente en `node_modules`.
- Se ejecutó `npm install`.
- Se validó la solución con `npm run build`.

**Resultado:** La aplicación volvió a compilar correctamente.

## 2. Búsqueda de variables y valores hardcodeados

**Solicitud:** Buscar variables, URLs, coordenadas y otros valores definidos directamente en el código.

**Hallazgos principales:**

- URL local del backend (`http://localhost:8000`).
- URLs de los proveedores de mapas y del worker de MapLibre.
- Coordenadas y datos iniciales de parcelas.
- Coordenadas y límites geográficos de La Frailesca.
- Umbrales agronómicos definidos como constantes.
- Colores visuales repetidos en los componentes.

**Resultado:** Se identificaron los valores que podrían trasladarse a configuración o datos externos. No se encontraron contraseñas, tokens ni API keys.

## 3. Eliminación de textos fijos de La Frailesca en el mapa

**Solicitud:** Evitar que el mapa mostrara “La Frailesca” aunque el usuario seleccionara un punto fuera de esa región.

**Cambios realizados:**

- El título del mapa pasó a usar dinámicamente el nombre de la parcela o ubicación actual.
- Se reemplazó “Centrar La Frailesca” por “Centrar punto”.
- Se eliminaron los textos fijos de municipios de La Frailesca.
- Se eliminaron el contorno y las capas geográficas fijas de esa región.
- El nombre alternativo pasó de “Punto seleccionado en La Frailesca” a “Punto seleccionado”.

**Archivos principales:**

- `components/MapboxViewer.tsx`
- `app/dashboard/page.tsx`

## 4. Botones de consulta por regiones de México

**Solicitud:** Agregar al menos ocho puntos preconfigurados para consultar distintas regiones mediante el backend.

**Puntos agregados:**

1. Villaflores — Frailesca.
2. La Concordia — Frailesca.
3. Tapachula — Sur de Chiapas.
4. Palenque — Norte de Chiapas.
5. Querétaro — Centro de México.
6. Ciudad de México — Centro de México.
7. Monterrey — Norte de México.
8. Mérida — Península de Yucatán.

**Comportamiento:**

- Cada botón actualiza el marcador y las coordenadas del mapa.
- La selección activa el modo de punto personalizado.
- Se consultan automáticamente los endpoints de recomendación y pronóstico.
- Se actualizan los indicadores climáticos y la recomendación mostrada.

**Validación:** Las modificaciones se verificaron con `npm run build`.
