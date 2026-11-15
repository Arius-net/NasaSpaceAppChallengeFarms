-- ============================================================================
-- K'INAL - Esquema Relacional PostgreSQL (Supabase)
-- Tablas transaccionales para usuarios, parcelas e historial agronómico (RAG)
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ----------------------------------------------------------------------------
-- 1. Tabla: users (Productores / Técnicos registrados en K'INAL)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT NOT NULL UNIQUE,
    full_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ----------------------------------------------------------------------------
-- 2. Tabla: farms (Parcelas registradas por usuario)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS farms (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    location_lat DOUBLE PRECISION NOT NULL CHECK (location_lat >= -90.0 AND location_lat <= 90.0),
    location_lon DOUBLE PRECISION NOT NULL CHECK (location_lon >= -180.0 AND location_lon <= 180.0),
    area_hectares NUMERIC(10, 2) CHECK (area_hectares > 0),
    soil_type TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_farms_user_id ON farms(user_id);
CREATE INDEX IF NOT EXISTS idx_farms_coordinates ON farms(location_lat, location_lon);

-- ----------------------------------------------------------------------------
-- 3. Tabla: crop_history (Historial agronómico por parcela para contexto RAG)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS crop_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    farm_id UUID NOT NULL REFERENCES farms(id) ON DELETE CASCADE,
    cycle_year INTEGER NOT NULL CHECK (cycle_year >= 1950 AND cycle_year <= 2100),
    crop_type TEXT NOT NULL DEFAULT 'maíz de temporal',
    planting_date DATE,
    harvest_date DATE,
    yield_notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_crop_history_dates CHECK (
        harvest_date IS NULL OR planting_date IS NULL OR harvest_date >= planting_date
    )
);

CREATE INDEX IF NOT EXISTS idx_crop_history_farm_id ON crop_history(farm_id);
CREATE INDEX IF NOT EXISTS idx_crop_history_farm_cycle ON crop_history(farm_id, cycle_year DESC);
