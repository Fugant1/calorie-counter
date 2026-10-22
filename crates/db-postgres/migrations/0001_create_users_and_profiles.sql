-- Migration 0001: Users personal/authentication information (CPF, phone, address, email, hashed password)
-- and user nutrition profile targets.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cpf             VARCHAR(14) NOT NULL UNIQUE,
    phone           VARCHAR(32) NOT NULL,
    address         TEXT NOT NULL,
    email           VARCHAR(255) NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    full_name       TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_users_cpf_nonempty CHECK (char_length(trim(cpf)) >= 11),
    CONSTRAINT chk_users_email_format CHECK (position('@' in email) > 1),
    CONSTRAINT chk_users_password_hash_nonempty CHECK (char_length(password_hash) >= 16)
);

CREATE INDEX IF NOT EXISTS idx_users_cpf ON users(cpf);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

CREATE TABLE IF NOT EXISTS user_nutrition_profiles (
    user_id                     UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    daily_calorie_target_kcal   REAL NOT NULL DEFAULT 2200.0,
    daily_protein_target_g      REAL NOT NULL DEFAULT 165.0,
    daily_carbs_target_g        REAL NOT NULL DEFAULT 220.0,
    daily_fat_target_g          REAL NOT NULL DEFAULT 65.0,
    daily_fiber_target_g        REAL NOT NULL DEFAULT 30.0,
    hydration_target_ml         INTEGER NOT NULL DEFAULT 2500,
    active_plan_id              UUID,
    goal                        TEXT,
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
