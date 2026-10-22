-- Migration 0002: Nutrition plans (parsed from .pdf/.txt), daily calorie exchanges,
-- foods eaten, and daily nutrition total calories rollup tables.

CREATE TABLE IF NOT EXISTS nutrition_plans (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                 UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title                   TEXT NOT NULL,
    source_filename         TEXT NOT NULL,
    source_format           VARCHAR(16) NOT NULL CHECK (source_format IN ('pdf', 'txt')),
    file_sha256             CHAR(64) NOT NULL,
    nutritionist_name       TEXT,
    nutritionist_crn        TEXT,
    target_calories_kcal    REAL NOT NULL,
    target_protein_g        REAL NOT NULL,
    target_carbs_g          REAL NOT NULL,
    target_fat_g            REAL NOT NULL,
    planned_calories_kcal   REAL NOT NULL,
    hydration_target_ml     INTEGER NOT NULL DEFAULT 2500,
    probable_daily_foods    JSONB NOT NULL DEFAULT '[]'::jsonb,
    general_guidelines      JSONB NOT NULL DEFAULT '[]'::jsonb,
    status                  VARCHAR(24) NOT NULL DEFAULT 'ACTIVE',
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_nutrition_plans_user_status ON nutrition_plans(user_id, status);

CREATE TABLE IF NOT EXISTS daily_calorie_exchanges (
    id                          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    plan_id                     UUID REFERENCES nutrition_plans(id) ON DELETE SET NULL,
    exchange_date               DATE NOT NULL DEFAULT CURRENT_DATE,
    meal_name                   TEXT NOT NULL,
    primary_food_name           TEXT NOT NULL,
    primary_grams               REAL NOT NULL,
    primary_calories_kcal       REAL NOT NULL,
    replacement_food_name       TEXT NOT NULL,
    replacement_grams           REAL NOT NULL,
    replacement_calories_kcal   REAL NOT NULL,
    calorie_delta_kcal          REAL NOT NULL,
    replacement_protein_g       REAL NOT NULL DEFAULT 0.0,
    replacement_carbs_g         REAL NOT NULL DEFAULT 0.0,
    replacement_fat_g           REAL NOT NULL DEFAULT 0.0,
    applied_today               BOOLEAN NOT NULL DEFAULT FALSE,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_daily_calorie_exchanges_user_date
    ON daily_calorie_exchanges(user_id, exchange_date);

CREATE TABLE IF NOT EXISTS foods_eaten (
    id                              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    plan_id                         UUID REFERENCES nutrition_plans(id) ON DELETE SET NULL,
    eaten_date                      DATE NOT NULL DEFAULT CURRENT_DATE,
    meal_name                       TEXT NOT NULL,
    food_name                       TEXT NOT NULL,
    canonical_key                   TEXT NOT NULL,
    grams                           REAL NOT NULL CHECK (grams > 0.0),
    calories_kcal                   REAL NOT NULL CHECK (calories_kcal >= 0.0),
    protein_g                       REAL NOT NULL DEFAULT 0.0,
    carbs_g                         REAL NOT NULL DEFAULT 0.0,
    fat_g                           REAL NOT NULL DEFAULT 0.0,
    fiber_g                         REAL NOT NULL DEFAULT 0.0,
    exchanged_from_food             TEXT,
    calorie_exchange_delta_kcal     REAL NOT NULL DEFAULT 0.0,
    logged_at                       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_foods_eaten_user_date ON foods_eaten(user_id, eaten_date);

CREATE TABLE IF NOT EXISTS daily_nutrition_totals (
    id                                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                             UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    log_date                            DATE NOT NULL DEFAULT CURRENT_DATE,
    total_calories_kcal                 REAL NOT NULL DEFAULT 0.0,
    target_calories_kcal                REAL NOT NULL DEFAULT 2200.0,
    remaining_calories_kcal             REAL NOT NULL DEFAULT 2200.0,
    calorie_exchange_daily_delta_kcal   REAL NOT NULL DEFAULT 0.0,
    total_protein_g                     REAL NOT NULL DEFAULT 0.0,
    total_carbs_g                       REAL NOT NULL DEFAULT 0.0,
    total_fat_g                         REAL NOT NULL DEFAULT 0.0,
    total_fiber_g                       REAL NOT NULL DEFAULT 0.0,
    foods_eaten_count                   INTEGER NOT NULL DEFAULT 0,
    exchanges_applied_count             INTEGER NOT NULL DEFAULT 0,
    updated_at                          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_daily_nutrition_totals_user_date UNIQUE (user_id, log_date)
);

CREATE INDEX IF NOT EXISTS idx_daily_nutrition_totals_user_date
    ON daily_nutrition_totals(user_id, log_date);
