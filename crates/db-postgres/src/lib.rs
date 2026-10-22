//! PostgreSQL Data Access Layer, Embedded SQLx Migrations, and Relational Repositories
//! for User Accounts (`cpf`, `phone`, `address`, `email`, `password_hash`), Nutrition Plans,
//! Daily Calorie Exchanges, Foods Eaten, Daily Total Calories, Chat Conversations/Messages,
//! and Agent Chat Memories.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use shared_types::{
    canonicalize_plan_food_key, hash_user_password, lookup_reference_macros_per_100g,
    validate_and_format_cpf, verify_user_password, CalorieExchangeOption, CreateUserRegistration,
    MacroSummary, ParsedNutritionPlan, ProbableDailyFood, SharedDomainError, UserAccountRecord,
};
use std::collections::BTreeMap;
use thiserror::Error;
use uuid::Uuid;

/// Embedded SQL migration 0001: Users (`cpf`, `phone`, `address`, `email`, `password_hash`) & Profiles.
pub const MIGRATION_0001_USERS_AND_PROFILES: &str =
    include_str!("../migrations/0001_create_users_and_profiles.sql");

/// Embedded SQL migration 0002: Nutrition Plans, Daily Calorie Exchanges, Foods Eaten, and Daily Totals.
pub const MIGRATION_0002_NUTRITION_TABLES: &str =
    include_str!("../migrations/0002_create_nutrition_plans_exchanges_and_foods_eaten.sql");

/// Embedded SQL migration 0003: Chat Conversations, Chat Messages, and Agent Chat Memories.
pub const MIGRATION_0003_CHAT_AND_AGENT_MEMORIES: &str =
    include_str!("../migrations/0003_create_chat_conversations_and_agent_memories.sql");

/// Descriptor for a sequential PostgreSQL migration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SqlMigration {
    pub version: u32,
    pub name: &'static str,
    pub sql: &'static str,
}

/// Returns all ordered PostgreSQL migrations in `crates/db-postgres/migrations/`.
pub fn registered_migrations() -> Vec<SqlMigration> {
    vec![
        SqlMigration {
            version: 1,
            name: "0001_create_users_and_profiles.sql",
            sql: MIGRATION_0001_USERS_AND_PROFILES,
        },
        SqlMigration {
            version: 2,
            name: "0002_create_nutrition_plans_exchanges_and_foods_eaten.sql",
            sql: MIGRATION_0002_NUTRITION_TABLES,
        },
        SqlMigration {
            version: 3,
            name: "0003_create_chat_conversations_and_agent_memories.sql",
            sql: MIGRATION_0003_CHAT_AND_AGENT_MEMORIES,
        },
    ]
}

/// Database repository errors.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PostgresDbError {
    #[error("Domain validation error: {0}")]
    Validation(#[from] SharedDomainError),

    #[error("Duplicate user constraint violation on '{field}': '{value}' already exists")]
    DuplicateUser { field: String, value: String },

    #[error("User not found: '{0}'")]
    UserNotFound(String),

    #[error("Nutrition plan not found: '{0}'")]
    PlanNotFound(String),
}

/// Row model for `user_nutrition_profiles`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserNutritionProfileRow {
    pub user_id: String,
    pub daily_calorie_target_kcal: f32,
    pub daily_protein_target_g: f32,
    pub daily_carbs_target_g: f32,
    pub daily_fat_target_g: f32,
    pub daily_fiber_target_g: f32,
    pub hydration_target_ml: u32,
    pub active_plan_id: Option<String>,
    pub goal: Option<String>,
}

/// Row model for `foods_eaten`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FoodEatenRow {
    pub id: String,
    pub user_id: String,
    pub plan_id: Option<String>,
    pub eaten_date: String,
    pub meal_name: String,
    pub food_name: String,
    pub canonical_key: String,
    pub grams: f32,
    pub calories_kcal: f32,
    pub protein_g: f32,
    pub carbs_g: f32,
    pub fat_g: f32,
    pub fiber_g: f32,
    pub exchanged_from_food: Option<String>,
    pub calorie_exchange_delta_kcal: f32,
}

/// Row model for `daily_calorie_exchanges`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DailyCalorieExchangeRow {
    pub id: String,
    pub user_id: String,
    pub plan_id: Option<String>,
    pub exchange_date: String,
    pub meal_name: String,
    pub primary_food_name: String,
    pub primary_grams: f32,
    pub primary_calories_kcal: f32,
    pub replacement_food_name: String,
    pub replacement_grams: f32,
    pub replacement_calories_kcal: f32,
    pub calorie_delta_kcal: f32,
    pub replacement_protein_g: f32,
    pub replacement_carbs_g: f32,
    pub replacement_fat_g: f32,
    pub applied_today: bool,
}

/// Row model for `daily_nutrition_totals`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DailyNutritionTotalsRow {
    pub id: String,
    pub user_id: String,
    pub log_date: String,
    pub total_calories_kcal: f32,
    pub target_calories_kcal: f32,
    pub remaining_calories_kcal: f32,
    pub calorie_exchange_daily_delta_kcal: f32,
    pub total_protein_g: f32,
    pub total_carbs_g: f32,
    pub total_fat_g: f32,
    pub total_fiber_g: f32,
    pub foods_eaten_count: u32,
    pub exchanges_applied_count: u32,
}

/// Input to log a food item eaten by the user (either custom, from plan, or swapped via calorie exchange).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogFoodEatenInput {
    pub user_id: String,
    pub eaten_date: Option<String>,
    pub meal_name: String,
    pub food_name: String,
    pub grams: f32,
    pub exchanged_from_food: Option<String>,
    pub primary_reference_calories_kcal: Option<f32>,
}

/// Row model for `chat_conversations` in PostgreSQL.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatConversationRow {
    pub id: String,
    pub user_id: String,
    pub title: String,
    pub language: String,
    pub message_count: u32,
    pub total_tokens_used: u32,
    pub created_at_iso: String,
    pub updated_at_iso: String,
}

/// Row model for `chat_messages` in PostgreSQL.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessageRow {
    pub id: String,
    pub conversation_id: String,
    pub user_id: String,
    pub sender_role: String,
    pub content: String,
    pub detected_intent: Option<String>,
    pub registered_meal_id: Option<String>,
    pub node_decision_path: Vec<String>,
    pub tokens_used: u32,
    pub created_at_iso: String,
}

/// Input payload to persist a chat message in PostgreSQL `chat_messages`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SaveChatMessageInput {
    pub conversation_id: String,
    pub user_id: String,
    pub sender_role: String,
    pub content: String,
    pub detected_intent: Option<String>,
    pub registered_meal_id: Option<String>,
    pub node_decision_path: Vec<String>,
    pub tokens_used: u32,
}

/// Row model for `agent_chat_memories` in PostgreSQL (`working`, `episodic`, `semantic` tiers).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentChatMemoryRow {
    pub id: String,
    pub user_id: String,
    pub conversation_id: Option<String>,
    pub source_message_id: Option<String>,
    pub memory_tier: String,
    pub memory_key: String,
    pub memory_value: String,
    pub created_at_iso: String,
}

/// Thread-safe PostgreSQL schema & repository store supporting migrations (`0001`, `0002`, `0003`),
/// unique CPF/email constraints, hashed passwords, nutrition plans, daily calorie exchanges,
/// foods eaten, daily totals, chat conversations/messages, and agent chat memories.
#[derive(Debug, Default)]
pub struct PostgresNutritionStore {
    applied_migrations: RwLock<Vec<u32>>,
    users: RwLock<BTreeMap<String, UserAccountRecord>>,
    profiles: RwLock<BTreeMap<String, UserNutritionProfileRow>>,
    plans: RwLock<BTreeMap<String, ParsedNutritionPlan>>,
    foods_eaten: RwLock<BTreeMap<String, Vec<FoodEatenRow>>>,
    calorie_exchanges: RwLock<BTreeMap<String, Vec<DailyCalorieExchangeRow>>>,
    daily_totals: RwLock<BTreeMap<(String, String), DailyNutritionTotalsRow>>,
    chat_conversations: RwLock<BTreeMap<String, ChatConversationRow>>,
    chat_messages: RwLock<BTreeMap<String, Vec<ChatMessageRow>>>,
    agent_chat_memories: RwLock<BTreeMap<String, Vec<AgentChatMemoryRow>>>,
}

impl PostgresNutritionStore {
    /// Initializes a new store and runs all embedded PostgreSQL migrations (`0001`, `0002`, `0003`).
    pub fn new_with_migrations() -> Self {
        let store = Self::default();
        store.run_migrations();
        store
    }

    /// Applies all pending migrations from `crates/db-postgres/migrations/`.
    pub fn run_migrations(&self) -> Vec<u32> {
        let mut applied = self.applied_migrations.write();
        for mig in registered_migrations() {
            if !applied.contains(&mig.version) && !mig.sql.trim().is_empty() {
                applied.push(mig.version);
            }
        }
        applied.clone()
    }

    pub fn applied_migration_versions(&self) -> Vec<u32> {
        self.applied_migrations.read().clone()
    }

    /// Creates a new user in `users` with validated CPF, phone, address, email, and hashed password,
    /// and initializes their `user_nutrition_profiles` and `daily_nutrition_totals` rows.
    pub fn create_user(
        &self,
        input: CreateUserRegistration,
    ) -> Result<UserAccountRecord, PostgresDbError> {
        let formatted_cpf = validate_and_format_cpf(&input.cpf)?;
        let email_clean = input.email.trim().to_lowercase();
        if !email_clean.contains('@') || email_clean.len() < 5 {
            return Err(PostgresDbError::Validation(
                SharedDomainError::ValidationError {
                    field: "email".to_string(),
                    message: "Invalid email address".to_string(),
                },
            ));
        }
        let phone_clean = input.phone.trim().to_string();
        if phone_clean.len() < 8 {
            return Err(PostgresDbError::Validation(
                SharedDomainError::ValidationError {
                    field: "phone".to_string(),
                    message: "Phone number must have at least 8 characters".to_string(),
                },
            ));
        }
        let address_clean = input.address.trim().to_string();
        if address_clean.is_empty() {
            return Err(PostgresDbError::Validation(
                SharedDomainError::ValidationError {
                    field: "address".to_string(),
                    message: "Address cannot be empty".to_string(),
                },
            ));
        }

        let password_hash = hash_user_password(&input.password, &email_clean)?;

        let mut users = self.users.write();
        for existing in users.values() {
            if existing.cpf == formatted_cpf {
                return Err(PostgresDbError::DuplicateUser {
                    field: "cpf".to_string(),
                    value: formatted_cpf,
                });
            }
            if existing.email == email_clean {
                return Err(PostgresDbError::DuplicateUser {
                    field: "email".to_string(),
                    value: email_clean,
                });
            }
        }

        let user_id = format!("usr-{}", &Uuid::new_v4().to_string()[..8]);
        let record = UserAccountRecord {
            id: user_id.clone(),
            cpf: formatted_cpf,
            phone: phone_clean,
            address: address_clean,
            email: email_clean,
            password_hash,
            full_name: input.full_name.map(|s| s.trim().to_string()),
            created_at_iso: "2026-10-22T17:15:00Z".to_string(),
        };
        users.insert(user_id.clone(), record.clone());
        drop(users);

        self.profiles.write().insert(
            user_id.clone(),
            UserNutritionProfileRow {
                user_id: user_id.clone(),
                daily_calorie_target_kcal: 2200.0,
                daily_protein_target_g: 165.0,
                daily_carbs_target_g: 220.0,
                daily_fat_target_g: 65.0,
                daily_fiber_target_g: 30.0,
                hydration_target_ml: 2500,
                active_plan_id: None,
                goal: Some("Maintenance / General Fitness".to_string()),
            },
        );

        self.recompute_daily_totals(&user_id, "2026-10-22");
        Ok(record)
    }

    /// Authenticates a user by email (or CPF) and raw password against the stored `password_hash`.
    pub fn authenticate_user(&self, login: &str, raw_password: &str) -> Option<UserAccountRecord> {
        let login_clean = login.trim().to_lowercase();
        let login_digits: String = login.chars().filter(|c| c.is_ascii_digit()).collect();
        let users = self.users.read();
        let user = users.values().find(|u| {
            let user_cpf_digits: String = u.cpf.chars().filter(|c| c.is_ascii_digit()).collect();
            u.email == login_clean || (!login_digits.is_empty() && user_cpf_digits == login_digits)
        })?;
        if verify_user_password(raw_password, &user.email, &user.password_hash) {
            Some(user.clone())
        } else {
            None
        }
    }

    pub fn get_user(&self, user_id: &str) -> Option<UserAccountRecord> {
        self.users.read().get(user_id).cloned()
    }

    pub fn list_users(&self) -> Vec<UserAccountRecord> {
        self.users.read().values().cloned().collect()
    }

    pub fn get_user_profile(&self, user_id: &str) -> Option<UserNutritionProfileRow> {
        self.profiles.read().get(user_id).cloned()
    }

    /// Saves a parsed `.pdf` or `.txt` nutrition plan into `nutrition_plans`, updates
    /// `user_nutrition_profiles` targets, and populates `daily_calorie_exchanges` for the user.
    pub fn save_parsed_nutrition_plan(
        &self,
        plan: ParsedNutritionPlan,
    ) -> Result<ParsedNutritionPlan, PostgresDbError> {
        if !self.users.read().contains_key(&plan.user_id) {
            return Err(PostgresDbError::UserNotFound(plan.user_id));
        }

        // Update user nutrition profile targets from the plan
        {
            let mut profiles = self.profiles.write();
            let entry = profiles
                .entry(plan.user_id.clone())
                .or_insert_with(|| UserNutritionProfileRow {
                    user_id: plan.user_id.clone(),
                    daily_calorie_target_kcal: plan.target_daily_macros.calories_kcal,
                    daily_protein_target_g: plan.target_daily_macros.protein_g,
                    daily_carbs_target_g: plan.target_daily_macros.carbs_g,
                    daily_fat_target_g: plan.target_daily_macros.fat_g,
                    daily_fiber_target_g: plan.target_daily_macros.fiber_g,
                    hydration_target_ml: plan.hydration_target_ml,
                    active_plan_id: Some(plan.plan_id.clone()),
                    goal: plan.extracted_user_profile.goal.clone(),
                });
            entry.daily_calorie_target_kcal = plan.target_daily_macros.calories_kcal;
            entry.daily_protein_target_g = plan.target_daily_macros.protein_g;
            entry.daily_carbs_target_g = plan.target_daily_macros.carbs_g;
            entry.daily_fat_target_g = plan.target_daily_macros.fat_g;
            entry.daily_fiber_target_g = plan.target_daily_macros.fiber_g;
            entry.hydration_target_ml = plan.hydration_target_ml;
            entry.active_plan_id = Some(plan.plan_id.clone());
            if plan.extracted_user_profile.goal.is_some() {
                entry.goal = plan.extracted_user_profile.goal.clone();
            }
        }

        // Populate `daily_calorie_exchanges` rows from the parsed plan
        {
            let mut ex_guard = self.calorie_exchanges.write();
            let user_exchanges = ex_guard.entry(plan.user_id.clone()).or_default();
            user_exchanges.clear();
            for ex in &plan.calorie_exchange_options {
                user_exchanges.push(Self::map_exchange_row(&plan.user_id, &plan.plan_id, ex));
            }
        }

        self.plans
            .write()
            .insert(plan.plan_id.clone(), plan.clone());
        self.recompute_daily_totals(&plan.user_id, "2026-10-22");
        Ok(plan)
    }

    fn map_exchange_row(
        user_id: &str,
        plan_id: &str,
        ex: &CalorieExchangeOption,
    ) -> DailyCalorieExchangeRow {
        DailyCalorieExchangeRow {
            id: ex.exchange_id.clone(),
            user_id: user_id.to_string(),
            plan_id: Some(plan_id.to_string()),
            exchange_date: "2026-10-22".to_string(),
            meal_name: ex.meal_name.clone(),
            primary_food_name: ex.primary_food_name.clone(),
            primary_grams: ex.primary_grams,
            primary_calories_kcal: ex.primary_calories_kcal,
            replacement_food_name: ex.replacement_food_name.clone(),
            replacement_grams: ex.replacement_grams,
            replacement_calories_kcal: ex.replacement_macros.calories_kcal,
            calorie_delta_kcal: ex.calorie_delta_kcal,
            replacement_protein_g: ex.replacement_macros.protein_g,
            replacement_carbs_g: ex.replacement_macros.carbs_g,
            replacement_fat_g: ex.replacement_macros.fat_g,
            applied_today: false,
        }
    }

    pub fn get_active_plan_for_user(&self, user_id: &str) -> Option<ParsedNutritionPlan> {
        let plan_id = self
            .profiles
            .read()
            .get(user_id)
            .and_then(|p| p.active_plan_id.clone())?;
        self.plans.read().get(&plan_id).cloned()
    }

    pub fn get_probable_daily_foods(&self, user_id: &str) -> Vec<ProbableDailyFood> {
        self.get_active_plan_for_user(user_id)
            .map(|p| p.probable_daily_foods)
            .unwrap_or_default()
    }

    pub fn get_daily_calorie_exchanges(&self, user_id: &str) -> Vec<DailyCalorieExchangeRow> {
        self.calorie_exchanges
            .read()
            .get(user_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Logs a food eaten by the user into `foods_eaten` and recalculates `daily_nutrition_totals`.
    pub fn log_food_eaten(
        &self,
        input: LogFoodEatenInput,
    ) -> Result<(FoodEatenRow, DailyNutritionTotalsRow), PostgresDbError> {
        if !self.users.read().contains_key(&input.user_id) {
            return Err(PostgresDbError::UserNotFound(input.user_id));
        }
        if input.grams <= 0.0 || !input.grams.is_finite() {
            return Err(PostgresDbError::Validation(
                SharedDomainError::ValidationError {
                    field: "grams".to_string(),
                    message: "Grams must be a positive number".to_string(),
                },
            ));
        }

        let date_str = input
            .eaten_date
            .unwrap_or_else(|| "2026-10-22".to_string());
        let canonical_key = canonicalize_plan_food_key(&input.food_name);
        let macros_100g = lookup_reference_macros_per_100g(&canonical_key);
        let scaled = macros_100g.scaled_for_grams(input.grams);

        let delta_kcal = match input.primary_reference_calories_kcal {
            Some(primary_kcal) => ((scaled.calories_kcal - primary_kcal) * 10.0).round() / 10.0,
            None => 0.0,
        };

        let active_plan_id = self
            .profiles
            .read()
            .get(&input.user_id)
            .and_then(|p| p.active_plan_id.clone());

        let row = FoodEatenRow {
            id: format!("fe-{}", &Uuid::new_v4().to_string()[..8]),
            user_id: input.user_id.clone(),
            plan_id: active_plan_id,
            eaten_date: date_str.clone(),
            meal_name: input.meal_name,
            food_name: input.food_name,
            canonical_key,
            grams: input.grams,
            calories_kcal: scaled.calories_kcal,
            protein_g: scaled.protein_g,
            carbs_g: scaled.carbs_g,
            fat_g: scaled.fat_g,
            fiber_g: scaled.fiber_g,
            exchanged_from_food: input.exchanged_from_food,
            calorie_exchange_delta_kcal: delta_kcal,
        };

        self.foods_eaten
            .write()
            .entry(input.user_id.clone())
            .or_default()
            .push(row.clone());

        let totals = self.recompute_daily_totals(&input.user_id, &date_str);
        Ok((row, totals))
    }

    /// Applies a daily calorie exchange option (`exchange_id`), logging the replacement food
    /// into `foods_eaten`, marking the exchange as applied today, and updating `daily_nutrition_totals`.
    pub fn apply_calorie_exchange(
        &self,
        user_id: &str,
        exchange_id: &str,
    ) -> Result<(FoodEatenRow, DailyNutritionTotalsRow), PostgresDbError> {
        let exchange = {
            let mut guard = self.calorie_exchanges.write();
            let list = guard
                .get_mut(user_id)
                .ok_or_else(|| PostgresDbError::UserNotFound(user_id.to_string()))?;
            let found = list
                .iter_mut()
                .find(|e| e.id == exchange_id)
                .ok_or_else(|| {
                    PostgresDbError::Validation(SharedDomainError::ValidationError {
                        field: "exchange_id".to_string(),
                        message: format!("Calorie exchange '{}' not found", exchange_id),
                    })
                })?;
            found.applied_today = true;
            found.clone()
        };

        self.log_food_eaten(LogFoodEatenInput {
            user_id: user_id.to_string(),
            eaten_date: Some(exchange.exchange_date),
            meal_name: exchange.meal_name,
            food_name: exchange.replacement_food_name,
            grams: exchange.replacement_grams,
            exchanged_from_food: Some(exchange.primary_food_name),
            primary_reference_calories_kcal: Some(exchange.primary_calories_kcal),
        })
    }

    pub fn list_foods_eaten(&self, user_id: &str) -> Vec<FoodEatenRow> {
        self.foods_eaten
            .read()
            .get(user_id)
            .cloned()
            .unwrap_or_default()
    }

    pub fn get_daily_totals(&self, user_id: &str, log_date: &str) -> DailyNutritionTotalsRow {
        if let Some(existing) = self
            .daily_totals
            .read()
            .get(&(user_id.to_string(), log_date.to_string()))
        {
            return existing.clone();
        }
        self.recompute_daily_totals(user_id, log_date)
    }

    fn recompute_daily_totals(&self, user_id: &str, log_date: &str) -> DailyNutritionTotalsRow {
        let target_kcal = self
            .profiles
            .read()
            .get(user_id)
            .map(|p| p.daily_calorie_target_kcal)
            .unwrap_or(2200.0);

        let foods = self.list_foods_eaten(user_id);
        let mut macros = MacroSummary::default();
        let mut exchange_delta = 0.0_f32;
        let mut count = 0_u32;
        let mut exchanges_count = 0_u32;

        for f in foods.iter().filter(|f| f.eaten_date == log_date) {
            macros.add_assign(&MacroSummary {
                calories_kcal: f.calories_kcal,
                protein_g: f.protein_g,
                carbs_g: f.carbs_g,
                fat_g: f.fat_g,
                fiber_g: f.fiber_g,
            });
            exchange_delta =
                ((exchange_delta + f.calorie_exchange_delta_kcal) * 10.0).round() / 10.0;
            count += 1;
            if f.exchanged_from_food.is_some() {
                exchanges_count += 1;
            }
        }

        let remaining = ((target_kcal - macros.calories_kcal) * 10.0).round() / 10.0;
        let row = DailyNutritionTotalsRow {
            id: format!("tot-{}-{}", user_id, log_date),
            user_id: user_id.to_string(),
            log_date: log_date.to_string(),
            total_calories_kcal: macros.calories_kcal,
            target_calories_kcal: target_kcal,
            remaining_calories_kcal: remaining,
            calorie_exchange_daily_delta_kcal: exchange_delta,
            total_protein_g: macros.protein_g,
            total_carbs_g: macros.carbs_g,
            total_fat_g: macros.fat_g,
            total_fiber_g: macros.fiber_g,
            foods_eaten_count: count,
            exchanges_applied_count: exchanges_count,
        };

        self.daily_totals
            .write()
            .insert((user_id.to_string(), log_date.to_string()), row.clone());
        row
    }

    /// Retrieves an existing `chat_conversations` record or creates a new one for `user_id`.
    pub fn get_or_create_conversation(
        &self,
        user_id: &str,
        conversation_id: Option<&str>,
        title: Option<&str>,
    ) -> Result<ChatConversationRow, PostgresDbError> {
        if !self.users.read().contains_key(user_id) {
            return Err(PostgresDbError::UserNotFound(user_id.to_string()));
        }

        let conv_id = conversation_id
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("conv-{}", user_id));

        let mut convs = self.chat_conversations.write();
        if let Some(existing) = convs.get(&conv_id) {
            return Ok(existing.clone());
        }

        let row = ChatConversationRow {
            id: conv_id.clone(),
            user_id: user_id.to_string(),
            title: title
                .unwrap_or("Chat Nutricional & Calorias")
                .to_string(),
            language: "pt".to_string(),
            message_count: 0,
            total_tokens_used: 0,
            created_at_iso: "2026-10-22T19:30:00Z".to_string(),
            updated_at_iso: "2026-10-22T19:30:00Z".to_string(),
        };
        convs.insert(conv_id, row.clone());
        Ok(row)
    }

    /// Persists a message (`user`, `assistant`, or `system`) in PostgreSQL `chat_messages`
    /// and updates the parent `chat_conversations` counters.
    pub fn save_chat_message(
        &self,
        input: SaveChatMessageInput,
    ) -> Result<ChatMessageRow, PostgresDbError> {
        if !self.users.read().contains_key(&input.user_id) {
            return Err(PostgresDbError::UserNotFound(input.user_id));
        }

        let _ = self.get_or_create_conversation(&input.user_id, Some(&input.conversation_id), None)?;

        let msg_row = ChatMessageRow {
            id: format!("msg-{}", &Uuid::new_v4().to_string()[..8]),
            conversation_id: input.conversation_id.clone(),
            user_id: input.user_id.clone(),
            sender_role: input.sender_role,
            content: input.content,
            detected_intent: input.detected_intent,
            registered_meal_id: input.registered_meal_id,
            node_decision_path: input.node_decision_path,
            tokens_used: input.tokens_used,
            created_at_iso: "2026-10-22T19:30:05Z".to_string(),
        };

        self.chat_messages
            .write()
            .entry(input.conversation_id.clone())
            .or_default()
            .push(msg_row.clone());

        if let Some(conv) = self.chat_conversations.write().get_mut(&input.conversation_id) {
            conv.message_count += 1;
            conv.total_tokens_used += msg_row.tokens_used;
            conv.updated_at_iso = "2026-10-22T19:30:05Z".to_string();
        }

        Ok(msg_row)
    }

    /// Lists all `chat_conversations` belonging to `user_id`.
    pub fn list_chat_conversations(&self, user_id: &str) -> Vec<ChatConversationRow> {
        self.chat_conversations
            .read()
            .values()
            .filter(|c| c.user_id == user_id)
            .cloned()
            .collect()
    }

    /// Lists all `chat_messages` in chronological order for a `conversation_id`.
    pub fn list_chat_messages(&self, conversation_id: &str) -> Vec<ChatMessageRow> {
        self.chat_messages
            .read()
            .get(conversation_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Lists all `chat_messages` across all conversations for `user_id`.
    pub fn list_user_chat_messages(&self, user_id: &str) -> Vec<ChatMessageRow> {
        let conv_ids: Vec<String> = self
            .list_chat_conversations(user_id)
            .into_iter()
            .map(|c| c.id)
            .collect();
        let guard = self.chat_messages.read();
        let mut all = Vec::new();
        for cid in conv_ids {
            if let Some(msgs) = guard.get(&cid) {
                all.extend(msgs.iter().cloned());
            }
        }
        all
    }

    /// Saves a chat-derived memory item into PostgreSQL `agent_chat_memories` (`working`, `episodic`, `semantic`)
    /// so agents can recall past chat turns and user facts.
    pub fn save_agent_chat_memory(
        &self,
        user_id: &str,
        conversation_id: Option<&str>,
        source_message_id: Option<&str>,
        memory_tier: &str,
        memory_key: &str,
        memory_value: &str,
    ) -> AgentChatMemoryRow {
        let row = AgentChatMemoryRow {
            id: format!("cmem-{}", &Uuid::new_v4().to_string()[..8]),
            user_id: user_id.to_string(),
            conversation_id: conversation_id.map(|s| s.to_string()),
            source_message_id: source_message_id.map(|s| s.to_string()),
            memory_tier: memory_tier.to_string(),
            memory_key: memory_key.to_string(),
            memory_value: memory_value.to_string(),
            created_at_iso: "2026-10-22T19:30:05Z".to_string(),
        };

        self.agent_chat_memories
            .write()
            .entry(user_id.to_string())
            .or_default()
            .push(row.clone());

        row
    }

    /// Lists all persisted `agent_chat_memories` rows for `user_id`.
    pub fn list_agent_chat_memories(&self, user_id: &str) -> Vec<AgentChatMemoryRow> {
        self.agent_chat_memories
            .read()
            .get(user_id)
            .cloned()
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::NutritionPlanDocumentReader;

    #[test]
    fn test_postgres_migrations_contain_required_tables_and_columns() {
        let migrations = registered_migrations();
        assert_eq!(migrations.len(), 3);

        let m1 = migrations[0].sql;
        assert!(m1.contains("CREATE TABLE IF NOT EXISTS users"));
        assert!(m1.contains("cpf"));
        assert!(m1.contains("phone"));
        assert!(m1.contains("address"));
        assert!(m1.contains("email"));
        assert!(m1.contains("password_hash"));

        let m2 = migrations[1].sql;
        assert!(m2.contains("CREATE TABLE IF NOT EXISTS nutrition_plans"));
        assert!(m2.contains("CREATE TABLE IF NOT EXISTS daily_calorie_exchanges"));
        assert!(m2.contains("CREATE TABLE IF NOT EXISTS foods_eaten"));
        assert!(m2.contains("CREATE TABLE IF NOT EXISTS daily_nutrition_totals"));

        let m3 = migrations[2].sql;
        assert!(m3.contains("CREATE TABLE IF NOT EXISTS chat_conversations"));
        assert!(m3.contains("CREATE TABLE IF NOT EXISTS chat_messages"));
        assert!(m3.contains("CREATE TABLE IF NOT EXISTS agent_chat_memories"));
    }

    #[test]
    fn test_user_creation_authentication_plan_storage_chat_persistence_and_daily_calorie_exchange() {
        let db = PostgresNutritionStore::new_with_migrations();
        assert_eq!(db.applied_migration_versions(), vec![1, 2, 3]);

        // 1. Register user with CPF, phone, address, email, and password
        let user = db
            .create_user(CreateUserRegistration {
                cpf: "12345678909".to_string(),
                phone: "+55 11 98888-7777".to_string(),
                address: "Rua Oscar Freire, 500 - Sao Paulo, SP".to_string(),
                email: "lucas@calorie.ai".to_string(),
                password: "StrongPassword#2026".to_string(),
                full_name: Some("Lucas Ferreira".to_string()),
            })
            .expect("User creation should succeed");

        assert_eq!(user.cpf, "123.456.789-09");
        assert!(user.password_hash.starts_with("$sha256-1024$"));
        assert_ne!(user.password_hash, "StrongPassword#2026");

        // Duplicate CPF must fail
        let dup = db.create_user(CreateUserRegistration {
            cpf: "123.456.789-09".to_string(),
            phone: "+55 11 90000-0000".to_string(),
            address: "Outro Endereco, 10".to_string(),
            email: "outro@calorie.ai".to_string(),
            password: "StrongPassword#2026".to_string(),
            full_name: None,
        });
        assert!(matches!(dup, Err(PostgresDbError::DuplicateUser { .. })));

        // Authenticate user
        assert!(db
            .authenticate_user("lucas@calorie.ai", "StrongPassword#2026")
            .is_some());
        assert!(db
            .authenticate_user("lucas@calorie.ai", "wrong-password")
            .is_none());

        // 2. Parse a nutrition plan (.txt) and store in db
        let plan_txt = r#"
Plano Alimentar: Hipertrofia
Nutricionista: Dra. Camila - CRN-3 11223
Meta: 2400 kcal | Proteina: 180g | Carboidratos: 260g | Gordura: 70g
Hidratacao: 3000 ml
08:00 - Cafe da Manha
- 3 unidades de Ovos Mexidos (150g) OU 30g de Whey Protein
- 60g de Aveia OU 80g de Pao Integral
12:30 - Almoco
- 180g de Peito de Frango Grelhado OU 220g de Tilapia
- 150g de Arroz Integral OU 180g de Batata Doce
"#;
        let parsed_plan = NutritionPlanDocumentReader::parse_nutrition_plan(
            &user.id,
            "plano_lucas.txt",
            plan_txt.as_bytes(),
        )
        .expect("Plan parse ok");

        db.save_parsed_nutrition_plan(parsed_plan)
            .expect("Save plan ok");

        let profile = db.get_user_profile(&user.id).expect("Profile exists");
        assert_eq!(profile.daily_calorie_target_kcal, 2400.0);
        assert_eq!(profile.daily_protein_target_g, 180.0);
        assert_eq!(profile.hydration_target_ml, 3000);

        let probable_foods = db.get_probable_daily_foods(&user.id);
        assert_eq!(probable_foods.len(), 4);

        let exchanges = db.get_daily_calorie_exchanges(&user.id);
        assert_eq!(exchanges.len(), 4);

        // 3. Log a primary food eaten and apply a daily calorie exchange swap
        let (_, totals_after_first) = db
            .log_food_eaten(LogFoodEatenInput {
                user_id: user.id.clone(),
                eaten_date: Some("2026-10-22".to_string()),
                meal_name: "08:00 - Cafe da Manha".to_string(),
                food_name: "Ovos Mexidos".to_string(),
                grams: 150.0,
                exchanged_from_food: None,
                primary_reference_calories_kcal: None,
            })
            .expect("Log food ok");
        assert!(totals_after_first.total_calories_kcal > 200.0);
        assert_eq!(totals_after_first.foods_eaten_count, 1);

        let tilapia_exchange_id = exchanges
            .iter()
            .find(|e| e.replacement_food_name.to_lowercase().contains("tilapia"))
            .expect("Tilapia exchange must exist")
            .id
            .clone();

        let (swapped_food, totals_after_swap) = db
            .apply_calorie_exchange(&user.id, &tilapia_exchange_id)
            .expect("Apply calorie exchange ok");

        assert!(swapped_food.exchanged_from_food.is_some());
        assert_eq!(totals_after_swap.foods_eaten_count, 2);
        assert_eq!(totals_after_swap.exchanges_applied_count, 1);
        assert!(totals_after_swap.total_calories_kcal > totals_after_first.total_calories_kcal);

        // 4. Test Chat Conversation, Chat Messages, and Agent Chat Memories persistence
        let conv = db
            .get_or_create_conversation(&user.id, Some("conv-lucas-1"), Some("Coaching Lucas"))
            .expect("Conversation created");
        let user_msg = db
            .save_chat_message(SaveChatMessageInput {
                conversation_id: conv.id.clone(),
                user_id: user.id.clone(),
                sender_role: "user".to_string(),
                content: "O que posso comer no almoco segundo meu plano?".to_string(),
                detected_intent: Some("AskNutrition".to_string()),
                registered_meal_id: None,
                node_decision_path: vec!["IntentRouter".to_string(), "CoachingAgent".to_string()],
                tokens_used: 120,
            })
            .expect("Save user chat message");

        db.save_agent_chat_memory(
            &user.id,
            Some(&conv.id),
            Some(&user_msg.id),
            "episodic",
            "chat_turn",
            "User asked about lunch options in their nutrition plan",
        );

        let msgs = db.list_chat_messages(&conv.id);
        assert_eq!(msgs.len(), 1);
        let memories = db.list_agent_chat_memories(&user.id);
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].memory_tier, "episodic");
    }
}
