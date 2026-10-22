//! HTTP REST Gateway & Application Service integrating User Registration (`cpf`, `phone`,
//! `address`, `email`, `password_hash`), `.pdf` / `.txt` Nutrition Plan Parsing (including
//! compressed FlateDecode/Type3 PDFs like `Plano Alimentar Teste.pdf`), Persisted User Chats
//! (`chat_conversations`, `chat_messages`, `agent_chat_memories`), Multi-Tier Agent Memory,
//! Probable Daily Foods, Daily Calorie Exchanges, and the Interactive Test UI.

use agent_graph::{
    state::{GraphInput, GraphState},
    GraphExecutor,
};
use db_postgres::{
    AgentChatMemoryRow, ChatConversationRow, ChatMessageRow, DailyCalorieExchangeRow,
    DailyNutritionTotalsRow, FoodEatenRow, LogFoodEatenInput, PostgresDbError,
    PostgresNutritionStore, SaveChatMessageInput, UserNutritionProfileRow,
};
use memory_engine::{NutritionPlanMemoryEngine, UserMemoryProfileSnapshot};
use serde::{Deserialize, Serialize};
use shared_types::{
    CreateUserRegistration, NutritionPlanDocumentReader, ParsedNutritionPlan, ProbableDailyFood,
    UserAccountRecord,
};
use std::sync::Arc;

/// Unified dashboard state returned to the UI for testing user profile, memory,
/// probable daily foods, daily calorie exchanges, foods eaten, total calories, and chats.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserDashboardResponse {
    pub user: UserAccountRecord,
    pub profile: UserNutritionProfileRow,
    pub memory_snapshot: UserMemoryProfileSnapshot,
    pub active_plan: Option<ParsedNutritionPlan>,
    pub probable_daily_foods: Vec<ProbableDailyFood>,
    pub daily_calorie_exchanges: Vec<DailyCalorieExchangeRow>,
    pub foods_eaten: Vec<FoodEatenRow>,
    pub daily_totals: DailyNutritionTotalsRow,
    pub chat_conversations: Vec<ChatConversationRow>,
    pub chat_messages: Vec<ChatMessageRow>,
    pub agent_chat_memories: Vec<AgentChatMemoryRow>,
}

/// Request payload for uploading a `.pdf` or `.txt` nutrition plan via the API/UI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadNutritionPlanRequest {
    pub user_id: String,
    pub filename: String,
    /// Optional hex-encoded raw binary bytes of the uploaded `.pdf` or `.txt` file.
    #[serde(default)]
    pub file_bytes_hex: Option<String>,
    /// Plain text content of the `.txt` plan, or text lines to encode/parse as a `.pdf` stream.
    #[serde(default)]
    pub content_text: String,
    /// When `true` or when `filename` ends with `.pdf`, encodes/parses through the PDF binary stream reader.
    #[serde(default)]
    pub as_pdf_binary: bool,
}

/// Request payload to load the workspace's `Plano Alimentar Teste.pdf` for a user.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadWorkspaceSamplePdfRequest {
    pub user_id: String,
}

/// Request payload to apply a daily calorie exchange (`Lista de Substituição`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplyCalorieExchangeRequest {
    pub user_id: String,
    pub exchange_id: String,
}

/// Request payload to send a chat message to the multi-agent graph and persist in PostgreSQL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SendChatMessageRequest {
    pub user_id: String,
    #[serde(default)]
    pub conversation_id: Option<String>,
    pub message: String,
}

/// Request payload for user authentication (`POST /api/v1/users/login`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoginUserRequest {
    pub login: String,
    pub password: String,
}

/// Shared application state wiring `PostgresNutritionStore`, `NutritionPlanMemoryEngine`, and `GraphExecutor`.
#[derive(Clone)]
pub struct AppServerState {
    pub db: Arc<PostgresNutritionStore>,
    pub memory: Arc<NutritionPlanMemoryEngine>,
    pub graph: Arc<GraphExecutor>,
}

impl Default for AppServerState {
    fn default() -> Self {
        Self::new()
    }
}

impl AppServerState {
    pub fn new() -> Self {
        Self {
            db: Arc::new(PostgresNutritionStore::new_with_migrations()),
            memory: Arc::new(NutritionPlanMemoryEngine::new()),
            graph: Arc::new(GraphExecutor::with_defaults()),
        }
    }

    /// Registers a new user (`cpf`, `phone`, `address`, `email`, `password` -> `password_hash`)
    /// in PostgreSQL and initializes their semantic profile memory and default chat conversation.
    pub fn register_user(
        &self,
        input: CreateUserRegistration,
    ) -> Result<UserDashboardResponse, String> {
        let user = self.db.create_user(input).map_err(|e| e.to_string())?;
        self.graph.context().memory_store.upsert_semantic_fact(
            &user.id,
            "user_cpf".to_string(),
            user.cpf.clone(),
        );
        self.graph.context().memory_store.upsert_semantic_fact(
            &user.id,
            "user_email".to_string(),
            user.email.clone(),
        );
        self.graph.context().memory_store.upsert_semantic_fact(
            &user.id,
            "user_phone".to_string(),
            user.phone.clone(),
        );
        self.graph.context().memory_store.upsert_semantic_fact(
            &user.id,
            "user_address".to_string(),
            user.address.clone(),
        );
        let _ = self
            .db
            .get_or_create_conversation(&user.id, None, Some("Chat Nutricional & Calorias"));
        self.get_user_dashboard(&user.id)
    }

    /// Authenticates an existing user by email or CPF + password (`POST /api/v1/users/login`).
    /// If the in-memory store is fresh and the user logs in with the default demo credentials,
    /// provisions the demo athlete account automatically for seamless testing.
    pub fn login_user(&self, req: LoginUserRequest) -> Result<UserDashboardResponse, String> {
        if let Some(user) = self.db.authenticate_user(&req.login, &req.password) {
            return self.get_user_dashboard(&user.id);
        }

        // If no users exist yet and the user submits the default demo login, seed the account
        if self.db.list_users().is_empty()
            && (req.login.trim().eq_ignore_ascii_case("gabriel.silva@calorie.ai")
                || req.login.trim() == "123.456.789-01"
                || req.login.trim() == "12345678901")
            && req.password == "NutriPlan#2026"
        {
            return self.register_user(CreateUserRegistration {
                full_name: Some("Gabriel Silva".to_string()),
                cpf: "123.456.789-01".to_string(),
                phone: "+55 11 99876-5432".to_string(),
                address: "Av. Paulista, 1000 - São Paulo, SP".to_string(),
                email: "gabriel.silva@calorie.ai".to_string(),
                password: req.password,
            });
        }

        Err("Invalid login or password / Login ou senha inválidos".to_string())
    }

    /// Parses a `.pdf` or `.txt` nutrition plan (from raw binary `file_bytes_hex` or text),
    /// saves user profile & probable daily foods in `memory-engine` and `agent-graph`,
    /// persists the plan and daily calorie exchanges in `db-postgres`, and records a memory entry.
    pub fn upload_and_parse_nutrition_plan(
        &self,
        req: UploadNutritionPlanRequest,
    ) -> Result<UserDashboardResponse, String> {
        let raw_bytes = if let Some(hex_str) = req
            .file_bytes_hex
            .as_ref()
            .filter(|s| !s.trim().is_empty())
        {
            hex::decode(hex_str.trim()).map_err(|e| format!("Invalid hex file payload: {}", e))?
        } else if req.as_pdf_binary || req.filename.to_lowercase().ends_with(".pdf") {
            if req.content_text.starts_with("%PDF-") {
                req.content_text.into_bytes()
            } else {
                let lines: Vec<&str> = req.content_text.lines().collect();
                NutritionPlanDocumentReader::build_pdf_document_from_text(&lines)
            }
        } else {
            req.content_text.into_bytes()
        };

        self.ingest_raw_plan_bytes(&req.user_id, &req.filename, &raw_bytes)
    }

    /// Loads `/home/fuganti/Projects/calorie-counter/Plano Alimentar Teste.pdf` from the workspace
    /// and ingests it for `user_id`.
    pub fn load_workspace_sample_pdf(&self, user_id: &str) -> Result<UserDashboardResponse, String> {
        let candidate_paths = [
            std::path::PathBuf::from("Plano Alimentar Teste.pdf"),
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Plano Alimentar Teste.pdf"),
        ];
        let mut pdf_bytes = None;
        for p in &candidate_paths {
            if let Ok(bytes) = std::fs::read(p) {
                pdf_bytes = Some(bytes);
                break;
            }
        }
        let bytes = pdf_bytes
            .ok_or_else(|| "Could not find 'Plano Alimentar Teste.pdf' in workspace".to_string())?;
        self.ingest_raw_plan_bytes(user_id, "Plano Alimentar Teste.pdf", &bytes)
    }

    fn ingest_raw_plan_bytes(
        &self,
        user_id: &str,
        filename: &str,
        raw_bytes: &[u8],
    ) -> Result<UserDashboardResponse, String> {
        // 1. Parse and ingest into Agent Graph (updates Graph Semantic/Episodic memory + warms RAG cache)
        let parsed_plan = self
            .graph
            .ingest_nutrition_plan_file(user_id, filename, raw_bytes)
            .map_err(|e| e.to_string())?;

        // 2. Save into Memory Engine (Working/Episodic/Semantic + probable daily foods)
        self.memory.save_parsed_plan_to_memory(parsed_plan.clone());

        // 3. Persist in PostgreSQL store (`nutrition_plans`, `user_nutrition_profiles`, `daily_calorie_exchanges`)
        self.db
            .save_parsed_nutrition_plan(parsed_plan.clone())
            .map_err(|e| e.to_string())?;

        // 4. Persist plan memory facts into PostgreSQL `agent_chat_memories` so agents recall them in chat
        self.db.save_agent_chat_memory(
            user_id,
            None,
            None,
            "semantic",
            "active_nutrition_plan",
            &format!(
                "{} ({} kcal, {} meals, {} probable daily foods, {} exchanges)",
                parsed_plan.title,
                parsed_plan.target_daily_macros.calories_kcal,
                parsed_plan.meals.len(),
                parsed_plan.probable_daily_foods.len(),
                parsed_plan.calorie_exchange_options.len()
            ),
        );

        self.get_user_dashboard(user_id)
    }

    /// Sends a user chat message through the `GraphExecutor`, persists both the user message
    /// and the assistant response in PostgreSQL (`chat_conversations` & `chat_messages`),
    /// logs any registered meal items into `foods_eaten`, and synchronizes the turn with
    /// PostgreSQL `agent_chat_memories` and `memory-engine` so agents remember the conversation.
    pub async fn send_chat_message(
        &self,
        req: SendChatMessageRequest,
    ) -> Result<(UserDashboardResponse, GraphState), String> {
        if req.message.trim().is_empty() {
            return Err("Chat message cannot be empty".to_string());
        }

        let conv = self
            .db
            .get_or_create_conversation(&req.user_id, req.conversation_id.as_deref(), None)
            .map_err(|e| e.to_string())?;

        // 1. Hydrate Agent Graph memory from persisted PostgreSQL `agent_chat_memories` & `chat_messages`
        let persisted_memories = self.db.list_agent_chat_memories(&req.user_id);
        for mem in &persisted_memories {
            if mem.memory_tier == "semantic" {
                self.graph.context().memory_store.upsert_semantic_fact(
                    &req.user_id,
                    mem.memory_key.clone(),
                    mem.memory_value.clone(),
                );
            }
        }

        // 2. Save the incoming user message to PostgreSQL `chat_messages`
        let saved_user_msg = self
            .db
            .save_chat_message(SaveChatMessageInput {
                conversation_id: conv.id.clone(),
                user_id: req.user_id.clone(),
                sender_role: "user".to_string(),
                content: req.message.trim().to_string(),
                detected_intent: None,
                registered_meal_id: None,
                node_decision_path: Vec::new(),
                tokens_used: 0,
            })
            .map_err(|e| e.to_string())?;

        // 3. Execute the multi-agent graph with the user's message
        let graph_state = self
            .graph
            .run(
                req.user_id.clone(),
                conv.id.clone(),
                GraphInput::text(req.message.trim()),
            )
            .await
            .map_err(|e| e.to_string())?;

        let intent_str = format!("{:?}", graph_state.plan.intent);
        let node_path: Vec<String> = graph_state
            .telemetry
            .node_metrics
            .iter()
            .map(|m| m.node_id.as_str().to_string())
            .collect();
        let tokens_used = graph_state.telemetry.total_token_usage.total_tokens;

        // 4. If the chat message registered a meal, also record the items in PostgreSQL `foods_eaten`
        let mut registered_meal_id = None;
        if let Some(reg) = &graph_state.meal_registration {
            if !reg.items.is_empty() {
                registered_meal_id = Some(reg.meal_id.clone());
                for item in &reg.items {
                    let _ = self.db.log_food_eaten(LogFoodEatenInput {
                        user_id: req.user_id.clone(),
                        eaten_date: Some("2026-10-22".to_string()),
                        meal_name: format!("Chat ({})", reg.meal_id),
                        food_name: item.display_name.clone(),
                        grams: item.grams,
                        exchanged_from_food: None,
                        primary_reference_calories_kcal: None,
                    });
                }
            }
        }

        let assistant_reply = graph_state
            .final_response
            .clone()
            .unwrap_or_else(|| "Mensagem processada pelo Calorie-Counter AI.".to_string());

        // 5. Save the assistant response to PostgreSQL `chat_messages`
        let saved_assistant_msg = self
            .db
            .save_chat_message(SaveChatMessageInput {
                conversation_id: conv.id.clone(),
                user_id: req.user_id.clone(),
                sender_role: "assistant".to_string(),
                content: assistant_reply.clone(),
                detected_intent: Some(intent_str.clone()),
                registered_meal_id,
                node_decision_path: node_path,
                tokens_used,
            })
            .map_err(|e| e.to_string())?;

        // 6. Persist Working, Episodic, and Semantic chat memory rows in PostgreSQL `agent_chat_memories`
        //    and update `NutritionPlanMemoryEngine` + `GraphExecutor` memory store!
        self.db.save_agent_chat_memory(
            &req.user_id,
            Some(&conv.id),
            Some(&saved_user_msg.id),
            "working",
            "chat_working_turn",
            &format!("User: {} | Intent: {}", req.message.trim(), intent_str),
        );
        self.db.save_agent_chat_memory(
            &req.user_id,
            Some(&conv.id),
            Some(&saved_assistant_msg.id),
            "episodic",
            "chat_episode",
            &format!(
                "User said '{}' -> Assistant ({})",
                req.message.trim(),
                intent_str
            ),
        );
        self.db.save_agent_chat_memory(
            &req.user_id,
            Some(&conv.id),
            Some(&saved_user_msg.id),
            "semantic",
            "last_user_chat_topic",
            req.message.trim(),
        );

        self.memory.record_chat_turn_in_memory(
            &req.user_id,
            &conv.id,
            req.message.trim(),
            &assistant_reply,
            &intent_str,
        );

        let dashboard = self.get_user_dashboard(&req.user_id)?;
        Ok((dashboard, graph_state))
    }

    /// Logs a food eaten by the user (either a probable daily food from the plan or a custom food)
    /// and updates PostgreSQL `foods_eaten` and `daily_nutrition_totals`.
    pub async fn log_food_eaten_and_update(
        &self,
        input: LogFoodEatenInput,
    ) -> Result<(UserDashboardResponse, GraphState), String> {
        let utterance = format!(
            "Comi {:.0}g de {} no {}",
            input.grams, input.food_name, input.meal_name
        );
        let graph_state = self
            .graph
            .run(
                input.user_id.clone(),
                format!("conv-{}", input.user_id),
                GraphInput::text(utterance),
            )
            .await
            .map_err(|e| e.to_string())?;

        let user_id = input.user_id.clone();
        self.db.log_food_eaten(input).map_err(|e| e.to_string())?;
        let dashboard = self.get_user_dashboard(&user_id)?;
        Ok((dashboard, graph_state))
    }

    /// Applies a daily calorie exchange option (`Lista de Substituição`), swapping the primary planned
    /// food for its replacement and updating `foods_eaten` and `daily_nutrition_totals`.
    pub fn apply_daily_calorie_exchange(
        &self,
        req: ApplyCalorieExchangeRequest,
    ) -> Result<UserDashboardResponse, String> {
        self.db
            .apply_calorie_exchange(&req.user_id, &req.exchange_id)
            .map_err(|e: PostgresDbError| e.to_string())?;
        self.get_user_dashboard(&req.user_id)
    }

    /// Builds the complete `UserDashboardResponse` for a user.
    pub fn get_user_dashboard(&self, user_id: &str) -> Result<UserDashboardResponse, String> {
        let user = self
            .db
            .get_user(user_id)
            .ok_or_else(|| format!("User '{}' not found", user_id))?;
        let profile = self
            .db
            .get_user_profile(user_id)
            .ok_or_else(|| format!("Profile for '{}' not found", user_id))?;
        let memory_snapshot = self.memory.get_user_snapshot(user_id);
        let active_plan = self.db.get_active_plan_for_user(user_id);
        let probable_daily_foods = self.db.get_probable_daily_foods(user_id);
        let daily_calorie_exchanges = self.db.get_daily_calorie_exchanges(user_id);
        let foods_eaten = self.db.list_foods_eaten(user_id);
        let daily_totals = self.db.get_daily_totals(user_id, "2026-10-22");
        let chat_conversations = self.db.list_chat_conversations(user_id);
        let chat_messages = self.db.list_user_chat_messages(user_id);
        let agent_chat_memories = self.db.list_agent_chat_memories(user_id);

        Ok(UserDashboardResponse {
            user,
            profile,
            memory_snapshot,
            active_plan,
            probable_daily_foods,
            daily_calorie_exchanges,
            foods_eaten,
            daily_totals,
            chat_conversations,
            chat_messages,
            agent_chat_memories,
        })
    }

    /// Processes an HTTP request (method, path, body) and returns `(status_code, content_type, response_body)`.
    pub async fn handle_http_request(
        &self,
        method: &str,
        path: &str,
        body: &str,
    ) -> (u16, &'static str, String) {
        if method == "GET" && (path == "/" || path == "/index.html") {
            return (
                200,
                "text/html; charset=utf-8",
                INTERACTIVE_TEST_UI_HTML.to_string(),
            );
        }

        if method == "GET" && path == "/api/v1/health" {
            return (
                200,
                "application/json",
                r#"{"status":"ok","migrations":[1,2,3]}"#.to_string(),
            );
        }

        if method == "GET" && path == "/api/v1/config/llms" {
            let configs = self.graph.context().llm_config.all_configs();
            return (
                200,
                "application/json",
                serde_json::to_string(&configs).unwrap_or_default(),
            );
        }

        if method == "POST" && path == "/api/v1/users/login" {
            match serde_json::from_str::<LoginUserRequest>(body) {
                Ok(req) => match self.login_user(req) {
                    Ok(dash) => (
                        200,
                        "application/json",
                        serde_json::to_string(&dash).unwrap_or_default(),
                    ),
                    Err(e) => (
                        401,
                        "application/json",
                        serde_json::json!({ "error": e }).to_string(),
                    ),
                },
                Err(e) => (
                    400,
                    "application/json",
                    serde_json::json!({ "error": e.to_string() }).to_string(),
                ),
            }
        } else if method == "POST" && path == "/api/v1/users" {
            match serde_json::from_str::<CreateUserRegistration>(body) {
                Ok(reg) => match self.register_user(reg) {
                    Ok(dash) => (
                        201,
                        "application/json",
                        serde_json::to_string(&dash).unwrap_or_default(),
                    ),
                    Err(e) => (
                        400,
                        "application/json",
                        serde_json::json!({ "error": e }).to_string(),
                    ),
                },
                Err(e) => (
                    400,
                    "application/json",
                    serde_json::json!({ "error": e.to_string() }).to_string(),
                ),
            }
        } else if method == "POST" && path == "/api/v1/plans/upload" {
            match serde_json::from_str::<UploadNutritionPlanRequest>(body) {
                Ok(req) => match self.upload_and_parse_nutrition_plan(req) {
                    Ok(dash) => (
                        200,
                        "application/json",
                        serde_json::to_string(&dash).unwrap_or_default(),
                    ),
                    Err(e) => (
                        400,
                        "application/json",
                        serde_json::json!({ "error": e }).to_string(),
                    ),
                },
                Err(e) => (
                    400,
                    "application/json",
                    serde_json::json!({ "error": e.to_string() }).to_string(),
                ),
            }
        } else if method == "POST" && path == "/api/v1/plans/load-sample-pdf" {
            match serde_json::from_str::<LoadWorkspaceSamplePdfRequest>(body) {
                Ok(req) => match self.load_workspace_sample_pdf(&req.user_id) {
                    Ok(dash) => (
                        200,
                        "application/json",
                        serde_json::to_string(&dash).unwrap_or_default(),
                    ),
                    Err(e) => (
                        400,
                        "application/json",
                        serde_json::json!({ "error": e }).to_string(),
                    ),
                },
                Err(e) => (
                    400,
                    "application/json",
                    serde_json::json!({ "error": e.to_string() }).to_string(),
                ),
            }
        } else if method == "POST" && path == "/api/v1/chat/send" {
            match serde_json::from_str::<SendChatMessageRequest>(body) {
                Ok(req) => match self.send_chat_message(req).await {
                    Ok((dash, _)) => (
                        200,
                        "application/json",
                        serde_json::to_string(&dash).unwrap_or_default(),
                    ),
                    Err(e) => (
                        400,
                        "application/json",
                        serde_json::json!({ "error": e }).to_string(),
                    ),
                },
                Err(e) => (
                    400,
                    "application/json",
                    serde_json::json!({ "error": e.to_string() }).to_string(),
                ),
            }
        } else if method == "POST" && path == "/api/v1/foods/log" {
            match serde_json::from_str::<LogFoodEatenInput>(body) {
                Ok(req) => match self.log_food_eaten_and_update(req).await {
                    Ok((dash, _)) => (
                        200,
                        "application/json",
                        serde_json::to_string(&dash).unwrap_or_default(),
                    ),
                    Err(e) => (
                        400,
                        "application/json",
                        serde_json::json!({ "error": e }).to_string(),
                    ),
                },
                Err(e) => (
                    400,
                    "application/json",
                    serde_json::json!({ "error": e.to_string() }).to_string(),
                ),
            }
        } else if method == "POST" && path == "/api/v1/exchanges/apply" {
            match serde_json::from_str::<ApplyCalorieExchangeRequest>(body) {
                Ok(req) => match self.apply_daily_calorie_exchange(req) {
                    Ok(dash) => (
                        200,
                        "application/json",
                        serde_json::to_string(&dash).unwrap_or_default(),
                    ),
                    Err(e) => (
                        400,
                        "application/json",
                        serde_json::json!({ "error": e }).to_string(),
                    ),
                },
                Err(e) => (
                    400,
                    "application/json",
                    serde_json::json!({ "error": e.to_string() }).to_string(),
                ),
            }
        } else if method == "GET"
            && path.starts_with("/api/v1/users/")
            && path.ends_with("/dashboard")
        {
            let user_id = path
                .trim_start_matches("/api/v1/users/")
                .trim_end_matches("/dashboard")
                .trim_matches('/');
            match self.get_user_dashboard(user_id) {
                Ok(dash) => (
                    200,
                    "application/json",
                    serde_json::to_string(&dash).unwrap_or_default(),
                ),
                Err(e) => (
                    404,
                    "application/json",
                    serde_json::json!({ "error": e }).to_string(),
                ),
            }
        } else {
            (
                404,
                "application/json",
                r#"{"error":"Endpoint not found"}"#.to_string(),
            )
        }
    }
}

/// White & Red Sports/Fitness Single-Page Application served at `GET /` with:
/// - **Entrance Screen**: Login & Password (`POST /api/v1/users/login`) and Athlete Sign-Up (`POST /api/v1/users`).
/// - **Athlete Home Hub**: Live Daily Macro Scoreboard + two main portals: **AI Nutrition & Meal Chat** and **My Nutrition Plan** (prompts to register a `.PDF` or `.TXT` plan if none is registered yet).
/// - **App Settings (`Configurações / Settings`)**: Full Language Switcher (`Português` / `English`), Athlete Profile, and Per-Node LLM Configuration Inspector.
pub const INTERACTIVE_TEST_UI_HTML: &str = r#"<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>APEX FIT AI — Sports Nutrition & Calorie Engine</title>
  <style>
    :root {
      --bg: #f8fafc;
      --surface: #ffffff;
      --surface-alt: #fff1f2;
      --red-primary: #dc2626;
      --red-dark: #991b1b;
      --red-bright: #ef4444;
      --red-soft: #fee2e2;
      --text-main: #0f172a;
      --text-muted: #64748b;
      --border: #e2e8f0;
      --border-red: #fca5a5;
      --shadow: 0 10px 25px -5px rgba(220, 38, 38, 0.08), 0 4px 10px -2px rgba(15, 23, 42, 0.05);
    }
    * { box-sizing: border-box; font-family: 'Inter', system-ui, -apple-system, sans-serif; }
    body {
      margin: 0;
      padding: 0;
      background: var(--bg);
      color: var(--text-main);
      min-height: 100vh;
    }
    /* Top Sports Navigation Bar */
    .topbar {
      background: linear-gradient(135deg, #991b1b 0%, #dc2626 55%, #ef4444 100%);
      color: #ffffff;
      padding: 14px 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      box-shadow: 0 4px 18px rgba(220, 38, 38, 0.28);
      position: sticky;
      top: 0;
      z-index: 50;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      font-weight: 900;
      font-size: 20px;
      letter-spacing: 0.8px;
      text-transform: uppercase;
      font-style: italic;
    }
    .brand-badge {
      background: #ffffff;
      color: var(--red-primary);
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 900;
      font-style: normal;
      letter-spacing: 1px;
    }
    .nav-tabs {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .nav-btn {
      background: rgba(255, 255, 255, 0.14);
      color: #ffffff;
      border: 1px solid rgba(255, 255, 255, 0.28);
      padding: 8px 16px;
      border-radius: 999px;
      font-weight: 800;
      font-size: 13px;
      cursor: pointer;
      transition: all 0.18s ease;
      text-transform: uppercase;
      letter-spacing: 0.4px;
    }
    .nav-btn:hover, .nav-btn.active {
      background: #ffffff;
      color: var(--red-primary);
      border-color: #ffffff;
      transform: translateY(-1px);
    }
    /* Entrance / Login Screen */
    .entrance-wrapper {
      min-height: calc(100vh - 64px);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 32px 20px;
      background:
        radial-gradient(circle at 15% 20%, rgba(220, 38, 38, 0.10), transparent 40%),
        radial-gradient(circle at 85% 80%, rgba(239, 68, 68, 0.12), transparent 45%),
        var(--bg);
    }
    .entrance-card {
      width: 100%;
      max-width: 480px;
      background: var(--surface);
      border-radius: 20px;
      border-top: 6px solid var(--red-primary);
      box-shadow: 0 20px 45px rgba(220, 38, 38, 0.14), 0 4px 12px rgba(15, 23, 42, 0.06);
      padding: 32px;
    }
    .sport-tag {
      display: inline-block;
      background: var(--red-soft);
      color: var(--red-primary);
      font-weight: 800;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      padding: 5px 12px;
      border-radius: 999px;
      margin-bottom: 12px;
    }
    .auth-toggle {
      display: grid;
      grid-template-columns: 1fr 1fr;
      background: #f1f5f9;
      padding: 4px;
      border-radius: 10px;
      margin: 18px 0 22px 0;
    }
    .auth-toggle button {
      background: transparent;
      color: var(--text-muted);
      border: none;
      padding: 10px;
      border-radius: 8px;
      font-weight: 800;
      font-size: 13px;
      cursor: pointer;
      text-transform: uppercase;
    }
    .auth-toggle button.active {
      background: var(--red-primary);
      color: #ffffff;
      box-shadow: 0 2px 8px rgba(220, 38, 38, 0.25);
    }
    /* Common Layout & Cards */
    .app-container {
      max-width: 1220px;
      margin: 0 auto;
      padding: 28px 20px 60px 20px;
    }
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 24px;
      box-shadow: var(--shadow);
      margin-bottom: 22px;
    }
    .card.red-accent {
      border-left: 6px solid var(--red-primary);
    }
    h1, h2, h3 { margin-top: 0; color: var(--text-main); }
    .section-title {
      font-size: 18px;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 16px;
    }
    label {
      display: block;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-muted);
      margin-bottom: 6px;
    }
    input, textarea, select {
      width: 100%;
      padding: 12px 14px;
      border-radius: 10px;
      border: 1.5px solid var(--border);
      background: #ffffff;
      color: var(--text-main);
      font-size: 14px;
      transition: border-color 0.15s;
      margin-bottom: 14px;
    }
    input:focus, textarea:focus, select:focus {
      outline: none;
      border-color: var(--red-primary);
      box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.12);
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }
    .btn-primary {
      background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
      color: #ffffff;
      font-weight: 800;
      border: none;
      padding: 12px 20px;
      border-radius: 10px;
      cursor: pointer;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      box-shadow: 0 4px 12px rgba(220, 38, 38, 0.25);
      transition: transform 0.15s, box-shadow 0.15s;
    }
    .btn-primary:hover {
      transform: translateY(-1px);
      box-shadow: 0 6px 16px rgba(220, 38, 38, 0.35);
    }
    .btn-outline {
      background: #ffffff;
      color: var(--red-primary);
      border: 2px solid var(--red-primary);
      font-weight: 800;
      padding: 10px 18px;
      border-radius: 10px;
      cursor: pointer;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .btn-outline:hover {
      background: var(--red-soft);
    }
    .chip {
      background: var(--surface-alt);
      color: var(--red-dark);
      border: 1px solid var(--border-red);
      padding: 7px 14px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
      margin-right: 6px;
      margin-bottom: 6px;
    }
    .chip:hover {
      background: var(--red-primary);
      color: #ffffff;
    }
    /* Scoreboard & Macro Bars */
    .scoreboard {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-bottom: 24px;
    }
    .score-card {
      background: var(--surface);
      border: 1.5px solid var(--border);
      border-top: 4px solid var(--red-primary);
      border-radius: 14px;
      padding: 18px;
      text-align: center;
      box-shadow: var(--shadow);
    }
    .score-val {
      font-size: 28px;
      font-weight: 900;
      color: var(--red-primary);
      font-style: italic;
    }
    .score-lbl {
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      color: var(--text-muted);
      letter-spacing: 0.6px;
      margin-top: 4px;
    }
    /* Home Action Portals */
    .portal-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 22px;
      margin-bottom: 24px;
    }
    .portal-card {
      background: var(--surface);
      border: 2px solid var(--border);
      border-radius: 18px;
      padding: 28px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      transition: all 0.2s ease;
      cursor: pointer;
      position: relative;
      overflow: hidden;
    }
    .portal-card:hover {
      border-color: var(--red-primary);
      transform: translateY(-3px);
      box-shadow: 0 16px 32px rgba(220, 38, 38, 0.14);
    }
    .portal-icon {
      width: 52px;
      height: 52px;
      border-radius: 14px;
      background: var(--red-soft);
      color: var(--red-primary);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 26px;
      margin-bottom: 16px;
    }
    .status-pill {
      display: inline-block;
      padding: 4px 12px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.6px;
    }
    .status-pill.active {
      background: #dcfce7;
      color: #15803d;
    }
    .status-pill.missing {
      background: var(--red-soft);
      color: var(--red-primary);
    }
    /* Chat Interface */
    .chat-box {
      background: #f8fafc;
      border: 1.5px solid var(--border);
      border-radius: 14px;
      height: 380px;
      overflow-y: auto;
      padding: 18px;
      margin-bottom: 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .msg {
      padding: 12px 16px;
      border-radius: 14px;
      max-width: 82%;
      font-size: 14px;
      line-height: 1.5;
      white-space: pre-wrap;
    }
    .msg.user {
      align-self: flex-end;
      background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
      color: #ffffff;
      border-bottom-right-radius: 4px;
    }
    .msg.assistant {
      align-self: flex-start;
      background: #ffffff;
      border: 1.5px solid var(--border);
      border-left: 4px solid var(--red-primary);
      color: var(--text-main);
      border-bottom-left-radius: 4px;
    }
    .msg-meta {
      font-size: 11px;
      opacity: 0.78;
      margin-top: 6px;
      font-weight: 600;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13.5px;
    }
    th, td {
      border-bottom: 1px solid var(--border);
      padding: 10px 8px;
      text-align: left;
    }
    th {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      color: var(--text-muted);
      font-weight: 800;
    }
    .hidden { display: none !important; }
    @media (max-width: 820px) {
      .portal-grid, .scoreboard, .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <!-- TOP NAVIGATION BAR -->
  <header class="topbar">
    <div class="brand">
      <span>⚡ APEX FIT AI</span>
      <span class="brand-badge" id="nav_subtitle">NUTRITION & PERFORMANCE</span>
    </div>
    <div class="nav-tabs" id="main_nav" style="display:none;">
      <button class="nav-btn active" id="tab_home" onclick="switchView('home')">🏠 Início</button>
      <button class="nav-btn" id="tab_chat" onclick="switchView('chat')">💬 Chat IA</button>
      <button class="nav-btn" id="tab_plan" onclick="switchView('plan')">📋 Plano Alimentar</button>
      <button class="nav-btn" id="tab_settings" onclick="switchView('settings')">⚙️ Configurações</button>
    </div>
    <div id="entrance_lang_switch">
      <button class="nav-btn" onclick="toggleLanguage()" id="quick_lang_btn">🌐 PT / EN</button>
    </div>
  </header>

  <!-- VIEW 1: ENTRANCE (LOGIN & SIGN UP) -->
  <section id="view_entrance" class="entrance-wrapper">
    <div class="entrance-card">
      <span class="sport-tag" id="ent_badge">PERFORMANCE NUTRITION APP</span>
      <h1 id="ent_title" style="font-size: 26px; font-weight: 900; margin-bottom: 6px;">Bem-vindo ao Apex Fit AI</h1>
      <p id="ent_desc" style="color: var(--text-muted); font-size: 14px; margin-bottom: 0;">
        Acesse sua conta para conversar com seu Coach IA ou consultar seu Plano Alimentar.
      </p>

      <div class="auth-toggle">
        <button id="btn_mode_login" class="active" onclick="setAuthMode('login')">Entrar (Login)</button>
        <button id="btn_mode_signup" onclick="setAuthMode('signup')">Criar Conta</button>
      </div>

      <!-- LOGIN FORM -->
      <div id="form_login">
        <label id="lbl_login_user">E-mail ou CPF</label>
        <input id="login_id" value="gabriel.silva@calorie.ai" placeholder="gabriel.silva@calorie.ai ou CPF" />
        <label id="lbl_login_pass">Senha</label>
        <input id="login_pass" type="password" value="NutriPlan#2026" placeholder="••••••••" onkeydown="if(event.key==='Enter') handleLogin()" />
        <button class="btn-primary" style="width: 100%; margin-top: 6px;" onclick="handleLogin()" id="btn_submit_login">
          🚀 Entrar no App
        </button>
      </div>

      <!-- SIGN UP FORM -->
      <div id="form_signup" class="hidden">
        <div class="row">
          <div><label id="lbl_su_name">Nome Completo</label><input id="u_name" value="Gabriel Silva" /></div>
          <div><label id="lbl_su_cpf">CPF</label><input id="u_cpf" value="123.456.789-01" /></div>
        </div>
        <div class="row">
          <div><label id="lbl_su_email">E-mail</label><input id="u_email" value="gabriel.silva@calorie.ai" /></div>
          <div><label id="lbl_su_phone">Telefone</label><input id="u_phone" value="+55 11 99876-5432" /></div>
        </div>
        <label id="lbl_su_addr">Endereço</label>
        <input id="u_address" value="Av. Paulista, 1000 - São Paulo, SP" />
        <label id="lbl_su_pass">Senha</label>
        <input id="u_pass" type="password" value="NutriPlan#2026" />
        <button class="btn-primary" style="width: 100%; margin-top: 6px;" onclick="handleRegisterUser()" id="btn_submit_signup">
          🔥 Criar Conta de Atleta
        </button>
      </div>

      <div id="auth_error" style="color: var(--red-primary); font-size: 13px; font-weight: 700; margin-top: 12px; text-align: center;"></div>
    </div>
  </section>

  <!-- MAIN AUTHENTICATED APP CONTAINER -->
  <main id="app_main" class="app-container hidden">

    <!-- VIEW 2: ATHLETE HOME HUB -->
    <section id="view_home">
      <div class="card red-accent" style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:14px;">
        <div>
          <span class="sport-tag" id="home_welcome_tag">PAINEL DO ATLETA</span>
          <h2 id="home_greeting" style="margin:0; font-size:24px; font-weight:900;">Olá, Atleta!</h2>
          <p id="home_sub" style="margin:4px 0 0 0; color:var(--text-muted); font-size:14px;">
            Acompanhe suas calorias do dia, registre refeições no Chat IA ou consulte seu Plano Alimentar.
          </p>
        </div>
        <div style="display:flex; gap:10px;">
          <button class="btn-primary" onclick="switchView('chat')" id="home_quick_chat_btn">💬 Ir para o Chat IA</button>
          <button class="btn-outline" onclick="switchView('plan')" id="home_quick_plan_btn">📋 Meu Plano Alimentar</button>
        </div>
      </div>

      <!-- SCOREBOARD -->
      <div class="scoreboard">
        <div class="score-card">
          <div class="score-val" id="st_eaten">0</div>
          <div class="score-lbl" id="lbl_st_eaten">Calorias Consumidas (kcal)</div>
        </div>
        <div class="score-card">
          <div class="score-val" id="st_target">2200</div>
          <div class="score-lbl" id="lbl_st_target">Meta Diária (kcal)</div>
        </div>
        <div class="score-card">
          <div class="score-val" id="st_rem">2200</div>
          <div class="score-lbl" id="lbl_st_rem">Calorias Restantes (kcal)</div>
        </div>
        <div class="score-card">
          <div class="score-val" id="st_delta">0.0</div>
          <div class="score-lbl" id="lbl_st_delta">Saldo de Substituições (kcal)</div>
        </div>
      </div>

      <!-- TWO PRIMARY HOME OPTIONS: 1) AI CHAT  2) NUTRITION PLAN -->
      <div class="portal-grid">
        <!-- OPTION 1: CHAT -->
        <div class="portal-card" onclick="switchView('chat')">
          <div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <div class="portal-icon">💬</div>
              <span class="status-pill active" id="chat_portal_badge">IA ATIVA • SMALL LLM JSON</span>
            </div>
            <h3 id="portal_chat_title" style="font-size:20px; font-weight:900; margin-bottom:8px;">
              Chat Nutricional & Registro de Refeições
            </h3>
            <p id="portal_chat_desc" style="color:var(--text-muted); font-size:14px; line-height:1.5;">
              Registre o que comeu em linguagem natural (ex: "Comi 3 ovos cozidos com 2 pães de forma") ou tire dúvidas com seu Coach IA. Tolera erros de digitação e converte medidas caseiras automaticamente.
            </p>
          </div>
          <div style="margin-top: 16px;">
            <span style="color:var(--red-primary); font-weight:800; font-size:13px; text-transform:uppercase;" id="portal_chat_cta">
              Abrir Chat Nutricional →
            </span>
          </div>
        </div>

        <!-- OPTION 2: NUTRITION PLAN -->
        <div class="portal-card" onclick="switchView('plan')">
          <div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <div class="portal-icon">📋</div>
              <span class="status-pill missing" id="plan_portal_badge">NENHUM PLANO REGISTRADO</span>
            </div>
            <h3 id="portal_plan_title" style="font-size:20px; font-weight:900; margin-bottom:8px;">
              Meu Plano Alimentar (.PDF / .TXT)
            </h3>
            <p id="portal_plan_desc" style="color:var(--text-muted); font-size:14px; line-height:1.5;">
              Nenhum plano alimentar cadastrado ainda. Clique aqui para enviar o PDF ou TXT do seu nutricionista e liberar suas refeições diárias e lista de substituições!
            </p>
          </div>
          <div style="margin-top: 16px;">
            <span style="color:var(--red-primary); font-weight:800; font-size:13px; text-transform:uppercase;" id="portal_plan_cta">
              Registrar / Ver Plano Alimentar →
            </span>
          </div>
        </div>
      </div>

      <!-- TODAY'S LOGGED FOODS ON HOME -->
      <div class="card">
        <div class="section-title" id="home_eaten_title">🔥 Refeições Registradas Hoje</div>
        <table>
          <thead>
            <tr>
              <th id="th_meal">Refeição</th>
              <th id="th_food">Alimento</th>
              <th id="th_grams">Gramas</th>
              <th id="th_kcal">Calorias</th>
              <th id="th_macros">Macros (P / C / G)</th>
            </tr>
          </thead>
          <tbody id="eaten_tbody">
            <tr><td colspan="5">Nenhuma refeição registrada hoje.</td></tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- VIEW 3: AI CHAT -->
    <section id="view_chat" class="hidden">
      <div class="card red-accent">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; flex-wrap:wrap; gap:8px;">
          <div class="section-title" style="margin:0;" id="chat_view_title">💬 Coach IA & Registro Inteligente de Refeições</div>
          <span class="sport-tag" style="margin:0;">MealParser: qwen2.5-1.5b-instruct-json</span>
        </div>
        <div style="margin-bottom: 12px;">
          <button class="chip" onclick="sendQuickChat('Comi 3 ovos cozidos com 2 pães de forma')" id="chip_1">🥚 "Comi 3 ovos cozidos com 2 pães de forma"</button>
          <button class="chip" onclick="sendQuickChat('Comi 3 ovio cozidos e 2 ovoos mexidos com 1 colhr de sopa de azeite')" id="chip_2">⚡ Testar Typos ("3 ovio cozidos e 2 ovoos")</button>
          <button class="chip" onclick="sendQuickChat('Quais são as refeições e calorias do meu plano alimentar?')" id="chip_3">📋 Perguntar sobre meu Plano</button>
        </div>
        <div class="chat-box" id="chat_box"></div>
        <div style="display: grid; grid-template-columns: 1fr auto; gap: 10px;">
          <input id="chat_input" style="margin:0;" placeholder="Digite sua refeição ou dúvida nutricional..." onkeydown="if(event.key==='Enter') sendChatMessage()" />
          <button class="btn-primary" onclick="sendChatMessage()" id="btn_send_chat">Enviar</button>
        </div>
      </div>
    </section>

    <!-- VIEW 4: NUTRITION PLAN (REGISTER IF MISSING, OR VIEW DETAILS) -->
    <section id="view_plan" class="hidden">
      <!-- ONBOARDING / UPLOAD CARD (Prominent when no plan is registered, or used to update plan) -->
      <div class="card red-accent" id="plan_upload_card">
        <span class="sport-tag" id="plan_status_tag">CADASTRO DE PLANO NUTRICIONAL</span>
        <h2 id="plan_upload_title" style="margin-bottom:6px;">Registre seu Plano Alimentar (.PDF ou .TXT)</h2>
        <p id="plan_upload_desc" style="color:var(--text-muted); font-size:14px; margin-bottom:18px;">
          Você ainda não possui um plano registrado. Envie o arquivo PDF ou TXT elaborado pelo seu nutricionista para extrair suas metas, refeições diárias e opções de substituição.
        </p>
        <div class="row">
          <div>
            <label id="lbl_select_file">Selecionar Arquivo (.PDF ou .TXT)</label>
            <input type="file" id="plan_file" accept=".pdf,.txt" onchange="loadSelectedFile(event)" />
          </div>
          <div style="display:flex; align-items:center;">
            <button class="btn-outline" style="width:100%;" onclick="loadWorkspaceSamplePdf()" id="btn_sample_pdf">
              📄 Carregar "Plano Alimentar Teste.pdf"
            </button>
          </div>
        </div>
        <label id="lbl_plan_preview">Texto Extraído / Editor do Plano Alimentar</label>
        <textarea id="plan_text" rows="5" placeholder="O texto extraído do seu PDF ou TXT aparecerá aqui..."></textarea>
        <div style="display:flex; gap:12px; align-items:center;">
          <button class="btn-primary" onclick="uploadNutritionPlan()" id="btn_save_plan">
            🔥 Salvar Plano Alimentar na Memória & Banco
          </button>
          <span id="plan_status" style="font-size:13px; font-weight:700; color:var(--red-primary);"></span>
        </div>
      </div>

      <!-- REGISTERED PLAN DETAILS (Shown once active_plan exists) -->
      <div id="plan_registered_content" class="hidden">
        <div class="row">
          <div class="card">
            <div class="section-title" id="plan_foods_title">🥗 Alimentos Prováveis do Dia (Plano Ativo)</div>
            <table>
              <thead>
                <tr><th>Refeição</th><th>Alimento</th><th>Gramas</th><th>Kcal</th><th>Ação</th></tr>
              </thead>
              <tbody id="probable_foods_tbody"></tbody>
            </table>
          </div>

          <div class="card">
            <div class="section-title" id="plan_exchanges_title">🔄 Lista de Substituições Calóricas</div>
            <table>
              <thead>
                <tr><th>Refeição</th><th>Original</th><th>Substituto</th><th>Delta</th><th>Ação</th></tr>
              </thead>
              <tbody id="exchanges_tbody"></tbody>
            </table>
          </div>
        </div>
      </div>
    </section>

    <!-- VIEW 5: SETTINGS (LANGUAGE SWITCHER, PROFILE, NODE LLM CONFIG) -->
    <section id="view_settings" class="hidden">
      <div class="card red-accent">
        <div class="section-title" id="settings_lang_title">🌐 Idioma do Aplicativo / App Language</div>
        <p id="settings_lang_desc" style="color:var(--text-muted); font-size:14px;">
          Escolha o idioma da interface esportiva (Português ou Inglês):
        </p>
        <div style="display:flex; gap:12px; margin-bottom:8px;">
          <button class="btn-primary" id="btn_lang_pt" onclick="setLanguage('pt')">🇧🇷 Português (Brasil)</button>
          <button class="btn-outline" id="btn_lang_en" onclick="setLanguage('en')">🇺🇸 English (US)</button>
        </div>
      </div>

      <div class="row">
        <div class="card">
          <div class="section-title" id="settings_profile_title">🏃 Perfil do Atleta</div>
          <div id="settings_profile_box" style="font-size:14px; line-height:1.8;"></div>
          <button class="btn-outline" style="margin-top:16px;" onclick="handleLogout()" id="btn_logout">
            🚪 Sair da Conta (Logout)
          </button>
        </div>

        <div class="card">
          <div class="section-title" id="settings_llm_title">🤖 Configuração de LLMs por Nó (`node_llm_config.json`)</div>
          <table>
            <thead>
              <tr><th>Nó (Node)</th><th>Papel</th><th>Modelo Configurado</th><th>Formato</th></tr>
            </thead>
            <tbody id="llm_config_tbody"></tbody>
          </table>
        </div>
      </div>
    </section>

  </main>

  <script>
    let currentUserId = null;
    let currentDashboard = null;
    let currentLang = localStorage.getItem('apex_fit_lang') || 'pt';
    let uploadedFilename = 'plano_nutricional.txt';
    let uploadedFileHex = null;

    const I18N = {
      pt: {
        tab_home: '🏠 Início',
        tab_chat: '💬 Chat IA',
        tab_plan: '📋 Plano Alimentar',
        tab_settings: '⚙️ Configurações',
        ent_badge: 'PERFORMANCE NUTRITION APP',
        ent_title: 'Bem-vindo ao Apex Fit AI',
        ent_desc: 'Informe seu login e senha para acessar o painel esportivo, conversar com o Coach IA ou ver seu Plano Alimentar.',
        btn_mode_login: 'Entrar (Login)',
        btn_mode_signup: 'Criar Conta',
        lbl_login_user: 'E-mail ou CPF',
        lbl_login_pass: 'Senha',
        btn_submit_login: '🚀 Entrar no App',
        lbl_su_name: 'Nome Completo',
        lbl_su_cpf: 'CPF',
        lbl_su_email: 'E-mail',
        lbl_su_phone: 'Telefone',
        lbl_su_addr: 'Endereço',
        lbl_su_pass: 'Senha',
        btn_submit_signup: '🔥 Criar Conta de Atleta',
        home_welcome_tag: 'PAINEL DO ATLETA',
        home_greeting: 'Olá',
        home_sub: 'Acompanhe suas calorias do dia, registre refeições no Chat IA ou consulte seu Plano Alimentar.',
        home_quick_chat_btn: '💬 Abrir Chat IA',
        home_quick_plan_btn: '📋 Meu Plano Alimentar',
        lbl_st_eaten: 'Calorias Consumidas (kcal)',
        lbl_st_target: 'Meta Diária (kcal)',
        lbl_st_rem: 'Calorias Restantes (kcal)',
        lbl_st_delta: 'Saldo de Substituições (kcal)',
        portal_chat_title: 'Chat Nutricional & Registro de Refeições',
        portal_chat_desc: 'Registre o que comeu em linguagem natural (ex: "Comi 3 ovos cozidos com 2 pães de forma") ou tire dúvidas com seu Coach IA. Corrige erros de digitação (ex: "ovio", "ovoos") via Small LLM JSON.',
        portal_chat_cta: 'Abrir Chat Nutricional →',
        portal_plan_title: 'Meu Plano Alimentar (.PDF / .TXT)',
        portal_plan_missing_badge: 'NENHUM PLANO — REGISTRE AGORA',
        portal_plan_active_badge: 'PLANO ATIVO REGISTRADO',
        portal_plan_missing_desc: 'Nenhum plano alimentar registrado ainda! Clique aqui para enviar seu PDF ou TXT e registrar seu plano.',
        portal_plan_cta_missing: 'Registrar Plano Alimentar Agora →',
        portal_plan_cta_active: 'Ver Meu Plano Alimentar →',
        home_eaten_title: '🔥 Refeições Registradas Hoje',
        chat_view_title: '💬 Coach IA & Registro Inteligente de Refeições',
        btn_send_chat: 'Enviar',
        plan_status_tag_missing: 'NENHUM PLANO CADASTRADO • AÇÃO NECESSÁRIA',
        plan_status_tag_active: 'PLANO ALIMENTAR ATIVO',
        plan_upload_title_missing: 'Registre seu Plano Alimentar (.PDF ou .TXT)',
        plan_upload_title_active: 'Atualizar / Substituir Plano Alimentar (.PDF ou .TXT)',
        plan_upload_desc: 'Envie o arquivo PDF ou TXT do seu nutricionista ou carregue o PDF de exemplo da workspace.',
        lbl_select_file: 'Selecionar Arquivo (.PDF ou .TXT)',
        btn_sample_pdf: '📄 Carregar "Plano Alimentar Teste.pdf"',
        lbl_plan_preview: 'Texto Extraído / Editor do Plano Alimentar',
        btn_save_plan: '🔥 Salvar Plano Alimentar',
        plan_foods_title: '🥗 Alimentos do Plano Para Hoje',
        plan_exchanges_title: '🔄 Lista de Substituições Calóricas',
        settings_lang_title: '🌐 Idioma do Aplicativo',
        settings_lang_desc: 'Altere o idioma de toda a plataforma entre Português e Inglês:',
        settings_profile_title: '🏃 Perfil do Atleta',
        settings_llm_title: '🤖 Configuração de LLMs por Nó (node_llm_config.json)',
        btn_logout: '🚪 Sair da Conta (Logout)',
        eat_btn: '+ Comer Hoje',
        swap_btn: 'Trocar & Registrar'
      },
      en: {
        tab_home: '🏠 Home',
        tab_chat: '💬 AI Chat',
        tab_plan: '📋 Nutrition Plan',
        tab_settings: '⚙️ Settings',
        ent_badge: 'PERFORMANCE NUTRITION APP',
        ent_title: 'Welcome to Apex Fit AI',
        ent_desc: 'Enter your login and password to access your athlete home, chat with your AI Coach, or view your registered Nutrition Plan.',
        btn_mode_login: 'Sign In (Login)',
        btn_mode_signup: 'Create Account',
        lbl_login_user: 'Email or CPF',
        lbl_login_pass: 'Password',
        btn_submit_login: '🚀 Enter App',
        lbl_su_name: 'Full Name',
        lbl_su_cpf: 'CPF (Tax ID)',
        lbl_su_email: 'Email',
        lbl_su_phone: 'Phone',
        lbl_su_addr: 'Address',
        lbl_su_pass: 'Password',
        btn_submit_signup: '🔥 Create Athlete Account',
        home_welcome_tag: 'ATHLETE PERFORMANCE HUB',
        home_greeting: 'Hello',
        home_sub: 'Track your daily calories, log meals via the AI Chat, or inspect your registered Nutrition Plan.',
        home_quick_chat_btn: '💬 Open AI Chat',
        home_quick_plan_btn: '📋 My Nutrition Plan',
        lbl_st_eaten: 'Calories Eaten (kcal)',
        lbl_st_target: 'Daily Target (kcal)',
        lbl_st_rem: 'Remaining Calories (kcal)',
        lbl_st_delta: 'Exchange Delta (kcal)',
        portal_chat_title: 'AI Nutrition & Meal Logger Chat',
        portal_chat_desc: 'Log meals in natural language (e.g. "I ate 3 boiled eggs with 2 slices of bread") or ask coaching questions. Automatically fixes typos ("ovio", "ovoos") via Small LLM JSON.',
        portal_chat_cta: 'Open AI Nutrition Chat →',
        portal_plan_title: 'My Nutrition Plan (.PDF / .TXT)',
        portal_plan_missing_badge: 'NO PLAN — REGISTER NOW',
        portal_plan_active_badge: 'ACTIVE PLAN REGISTERED',
        portal_plan_missing_desc: 'No nutrition plan registered yet! Click here to upload your nutritionist PDF or TXT file and register your plan.',
        portal_plan_cta_missing: 'Register Nutrition Plan Now →',
        portal_plan_cta_active: 'View Registered Nutrition Plan →',
        home_eaten_title: '🔥 Foods Eaten Today',
        chat_view_title: '💬 AI Coach & Smart Meal Logger',
        btn_send_chat: 'Send',
        plan_status_tag_missing: 'NO PLAN REGISTERED • ACTION REQUIRED',
        plan_status_tag_active: 'REGISTERED NUTRITION PLAN',
        plan_upload_title_missing: 'Register Your Nutrition Plan (.PDF or .TXT)',
        plan_upload_title_active: 'Update / Replace Nutrition Plan (.PDF or .TXT)',
        plan_upload_desc: 'Upload your nutritionist PDF or TXT file, or load the workspace sample PDF.',
        lbl_select_file: 'Select File (.PDF or .TXT)',
        btn_sample_pdf: '📄 Load "Plano Alimentar Teste.pdf"',
        lbl_plan_preview: 'Extracted Text / Nutrition Plan Editor',
        btn_save_plan: '🔥 Save Nutrition Plan',
        plan_foods_title: '🥗 Probable Daily Foods (Active Plan)',
        plan_exchanges_title: '🔄 Daily Calorie Exchanges',
        settings_lang_title: '🌐 App Language / Idioma',
        settings_lang_desc: 'Switch the application language between Portuguese and English:',
        settings_profile_title: '🏃 Athlete Profile',
        settings_llm_title: '🤖 Per-Node LLM Configuration (node_llm_config.json)',
        btn_logout: '🚪 Sign Out (Logout)',
        eat_btn: '+ Eat Today',
        swap_btn: 'Swap & Log'
      }
    };

    function t(key) {
      return (I18N[currentLang] && I18N[currentLang][key]) || I18N.pt[key] || key;
    }

    function setLanguage(lang) {
      currentLang = lang;
      localStorage.setItem('apex_fit_lang', lang);
      document.getElementById('btn_lang_pt').className = lang === 'pt' ? 'btn-primary' : 'btn-outline';
      document.getElementById('btn_lang_en').className = lang === 'en' ? 'btn-primary' : 'btn-outline';
      applyTranslations();
      if (currentDashboard) renderDashboard(currentDashboard);
    }

    function toggleLanguage() {
      setLanguage(currentLang === 'pt' ? 'en' : 'pt');
    }

    function applyTranslations() {
      const ids = [
        'tab_home', 'tab_chat', 'tab_plan', 'tab_settings',
        'ent_badge', 'ent_title', 'ent_desc',
        'btn_mode_login', 'btn_mode_signup',
        'lbl_login_user', 'lbl_login_pass', 'btn_submit_login',
        'lbl_su_name', 'lbl_su_cpf', 'lbl_su_email', 'lbl_su_phone', 'lbl_su_addr', 'lbl_su_pass', 'btn_submit_signup',
        'home_welcome_tag', 'home_sub', 'home_quick_chat_btn', 'home_quick_plan_btn',
        'lbl_st_eaten', 'lbl_st_target', 'lbl_st_rem', 'lbl_st_delta',
        'portal_chat_title', 'portal_chat_desc', 'portal_chat_cta', 'portal_plan_title',
        'home_eaten_title', 'chat_view_title', 'btn_send_chat',
        'plan_upload_desc', 'lbl_select_file', 'btn_sample_pdf', 'lbl_plan_preview', 'btn_save_plan',
        'plan_foods_title', 'plan_exchanges_title',
        'settings_lang_title', 'settings_lang_desc', 'settings_profile_title', 'settings_llm_title', 'btn_logout'
      ];
      ids.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.textContent = t(id);
      });
      document.getElementById('quick_lang_btn').textContent = currentLang === 'pt' ? '🇧🇷 PT | 🇺🇸 EN' : '🇺🇸 EN | 🇧🇷 PT';
    }

    function setAuthMode(mode) {
      document.getElementById('auth_error').textContent = '';
      if (mode === 'login') {
        document.getElementById('btn_mode_login').classList.add('active');
        document.getElementById('btn_mode_signup').classList.remove('active');
        document.getElementById('form_login').classList.remove('hidden');
        document.getElementById('form_signup').classList.add('hidden');
      } else {
        document.getElementById('btn_mode_signup').classList.add('active');
        document.getElementById('btn_mode_login').classList.remove('active');
        document.getElementById('form_signup').classList.remove('hidden');
        document.getElementById('form_login').classList.add('hidden');
      }
    }

    function switchView(view) {
      ['home', 'chat', 'plan', 'settings'].forEach(v => {
        document.getElementById('view_' + v).classList.toggle('hidden', v !== view);
        document.getElementById('tab_' + v).classList.toggle('active', v === view);
      });
      if (view === 'settings') loadLlmConfigTable();
    }

    async function handleLogin() {
      document.getElementById('auth_error').textContent = '';
      const res = await fetch('/api/v1/users/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          login: document.getElementById('login_id').value,
          password: document.getElementById('login_pass').value
        })
      });
      const data = await res.json();
      if (data.error) {
        document.getElementById('auth_error').textContent = data.error;
        return;
      }
      enterAuthenticatedApp(data);
    }

    async function handleRegisterUser() {
      document.getElementById('auth_error').textContent = '';
      const res = await fetch('/api/v1/users', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          full_name: document.getElementById('u_name').value,
          cpf: document.getElementById('u_cpf').value,
          email: document.getElementById('u_email').value,
          phone: document.getElementById('u_phone').value,
          address: document.getElementById('u_address').value,
          password: document.getElementById('u_pass').value,
        })
      });
      const data = await res.json();
      if (data.error) {
        document.getElementById('auth_error').textContent = data.error;
        return;
      }
      enterAuthenticatedApp(data);
    }

    function enterAuthenticatedApp(data) {
      currentUserId = data.user.id;
      document.getElementById('view_entrance').classList.add('hidden');
      document.getElementById('app_main').classList.remove('hidden');
      document.getElementById('main_nav').style.display = 'flex';
      renderDashboard(data);
      switchView('home');
    }

    function handleLogout() {
      currentUserId = null;
      currentDashboard = null;
      document.getElementById('app_main').classList.add('hidden');
      document.getElementById('main_nav').style.display = 'none';
      document.getElementById('view_entrance').classList.remove('hidden');
    }

    function bytesToHex(buffer) {
      const byteArray = new Uint8Array(buffer);
      const hexParts = [];
      for (let i = 0; i < byteArray.length; i++) {
        hexParts.push(byteArray[i].toString(16).padStart(2, '0'));
      }
      return hexParts.join('');
    }

    function loadSelectedFile(event) {
      const file = event.target.files[0];
      if (!file) return;
      uploadedFilename = file.name;
      const reader = new FileReader();
      reader.onload = async (e) => {
        uploadedFileHex = bytesToHex(e.target.result);
        document.getElementById('plan_status').textContent = `Arquivo carregado: ${file.name} (${file.size} bytes)`;
        await uploadNutritionPlan();
      };
      reader.readAsArrayBuffer(file);
    }

    async function loadWorkspaceSamplePdf() {
      const res = await fetch('/api/v1/plans/load-sample-pdf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: currentUserId })
      });
      const data = await res.json();
      if (data.error) { alert(data.error); return; }
      if (data.active_plan) {
        document.getElementById('plan_text').value = data.active_plan.raw_extracted_text;
        document.getElementById('plan_status').textContent =
          `✓ "${data.active_plan.title}" (${data.probable_daily_foods.length} itens)`;
      }
      renderDashboard(data);
    }

    async function uploadNutritionPlan() {
      const payload = {
        user_id: currentUserId,
        filename: uploadedFilename,
        file_bytes_hex: uploadedFileHex,
        content_text: document.getElementById('plan_text').value,
        as_pdf_binary: uploadedFilename.toLowerCase().endsWith('.pdf')
      };
      const res = await fetch('/api/v1/plans/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      uploadedFileHex = null;
      if (data.error) { alert(data.error); return; }
      if (data.active_plan) {
        document.getElementById('plan_text').value = data.active_plan.raw_extracted_text;
        document.getElementById('plan_status').textContent =
          `✓ "${data.active_plan.title}" (${data.probable_daily_foods.length} itens)`;
      }
      renderDashboard(data);
    }

    async function sendQuickChat(text) {
      document.getElementById('chat_input').value = text;
      await sendChatMessage();
    }

    async function sendChatMessage() {
      const inputEl = document.getElementById('chat_input');
      const message = inputEl.value.trim();
      if (!message) return;
      inputEl.value = '';

      const res = await fetch('/api/v1/chat/send', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: currentUserId, message })
      });
      const data = await res.json();
      if (data.error) { alert(data.error); return; }
      renderDashboard(data);
    }

    async function logProbableFood(mealName, foodName, grams) {
      const res = await fetch('/api/v1/foods/log', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: currentUserId,
          meal_name: mealName,
          food_name: foodName,
          grams: grams
        })
      });
      const data = await res.json();
      if (!data.error) renderDashboard(data);
    }

    async function applyExchange(exchangeId) {
      const res = await fetch('/api/v1/exchanges/apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: currentUserId, exchange_id: exchangeId })
      });
      const data = await res.json();
      if (!data.error) renderDashboard(data);
    }

    async function loadLlmConfigTable() {
      const res = await fetch('/api/v1/config/llms');
      const configs = await res.json();
      const tbody = document.getElementById('llm_config_tbody');
      if (Array.isArray(configs)) {
        tbody.innerHTML = configs.map(c => `
          <tr>
            <td><strong>${c.node_id}</strong></td>
            <td>${c.hierarchy_role}</td>
            <td><span class="status-pill missing">${c.model_id}</span></td>
            <td>${c.response_format}</td>
          </tr>
        `).join('');
      }
    }

    function renderDashboard(d) {
      currentDashboard = d;
      const athleteName = d.user.full_name || d.user.email;
      document.getElementById('home_greeting').textContent = `${t('home_greeting')}, ${athleteName}!`;

      document.getElementById('st_eaten').textContent = d.daily_totals.total_calories_kcal.toFixed(1);
      document.getElementById('st_target').textContent = d.daily_totals.target_calories_kcal.toFixed(0);
      document.getElementById('st_rem').textContent = d.daily_totals.remaining_calories_kcal.toFixed(1);
      const delta = d.daily_totals.calorie_exchange_daily_delta_kcal;
      document.getElementById('st_delta').textContent = (delta >= 0 ? '+' : '') + delta.toFixed(1);

      // Update Nutrition Plan Status on Home Portal & Plan View
      const planBadge = document.getElementById('plan_portal_badge');
      const planDesc = document.getElementById('portal_plan_desc');
      const planCta = document.getElementById('portal_plan_cta');
      const planRegContent = document.getElementById('plan_registered_content');
      const planStatusTag = document.getElementById('plan_status_tag');
      const planUploadTitle = document.getElementById('plan_upload_title');

      if (d.active_plan) {
        planBadge.className = 'status-pill active';
        planBadge.textContent = t('portal_plan_active_badge');
        planDesc.textContent = `${d.active_plan.title} — ${d.daily_totals.target_calories_kcal.toFixed(0)} kcal/dia (${d.probable_daily_foods.length} alimentos, ${d.daily_calorie_exchanges.length} substituições).`;
        planCta.textContent = t('portal_plan_cta_active');
        planRegContent.classList.remove('hidden');
        planStatusTag.textContent = t('plan_status_tag_active');
        planUploadTitle.textContent = t('plan_upload_title_active');
      } else {
        planBadge.className = 'status-pill missing';
        planBadge.textContent = t('portal_plan_missing_badge');
        planDesc.textContent = t('portal_plan_missing_desc');
        planCta.textContent = t('portal_plan_cta_missing');
        planRegContent.classList.add('hidden');
        planStatusTag.textContent = t('plan_status_tag_missing');
        planUploadTitle.textContent = t('plan_upload_title_missing');
      }

      // Render Chat
      const chatBox = document.getElementById('chat_box');
      if (d.chat_messages && d.chat_messages.length > 0) {
        chatBox.innerHTML = d.chat_messages.map(m => `
          <div class="msg ${m.sender_role}">
            <div>${m.content}</div>
            <div class="msg-meta">
              ${m.sender_role.toUpperCase()}
              ${m.detected_intent ? ` • ${m.detected_intent}` : ''}
              ${m.node_decision_path && m.node_decision_path.length ? ` • ${m.node_decision_path.join(' → ')}` : ''}
            </div>
          </div>
        `).join('');
        chatBox.scrollTop = chatBox.scrollHeight;
      } else {
        chatBox.innerHTML = `<div style="color:var(--text-muted); font-size:13.5px;">${t('portal_chat_desc')}</div>`;
      }

      // Render Probable Daily Foods
      const pfBody = document.getElementById('probable_foods_tbody');
      pfBody.innerHTML = d.probable_daily_foods.map(f => `
        <tr>
          <td>${f.meal_name}</td>
          <td><strong>${f.display_name}</strong></td>
          <td>${f.estimated_grams}g</td>
          <td>${f.macros.calories_kcal} kcal</td>
          <td><button class="btn-primary" style="padding:6px 12px; font-size:12px;" onclick="logProbableFood('${f.meal_name}', '${f.display_name}', ${f.estimated_grams})">${t('eat_btn')}</button></td>
        </tr>
      `).join('');

      // Render Daily Calorie Exchanges
      const exBody = document.getElementById('exchanges_tbody');
      exBody.innerHTML = d.daily_calorie_exchanges.map(ex => `
        <tr>
          <td>${ex.meal_name}</td>
          <td>${ex.primary_food_name} (${ex.primary_grams}g)</td>
          <td><strong>${ex.replacement_food_name}</strong> (${ex.replacement_grams}g)</td>
          <td><span class="status-pill missing">${ex.calorie_delta_kcal >= 0 ? '+' : ''}${ex.calorie_delta_kcal} kcal</span></td>
          <td><button class="btn-outline" style="padding:6px 12px; font-size:12px;" onclick="applyExchange('${ex.id}')">${t('swap_btn')}</button></td>
        </tr>
      `).join('');

      // Render Foods Eaten Today
      const eatenBody = document.getElementById('eaten_tbody');
      eatenBody.innerHTML = d.foods_eaten.map(fe => `
        <tr>
          <td>${fe.meal_name}</td>
          <td><strong>${fe.food_name}</strong></td>
          <td>${fe.grams}g</td>
          <td><strong>${fe.calories_kcal} kcal</strong></td>
          <td>P: ${fe.protein_g}g | C: ${fe.carbs_g}g | F: ${fe.fat_g}g</td>
        </tr>
      `).join('') || `<tr><td colspan="5" style="color:var(--text-muted);">—</td></tr>`;

      // Render Settings Athlete Profile
      document.getElementById('settings_profile_box').innerHTML = `
        <div><strong>ID:</strong> ${d.user.id}</div>
        <div><strong>Nome / Name:</strong> ${d.user.full_name || '-'}</div>
        <div><strong>CPF:</strong> ${d.user.cpf}</div>
        <div><strong>E-mail:</strong> ${d.user.email}</div>
        <div><strong>Telefone / Phone:</strong> ${d.user.phone}</div>
        <div><strong>Endereço / Address:</strong> ${d.user.address}</div>
      `;
    }

    applyTranslations();
  </script>
</body>
</html>
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_api_server_plano_alimentar_teste_pdf_and_persisted_user_chat_with_agent_memory() {
        let app = AppServerState::new();

        // 1. Register user
        let create_body = serde_json::json!({
            "cpf": "12345678901",
            "phone": "+55 11 99876-5432",
            "address": "Av. Paulista, 1000 - Sao Paulo, SP",
            "email": "gabriel@calorie.ai",
            "password": "SafePassword123!",
            "full_name": "Gabriel Silva"
        })
        .to_string();

        let (status, _, resp_json) = app
            .handle_http_request("POST", "/api/v1/users", &create_body)
            .await;
        assert_eq!(status, 201);
        let dash: UserDashboardResponse =
            serde_json::from_str(&resp_json).expect("Valid dashboard JSON");
        let user_id = dash.user.id.clone();

        // 2. Ingest the workspace's real `Plano Alimentar Teste.pdf` via `/api/v1/plans/load-sample-pdf`
        let sample_body = serde_json::json!({ "user_id": user_id }).to_string();
        let (status_pdf, _, pdf_json) = app
            .handle_http_request("POST", "/api/v1/plans/load-sample-pdf", &sample_body)
            .await;
        assert_eq!(status_pdf, 200);
        let dash_pdf: UserDashboardResponse =
            serde_json::from_str(&pdf_json).expect("Valid dashboard JSON after sample PDF");
        assert_eq!(dash_pdf.profile.daily_calorie_target_kcal, 2100.0);
        assert_eq!(dash_pdf.probable_daily_foods.len(), 28);
        assert!(!dash_pdf.daily_calorie_exchanges.is_empty());

        // 3. User sends a Chat Message asking about their nutrition plan -> saved in Postgres & used in Agent Memory
        let chat_q_body = serde_json::json!({
            "user_id": user_id,
            "message": "Quais sao as calorias e opcoes do meu plano alimentar?"
        })
        .to_string();
        let (status_chat1, _, chat1_json) = app
            .handle_http_request("POST", "/api/v1/chat/send", &chat_q_body)
            .await;
        assert_eq!(status_chat1, 200);
        let dash_chat1: UserDashboardResponse =
            serde_json::from_str(&chat1_json).expect("Valid dashboard JSON after chat 1");

        assert_eq!(dash_chat1.chat_messages.len(), 2); // 1 user + 1 assistant
        assert_eq!(dash_chat1.chat_messages[0].sender_role, "user");
        assert_eq!(dash_chat1.chat_messages[1].sender_role, "assistant");
        // Verify the assistant's coaching response includes the active plan from agent memory
        assert!(dash_chat1.chat_messages[1]
            .content
            .contains("Plano Alimentar Individualizado"));

        // 4. User sends a Chat Message logging a meal -> automatically logs into `foods_eaten` & updates `agent_chat_memories`
        let chat_meal_body = serde_json::json!({
            "user_id": user_id,
            "message": "Comi 120g de peito de frango grelhado e 100g de arroz integral no almoco"
        })
        .to_string();
        let (status_chat2, _, chat2_json) = app
            .handle_http_request("POST", "/api/v1/chat/send", &chat_meal_body)
            .await;
        assert_eq!(status_chat2, 200);
        let dash_chat2: UserDashboardResponse =
            serde_json::from_str(&chat2_json).expect("Valid dashboard JSON after chat 2");

        assert_eq!(dash_chat2.chat_messages.len(), 4);
        assert_eq!(dash_chat2.foods_eaten.len(), 2);
        assert!(dash_chat2.daily_totals.total_calories_kcal > 300.0);
        assert!(dash_chat2.agent_chat_memories.len() >= 7);
        assert_eq!(dash_chat2.memory_snapshot.working_turns.len(), 2);

        // 5. Verify entrance authentication (`POST /api/v1/users/login`) by email or CPF
        let login_ok_body = serde_json::json!({
            "login": "gabriel@calorie.ai",
            "password": "SafePassword123!"
        })
        .to_string();
        let (status_login, _, login_json) = app
            .handle_http_request("POST", "/api/v1/users/login", &login_ok_body)
            .await;
        assert_eq!(status_login, 200);
        let dash_login: UserDashboardResponse =
            serde_json::from_str(&login_json).expect("Valid dashboard JSON on login");
        assert_eq!(dash_login.user.id, user_id);

        let login_bad_body = serde_json::json!({
            "login": "gabriel@calorie.ai",
            "password": "WrongPassword!"
        })
        .to_string();
        let (status_bad, _, _) = app
            .handle_http_request("POST", "/api/v1/users/login", &login_bad_body)
            .await;
        assert_eq!(status_bad, 401);

        // 6. Verify per-node LLM configuration endpoint (`GET /api/v1/config/llms`)
        let (status_llms, _, llms_json) =
            app.handle_http_request("GET", "/api/v1/config/llms", "").await;
        assert_eq!(status_llms, 200);
        assert!(llms_json.contains("qwen2.5-1.5b-instruct-json"));
    }
}
