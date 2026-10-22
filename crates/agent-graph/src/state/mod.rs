//! Immutable/versioned `GraphState`, hierarchy contracts, checkpoint snapshots, and
//! harness telemetry structures passed across nodes in `agent-graph`.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use uuid::Uuid;

/// Identifier for every specialized agent node in the graph (matching `README.md` Section 5.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum GraphNodeId {
    /// Entry point: classifies user intent, creates the initial `ExecutionPlan`,
    /// and routes audio inputs to `AudioTranscriber`.
    IntentRouter,
    /// Tool-calling node that invokes Whisper VAD + transcription on voice inputs.
    AudioTranscriber,
    /// Planner/decomposition node that extracts structured `FoodEntity` items and serving sizes.
    MealParser,
    /// Tool-calling node (`nutrition_lookup`) that checks Redis L1 (exact/barcode) and L2 (semantic RAG) caches.
    CacheLookup,
    /// Tool-calling node (`web_researcher`) that performs concurrent web search fan-out,
    /// ingests nutrition tables into Hybrid RAG, and populates the L1/L2 cache for future users.
    WebSearchAndRAG,
    /// Planner/critic node (`critic_verifier`) that validates retrieved nutrition facts,
    /// scores confidence, and triggers cyclic retries or advances to the Executioner tier.
    NutritionVerifier,
    /// Executioner node that scales servings and calculates total meal macros & calories.
    MacroCalculator,
    /// Tool-calling & domain coaching node (`fitness_coach`) that answers nutrition/fitness questions
    /// using Episodic/Semantic memory and grounded RAG context.
    CoachingAgent,
    /// Executioner node that persists new meal registrations in the calorie counter ledger
    /// and updates Working/Episodic/Semantic memory.
    MemoryUpdater,
    /// Executioner node that synthesizes the final user-facing response and telemetry summary.
    ResponseSynthesizer,
    /// Terminal graph sink (`[*]`).
    Terminal,
}

impl GraphNodeId {
    /// Canonical string identifier used in checkpoints, logs, and the token ledger.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::IntentRouter => "IntentRouter",
            Self::AudioTranscriber => "AudioTranscriber",
            Self::MealParser => "MealParser",
            Self::CacheLookup => "CacheLookup",
            Self::WebSearchAndRAG => "WebSearchAndRAG",
            Self::NutritionVerifier => "NutritionVerifier",
            Self::MacroCalculator => "MacroCalculator",
            Self::CoachingAgent => "CoachingAgent",
            Self::MemoryUpdater => "MemoryUpdater",
            Self::ResponseSynthesizer => "ResponseSynthesizer",
            Self::Terminal => "Terminal",
        }
    }

    /// Returns the node's tier in the **Planner -> Tool Calling -> Executioner** hierarchy.
    pub fn hierarchy_role(&self) -> AgentHierarchyRole {
        match self {
            Self::IntentRouter | Self::MealParser | Self::NutritionVerifier => {
                AgentHierarchyRole::Planner
            }
            Self::AudioTranscriber
            | Self::CacheLookup
            | Self::WebSearchAndRAG
            | Self::CoachingAgent => AgentHierarchyRole::ToolCaller,
            Self::MacroCalculator
            | Self::MemoryUpdater
            | Self::ResponseSynthesizer
            | Self::Terminal => AgentHierarchyRole::Executioner,
        }
    }

    /// Default model routing tier recommended for this node (`README.md` Section 5.2).
    pub fn default_model_tier(&self) -> ModelTier {
        match self {
            Self::AudioTranscriber | Self::CacheLookup | Self::MacroCalculator | Self::Terminal => {
                ModelTier::Tier0Deterministic
            }
            Self::IntentRouter
            | Self::MealParser
            | Self::WebSearchAndRAG
            | Self::MemoryUpdater
            | Self::ResponseSynthesizer => ModelTier::Tier1FastNano,
            Self::NutritionVerifier | Self::CoachingAgent => ModelTier::Tier2Reasoning,
        }
    }
}

impl std::fmt::Display for GraphNodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Three-tier agent hierarchy governing planning, tool execution, and state mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentHierarchyRole {
    /// Analyzes user intent, decomposes tasks, builds the `ExecutionPlan`, and verifies quality/confidence.
    Planner,
    /// Invokes external or internal tools (Whisper STT, Redis L1/L2 RAG cache, Web Search fan-out, RAG ingestion, Memory retrieval).
    ToolCaller,
    /// Computes deterministic macro totals, commits meal registrations to the calorie counter, updates memory, and synthesizes final output.
    Executioner,
}

impl std::fmt::Display for AgentHierarchyRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Planner => f.write_str("Planner"),
            Self::ToolCaller => f.write_str("ToolCaller"),
            Self::Executioner => f.write_str("Executioner"),
        }
    }
}

/// Detected natural language of the user request (English or Portuguese).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum DetectedLanguage {
    #[default]
    English,
    Portuguese,
}

/// Dynamic model routing tiers enforced by `agent-harness` (`README.md` Section 5.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ModelTier {
    /// Tier 0: Local/deterministic logic, regex, Redis cache hits, local Whisper/embeddings (`$0` LLM token cost).
    Tier0Deterministic,
    /// Tier 1: Fast/Nano LLM for routing, extraction, query rewriting, and concise synthesis.
    Tier1FastNano,
    /// Tier 2: Deep reasoning LLM for conflicting nutrition verification and personalized metabolic coaching.
    Tier2Reasoning,
}

impl ModelTier {
    pub fn default_model_id(&self) -> &'static str {
        match self {
            Self::Tier0Deterministic => "tier0-local-deterministic",
            Self::Tier1FastNano => "tier1-fast-nano-llm",
            Self::Tier2Reasoning => "tier2-reasoning-llm",
        }
    }

    /// Cost rate in USD micro-dollars ($10^{-6}$ USD) per 1,000 prompt tokens.
    pub fn prompt_cost_micros_per_1k(&self) -> u64 {
        match self {
            Self::Tier0Deterministic => 0,
            Self::Tier1FastNano => 150,     // $0.00015 / 1K tokens
            Self::Tier2Reasoning => 2_500,  // $0.00250 / 1K tokens
        }
    }

    /// Cost rate in USD micro-dollars ($10^{-6}$ USD) per 1,000 completion tokens.
    pub fn completion_cost_micros_per_1k(&self) -> u64 {
        match self {
            Self::Tier0Deterministic => 0,
            Self::Tier1FastNano => 600,     // $0.00060 / 1K tokens
            Self::Tier2Reasoning => 10_000, // $0.01000 / 1K tokens
        }
    }
}

/// Audio encoding format submitted to the graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioFormat {
    Pcm16kHzMonoF32,
    OpusWebm,
    Wav,
    Aac,
}

/// Raw or pre-chunked audio payload for voice meal logging or voice coaching questions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioPayload {
    pub audio_id: String,
    pub format: AudioFormat,
    pub sample_rate_hz: u32,
    pub duration_ms: u64,
    /// Raw audio bytes (or encoded frame bytes).
    pub bytes: Vec<u8>,
    /// Optional domain hint or simulated speech content for deterministic testing of the Whisper pipeline.
    pub speech_hint: Option<String>,
}

/// Primary input into the agent graph: either an audio recording or text message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphInput {
    /// Natural language text input (English or Portuguese).
    Text { text: String },
    /// Voice recording input to be transcribed by `AudioTranscriber` using Whisper.
    Audio { payload: AudioPayload },
}

impl GraphInput {
    pub fn text(input: impl Into<String>) -> Self {
        Self::Text { text: input.into() }
    }

    pub fn audio(
        bytes: Vec<u8>,
        duration_ms: u64,
        speech_hint: Option<String>,
    ) -> Self {
        Self::Audio {
            payload: AudioPayload {
                audio_id: Uuid::new_v4().to_string(),
                format: AudioFormat::Pcm16kHzMonoF32,
                sample_rate_hz: 16_000,
                duration_ms,
                bytes,
                speech_hint,
            },
        }
    }

    pub fn is_audio(&self) -> bool {
        matches!(self, Self::Audio { .. })
    }
}

/// High-level user intent classified by the `IntentRouter` Planner node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UserIntent {
    /// Register a new meal / food items and calories in the calorie counter.
    RegisterMeal,
    /// Ask a question about nutrition, calories, macros, or fitness coaching.
    AskNutrition,
    /// Both register a meal AND answer a nutritional/coaching question in the same turn.
    RegisterMealAndAskNutrition,
}

/// A tool invocation scheduled by the Planner tier for the ToolCaller tier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedToolCall {
    pub tool_name: String,
    pub assigned_node: GraphNodeId,
    pub target_entity_or_query: String,
    pub purpose: String,
    pub completed: bool,
}

/// Structured execution plan produced and refined by the **Planner** hierarchy tier
/// (`IntentRouter`, `MealParser`, `NutritionVerifier`) to coordinate **ToolCaller** and **Executioner** nodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub plan_id: String,
    pub intent: UserIntent,
    pub language: DetectedLanguage,
    pub current_stage: AgentHierarchyRole,
    pub requires_audio_transcription: bool,
    pub requires_meal_registration: bool,
    pub requires_nutrition_coaching: bool,
    pub confidence_threshold: f32,
    pub max_verification_retries: u32,
    pub planner_notes: Vec<String>,
    pub scheduled_tool_calls: Vec<PlannedToolCall>,
    pub executioner_actions: Vec<String>,
}

impl Default for ExecutionPlan {
    fn default() -> Self {
        Self {
            plan_id: Uuid::new_v4().to_string(),
            intent: UserIntent::RegisterMeal,
            language: DetectedLanguage::English,
            current_stage: AgentHierarchyRole::Planner,
            requires_audio_transcription: false,
            requires_meal_registration: true,
            requires_nutrition_coaching: false,
            confidence_threshold: 0.78,
            max_verification_retries: 2,
            planner_notes: Vec::new(),
            scheduled_tool_calls: Vec::new(),
            executioner_actions: Vec::new(),
        }
    }
}

/// Structured food entity extracted by `MealParser` from text or voice transcript.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FoodEntity {
    pub entity_id: String,
    pub raw_mention: String,
    /// Canonical normalized key used for Redis L1/L2 cache indexing (e.g., `"grilled_chicken_breast"`).
    pub canonical_key: String,
    pub display_name: String,
    pub brand: Option<String>,
    pub barcode_ean: Option<String>,
    pub quantity: f32,
    pub unit: String,
    /// Estimated weight in grams for macro scaling.
    pub estimated_grams: f32,
    pub preparation: Option<String>,
}

/// Macro and key micronutrient breakdown (either per 100g or scaled to a serving).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MacroNutrients {
    pub calories_kcal: f32,
    pub protein_g: f32,
    pub carbs_g: f32,
    pub fat_g: f32,
    pub fiber_g: f32,
    pub sugar_g: f32,
    pub sodium_mg: f32,
}

impl MacroNutrients {
    /// Scale per-100g macros to a specific gram weight.
    pub fn scaled_for_grams(&self, grams: f32) -> Self {
        let safe_grams = if grams.is_finite() {
            grams.max(0.0)
        } else {
            0.0
        };
        let factor = safe_grams / 100.0;
        Self {
            calories_kcal: round_one_decimal(self.calories_kcal * factor),
            protein_g: round_one_decimal(self.protein_g * factor),
            carbs_g: round_one_decimal(self.carbs_g * factor),
            fat_g: round_one_decimal(self.fat_g * factor),
            fiber_g: round_one_decimal(self.fiber_g * factor),
            sugar_g: round_one_decimal(self.sugar_g * factor),
            sodium_mg: round_one_decimal(self.sodium_mg * factor),
        }
    }

    /// Accumulate another macro profile into `self`.
    pub fn add_assign(&mut self, other: &Self) {
        self.calories_kcal = round_one_decimal(self.calories_kcal + other.calories_kcal);
        self.protein_g = round_one_decimal(self.protein_g + other.protein_g);
        self.carbs_g = round_one_decimal(self.carbs_g + other.carbs_g);
        self.fat_g = round_one_decimal(self.fat_g + other.fat_g);
        self.fiber_g = round_one_decimal(self.fiber_g + other.fiber_g);
        self.sugar_g = round_one_decimal(self.sugar_g + other.sugar_g);
        self.sodium_mg = round_one_decimal(self.sodium_mg + other.sodium_mg);
    }

    /// Computes expected kcal via the Atwater system ($4P + 4C + 9F$).
    pub fn atwater_estimated_kcal(&self) -> f32 {
        (self.protein_g * 4.0) + (self.carbs_g * 4.0) + (self.fat_g * 9.0)
    }
}

fn round_one_decimal(val: f32) -> f32 {
    if !val.is_finite() {
        return 0.0;
    }
    (val * 10.0).round() / 10.0
}

/// Origin of a nutritional RAG document or cache hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RagSourceType {
    /// Redis L1 exact canonical key or EAN barcode cache hit.
    RedisL1ExactCache,
    /// Redis L2 semantic vector similarity cache hit.
    RedisL2SemanticCache,
    /// PostgreSQL Hybrid RAG (`pgvector` HNSW + BM25 `tsvector`).
    PostgresHybridRag,
    /// SearXNG metasearch web scrape.
    WebSearchSearXNG,
    /// USDA FoodData Central / TACO (Tabela Brasileira de Composição de Alimentos).
    UsdaFoodDataCentral,
    /// OpenFoodFacts product & barcode database.
    OpenFoodFacts,
}

/// Grounded document chunk stored in and retrieved from the Hybrid RAG & Redis L1/L2 cache.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RagDocumentChunk {
    pub chunk_id: String,
    pub canonical_food_key: String,
    pub display_name: String,
    pub barcode_ean: Option<String>,
    pub source_type: RagSourceType,
    pub source_url: String,
    pub content_markdown: String,
    pub macros_per_100g: MacroNutrients,
    /// Compact dense embedding vector used for semantic similarity lookup in RAG cache.
    pub embedding: Vec<f32>,
    pub relevance_score: f32,
}

/// Verified food item with scaled macros ready for registration by the Executioner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifiedFoodItem {
    pub entity_id: String,
    pub canonical_key: String,
    pub display_name: String,
    pub quantity: f32,
    pub unit: String,
    pub grams: f32,
    pub macros_per_100g: MacroNutrients,
    pub scaled_macros: MacroNutrients,
    pub confidence: f32,
    pub source_type: RagSourceType,
    pub citation_url: String,
    pub rag_chunk_id: String,
    pub served_from_cache: bool,
}

/// Completed meal registration record persisted by `MemoryUpdater` (`Executioner` tier).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MealRegistration {
    pub meal_id: String,
    pub run_id: String,
    pub user_id: String,
    pub conversation_id: String,
    pub raw_input_summary: String,
    pub items: Vec<VerifiedFoodItem>,
    pub total_macros: MacroNutrients,
    pub average_confidence: f32,
    pub newly_cached_rag_keys: Vec<String>,
    pub cache_hit_keys: Vec<String>,
}

/// Token consumption and cost accounting for a single node or cumulative graph run.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub cached_tokens: u32,
    pub total_tokens: u32,
    /// Estimated cost in micro-dollars ($10^{-6}$ USD).
    pub cost_usd_micros: u64,
}

impl TokenUsage {
    pub fn new(
        prompt_tokens: u32,
        completion_tokens: u32,
        cached_tokens: u32,
        tier: ModelTier,
    ) -> Self {
        let billable_prompt = prompt_tokens.saturating_sub(cached_tokens / 2);
        let prompt_cost =
            (billable_prompt as u64 * tier.prompt_cost_micros_per_1k()).div_ceil(1_000);
        let completion_cost =
            (completion_tokens as u64 * tier.completion_cost_micros_per_1k()).div_ceil(1_000);
        Self {
            prompt_tokens,
            completion_tokens,
            cached_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            cost_usd_micros: prompt_cost + completion_cost,
        }
    }

    pub fn accumulate(&mut self, other: &Self) {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.cached_tokens += other.cached_tokens;
        self.total_tokens += other.total_tokens;
        self.cost_usd_micros += other.cost_usd_micros;
    }
}

/// Per-node execution telemetry recorded by the Harness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeExecutionMetric {
    pub step_index: u32,
    pub node_id: GraphNodeId,
    pub hierarchy_role: AgentHierarchyRole,
    pub model_tier: ModelTier,
    pub model_id: String,
    pub downgraded_by_harness: bool,
    pub is_mock_model: bool,
    pub token_usage: TokenUsage,
    pub latency_ms: u64,
    pub tools_called: u32,
}

/// Record of an individual tool invocation within a graph node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolUsageRecord {
    pub call_id: String,
    pub step_index: u32,
    pub caller_node: GraphNodeId,
    pub hierarchy_role: AgentHierarchyRole,
    pub tool_name: String,
    pub input_summary: String,
    pub output_summary: String,
    /// `Some(true)` if the tool hit the Redis L1/L2 RAG cache, `Some(false)` on cache miss.
    pub cache_hit: Option<bool>,
    pub latency_ms: u64,
    pub tokens_used: u32,
    pub success: bool,
}

/// Audit record of an edge evaluation / routing decision in the agent graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionPathStep {
    pub step_index: u32,
    pub from_node: GraphNodeId,
    pub to_node: GraphNodeId,
    pub from_role: AgentHierarchyRole,
    pub to_role: AgentHierarchyRole,
    pub condition_label: String,
    pub rationale: String,
    pub confidence_score: Option<f32>,
    pub retry_iteration: u32,
    pub evaluation_latency_us: u64,
}

/// Aggregated harness telemetry across the entire graph execution.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ExecutionTelemetry {
    /// Simulated/accounted token usage tracked by the Harness budget ledger.
    pub total_token_usage: TokenUsage,
    /// Real external API tokens consumed (`0` when running with `MockLlmModelProvider`).
    pub real_api_tokens_consumed: u32,
    /// Number of mocked LLM invocations executed without external API calls.
    pub mocked_model_calls: u32,
    pub total_latency_ms: u64,
    pub node_metrics: Vec<NodeExecutionMetric>,
    pub tool_usage_log: Vec<ToolUsageRecord>,
    pub decision_path: Vec<DecisionPathStep>,
}

/// Budget & circuit-breaker limits enforced by `agent-harness` (`README.md` Section 5.2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HarnessBudgetConfig {
    /// Maximum output/total tokens allowed for a single node call.
    pub default_max_node_tokens: u32,
    /// Maximum cumulative tokens across all graph steps in a single user turn (default: 12,000).
    pub max_turn_tokens: u32,
    /// Maximum cumulative USD cost in micro-dollars per turn (default: 15,000 = $0.015).
    pub max_turn_cost_usd_micros: u64,
    /// User's daily token quota and current usage (for 80% automatic tier downgrade).
    pub user_daily_token_limit: u32,
    pub user_daily_tokens_used: u32,
    /// Ratio (e.g. 0.80) at which Tier 2 calls are downgraded to Tier 1.
    pub tier_downgrade_threshold: f32,
    /// Hard maximum number of node transitions per run (default: 12).
    pub max_graph_steps: u32,
    /// Maximum number of times any single `GraphNodeId` may be visited in one run before loop detection triggers.
    pub max_node_visits: u32,
}

impl Default for HarnessBudgetConfig {
    fn default() -> Self {
        Self {
            default_max_node_tokens: 2_000,
            max_turn_tokens: 12_000,
            max_turn_cost_usd_micros: 15_000, // $0.015
            user_daily_token_limit: 100_000,
            user_daily_tokens_used: 0,
            tier_downgrade_threshold: 0.80,
            max_graph_steps: 12,
            max_node_visits: 3,
        }
    }
}

/// Immutable/versioned state passed across nodes in `agent-graph`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphState {
    pub run_id: String,
    pub conversation_id: String,
    pub user_id: String,
    pub state_version: u32,
    pub step_index: u32,
    pub current_node: GraphNodeId,
    pub input: GraphInput,
    /// Populated either directly from `GraphInput::Text` or by `AudioTranscriber` from `GraphInput::Audio`.
    pub transcript: Option<String>,
    /// Planner's structured execution plan governing ToolCallers and Executioners.
    pub plan: ExecutionPlan,
    /// Structured food entities extracted by `MealParser`.
    pub extracted_foods: Vec<FoodEntity>,
    /// Canonical keys of entities that missed the Redis L1/L2 RAG cache and require `WebSearchAndRAG`.
    pub cache_miss_keys: Vec<String>,
    /// Grounded nutrition chunks retrieved from Redis L1/L2 cache or WebSearch + Hybrid RAG.
    pub retrieved_rag_chunks: BTreeMap<String, RagDocumentChunk>,
    /// Canonical food keys that were newly scraped from Web Search and written into the RAG cache during this run.
    pub newly_populated_rag_cache_keys: Vec<String>,
    /// Canonical food keys that were served directly from the RAG cache (L1 or L2).
    pub cache_hit_keys: Vec<String>,
    /// Canonical food keys where a concurrent in-flight web search race condition was coalesced into a cache hit.
    pub race_coalesced_keys: Vec<String>,
    /// Warnings generated when sanitizing weird, malformed, negative, or adversarial inputs.
    pub sanitization_warnings: Vec<String>,
    /// Per-node visit counter tracked by the Harness to detect oscillation loops.
    pub node_visit_counts: BTreeMap<GraphNodeId, u32>,
    /// Verified food items with portion-scaled macros and confidence scores.
    pub verified_items: Vec<VerifiedFoodItem>,
    /// Overall nutrition verification confidence computed by `NutritionVerifier` (`0.0..=1.0`).
    pub verification_confidence: f32,
    /// Number of cyclic retries from `NutritionVerifier` back to `WebSearchAndRAG`.
    pub verification_retries: u32,
    /// SHA-256 hashes of executed web search queries to prevent duplicate search loops.
    pub executed_search_query_hashes: Vec<String>,
    /// Final meal registration record computed by `MacroCalculator` and persisted by `MemoryUpdater`.
    pub meal_registration: Option<MealRegistration>,
    /// Grounded nutrition/fitness coaching response prepared by `CoachingAgent`.
    pub coaching_notes: Option<String>,
    /// Working/Episodic/Semantic memory keys updated by `MemoryUpdater`.
    pub updated_memory_keys: Vec<String>,
    /// Final synthesized user-facing message produced by `ResponseSynthesizer`.
    pub final_response: Option<String>,
    /// Harness budget configuration for this run.
    pub budget_config: HarnessBudgetConfig,
    /// Full telemetry of token usage, latency, tool calls, and decision path transitions.
    pub telemetry: ExecutionTelemetry,
}

impl GraphState {
    /// Creates a new initial `GraphState` starting at `GraphNodeId::IntentRouter`.
    pub fn new(
        user_id: impl Into<String>,
        conversation_id: impl Into<String>,
        input: GraphInput,
    ) -> Self {
        let transcript = match &input {
            GraphInput::Text { text } => Some(text.clone()),
            GraphInput::Audio { .. } => None,
        };
        Self {
            run_id: Uuid::new_v4().to_string(),
            conversation_id: conversation_id.into(),
            user_id: user_id.into(),
            state_version: 1,
            step_index: 0,
            current_node: GraphNodeId::IntentRouter,
            input,
            transcript,
            plan: ExecutionPlan::default(),
            extracted_foods: Vec::new(),
            cache_miss_keys: Vec::new(),
            retrieved_rag_chunks: BTreeMap::new(),
            newly_populated_rag_cache_keys: Vec::new(),
            cache_hit_keys: Vec::new(),
            race_coalesced_keys: Vec::new(),
            sanitization_warnings: Vec::new(),
            node_visit_counts: BTreeMap::new(),
            verified_items: Vec::new(),
            verification_confidence: 0.0,
            verification_retries: 0,
            executed_search_query_hashes: Vec::new(),
            meal_registration: None,
            coaching_notes: None,
            updated_memory_keys: Vec::new(),
            final_response: None,
            budget_config: HarnessBudgetConfig::default(),
            telemetry: ExecutionTelemetry::default(),
        }
    }

    /// Configures custom harness budget parameters.
    pub fn with_budget_config(mut self, config: HarnessBudgetConfig) -> Self {
        self.budget_config = config;
        self
    }

    /// Records a tool invocation in the state's telemetry log.
    pub fn record_tool_call(&mut self, record: ToolUsageRecord) {
        self.telemetry.tool_usage_log.push(record);
    }

    /// Marks a planned tool call as completed if present in `self.plan`.
    pub fn mark_planned_tool_completed(&mut self, tool_name: &str) {
        for call in &mut self.plan.scheduled_tool_calls {
            if call.tool_name == tool_name && !call.completed {
                call.completed = true;
            }
        }
    }

    /// Computes a deterministic SHA-256 hex digest of the current state for checkpoint integrity.
    pub fn compute_state_hash(&self) -> String {
        let serialized = serde_json::to_vec(self).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(&serialized);
        hex::encode(hasher.finalize())
    }
}

/// Serializable checkpoint snapshot captured after every node transition (`README.md` Section 5.1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphCheckpoint {
    pub checkpoint_id: String,
    pub conversation_id: String,
    pub run_id: String,
    pub step_index: u32,
    pub state_version: u32,
    pub completed_node: GraphNodeId,
    pub next_node: GraphNodeId,
    pub hierarchy_role: AgentHierarchyRole,
    pub state_hash: String,
    pub cumulative_tokens: u32,
    pub cumulative_latency_ms: u64,
    pub state_snapshot: GraphState,
}

impl GraphCheckpoint {
    /// Creates a new checkpoint snapshot after `completed_node` transitions to `next_node`.
    pub fn from_transition(
        completed_node: GraphNodeId,
        next_node: GraphNodeId,
        state: &GraphState,
    ) -> Self {
        Self {
            checkpoint_id: format!("chk-{}-step-{}", state.run_id, state.step_index),
            conversation_id: state.conversation_id.clone(),
            run_id: state.run_id.clone(),
            step_index: state.step_index,
            state_version: state.state_version,
            completed_node,
            next_node,
            hierarchy_role: completed_node.hierarchy_role(),
            state_hash: state.compute_state_hash(),
            cumulative_tokens: state.telemetry.total_token_usage.total_tokens,
            cumulative_latency_ms: state.telemetry.total_latency_ms,
            state_snapshot: state.clone(),
        }
    }
}
