//! Specialized Agent Nodes implementing the **Planner -> Tool Calling -> Executioner** hierarchy,
//! Mock LLM Model Provider (`MockLlmModelProvider` for zero-API-token testing), and thread-safe
//! RAG cache & calorie ledger stores with race-condition coalescing and idempotency guards.

pub mod critic_verifier;
pub mod executioner;
pub mod fitness_coach;
pub mod intent_router;
pub mod meal_parser;
pub mod nutrition_lookup;
pub mod web_researcher;

pub use critic_verifier::CriticVerifierNode;
pub use executioner::{MacroCalculatorNode, MemoryUpdaterNode, ResponseSynthesizerNode};
pub use fitness_coach::FitnessCoachNode;
pub use intent_router::{AudioTranscriberNode, IntentRouterNode};
pub use meal_parser::MealParserNode;
pub use nutrition_lookup::NutritionLookupNode;
pub use web_researcher::WebResearcherNode;

use crate::state::{
    AgentHierarchyRole, DetectedLanguage, GraphNodeId, GraphState, MacroNutrients,
    MealRegistration, ModelTier, RagDocumentChunk, RagSourceType,
};
use async_trait::async_trait;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use thiserror::Error;

/// Errors produced during individual node execution.
#[derive(Debug, Error)]
pub enum GraphNodeError {
    #[error("Missing transcript or text input at node {node}")]
    MissingTranscript { node: GraphNodeId },

    #[error("Tool execution failed in node {node} ({tool}): {reason}")]
    ToolExecutionFailed {
        node: GraphNodeId,
        tool: String,
        reason: String,
    },

    #[error("Invalid graph state at node {node}: {details}")]
    InvalidState { node: GraphNodeId, details: String },
}

/// Output returned by an `AgentNode::execute` invocation before harness post-processing.
#[derive(Debug, Clone)]
pub struct NodeExecutionResult {
    pub state: GraphState,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub cached_tokens: u32,
    pub is_mock_model: bool,
}

impl NodeExecutionResult {
    pub fn new(
        state: GraphState,
        prompt_tokens: u32,
        completion_tokens: u32,
        cached_tokens: u32,
    ) -> Self {
        Self {
            state,
            prompt_tokens,
            completion_tokens,
            cached_tokens,
            is_mock_model: true,
        }
    }

    pub fn with_mock_flag(mut self, is_mock_model: bool) -> Self {
        self.is_mock_model = is_mock_model;
        self
    }
}

/// Request sent to an `LlmModelProvider` by a graph node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCompletionRequest {
    pub node_id: GraphNodeId,
    pub model_tier: ModelTier,
    pub language: DetectedLanguage,
    pub system_prompt: String,
    pub user_prompt: String,
}

/// Response returned by an `LlmModelProvider` (mocked in tests to consume 0 real API tokens).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCompletionResponse {
    pub content: String,
    pub simulated_prompt_tokens: u32,
    pub simulated_completion_tokens: u32,
    pub cached_tokens: u32,
    /// `true` when produced by `MockLlmModelProvider` (0 real API tokens consumed).
    pub is_mock: bool,
    /// Optional search query override used in tests to simulate duplicate-query loop bugs.
    pub forced_search_query_override: Option<String>,
}

/// Pluggable LLM provider interface. Using `MockLlmModelProvider` guarantees zero external API calls
/// and zero real API token consumption during testing while still feeding simulated token counts
/// into the `CostAndTokenHarness`.
#[async_trait]
pub trait LlmModelProvider: Send + Sync {
    /// Returns `true` if this provider mocks model responses and consumes 0 real API tokens.
    fn is_mock(&self) -> bool;

    /// Executes a model completion (or returns a deterministic mocked response).
    async fn complete(
        &self,
        request: ModelCompletionRequest,
    ) -> Result<ModelCompletionResponse, GraphNodeError>;

    /// Total real API tokens billed against an external LLM provider (`0` for mock providers).
    fn real_api_tokens_consumed(&self) -> u32;
}

/// Deterministic mock LLM provider that never calls external APIs (`real_api_tokens_consumed == 0`).
///
/// Supports bilingual (Portuguese & English) mock responses, custom canned responses per node,
/// simulated token usage overrides, artificial async delays (for testing race conditions),
/// and forced duplicate search queries (for testing loop circuit breakers).
#[derive(Debug, Default)]
pub struct MockLlmModelProvider {
    invocations: RwLock<Vec<ModelCompletionRequest>>,
    canned_responses: RwLock<BTreeMap<GraphNodeId, ModelCompletionResponse>>,
    simulated_delay_ms: AtomicUsize,
    force_duplicate_search_query: RwLock<Option<String>>,
}

impl MockLlmModelProvider {
    pub fn new() -> Self {
        Self::default()
    }

    /// Configures a custom canned response for a specific `GraphNodeId`.
    pub fn with_canned_node_response(
        self,
        node_id: GraphNodeId,
        response: ModelCompletionResponse,
    ) -> Self {
        self.canned_responses.write().insert(node_id, response);
        self
    }

    /// Configures an artificial async delay (in milliseconds) to test concurrent race conditions.
    pub fn with_simulated_delay_ms(self, delay_ms: usize) -> Self {
        self.simulated_delay_ms.store(delay_ms, Ordering::SeqCst);
        self
    }

    /// Forces `WebSearchAndRAG` to reuse a static search query string so duplicate-query loop
    /// detection in the Harness can be deterministically tested.
    pub fn with_forced_duplicate_search_query(self, query: impl Into<String>) -> Self {
        *self.force_duplicate_search_query.write() = Some(query.into());
        self
    }

    /// Returns a snapshot of all model requests captured by this mock provider.
    pub fn recorded_invocations(&self) -> Vec<ModelCompletionRequest> {
        self.invocations.read().clone()
    }
}

#[async_trait]
impl LlmModelProvider for MockLlmModelProvider {
    fn is_mock(&self) -> bool {
        true
    }

    async fn complete(
        &self,
        request: ModelCompletionRequest,
    ) -> Result<ModelCompletionResponse, GraphNodeError> {
        let delay = self.simulated_delay_ms.load(Ordering::SeqCst);
        if delay > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(delay as u64)).await;
        }

        self.invocations.write().push(request.clone());

        if let Some(canned) = self.canned_responses.read().get(&request.node_id) {
            return Ok(canned.clone());
        }

        let prompt_tokens = ((request.system_prompt.len() + request.user_prompt.len()) as u32 / 4)
            .clamp(30, 450);
        let completion_tokens = match request.node_id {
            GraphNodeId::IntentRouter => 42,
            GraphNodeId::MealParser => 55,
            GraphNodeId::WebSearchAndRAG => 60,
            GraphNodeId::NutritionVerifier => 45,
            GraphNodeId::CoachingAgent => 95,
            GraphNodeId::MemoryUpdater => 30,
            GraphNodeId::ResponseSynthesizer => 65,
            _ => 20,
        };

        let content = match (request.node_id, request.language) {
            (GraphNodeId::IntentRouter, DetectedLanguage::Portuguese) => {
                "MOCK_PT_INTENT_ROUTED: Intenção classificada e plano hierárquico criado.".to_string()
            }
            (GraphNodeId::IntentRouter, DetectedLanguage::English) => {
                "MOCK_EN_INTENT_ROUTED: Intent classified and hierarchical plan created.".to_string()
            }
            (GraphNodeId::CoachingAgent, DetectedLanguage::Portuguese) => {
                "MOCK_PT_COACH: Recomendações nutricionais baseadas na tabela TACO/USDA e metas de TDEE.".to_string()
            }
            (GraphNodeId::CoachingAgent, DetectedLanguage::English) => {
                "MOCK_EN_COACH: Grounded nutrition coaching synthesized from USDA/RAG and TDEE profile.".to_string()
            }
            _ => format!("MOCK_MODEL_OK:{}:{}", request.node_id, request.model_tier.default_model_id()),
        };

        let forced_query = self.force_duplicate_search_query.read().clone();

        Ok(ModelCompletionResponse {
            content,
            simulated_prompt_tokens: prompt_tokens,
            simulated_completion_tokens: completion_tokens,
            cached_tokens: 0,
            is_mock: true,
            forced_search_query_override: forced_query,
        })
    }

    fn real_api_tokens_consumed(&self) -> u32 {
        0
    }
}

/// Trait implemented by every node in `agent-graph`.
#[async_trait]
pub trait AgentNode: Send + Sync {
    /// Unique identifier for the node in the state machine.
    fn id(&self) -> GraphNodeId;

    /// Role in the **Planner -> ToolCaller -> Executioner** agent hierarchy.
    fn hierarchy_role(&self) -> AgentHierarchyRole {
        self.id().hierarchy_role()
    }

    /// Default LLM or deterministic tier used by the Harness router.
    fn default_model_tier(&self) -> ModelTier {
        self.id().default_model_tier()
    }

    /// Maximum token budget allocated to a single invocation of this node.
    fn node_token_budget(&self) -> u32 {
        match self.id() {
            GraphNodeId::IntentRouter => 350,
            GraphNodeId::AudioTranscriber => 100,
            GraphNodeId::MealParser => 650,
            GraphNodeId::CacheLookup => 150,
            GraphNodeId::WebSearchAndRAG => 1_400,
            GraphNodeId::NutritionVerifier => 850,
            GraphNodeId::MacroCalculator => 100,
            GraphNodeId::CoachingAgent => 1_500,
            GraphNodeId::MemoryUpdater => 450,
            GraphNodeId::ResponseSynthesizer => 950,
            GraphNodeId::Terminal => 0,
        }
    }

    /// Tools that this node is authorized to invoke (tracked by the Harness).
    fn declared_tools(&self) -> &'static [&'static str];

    /// Executes the node's logic against the current `GraphState`.
    async fn execute(
        &self,
        state: GraphState,
        ctx: &NodeExecutionContext,
    ) -> Result<NodeExecutionResult, GraphNodeError>;
}

/// Multi-layer Redis L1/L2 + PostgreSQL Hybrid RAG cache interface (`README.md` Section 8).
pub trait RagNutritionCacheStore: Send + Sync {
    /// L1 Exact canonical key or EAN barcode cache lookup (<2ms in Redis).
    fn lookup_l1_exact(
        &self,
        canonical_key: &str,
        barcode_ean: Option<&str>,
    ) -> Option<RagDocumentChunk>;

    /// L2 Semantic vector similarity cache lookup (<10ms in Redis / pgvector).
    fn lookup_l2_semantic(
        &self,
        query_embedding: &[f32],
        min_cosine_similarity: f32,
    ) -> Option<RagDocumentChunk>;

    /// Hybrid RAG search combining keyword match + dense vector similarity.
    fn search_hybrid_rag(&self, query: &str, top_k: usize) -> Vec<RagDocumentChunk>;

    /// Ingests a newly scraped/verified web nutrition chunk into the Hybrid RAG and populates L1/L2 cache.
    fn ingest_and_cache(&self, chunk: RagDocumentChunk);

    /// Lists all canonical food keys currently stored in the RAG cache.
    fn cached_food_keys(&self) -> Vec<String>;

    /// Acquires a per-food-key async lock to coalesce concurrent web searches for the same uncached food
    /// (preventing cache stampede race conditions).
    fn acquire_in_flight_lock(&self, canonical_key: &str) -> Arc<tokio::sync::Mutex<()>>;

    /// Records and returns the count of concurrent web search requests coalesced by singleflight locking.
    fn record_coalesced_race_hit(&self) -> usize;

    /// Total number of concurrent cache stampede races prevented by singleflight coalescing.
    fn coalesced_race_hits(&self) -> usize;
}

/// Thread-safe in-memory implementation of the L1/L2 + Hybrid RAG cache with singleflight
/// per-key lock coalescing to prevent concurrent web-search race conditions.
#[derive(Debug, Default)]
pub struct InMemoryRagCacheStore {
    by_canonical_key: RwLock<BTreeMap<String, RagDocumentChunk>>,
    by_barcode: RwLock<BTreeMap<String, String>>,
    in_flight_locks: Mutex<BTreeMap<String, Arc<tokio::sync::Mutex<()>>>>,
    coalesced_races: AtomicUsize,
}

impl InMemoryRagCacheStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_seed_chunks(chunks: Vec<RagDocumentChunk>) -> Self {
        let store = Self::new();
        for chunk in chunks {
            store.ingest_and_cache(chunk);
        }
        store
    }
}

impl RagNutritionCacheStore for InMemoryRagCacheStore {
    fn lookup_l1_exact(
        &self,
        canonical_key: &str,
        barcode_ean: Option<&str>,
    ) -> Option<RagDocumentChunk> {
        let map = self.by_canonical_key.read();
        if let Some(chunk) = map.get(canonical_key) {
            let mut hit = chunk.clone();
            hit.source_type = RagSourceType::RedisL1ExactCache;
            return Some(hit);
        }
        if let Some(ean) = barcode_ean {
            let barcodes = self.by_barcode.read();
            if let Some(mapped_key) = barcodes.get(ean) {
                if let Some(chunk) = map.get(mapped_key) {
                    let mut hit = chunk.clone();
                    hit.source_type = RagSourceType::RedisL1ExactCache;
                    return Some(hit);
                }
            }
        }
        None
    }

    fn lookup_l2_semantic(
        &self,
        query_embedding: &[f32],
        min_cosine_similarity: f32,
    ) -> Option<RagDocumentChunk> {
        let map = self.by_canonical_key.read();
        let mut best: Option<(f32, RagDocumentChunk)> = None;
        for chunk in map.values() {
            let sim = cosine_similarity(query_embedding, &chunk.embedding);
            if sim >= min_cosine_similarity {
                match &best {
                    Some((best_sim, _)) if sim <= *best_sim => {}
                    _ => {
                        let mut hit = chunk.clone();
                        hit.source_type = RagSourceType::RedisL2SemanticCache;
                        hit.relevance_score = sim;
                        best = Some((sim, hit));
                    }
                }
            }
        }
        best.map(|(_, chunk)| chunk)
    }

    fn search_hybrid_rag(&self, query: &str, top_k: usize) -> Vec<RagDocumentChunk> {
        let query_norm = normalize_food_key(query);
        let query_emb = compute_deterministic_embedding(&query_norm);
        let map = self.by_canonical_key.read();

        let mut scored: Vec<(f32, RagDocumentChunk)> = map
            .values()
            .map(|chunk| {
                let dense_sim = cosine_similarity(&query_emb, &chunk.embedding);
                let keyword_bonus = if !query_norm.is_empty()
                    && (chunk.canonical_food_key.contains(&query_norm)
                        || query_norm.contains(&chunk.canonical_food_key))
                {
                    0.25
                } else {
                    0.0
                };
                let rrf_score = (dense_sim + keyword_bonus).min(1.0);
                let mut item = chunk.clone();
                item.relevance_score = rrf_score;
                (rrf_score, item)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .take(top_k)
            .map(|(_, chunk)| chunk)
            .collect()
    }

    fn ingest_and_cache(&self, chunk: RagDocumentChunk) {
        if let Some(ean) = &chunk.barcode_ean {
            self.by_barcode
                .write()
                .insert(ean.clone(), chunk.canonical_food_key.clone());
        }
        self.by_canonical_key
            .write()
            .insert(chunk.canonical_food_key.clone(), chunk);
    }

    fn cached_food_keys(&self) -> Vec<String> {
        self.by_canonical_key.read().keys().cloned().collect()
    }

    fn acquire_in_flight_lock(&self, canonical_key: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut guard = self.in_flight_locks.lock();
        guard
            .entry(canonical_key.to_string())
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .clone()
    }

    fn record_coalesced_race_hit(&self) -> usize {
        self.coalesced_races.fetch_add(1, Ordering::SeqCst) + 1
    }

    fn coalesced_race_hits(&self) -> usize {
        self.coalesced_races.load(Ordering::SeqCst)
    }
}

/// Calorie counter ledger store persisting user meal registrations with idempotency protection.
pub trait CalorieCounterLedgerStore: Send + Sync {
    /// Registers a meal idempotently (returns `true` if newly inserted, `false` if duplicate `run_id` or `meal_id` was already committed).
    fn register_meal(&self, registration: MealRegistration) -> bool;
    fn user_daily_totals(&self, user_id: &str) -> MacroNutrients;
    fn user_meals(&self, user_id: &str) -> Vec<MealRegistration>;
}

/// Thread-safe in-memory calorie counter ledger with idempotency deduplication against concurrent duplicate writes.
#[derive(Debug, Default)]
pub struct InMemoryCalorieLedgerStore {
    meals_by_user: RwLock<BTreeMap<String, Vec<MealRegistration>>>,
    committed_ids: RwLock<HashSet<String>>,
}

impl InMemoryCalorieLedgerStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl CalorieCounterLedgerStore for InMemoryCalorieLedgerStore {
    fn register_meal(&self, registration: MealRegistration) -> bool {
        let mut committed = self.committed_ids.write();
        let idempotency_key = format!("{}:{}", registration.user_id, registration.run_id);
        if !committed.insert(idempotency_key) || !committed.insert(registration.meal_id.clone()) {
            // Concurrent or resumed duplicate registration prevented from double-charging calories
            return false;
        }
        drop(committed);

        self.meals_by_user
            .write()
            .entry(registration.user_id.clone())
            .or_default()
            .push(registration);
        true
    }

    fn user_daily_totals(&self, user_id: &str) -> MacroNutrients {
        let guard = self.meals_by_user.read();
        let mut totals = MacroNutrients::default();
        if let Some(meals) = guard.get(user_id) {
            for meal in meals {
                totals.add_assign(&meal.total_macros);
            }
        }
        totals
    }

    fn user_meals(&self, user_id: &str) -> Vec<MealRegistration> {
        self.meals_by_user
            .read()
            .get(user_id)
            .cloned()
            .unwrap_or_default()
    }
}

/// Multi-tier conversation memory store (Working, Episodic, Semantic) (`README.md` Section 7.2).
pub trait ConversationMemoryStore: Send + Sync {
    fn append_working_turn(&self, conversation_id: &str, summary: String);
    fn store_episodic_event(&self, user_id: &str, event: String);
    fn get_user_semantic_profile(&self, user_id: &str) -> BTreeMap<String, String>;
    fn upsert_semantic_fact(&self, user_id: &str, key: String, value: String);
    fn get_episodic_events(&self, user_id: &str) -> Vec<String>;
}

/// In-memory multi-tier conversation memory store.
#[derive(Debug, Default)]
pub struct InMemoryConversationMemoryStore {
    working_turns: RwLock<BTreeMap<String, Vec<String>>>,
    episodic_events: RwLock<BTreeMap<String, Vec<String>>>,
    semantic_facts: RwLock<BTreeMap<String, BTreeMap<String, String>>>,
}

impl InMemoryConversationMemoryStore {
    pub fn new() -> Self {
        let store = Self::default();
        store.upsert_semantic_fact(
            "default",
            "daily_calorie_target_kcal".to_string(),
            "2200".to_string(),
        );
        store.upsert_semantic_fact(
            "default",
            "daily_protein_target_g".to_string(),
            "165".to_string(),
        );
        store
    }
}

impl ConversationMemoryStore for InMemoryConversationMemoryStore {
    fn append_working_turn(&self, conversation_id: &str, summary: String) {
        self.working_turns
            .write()
            .entry(conversation_id.to_string())
            .or_default()
            .push(summary);
    }

    fn store_episodic_event(&self, user_id: &str, event: String) {
        self.episodic_events
            .write()
            .entry(user_id.to_string())
            .or_default()
            .push(event);
    }

    fn get_user_semantic_profile(&self, user_id: &str) -> BTreeMap<String, String> {
        let guard = self.semantic_facts.read();
        let mut profile = guard.get("default").cloned().unwrap_or_default();
        if let Some(user_facts) = guard.get(user_id) {
            for (k, v) in user_facts {
                profile.insert(k.clone(), v.clone());
            }
        }
        profile
    }

    fn upsert_semantic_fact(&self, user_id: &str, key: String, value: String) {
        self.semantic_facts
            .write()
            .entry(user_id.to_string())
            .or_default()
            .insert(key, value);
    }

    fn get_episodic_events(&self, user_id: &str) -> Vec<String> {
        self.episodic_events
            .read()
            .get(user_id)
            .cloned()
            .unwrap_or_default()
    }
}

/// Shared execution context passed to every `AgentNode` during graph execution.
///
/// Uses `MockLlmModelProvider` by default so testing never consumes external API tokens.
#[derive(Clone)]
pub struct NodeExecutionContext {
    pub rag_cache: Arc<dyn RagNutritionCacheStore>,
    pub calorie_ledger: Arc<dyn CalorieCounterLedgerStore>,
    pub memory_store: Arc<dyn ConversationMemoryStore>,
    pub llm_provider: Arc<dyn LlmModelProvider>,
}

impl Default for NodeExecutionContext {
    fn default() -> Self {
        Self {
            rag_cache: Arc::new(InMemoryRagCacheStore::new()),
            calorie_ledger: Arc::new(InMemoryCalorieLedgerStore::new()),
            memory_store: Arc::new(InMemoryConversationMemoryStore::new()),
            llm_provider: Arc::new(MockLlmModelProvider::new()),
        }
    }
}

impl NodeExecutionContext {
    pub fn new(
        rag_cache: Arc<dyn RagNutritionCacheStore>,
        calorie_ledger: Arc<dyn CalorieCounterLedgerStore>,
        memory_store: Arc<dyn ConversationMemoryStore>,
    ) -> Self {
        Self {
            rag_cache,
            calorie_ledger,
            memory_store,
            llm_provider: Arc::new(MockLlmModelProvider::new()),
        }
    }

    pub fn with_llm_provider(mut self, llm_provider: Arc<dyn LlmModelProvider>) -> Self {
        self.llm_provider = llm_provider;
        self
    }
}

/// Strips Portuguese and Latin diacritics (`áàâã -> a`, `éê -> e`, `í -> i`, `óôõ -> o`, `ú -> u`, `ç -> c`).
pub fn strip_portuguese_diacritics(input: &str) -> String {
    input
        .chars()
        .map(|c| match c {
            'á' | 'à' | 'â' | 'ã' | 'ä' | 'Á' | 'À' | 'Â' | 'Ã' | 'Ä' => 'a',
            'é' | 'è' | 'ê' | 'ë' | 'É' | 'È' | 'Ê' | 'Ë' => 'e',
            'í' | 'ì' | 'î' | 'ï' | 'Í' | 'Ì' | 'Î' | 'Ï' => 'i',
            'ó' | 'ò' | 'ô' | 'õ' | 'ö' | 'Ó' | 'Ò' | 'Ô' | 'Õ' | 'Ö' => 'o',
            'ú' | 'ù' | 'û' | 'ü' | 'Ú' | 'Ù' | 'Û' | 'Ü' => 'u',
            'ç' | 'Ç' => 'c',
            'ñ' | 'Ñ' => 'n',
            other => other,
        })
        .collect()
}

/// Normalizes a food mention (English or Portuguese) into a canonical cache key.
///
/// Strips Portuguese accents, removes English and Portuguese stop words, and canonicalizes
/// common bilingual food synonyms so cache lookups are consistent across phrasings.
pub fn normalize_food_key(raw: &str) -> String {
    let ascii = strip_portuguese_diacritics(raw).to_lowercase();
    let cleaned: String = ascii
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { ' ' })
        .collect();

    let stop_words = [
        // English stop words
        "of", "with", "and", "a", "an", "the", "some", "fresh", "raw", "cooked", "i", "ate",
        "had", "just", "logged", "register", "add", "calories", "in", "about", "how", "many",
        "much", "protein", "carbs", "fat", "is", "are", "should", "eat", "on", "cut", "meal",
        "meals", "table", "script", "instructions", "previous", "ignore", "drop", "delete",
        // Portuguese stop words
        "de", "do", "da", "dos", "das", "com", "e", "ou", "um", "uma", "uns", "umas", "o", "os",
        "as", "no", "na", "nos", "nas", "pro", "pra", "para", "por", "meu", "minha", "eu", "comi",
        "almocei", "jantei", "lanchei", "tomei", "bebi", "ingeri", "acabei", "comer", "adicione",
        "adicionar", "registra", "registrar", "anota", "anotar", "caloria", "calorias", "em",
        "sobre", "quantas", "quantos", "quanto", "qual", "quais", "proteina", "proteinas",
        "carboidrato", "carboidratos", "gordura", "gorduras", "tem", "devo", "posso", "na",
        "dieta", "hoje", "meu", "minha", "refeicao", "almoco", "jantar", "cafe", "manha",
    ];

    let joined = cleaned
        .split_whitespace()
        .filter(|w| !stop_words.contains(w))
        .collect::<Vec<_>>()
        .join("_");

    canonicalize_bilingual_food_alias(&joined)
}

/// Maps common Portuguese and English food synonyms to unified canonical keys so that
/// RAG cache entries are shared seamlessly across Portuguese and English queries.
pub fn canonicalize_bilingual_food_alias(key: &str) -> String {
    match key {
        "peito_frango_grelhado"
        | "frango_grelhado"
        | "peito_frango"
        | "file_frango_grelhado"
        | "file_peito_frango"
        | "grilled_chicken_breast"
        | "chicken_breast" => "grilled_chicken_breast".to_string(),

        "azeite_oliva" | "azeite" | "azeite_extra_virgem" | "olive_oil" | "extra_virgin_olive_oil" => {
            "olive_oil".to_string()
        }

        "batata_doce"
        | "batata_doce_cozida"
        | "batata_doce_assada"
        | "sweet_potato" => "sweet_potato".to_string(),

        "salmao" | "salmao_grelhado" | "salmon" | "grilled_salmon" => "salmon".to_string(),

        "arroz_branco" | "arroz_cozido" | "arroz" | "white_rice" | "rice" => "rice".to_string(),

        "arroz_integral" | "brown_rice" => "brown_rice".to_string(),

        "feijao" | "feijao_carioca" | "feijao_preto" | "feijao_cozido" | "black_beans" | "beans" => {
            "feijao".to_string()
        }

        "ovo" | "ovos" | "ovo_cozido" | "ovo_mexido" | "ovo_frito" | "egg" | "eggs" | "boiled_egg" => {
            "egg".to_string()
        }

        "aveia" | "flocos_aveia" | "farinha_aveia" | "mingau_aveia" | "oats" | "oatmeal" | "rolled_oats" => {
            "oats".to_string()
        }

        "banana" | "banana_prata" | "banana_nanica" => "banana".to_string(),

        "abacate" | "avocado" => "avocado".to_string(),

        "pao_queijo" | "pao_de_queijo" | "cheese_bread" => "pao_de_queijo".to_string(),

        "pao_frances" | "pao_sal" | "french_bread" => "pao_frances".to_string(),

        "tapioca" | "goma_tapioca" | "beiju" => "tapioca".to_string(),

        "acai" | "polpa_acai" | "tigela_acai" => "acai".to_string(),

        "picanha" | "picanha_grelhada" => "picanha".to_string(),

        "carne_moida" | "patinho_moido" | "acem_moido" | "ground_beef" => "ground_beef".to_string(),

        "mandioca" | "aipim" | "macaxeira" | "cassava" => "mandioca".to_string(),

        "pasta_amendoim" | "manteiga_amendoim" | "peanut_butter" => "peanut_butter".to_string(),

        "whey" | "whey_protein" | "proteina_isolada" | "shake_whey" => "whey_protein".to_string(),

        other => other.to_string(),
    }
}

/// Generates a deterministic 16-dimensional unit-normalized embedding vector from text
/// so semantic L2 cache similarity works deterministically without external API keys.
pub fn compute_deterministic_embedding(text: &str) -> Vec<f32> {
    let normalized = normalize_food_key(text);
    let mut vec = vec![0.0_f32; 16];
    for (idx, token) in normalized.split('_').filter(|s| !s.is_empty()).enumerate() {
        for (c_idx, b) in token.bytes().enumerate() {
            let slot = (b as usize + c_idx * 3 + idx * 7) % 16;
            vec[slot] += 1.0;
        }
    }
    let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut vec {
            *v /= norm;
        }
    }
    vec
}

/// Computes cosine similarity between two embedding slices.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}
