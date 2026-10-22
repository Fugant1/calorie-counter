//! `FitnessCoachNode` (`CoachingAgent` in `README.md` Section 5.1 — **ToolCaller** tier)
//! with English and Portuguese coaching synthesis and `LlmModelProvider` integration.

use crate::nodes::{
    compute_deterministic_embedding, meal_parser::MealParserNode,
    web_researcher::WebResearcherNode, AgentNode, GraphNodeError, ModelCompletionRequest,
    NodeExecutionContext, NodeExecutionResult,
};
use crate::state::{
    AgentHierarchyRole, DetectedLanguage, GraphNodeId, GraphState, RagDocumentChunk,
    ToolUsageRecord,
};
use async_trait::async_trait;
use uuid::Uuid;

/// **ToolCaller** node that retrieves user semantic/episodic context and grounded RAG
/// nutrition facts to answer nutrition and fitness coaching inquiries (EN & PT).
#[derive(Debug, Default, Clone)]
pub struct FitnessCoachNode;

impl FitnessCoachNode {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl AgentNode for FitnessCoachNode {
    fn id(&self) -> GraphNodeId {
        GraphNodeId::CoachingAgent
    }

    fn declared_tools(&self) -> &'static [&'static str] {
        &[
            "semantic_and_episodic_memory_retrieve",
            "hybrid_rag_nutrition_search",
        ]
    }

    async fn execute(
        &self,
        mut state: GraphState,
        ctx: &NodeExecutionContext,
    ) -> Result<NodeExecutionResult, GraphNodeError> {
        let query =
            state
                .transcript
                .clone()
                .ok_or_else(|| GraphNodeError::MissingTranscript {
                    node: GraphNodeId::CoachingAgent,
                })?;

        // 1. Tool Call: Retrieve user Semantic Profile, Episodic History & Daily Calorie Ledger
        let profile = ctx.memory_store.get_user_semantic_profile(&state.user_id);
        let daily_totals = ctx.calorie_ledger.user_daily_totals(&state.user_id);
        let target_kcal = profile
            .get("daily_calorie_target_kcal")
            .cloned()
            .unwrap_or_else(|| "2200".to_string());
        let target_protein = profile
            .get("daily_protein_target_g")
            .cloned()
            .unwrap_or_else(|| "165".to_string());

        state.record_tool_call(ToolUsageRecord {
            call_id: Uuid::new_v4().to_string(),
            step_index: state.step_index,
            caller_node: GraphNodeId::CoachingAgent,
            hierarchy_role: AgentHierarchyRole::ToolCaller,
            tool_name: "semantic_and_episodic_memory_retrieve".to_string(),
            input_summary: format!("user_id='{}'", state.user_id),
            output_summary: format!(
                "targets=({} kcal, {}g protein), logged_today=({} kcal, {}g protein)",
                target_kcal, target_protein, daily_totals.calories_kcal, daily_totals.protein_g
            ),
            cache_hit: Some(true),
            latency_ms: 3,
            tokens_used: 45,
            success: true,
        });

        // 2. Tool Call: Search Hybrid RAG for any food items referenced in the nutrition question
        let rag_hits = ctx.rag_cache.search_hybrid_rag(&query, 3);
        let mut grounded_facts = Vec::new();
        let mut rag_cache_hit = !rag_hits.is_empty();

        for hit in &rag_hits {
            if hit.relevance_score >= 0.45 {
                grounded_facts.push(format!(
                    "{}: {} kcal, {:.1}g protein, {:.1}g carbs, {:.1}g fat per 100g ({})",
                    hit.display_name,
                    hit.macros_per_100g.calories_kcal,
                    hit.macros_per_100g.protein_g,
                    hit.macros_per_100g.carbs_g,
                    hit.macros_per_100g.fat_g,
                    hit.source_url
                ));
                state
                    .retrieved_rag_chunks
                    .insert(hit.canonical_food_key.clone(), hit.clone());
            }
        }

        // If the user asked about a food not yet in the RAG cache, resolve via web nutrition & cache it
        if grounded_facts.is_empty() {
            rag_cache_hit = false;
            let candidate_entities = MealParserNode::parse_food_entities(&query);
            for entity in candidate_entities {
                let (macros_100g, source_type, source_url) =
                    WebResearcherNode::resolve_web_nutrition(&entity, 0);
                let chunk = RagDocumentChunk {
                    chunk_id: format!("rag-coach-{}", &Uuid::new_v4().to_string()[..8]),
                    canonical_food_key: entity.canonical_key.clone(),
                    display_name: entity.display_name.clone(),
                    barcode_ean: None,
                    source_type,
                    source_url: source_url.clone(),
                    content_markdown: format!(
                        "{}: {} kcal/100g",
                        entity.display_name, macros_100g.calories_kcal
                    ),
                    macros_per_100g: macros_100g.clone(),
                    embedding: compute_deterministic_embedding(&entity.canonical_key),
                    relevance_score: 0.90,
                };
                ctx.rag_cache.ingest_and_cache(chunk.clone());
                if !state
                    .newly_populated_rag_cache_keys
                    .contains(&entity.canonical_key)
                {
                    state
                        .newly_populated_rag_cache_keys
                        .push(entity.canonical_key.clone());
                }
                grounded_facts.push(format!(
                    "{}: {} kcal, {:.1}g protein, {:.1}g carbs, {:.1}g fat per 100g ({})",
                    entity.display_name,
                    macros_100g.calories_kcal,
                    macros_100g.protein_g,
                    macros_100g.carbs_g,
                    macros_100g.fat_g,
                    source_url
                ));
                state
                    .retrieved_rag_chunks
                    .insert(entity.canonical_key, chunk);
            }
        }

        let llm_resp = ctx
            .llm_provider
            .complete(ModelCompletionRequest {
                node_id: GraphNodeId::CoachingAgent,
                model_tier: self.default_model_tier(),
                language: state.plan.language,
                system_prompt:
                    "Provide grounded nutrition & fitness coaching based on RAG and TDEE targets."
                        .to_string(),
                user_prompt: query.chars().take(512).collect(),
            })
            .await?;

        state.record_tool_call(ToolUsageRecord {
            call_id: Uuid::new_v4().to_string(),
            step_index: state.step_index,
            caller_node: GraphNodeId::CoachingAgent,
            hierarchy_role: AgentHierarchyRole::ToolCaller,
            tool_name: "hybrid_rag_nutrition_search".to_string(),
            input_summary: format!("query='{}'", query),
            output_summary: format!("retrieved {} grounded nutrition facts", grounded_facts.len()),
            cache_hit: Some(rag_cache_hit),
            latency_ms: if rag_cache_hit { 5 } else { 28 },
            tokens_used: llm_resp.simulated_prompt_tokens + llm_resp.simulated_completion_tokens,
            success: true,
        });

        let facts_summary = if grounded_facts.is_empty() {
            match state.plan.language {
                DetectedLanguage::Portuguese => {
                    "Orientação metabólica baseada no perfil de TDEE do usuário.".to_string()
                }
                DetectedLanguage::English => {
                    "General metabolic guidance applied based on user TDEE profile.".to_string()
                }
            }
        } else {
            grounded_facts.join(" | ")
        };

        let header = match state.plan.language {
            DetectedLanguage::Portuguese => "Análise Nutricional & Coaching",
            DetectedLanguage::English => "Nutrition & Coaching Analysis",
        };

        state.coaching_notes = Some(format!(
            "{} (Nutrition & Coaching Analysis): [{}] | Meta/Target: {} kcal ({}g protein) | Consumido Hoje/Logged Today: {:.1} kcal ({:.1}g protein) | {}",
            header,
            facts_summary,
            target_kcal,
            target_protein,
            daily_totals.calories_kcal,
            daily_totals.protein_g,
            llm_resp.content
        ));

        state.mark_planned_tool_completed("rag_nutrition_and_memory_lookup");
        state.plan.current_stage = AgentHierarchyRole::Executioner;

        Ok(NodeExecutionResult::new(
            state,
            llm_resp.simulated_prompt_tokens,
            llm_resp.simulated_completion_tokens,
            30,
        )
        .with_mock_flag(llm_resp.is_mock))
    }
}
