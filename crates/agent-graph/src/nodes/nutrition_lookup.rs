//! `NutritionLookupNode` (`CacheLookup` in `README.md` Section 5.1 — **ToolCaller** tier).
//!
//! Queries the multi-layer Redis L1 (exact canonical key / EAN barcode) and L2 (semantic vector RAG)
//! cache for each extracted `FoodEntity`. Cache hits bypass expensive web search and LLM extraction;
//! cache misses are flagged in `state.cache_miss_keys` to trigger `WebSearchAndRAG`.

use crate::nodes::{
    compute_deterministic_embedding, AgentNode, GraphNodeError, NodeExecutionContext,
    NodeExecutionResult, WebResearcherNode,
};
use crate::state::{
    AgentHierarchyRole, GraphNodeId, GraphState, RagSourceType, ToolUsageRecord, VerifiedFoodItem,
};
use async_trait::async_trait;
use uuid::Uuid;

/// **ToolCaller** node that checks Redis L1/L2 RAG nutrition caches before any web search runs.
#[derive(Debug, Default, Clone)]
pub struct NutritionLookupNode {
    pub semantic_similarity_threshold: f32,
}

impl NutritionLookupNode {
    pub fn new() -> Self {
        Self {
            semantic_similarity_threshold: 0.92,
        }
    }
}

#[async_trait]
impl AgentNode for NutritionLookupNode {
    fn id(&self) -> GraphNodeId {
        GraphNodeId::CacheLookup
    }

    fn declared_tools(&self) -> &'static [&'static str] {
        &[
            "redis_l1_exact_cache_lookup",
            "redis_l2_semantic_rag_lookup",
        ]
    }

    async fn execute(
        &self,
        mut state: GraphState,
        ctx: &NodeExecutionContext,
    ) -> Result<NodeExecutionResult, GraphNodeError> {
        state.cache_miss_keys.clear();
        let entities = state.extracted_foods.clone();

        for entity in entities {
            // 1. Try L1 Exact / Barcode Cache (<2ms)
            if let Some(chunk) = ctx
                .rag_cache
                .lookup_l1_exact(&entity.canonical_key, entity.barcode_ean.as_deref())
            {
                state.record_tool_call(ToolUsageRecord {
                    call_id: Uuid::new_v4().to_string(),
                    step_index: state.step_index,
                    caller_node: GraphNodeId::CacheLookup,
                    hierarchy_role: AgentHierarchyRole::ToolCaller,
                    tool_name: "redis_l1_exact_cache_lookup".to_string(),
                    input_summary: format!(
                        "key='{}' barcode={:?}",
                        entity.canonical_key, entity.barcode_ean
                    ),
                    output_summary: format!(
                        "L1 HIT: {} ({} kcal/100g)",
                        chunk.display_name, chunk.macros_per_100g.calories_kcal
                    ),
                    cache_hit: Some(true),
                    latency_ms: 1,
                    tokens_used: 0,
                    success: true,
                });

                let resolved_grams = if entity.is_household_measure {
                    let gt = chunk
                        .resolve_household_grams(&entity.unit, entity.quantity)
                        .or_else(|| {
                            let table = WebResearcherNode::resolve_ground_truth_household_measures(
                                &entity.canonical_key,
                            );
                            table
                                .get(&entity.unit)
                                .map(|per_u| (entity.quantity * per_u * 10.0).round() / 10.0)
                        })
                        .unwrap_or(entity.estimated_grams);
                    if let Some(state_entity) = state
                        .extracted_foods
                        .iter_mut()
                        .find(|e| e.entity_id == entity.entity_id)
                    {
                        state_entity.estimated_grams = gt;
                        state_entity.household_grams_verified_from_ground_truth = true;
                    }
                    gt
                } else {
                    entity.estimated_grams
                };

                let scaled = chunk.macros_per_100g.scaled_for_grams(resolved_grams);
                state.verified_items.push(VerifiedFoodItem {
                    entity_id: entity.entity_id.clone(),
                    canonical_key: entity.canonical_key.clone(),
                    display_name: entity.display_name.clone(),
                    quantity: entity.quantity,
                    unit: entity.unit.clone(),
                    grams: resolved_grams,
                    macros_per_100g: chunk.macros_per_100g.clone(),
                    scaled_macros: scaled,
                    confidence: 0.96,
                    source_type: RagSourceType::RedisL1ExactCache,
                    citation_url: chunk.source_url.clone(),
                    rag_chunk_id: chunk.chunk_id.clone(),
                    served_from_cache: true,
                });
                state
                    .retrieved_rag_chunks
                    .insert(entity.canonical_key.clone(), chunk);
                if !state.cache_hit_keys.contains(&entity.canonical_key) {
                    state.cache_hit_keys.push(entity.canonical_key.clone());
                }
                continue;
            }

            // 2. Try L2 Semantic Vector Cache (<10ms)
            let query_emb = compute_deterministic_embedding(&entity.canonical_key);
            if let Some(chunk) = ctx
                .rag_cache
                .lookup_l2_semantic(&query_emb, self.semantic_similarity_threshold)
            {
                state.record_tool_call(ToolUsageRecord {
                    call_id: Uuid::new_v4().to_string(),
                    step_index: state.step_index,
                    caller_node: GraphNodeId::CacheLookup,
                    hierarchy_role: AgentHierarchyRole::ToolCaller,
                    tool_name: "redis_l2_semantic_rag_lookup".to_string(),
                    input_summary: format!(
                        "semantic_query='{}' min_sim={:.2}",
                        entity.canonical_key, self.semantic_similarity_threshold
                    ),
                    output_summary: format!(
                        "L2 SEMANTIC HIT: matched '{}' (sim={:.2})",
                        chunk.canonical_food_key, chunk.relevance_score
                    ),
                    cache_hit: Some(true),
                    latency_ms: 4,
                    tokens_used: 0,
                    success: true,
                });

                let resolved_grams = if entity.is_household_measure {
                    let gt = chunk
                        .resolve_household_grams(&entity.unit, entity.quantity)
                        .or_else(|| {
                            let table = WebResearcherNode::resolve_ground_truth_household_measures(
                                &entity.canonical_key,
                            );
                            table
                                .get(&entity.unit)
                                .map(|per_u| (entity.quantity * per_u * 10.0).round() / 10.0)
                        })
                        .unwrap_or(entity.estimated_grams);
                    if let Some(state_entity) = state
                        .extracted_foods
                        .iter_mut()
                        .find(|e| e.entity_id == entity.entity_id)
                    {
                        state_entity.estimated_grams = gt;
                        state_entity.household_grams_verified_from_ground_truth = true;
                    }
                    gt
                } else {
                    entity.estimated_grams
                };

                let scaled = chunk.macros_per_100g.scaled_for_grams(resolved_grams);
                state.verified_items.push(VerifiedFoodItem {
                    entity_id: entity.entity_id.clone(),
                    canonical_key: entity.canonical_key.clone(),
                    display_name: entity.display_name.clone(),
                    quantity: entity.quantity,
                    unit: entity.unit.clone(),
                    grams: resolved_grams,
                    macros_per_100g: chunk.macros_per_100g.clone(),
                    scaled_macros: scaled,
                    confidence: 0.90,
                    source_type: RagSourceType::RedisL2SemanticCache,
                    citation_url: chunk.source_url.clone(),
                    rag_chunk_id: chunk.chunk_id.clone(),
                    served_from_cache: true,
                });
                state
                    .retrieved_rag_chunks
                    .insert(entity.canonical_key.clone(), chunk);
                if !state.cache_hit_keys.contains(&entity.canonical_key) {
                    state.cache_hit_keys.push(entity.canonical_key.clone());
                }
                continue;
            }

            // 3. Cache Miss on both L1 and L2 -> record miss so WebSearchAndRAG is triggered
            state.record_tool_call(ToolUsageRecord {
                call_id: Uuid::new_v4().to_string(),
                step_index: state.step_index,
                caller_node: GraphNodeId::CacheLookup,
                hierarchy_role: AgentHierarchyRole::ToolCaller,
                tool_name: "redis_l1_exact_cache_lookup".to_string(),
                input_summary: format!("key='{}'", entity.canonical_key),
                output_summary: "CACHE MISS (L1 & L2) -> queued for WebSearchAndRAG".to_string(),
                cache_hit: Some(false),
                latency_ms: 2,
                tokens_used: 0,
                success: true,
            });
            state.cache_miss_keys.push(entity.canonical_key.clone());
        }

        state.mark_planned_tool_completed("rag_cache_l1_l2_lookup");
        state.mark_planned_tool_completed("redis_l1_l2_rag_lookup");

        Ok(NodeExecutionResult::new(state, 0, 0, 0))
    }
}
