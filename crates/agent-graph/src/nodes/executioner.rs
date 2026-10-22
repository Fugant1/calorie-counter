//! **Executioner** hierarchy tier nodes (`MacroCalculatorNode`, `MemoryUpdaterNode`, `ResponseSynthesizerNode`).
//!
//! Responsibilities:
//! - `MacroCalculatorNode`: Deterministic serving scaling and meal-wide macro/calorie summation (`Tier0Deterministic`).
//! - `MemoryUpdaterNode`: Idempotently commits new `MealRegistration` records to the calorie counter ledger
//!   (preventing duplicate concurrent writes) and updates Working, Episodic, and Semantic conversation memory.
//! - `ResponseSynthesizerNode`: Synthesizes the final grounded user response (English or Portuguese)
//!   following the Planner's `ExecutionPlan`.

use crate::nodes::{
    AgentNode, GraphNodeError, ModelCompletionRequest, NodeExecutionContext, NodeExecutionResult,
};
use crate::state::{
    AgentHierarchyRole, DetectedLanguage, GraphNodeId, GraphState, MacroNutrients,
    MealRegistration, ToolUsageRecord,
};
use async_trait::async_trait;
use uuid::Uuid;

/// **Executioner** node (`MacroCalculator` in `README.md`) that computes deterministic macro
/// and calorie totals across all verified food items.
#[derive(Debug, Default, Clone)]
pub struct MacroCalculatorNode;

impl MacroCalculatorNode {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl AgentNode for MacroCalculatorNode {
    fn id(&self) -> GraphNodeId {
        GraphNodeId::MacroCalculator
    }

    fn declared_tools(&self) -> &'static [&'static str] {
        &["deterministic_macro_aggregator"]
    }

    async fn execute(
        &self,
        mut state: GraphState,
        _ctx: &NodeExecutionContext,
    ) -> Result<NodeExecutionResult, GraphNodeError> {
        let mut total_macros = MacroNutrients::default();
        for item in &state.verified_items {
            total_macros.add_assign(&item.scaled_macros);
        }

        let registration = MealRegistration {
            meal_id: format!("meal-{}", &state.run_id[..8.min(state.run_id.len())]),
            run_id: state.run_id.clone(),
            user_id: state.user_id.clone(),
            conversation_id: state.conversation_id.clone(),
            raw_input_summary: state.transcript.clone().unwrap_or_default(),
            items: state.verified_items.clone(),
            total_macros: total_macros.clone(),
            average_confidence: state.verification_confidence,
            newly_cached_rag_keys: state.newly_populated_rag_cache_keys.clone(),
            cache_hit_keys: state.cache_hit_keys.clone(),
        };

        state.record_tool_call(ToolUsageRecord {
            call_id: Uuid::new_v4().to_string(),
            step_index: state.step_index,
            caller_node: GraphNodeId::MacroCalculator,
            hierarchy_role: AgentHierarchyRole::Executioner,
            tool_name: "deterministic_macro_aggregator".to_string(),
            input_summary: format!("items={}", registration.items.len()),
            output_summary: format!(
                "total_kcal={:.1} P={:.1}g C={:.1}g F={:.1}g",
                total_macros.calories_kcal,
                total_macros.protein_g,
                total_macros.carbs_g,
                total_macros.fat_g
            ),
            cache_hit: None,
            latency_ms: 1,
            tokens_used: 0,
            success: true,
        });

        state.meal_registration = Some(registration);
        state.plan.current_stage = AgentHierarchyRole::Executioner;

        Ok(NodeExecutionResult::new(state, 0, 0, 0))
    }
}

/// **Executioner** node (`MemoryUpdater` in `README.md`) that persists the registered meal
/// into the calorie counter ledger with idempotency protection and updates Working/Episodic/Semantic memory.
#[derive(Debug, Default, Clone)]
pub struct MemoryUpdaterNode;

impl MemoryUpdaterNode {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl AgentNode for MemoryUpdaterNode {
    fn id(&self) -> GraphNodeId {
        GraphNodeId::MemoryUpdater
    }

    fn declared_tools(&self) -> &'static [&'static str] {
        &[
            "calorie_counter_register_meal",
            "memory_working_episodic_persist",
        ]
    }

    async fn execute(
        &self,
        mut state: GraphState,
        ctx: &NodeExecutionContext,
    ) -> Result<NodeExecutionResult, GraphNodeError> {
        if let Some(registration) = state.meal_registration.clone() {
            if !registration.items.is_empty() {
                let inserted = ctx.calorie_ledger.register_meal(registration.clone());

                state.record_tool_call(ToolUsageRecord {
                    call_id: Uuid::new_v4().to_string(),
                    step_index: state.step_index,
                    caller_node: GraphNodeId::MemoryUpdater,
                    hierarchy_role: AgentHierarchyRole::Executioner,
                    tool_name: "calorie_counter_register_meal".to_string(),
                    input_summary: format!(
                        "meal_id={} run_id={} user_id={} items={}",
                        registration.meal_id,
                        registration.run_id,
                        registration.user_id,
                        registration.items.len()
                    ),
                    output_summary: if inserted {
                        format!(
                            "Registered {:.1} kcal in calorie counter ledger",
                            registration.total_macros.calories_kcal
                        )
                    } else {
                        "Idempotent duplicate meal registration skipped (already committed)".to_string()
                    },
                    cache_hit: None,
                    latency_ms: 4,
                    tokens_used: 0,
                    success: true,
                });

                if inserted {
                    let episode_summary = format!(
                        "Meal {}: {:.1} kcal ({:.1}g P, {:.1}g C, {:.1}g F)",
                        registration.meal_id,
                        registration.total_macros.calories_kcal,
                        registration.total_macros.protein_g,
                        registration.total_macros.carbs_g,
                        registration.total_macros.fat_g
                    );
                    ctx.memory_store
                        .store_episodic_event(&state.user_id, episode_summary);
                    state
                        .updated_memory_keys
                        .push(format!("ledger:meal:{}", registration.meal_id));
                }
            }
        }

        let llm_resp = ctx
            .llm_provider
            .complete(ModelCompletionRequest {
                node_id: GraphNodeId::MemoryUpdater,
                model_tier: self.default_model_tier(),
                language: state.plan.language,
                system_prompt: "Compact working turn and extract episodic/semantic facts."
                    .to_string(),
                user_prompt: format!("conv={}", state.conversation_id),
            })
            .await?;

        let working_turn = format!(
            "turn:step={}:intent={:?}:input={}",
            state.step_index,
            state.plan.intent,
            state.transcript.as_deref().unwrap_or("")
        );
        ctx.memory_store
            .append_working_turn(&state.conversation_id, working_turn);
        state
            .updated_memory_keys
            .push(format!("mem:work:{}", state.conversation_id));
        state
            .updated_memory_keys
            .push(format!("mem:episodic:{}", state.user_id));

        state.record_tool_call(ToolUsageRecord {
            call_id: Uuid::new_v4().to_string(),
            step_index: state.step_index,
            caller_node: GraphNodeId::MemoryUpdater,
            hierarchy_role: AgentHierarchyRole::Executioner,
            tool_name: "memory_working_episodic_persist".to_string(),
            input_summary: format!(
                "conv_id={} keys={}",
                state.conversation_id,
                state.updated_memory_keys.len()
            ),
            output_summary: "Updated Working & Episodic memory tiers".to_string(),
            cache_hit: None,
            latency_ms: 3,
            tokens_used: llm_resp.simulated_prompt_tokens + llm_resp.simulated_completion_tokens,
            success: true,
        });

        Ok(NodeExecutionResult::new(
            state,
            llm_resp.simulated_prompt_tokens,
            llm_resp.simulated_completion_tokens,
            15,
        )
        .with_mock_flag(llm_resp.is_mock))
    }
}

/// **Executioner** node (`ResponseSynthesizer` in `README.md`) that formats the final
/// user response (English or Portuguese) combining meal registration totals, RAG cache provenance, and coaching advice.
#[derive(Debug, Default, Clone)]
pub struct ResponseSynthesizerNode;

impl ResponseSynthesizerNode {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl AgentNode for ResponseSynthesizerNode {
    fn id(&self) -> GraphNodeId {
        GraphNodeId::ResponseSynthesizer
    }

    fn declared_tools(&self) -> &'static [&'static str] {
        &["final_response_formatter"]
    }

    async fn execute(
        &self,
        mut state: GraphState,
        ctx: &NodeExecutionContext,
    ) -> Result<NodeExecutionResult, GraphNodeError> {
        let mut sections = Vec::new();

        if let Some(reg) = &state.meal_registration {
            let item_lines: Vec<String> = reg
                .items
                .iter()
                .map(|i| {
                    let origin = if i.served_from_cache {
                        "RAG Cache Hit"
                    } else {
                        "WebSearch -> RAG Cached"
                    };
                    format!(
                        "- {} ({:.0}g): {:.1} kcal | P: {:.1}g, C: {:.1}g, F: {:.1}g [{}]",
                        i.display_name,
                        i.grams,
                        i.scaled_macros.calories_kcal,
                        i.scaled_macros.protein_g,
                        i.scaled_macros.carbs_g,
                        i.scaled_macros.fat_g,
                        origin
                    )
                })
                .collect();

            let daily_totals = ctx.calorie_ledger.user_daily_totals(&state.user_id);
            let title = match state.plan.language {
                DetectedLanguage::Portuguese => "Refeição Registrada / Registered Meal",
                DetectedLanguage::English => "Registered Meal",
            };
            sections.push(format!(
                "{} ({}):\n{}\nTotal Meal: {:.1} kcal (Protein: {:.1}g, Carbs: {:.1}g, Fat: {:.1}g, Fiber: {:.1}g)\nDaily Cumulative Total: {:.1} kcal (Protein: {:.1}g)",
                title,
                reg.meal_id,
                item_lines.join("\n"),
                reg.total_macros.calories_kcal,
                reg.total_macros.protein_g,
                reg.total_macros.carbs_g,
                reg.total_macros.fat_g,
                reg.total_macros.fiber_g,
                daily_totals.calories_kcal,
                daily_totals.protein_g
            ));
        }

        if let Some(coaching) = &state.coaching_notes {
            sections.push(coaching.clone());
        }

        let llm_resp = ctx
            .llm_provider
            .complete(ModelCompletionRequest {
                node_id: GraphNodeId::ResponseSynthesizer,
                model_tier: self.default_model_tier(),
                language: state.plan.language,
                system_prompt: "Synthesize final user-facing response (EN/PT).".to_string(),
                user_prompt: format!("sections={}", sections.len()),
            })
            .await?;

        let response_text = if sections.is_empty() {
            "Request processed by Calorie-Counter AI.".to_string()
        } else {
            sections.join("\n\n")
        };

        state.record_tool_call(ToolUsageRecord {
            call_id: Uuid::new_v4().to_string(),
            step_index: state.step_index,
            caller_node: GraphNodeId::ResponseSynthesizer,
            hierarchy_role: AgentHierarchyRole::Executioner,
            tool_name: "final_response_formatter".to_string(),
            input_summary: format!("sections={}", sections.len()),
            output_summary: format!("synthesized {} chars", response_text.len()),
            cache_hit: None,
            latency_ms: 4,
            tokens_used: llm_resp.simulated_prompt_tokens + llm_resp.simulated_completion_tokens,
            success: true,
        });

        state.final_response = Some(response_text);

        Ok(NodeExecutionResult::new(
            state,
            llm_resp.simulated_prompt_tokens,
            llm_resp.simulated_completion_tokens,
            25,
        )
        .with_mock_flag(llm_resp.is_mock))
    }
}
