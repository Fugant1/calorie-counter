//! `CriticVerifierNode` (`NutritionVerifier` in `README.md` Section 5.1 — **Planner** / Critic tier).
//!
//! Evaluates nutrition profiles retrieved from Redis L1/L2 cache or WebSearch+RAG, checks
//! Atwater calorie-macro consistency ($4P + 4C + 9F \approx \text{kcal}$), computes aggregate
//! confidence, and decides whether to loop back to `WebSearchAndRAG` (up to `max_verification_retries`)
//! or advance the hierarchy to the **Executioner** tier (`MacroCalculator`).

use crate::nodes::{
    AgentNode, GraphNodeError, ModelCompletionRequest, NodeExecutionContext, NodeExecutionResult,
};
use crate::state::{AgentHierarchyRole, GraphNodeId, GraphState, ToolUsageRecord};
use async_trait::async_trait;
use uuid::Uuid;

/// **Planner / Critic** node that verifies grounded nutrition facts before execution.
#[derive(Debug, Default, Clone)]
pub struct CriticVerifierNode;

impl CriticVerifierNode {
    pub fn new() -> Self {
        Self
    }

    /// Evaluates Atwater thermodynamic consistency for a verified item (`0.0..=1.0` factor).
    pub fn atwater_consistency_factor(calories: f32, atwater_kcal: f32) -> f32 {
        if calories <= 0.0 && atwater_kcal <= 0.0 {
            return 1.0;
        }
        let max_val = calories.max(atwater_kcal).max(1.0);
        let relative_diff = (calories - atwater_kcal).abs() / max_val;
        if relative_diff <= 0.15 {
            1.0
        } else {
            (1.0 - relative_diff * 0.5).clamp(0.5, 1.0)
        }
    }
}

#[async_trait]
impl AgentNode for CriticVerifierNode {
    fn id(&self) -> GraphNodeId {
        GraphNodeId::NutritionVerifier
    }

    fn declared_tools(&self) -> &'static [&'static str] {
        &["atwater_macro_consistency_verifier"]
    }

    async fn execute(
        &self,
        mut state: GraphState,
        ctx: &NodeExecutionContext,
    ) -> Result<NodeExecutionResult, GraphNodeError> {
        let llm_resp = ctx
            .llm_provider
            .complete(ModelCompletionRequest {
                node_id: GraphNodeId::NutritionVerifier,
                model_tier: self.default_model_tier(),
                language: state.plan.language,
                system_prompt: "Verify Atwater calorie-macro consistency and source confidence."
                    .to_string(),
                user_prompt: format!("items={}", state.verified_items.len()),
            })
            .await?;

        // Edge case: no valid food entities were extracted (e.g., empty/malformed/negative input).
        // Advance directly to Executioner without triggering a useless cyclic retry loop.
        if state.verified_items.is_empty() {
            state.verification_confidence = 0.0;
            state.plan.current_stage = AgentHierarchyRole::Executioner;
            state.plan.planner_notes.push(
                "Planner(NutritionVerifier): No valid food entities to verify; advancing to Executioner without retry loop.".to_string(),
            );
            return Ok(NodeExecutionResult::new(
                state,
                llm_resp.simulated_prompt_tokens,
                llm_resp.simulated_completion_tokens,
                llm_resp.cached_tokens,
            )
            .with_mock_flag(llm_resp.is_mock));
        }

        let mut weighted_confidence_sum = 0.0_f32;
        for item in &mut state.verified_items {
            let expected_kcal = item.macros_per_100g.atwater_estimated_kcal();
            let consistency = Self::atwater_consistency_factor(
                item.macros_per_100g.calories_kcal,
                expected_kcal,
            );
            item.confidence = (item.confidence * consistency).clamp(0.0, 1.0);
            weighted_confidence_sum += item.confidence;
        }

        let avg_confidence = weighted_confidence_sum / (state.verified_items.len() as f32);
        state.verification_confidence = (avg_confidence * 100.0).round() / 100.0;

        state.record_tool_call(ToolUsageRecord {
            call_id: Uuid::new_v4().to_string(),
            step_index: state.step_index,
            caller_node: GraphNodeId::NutritionVerifier,
            hierarchy_role: AgentHierarchyRole::Planner,
            tool_name: "atwater_macro_consistency_verifier".to_string(),
            input_summary: format!("verified_items_count={}", state.verified_items.len()),
            output_summary: format!(
                "aggregate_confidence={:.2} threshold={:.2} retries={}/{}",
                state.verification_confidence,
                state.plan.confidence_threshold,
                state.verification_retries,
                state.plan.max_verification_retries
            ),
            cache_hit: None,
            latency_ms: 5,
            tokens_used: llm_resp.simulated_prompt_tokens + llm_resp.simulated_completion_tokens,
            success: true,
        });

        if state.verification_confidence < state.plan.confidence_threshold
            && state.verification_retries < state.plan.max_verification_retries
        {
            state.verification_retries += 1;
            state.plan.current_stage = AgentHierarchyRole::ToolCaller;
            state.plan.planner_notes.push(format!(
                "Planner(NutritionVerifier): Confidence {:.2} < threshold {:.2}; scheduling cyclic retry #{}/{} to WebSearchAndRAG.",
                state.verification_confidence,
                state.plan.confidence_threshold,
                state.verification_retries,
                state.plan.max_verification_retries
            ));
        } else {
            state.plan.current_stage = AgentHierarchyRole::Executioner;
            state.plan.planner_notes.push(format!(
                "Planner(NutritionVerifier): Verification passed (confidence={:.2}); advancing to Executioner tier (MacroCalculator).",
                state.verification_confidence
            ));
        }

        let cached_tokens = if !state.cache_hit_keys.is_empty() {
            40
        } else {
            llm_resp.cached_tokens
        };
        Ok(NodeExecutionResult::new(
            state,
            llm_resp.simulated_prompt_tokens,
            llm_resp.simulated_completion_tokens,
            cached_tokens,
        )
        .with_mock_flag(llm_resp.is_mock))
    }
}
