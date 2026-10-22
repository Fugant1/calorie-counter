//! Conditional routing predicates and cyclic retry guards for `agent-graph`.
//!
//! Every node transition is evaluated through typed `GraphEdgeRouter` rules that produce a
//! `DecisionPathStep` audit record for `agent-harness` (tracking source/target nodes,
//! `Planner -> ToolCaller -> Executioner` hierarchy transitions, confidence scores,
//! retry counts, and evaluation latency in microseconds).

use crate::state::{DecisionPathStep, GraphNodeId, GraphState, UserIntent};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Descriptor of a directed edge in the agent graph topology.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphEdgeDescriptor {
    pub from: GraphNodeId,
    pub to: GraphNodeId,
    pub condition_label: String,
    pub is_cyclic_retry: bool,
}

/// Evaluates conditional outgoing edges from `current_node` based on `GraphState`
/// and produces a `DecisionPathStep` for the Harness audit log.
#[derive(Debug, Default, Clone)]
pub struct GraphEdgeRouter;

impl GraphEdgeRouter {
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the next node to transition to from `current_node` given `state`,
    /// returning `(next_node, DecisionPathStep)`.
    pub fn resolve_next_node(
        &self,
        current_node: GraphNodeId,
        state: &GraphState,
    ) -> (GraphNodeId, DecisionPathStep) {
        let start = Instant::now();

        let (to_node, label, rationale, confidence) = match current_node {
            GraphNodeId::IntentRouter => {
                if state.input.is_audio() && state.transcript.is_none() {
                    (
                        GraphNodeId::AudioTranscriber,
                        "Voice Input Detected",
                        "Planner(IntentRouter) detected raw audio input; dispatching to ToolCaller(AudioTranscriber) for VAD + Whisper STT.".to_string(),
                        None,
                    )
                } else {
                    match state.plan.intent {
                        UserIntent::AskNutrition => (
                            GraphNodeId::CoachingAgent,
                            "Fitness / Diet Question",
                            "Planner(IntentRouter) classified intent as AskNutrition; dispatching to ToolCaller(CoachingAgent).".to_string(),
                            None,
                        ),
                        UserIntent::RegisterMeal | UserIntent::RegisterMealAndAskNutrition => (
                            GraphNodeId::MealParser,
                            "Text Meal Log",
                            format!(
                                "Planner(IntentRouter) classified intent as {:?}; advancing to Planner(MealParser) for food entity decomposition.",
                                state.plan.intent
                            ),
                            None,
                        ),
                    }
                }
            }

            GraphNodeId::AudioTranscriber => match state.plan.intent {
                UserIntent::AskNutrition => (
                    GraphNodeId::CoachingAgent,
                    "Transcript Ready (Nutrition Question)",
                    "ToolCaller(AudioTranscriber) completed Whisper transcription for nutrition question; routing to ToolCaller(CoachingAgent).".to_string(),
                    None,
                ),
                UserIntent::RegisterMeal | UserIntent::RegisterMealAndAskNutrition => (
                    GraphNodeId::MealParser,
                    "Transcript Ready",
                    "ToolCaller(AudioTranscriber) completed Whisper transcription; routing to Planner(MealParser) to extract food entities.".to_string(),
                    None,
                ),
            },

            GraphNodeId::MealParser => (
                GraphNodeId::CacheLookup,
                "Extracted Food Entities",
                format!(
                    "Planner(MealParser) extracted {} food entities; dispatching to ToolCaller(CacheLookup) for Redis L1/L2 RAG lookup.",
                    state.extracted_foods.len()
                ),
                None,
            ),

            GraphNodeId::CacheLookup => {
                if state.cache_miss_keys.is_empty() {
                    (
                        GraphNodeId::NutritionVerifier,
                        "Cache Hit (Redis L1/L2)",
                        format!(
                            "ToolCaller(CacheLookup) resolved {} food items from Redis L1/L2 RAG cache (0 cache misses); advancing to Planner(NutritionVerifier).",
                            state.cache_hit_keys.len()
                        ),
                        Some(0.95),
                    )
                } else {
                    (
                        GraphNodeId::WebSearchAndRAG,
                        "Cache Miss / Ambiguous Brand",
                        format!(
                            "ToolCaller(CacheLookup) missed {} food keys {:?} in L1/L2 cache; routing to ToolCaller(WebSearchAndRAG) to scrape web & populate RAG cache.",
                            state.cache_miss_keys.len(),
                            state.cache_miss_keys
                        ),
                        None,
                    )
                }
            }

            GraphNodeId::WebSearchAndRAG => (
                GraphNodeId::NutritionVerifier,
                "Retrieved Grounded Chunks",
                format!(
                    "ToolCaller(WebSearchAndRAG) retrieved and cached {} RAG chunks; routing to Planner(NutritionVerifier) for verification.",
                    state.retrieved_rag_chunks.len()
                ),
                Some(state.verification_confidence),
            ),

            GraphNodeId::NutritionVerifier => {
                // Note: CriticVerifierNode already incremented `verification_retries` if a retry was scheduled
                // and set `plan.current_stage` to `ToolCaller` on retry or `Executioner` on pass.
                if state.plan.current_stage == crate::state::AgentHierarchyRole::ToolCaller
                    && state.verification_confidence < state.plan.confidence_threshold
                {
                    (
                        GraphNodeId::WebSearchAndRAG,
                        "Confidence < Threshold (Retry Edge, Max 2)",
                        format!(
                            "Planner(NutritionVerifier) confidence {:.2} < {:.2}; triggering cyclic retry #{} to ToolCaller(WebSearchAndRAG).",
                            state.verification_confidence,
                            state.plan.confidence_threshold,
                            state.verification_retries
                        ),
                        Some(state.verification_confidence),
                    )
                } else {
                    (
                        GraphNodeId::MacroCalculator,
                        "Verified Serving & Macros",
                        format!(
                            "Planner(NutritionVerifier) verified nutrition (confidence={:.2}); transitioning to Executioner(MacroCalculator).",
                            state.verification_confidence
                        ),
                        Some(state.verification_confidence),
                    )
                }
            }

            GraphNodeId::MacroCalculator => {
                if state.plan.requires_nutrition_coaching && state.coaching_notes.is_none() {
                    (
                        GraphNodeId::CoachingAgent,
                        "Hybrid Meal + Nutrition Question",
                        "Executioner(MacroCalculator) finished meal macro totals; dispatching to ToolCaller(CoachingAgent) for the user's nutrition question.".to_string(),
                        Some(state.verification_confidence),
                    )
                } else {
                    (
                        GraphNodeId::MemoryUpdater,
                        "Persist Meal & Update Daily Totals",
                        "Executioner(MacroCalculator) computed meal macros; advancing to Executioner(MemoryUpdater) to register meal in calorie counter.".to_string(),
                        Some(state.verification_confidence),
                    )
                }
            }

            GraphNodeId::CoachingAgent => (
                GraphNodeId::MemoryUpdater,
                "Retrieve Episodic/Semantic Context",
                "ToolCaller(CoachingAgent) gathered grounded RAG & memory context; advancing to Executioner(MemoryUpdater).".to_string(),
                None,
            ),

            GraphNodeId::MemoryUpdater => (
                GraphNodeId::ResponseSynthesizer,
                "Memory & Ledger Updated",
                "Executioner(MemoryUpdater) persisted meal & memory state; advancing to Executioner(ResponseSynthesizer).".to_string(),
                None,
            ),

            GraphNodeId::ResponseSynthesizer | GraphNodeId::Terminal => (
                GraphNodeId::Terminal,
                "Complete ([*])",
                "Executioner(ResponseSynthesizer) produced final response; terminating graph run.".to_string(),
                Some(state.verification_confidence),
            ),
        };

        let elapsed_us = start.elapsed().as_micros() as u64;
        let decision = DecisionPathStep {
            step_index: state.step_index,
            from_node: current_node,
            to_node,
            from_role: current_node.hierarchy_role(),
            to_role: to_node.hierarchy_role(),
            condition_label: label.to_string(),
            rationale,
            confidence_score: confidence,
            retry_iteration: state.verification_retries,
            evaluation_latency_us: elapsed_us.max(1),
        };

        (to_node, decision)
    }

    /// Returns the static topological edge table matching `README.md` Section 5.1.
    pub fn canonical_edges() -> Vec<GraphEdgeDescriptor> {
        vec![
            GraphEdgeDescriptor {
                from: GraphNodeId::IntentRouter,
                to: GraphNodeId::AudioTranscriber,
                condition_label: "Voice Input Detected".to_string(),
                is_cyclic_retry: false,
            },
            GraphEdgeDescriptor {
                from: GraphNodeId::AudioTranscriber,
                to: GraphNodeId::MealParser,
                condition_label: "Transcript Ready".to_string(),
                is_cyclic_retry: false,
            },
            GraphEdgeDescriptor {
                from: GraphNodeId::AudioTranscriber,
                to: GraphNodeId::CoachingAgent,
                condition_label: "Transcript Ready (Nutrition Question)".to_string(),
                is_cyclic_retry: false,
            },
            GraphEdgeDescriptor {
                from: GraphNodeId::IntentRouter,
                to: GraphNodeId::MealParser,
                condition_label: "Text Meal Log".to_string(),
                is_cyclic_retry: false,
            },
            GraphEdgeDescriptor {
                from: GraphNodeId::IntentRouter,
                to: GraphNodeId::CoachingAgent,
                condition_label: "Fitness / Diet Question".to_string(),
                is_cyclic_retry: false,
            },
            GraphEdgeDescriptor {
                from: GraphNodeId::MealParser,
                to: GraphNodeId::CacheLookup,
                condition_label: "Extracted Food Entities".to_string(),
                is_cyclic_retry: false,
            },
            GraphEdgeDescriptor {
                from: GraphNodeId::CacheLookup,
                to: GraphNodeId::NutritionVerifier,
                condition_label: "Cache Hit (Redis L1/L2)".to_string(),
                is_cyclic_retry: false,
            },
            GraphEdgeDescriptor {
                from: GraphNodeId::CacheLookup,
                to: GraphNodeId::WebSearchAndRAG,
                condition_label: "Cache Miss / Ambiguous Brand".to_string(),
                is_cyclic_retry: false,
            },
            GraphEdgeDescriptor {
                from: GraphNodeId::WebSearchAndRAG,
                to: GraphNodeId::NutritionVerifier,
                condition_label: "Retrieved Grounded Chunks".to_string(),
                is_cyclic_retry: false,
            },
            GraphEdgeDescriptor {
                from: GraphNodeId::NutritionVerifier,
                to: GraphNodeId::WebSearchAndRAG,
                condition_label: "Confidence < Threshold (Retry Edge, Max 2)".to_string(),
                is_cyclic_retry: true,
            },
            GraphEdgeDescriptor {
                from: GraphNodeId::NutritionVerifier,
                to: GraphNodeId::MacroCalculator,
                condition_label: "Verified Serving & Macros".to_string(),
                is_cyclic_retry: false,
            },
            GraphEdgeDescriptor {
                from: GraphNodeId::MacroCalculator,
                to: GraphNodeId::MemoryUpdater,
                condition_label: "Persist Meal & Update Daily Totals".to_string(),
                is_cyclic_retry: false,
            },
            GraphEdgeDescriptor {
                from: GraphNodeId::MacroCalculator,
                to: GraphNodeId::CoachingAgent,
                condition_label: "Hybrid Meal + Nutrition Question".to_string(),
                is_cyclic_retry: false,
            },
            GraphEdgeDescriptor {
                from: GraphNodeId::CoachingAgent,
                to: GraphNodeId::MemoryUpdater,
                condition_label: "Retrieve Episodic/Semantic Context".to_string(),
                is_cyclic_retry: false,
            },
            GraphEdgeDescriptor {
                from: GraphNodeId::MemoryUpdater,
                to: GraphNodeId::ResponseSynthesizer,
                condition_label: "Memory & Ledger Updated".to_string(),
                is_cyclic_retry: false,
            },
            GraphEdgeDescriptor {
                from: GraphNodeId::ResponseSynthesizer,
                to: GraphNodeId::Terminal,
                condition_label: "Complete ([*])".to_string(),
                is_cyclic_retry: false,
            },
        ]
    }
}
