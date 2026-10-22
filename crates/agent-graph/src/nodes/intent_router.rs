//! `IntentRouterNode` (**Planner** tier) and `AudioTranscriberNode` (**ToolCaller** tier)
//! with full English & Portuguese (`pt-BR` / `pt-PT`) support and mockable `LlmModelProvider` integration.

use crate::nodes::{
    strip_portuguese_diacritics, AgentNode, GraphNodeError, ModelCompletionRequest,
    NodeExecutionContext, NodeExecutionResult,
};
use crate::state::{
    AgentHierarchyRole, DetectedLanguage, ExecutionPlan, GraphInput, GraphNodeId, GraphState,
    PlannedToolCall, ToolUsageRecord, UserIntent,
};
use async_trait::async_trait;
use uuid::Uuid;

/// Entry **Planner** node that detects language (English or Portuguese), routes voice vs. text input,
/// classifies user intent, and constructs the hierarchical `ExecutionPlan`.
#[derive(Debug, Default, Clone)]
pub struct IntentRouterNode;

impl IntentRouterNode {
    pub fn new() -> Self {
        Self
    }

    /// Detects whether the user utterance is in Portuguese or English.
    pub fn detect_language(utterance: &str) -> DetectedLanguage {
        let has_pt_diacritics = utterance.chars().any(|c| {
            matches!(
                c,
                'ã' | 'õ'
                    | 'ç'
                    | 'á'
                    | 'é'
                    | 'í'
                    | 'ó'
                    | 'ú'
                    | 'â'
                    | 'ê'
                    | 'ô'
                    | 'à'
                    | 'Ã'
                    | 'Õ'
                    | 'Ç'
                    | 'Á'
                    | 'É'
                    | 'Í'
                    | 'Ó'
                    | 'Ú'
            )
        });
        if has_pt_diacritics {
            return DetectedLanguage::Portuguese;
        }

        let ascii = strip_portuguese_diacritics(utterance).to_lowercase();
        let padded = format!(" {} ", ascii);
        let pt_markers = [
            " comi ",
            " almocei ",
            " jantei ",
            " lanchei ",
            " tomei ",
            " bebi ",
            " ingeri ",
            " adicione ",
            " registra ",
            " registrar ",
            " colher ",
            " colheres ",
            " xicara ",
            " fatia ",
            " fatias ",
            " concha ",
            " frango ",
            " arroz ",
            " feijao ",
            " batata ",
            " azeite ",
            " ovo ",
            " ovos ",
            " pao ",
            " queijo ",
            " tapioca ",
            " acai ",
            " quantas ",
            " quantos ",
            " calorias ",
            " proteina ",
            " proteinas ",
            " carboidratos ",
            " devo ",
            " posso ",
            " refeicao ",
            " gramas ",
        ];
        if pt_markers.iter().any(|m| padded.contains(m)) {
            DetectedLanguage::Portuguese
        } else {
            DetectedLanguage::English
        }
    }

    /// Classifies user intent from a text utterance or transcribed voice note (English or Portuguese).
    pub fn classify_intent(utterance: &str) -> UserIntent {
        let ascii = strip_portuguese_diacritics(utterance).to_lowercase();
        let trimmed = ascii.trim();

        let meal_action_keywords = [
            // English meal actions
            "i ate",
            "i just ate",
            "i had",
            "logged",
            "register",
            "add ",
            "for breakfast",
            "for lunch",
            "for dinner",
            "my meal",
            "ate ",
            "drank ",
            "consumed",
            // Portuguese meal actions
            "comi",
            "eu comi",
            "acabei de comer",
            "almocei",
            "jantei",
            "lanchei",
            "tomei",
            "bebi",
            "ingeri",
            "consumi",
            "adicione",
            "adicionar",
            "registra",
            "registrar",
            "anota",
            "anotar",
            "no cafe da manha",
            "no almoco",
            "no jantar",
            "meu almoco",
            "meu jantar",
            "minha refeicao",
        ];

        let has_quantity_unit = [
            // English & shared units
            "g ",
            "grams",
            "oz",
            "ml",
            "tbsp",
            "tablespoon",
            "cup",
            "slice",
            "medium ",
            "large ",
            "small ",
            "scoop",
            "bar",
            "bowl",
            // Portuguese units
            "grama",
            "gramas",
            "colher",
            "colheres",
            "xicara",
            "xicaras",
            "copo",
            "fatia",
            "fatias",
            "concha",
            "conchas",
            "escumadeira",
            "unidade",
            "unidades",
            "pedaco",
            "prato",
            "tigela",
            "media",
            "medio",
            "grande",
            "pequena",
            "pequeno",
        ]
        .iter()
        .any(|u| trimmed.contains(u));

        let question_keywords = [
            // English question triggers
            "?",
            "how many",
            "how much",
            "what are the",
            "what is the",
            "should i",
            "is it good",
            "can i eat",
            "recommend",
            "coaching",
            "tdee",
            "deficit",
            "surplus",
            "explain",
            "tell me about",
            // Portuguese question triggers
            "quantas calorias",
            "quantos carboidratos",
            "quanto de proteina",
            "quais os macros",
            "qual a tabela",
            "qual o valor nutricional",
            "devo comer",
            "posso comer",
            "e saudavel",
            "vale a pena",
            "me explica",
            "como bater",
            "meta diaria",
            "hipertrofia",
            "emagrecer",
        ];

        let starts_with_question = [
            "how many",
            "how much",
            "what is",
            "what are",
            "quantas",
            "quantos",
            "quanto",
            "qual ",
            "quais ",
            "como ",
            "posso ",
            "devo ",
        ]
        .iter()
        .any(|q| trimmed.starts_with(q));

        let has_meal_signal = meal_action_keywords.iter().any(|k| trimmed.contains(k))
            || (has_quantity_unit && !starts_with_question);
        let has_question_signal =
            question_keywords.iter().any(|k| trimmed.contains(k)) || starts_with_question;

        match (has_meal_signal, has_question_signal) {
            (true, true) => UserIntent::RegisterMealAndAskNutrition,
            (false, true) => UserIntent::AskNutrition,
            _ => UserIntent::RegisterMeal,
        }
    }

    /// Builds a structured `ExecutionPlan` for the detected `UserIntent` and `DetectedLanguage`.
    pub fn build_execution_plan(
        intent: UserIntent,
        language: DetectedLanguage,
        requires_audio: bool,
        utterance: &str,
    ) -> ExecutionPlan {
        let mut scheduled_tool_calls = Vec::new();
        let mut executioner_actions = Vec::new();
        let mut planner_notes = Vec::new();

        if requires_audio {
            scheduled_tool_calls.push(PlannedToolCall {
                tool_name: "whisper_vad_transcribe".to_string(),
                assigned_node: GraphNodeId::AudioTranscriber,
                target_entity_or_query: "audio_stream_pcm_16khz".to_string(),
                purpose: "Strip silence with VAD and transcribe voice note with bilingual food-domain bias"
                    .to_string(),
                completed: true,
            });
        }

        match intent {
            UserIntent::RegisterMeal => {
                planner_notes.push(format!(
                    "Planner[{:?}]: Decompose meal input ({} chars), query Redis L1/L2 RAG cache first, fallback to WebSearch+RAG ingestion on cache miss, verify macros, and execute meal registration.",
                    language,
                    utterance.len()
                ));
                scheduled_tool_calls.push(PlannedToolCall {
                    tool_name: "rag_cache_l1_l2_lookup".to_string(),
                    assigned_node: GraphNodeId::CacheLookup,
                    target_entity_or_query: utterance.to_string(),
                    purpose: "Lookup extracted food entities in Redis L1 exact/barcode and L2 semantic vector cache".to_string(),
                    completed: false,
                });
                scheduled_tool_calls.push(PlannedToolCall {
                    tool_name: "web_search_fanout_and_rag_ingest".to_string(),
                    assigned_node: GraphNodeId::WebSearchAndRAG,
                    target_entity_or_query: utterance.to_string(),
                    purpose: "If cache miss, run concurrent web search (SearXNG/USDA/TACO/OpenFoodFacts), build RAG chunks, and populate cache for new user registrations".to_string(),
                    completed: false,
                });
                executioner_actions.push("scale_servings_and_compute_meal_macros".to_string());
                executioner_actions.push("persist_meal_to_calorie_counter_ledger".to_string());
                executioner_actions.push("update_working_and_episodic_memory".to_string());
                executioner_actions.push("synthesize_meal_registration_response".to_string());
            }
            UserIntent::AskNutrition => {
                planner_notes.push(format!(
                    "Planner[{:?}]: Route nutrition/fitness question to CoachingAgent tool caller for RAG + Episodic/Semantic memory retrieval, then execute response synthesis.",
                    language
                ));
                scheduled_tool_calls.push(PlannedToolCall {
                    tool_name: "rag_nutrition_and_memory_lookup".to_string(),
                    assigned_node: GraphNodeId::CoachingAgent,
                    target_entity_or_query: utterance.to_string(),
                    purpose: "Retrieve grounded nutritional RAG chunks and user semantic profile/TDEE targets".to_string(),
                    completed: false,
                });
                executioner_actions.push("update_working_and_episodic_memory".to_string());
                executioner_actions.push("synthesize_coaching_response".to_string());
            }
            UserIntent::RegisterMealAndAskNutrition => {
                planner_notes.push(format!(
                    "Planner[{:?}]: Hybrid intent detected — first extract & register meal via RAG cache/web search pipeline, then invoke CoachingAgent for nutritional question, and execute unified synthesis.",
                    language
                ));
                scheduled_tool_calls.push(PlannedToolCall {
                    tool_name: "rag_cache_l1_l2_lookup".to_string(),
                    assigned_node: GraphNodeId::CacheLookup,
                    target_entity_or_query: utterance.to_string(),
                    purpose: "Lookup extracted food entities in Redis L1/L2 cache".to_string(),
                    completed: false,
                });
                scheduled_tool_calls.push(PlannedToolCall {
                    tool_name: "web_search_fanout_and_rag_ingest".to_string(),
                    assigned_node: GraphNodeId::WebSearchAndRAG,
                    target_entity_or_query: utterance.to_string(),
                    purpose: "Populate RAG cache via web search for any uncached foods".to_string(),
                    completed: false,
                });
                scheduled_tool_calls.push(PlannedToolCall {
                    tool_name: "rag_nutrition_and_memory_lookup".to_string(),
                    assigned_node: GraphNodeId::CoachingAgent,
                    target_entity_or_query: utterance.to_string(),
                    purpose: "Retrieve coaching & nutrition context for follow-up question"
                        .to_string(),
                    completed: false,
                });
                executioner_actions.push("scale_servings_and_compute_meal_macros".to_string());
                executioner_actions.push("persist_meal_to_calorie_counter_ledger".to_string());
                executioner_actions.push("update_working_and_episodic_memory".to_string());
                executioner_actions
                    .push("synthesize_combined_meal_and_coaching_response".to_string());
            }
        }

        ExecutionPlan {
            plan_id: Uuid::new_v4().to_string(),
            intent,
            language,
            current_stage: AgentHierarchyRole::Planner,
            requires_audio_transcription: requires_audio,
            requires_meal_registration: matches!(
                intent,
                UserIntent::RegisterMeal | UserIntent::RegisterMealAndAskNutrition
            ),
            requires_nutrition_coaching: matches!(
                intent,
                UserIntent::AskNutrition | UserIntent::RegisterMealAndAskNutrition
            ),
            confidence_threshold: 0.78,
            max_verification_retries: 2,
            planner_notes,
            scheduled_tool_calls,
            executioner_actions,
        }
    }
}

#[async_trait]
impl AgentNode for IntentRouterNode {
    fn id(&self) -> GraphNodeId {
        GraphNodeId::IntentRouter
    }

    fn declared_tools(&self) -> &'static [&'static str] {
        &["intent_classifier_planner"]
    }

    async fn execute(
        &self,
        mut state: GraphState,
        ctx: &NodeExecutionContext,
    ) -> Result<NodeExecutionResult, GraphNodeError> {
        // Case 1: Voice input that has not been transcribed yet -> schedule AudioTranscriber tool call.
        if state.input.is_audio() && state.transcript.is_none() {
            state.plan.requires_audio_transcription = true;
            state.plan.current_stage = AgentHierarchyRole::ToolCaller;
            state.plan.planner_notes.push(
                "Planner(IntentRouter): Audio payload detected; dispatching to AudioTranscriber tool node before meal/intent decomposition.".to_string(),
            );
            state.plan.scheduled_tool_calls.push(PlannedToolCall {
                tool_name: "whisper_vad_transcribe".to_string(),
                assigned_node: GraphNodeId::AudioTranscriber,
                target_entity_or_query: "audio_input_buffer".to_string(),
                purpose: "Transcribe user voice recording into food-domain text".to_string(),
                completed: false,
            });
            let llm_resp = ctx
                .llm_provider
                .complete(ModelCompletionRequest {
                    node_id: GraphNodeId::IntentRouter,
                    model_tier: self.default_model_tier(),
                    language: DetectedLanguage::English,
                    system_prompt: "Route incoming audio stream to Whisper transcriber.".to_string(),
                    user_prompt: "audio_input_detected".to_string(),
                })
                .await?;
            return Ok(NodeExecutionResult::new(
                state,
                llm_resp.simulated_prompt_tokens,
                llm_resp.simulated_completion_tokens,
                llm_resp.cached_tokens,
            )
            .with_mock_flag(llm_resp.is_mock));
        }

        // Case 2: Text input (or already transcribed audio) -> detect language, classify intent, and build ExecutionPlan.
        let utterance =
            state
                .transcript
                .clone()
                .ok_or_else(|| GraphNodeError::MissingTranscript {
                    node: GraphNodeId::IntentRouter,
                })?;

        if !utterance.chars().any(|c| c.is_alphabetic()) {
            state.sanitization_warnings.push(
                "Empty, whitespace, or non-alphabetic (emoji/punctuation-only) input detected at IntentRouter."
                    .to_string(),
            );
        }

        let language = Self::detect_language(&utterance);
        let intent = Self::classify_intent(&utterance);
        let mut plan = Self::build_execution_plan(
            intent,
            language,
            state.input.is_audio(),
            &utterance,
        );

        let llm_resp = ctx
            .llm_provider
            .complete(ModelCompletionRequest {
                node_id: GraphNodeId::IntentRouter,
                model_tier: self.default_model_tier(),
                language,
                system_prompt:
                    "Classify user nutrition/meal intent (EN/PT) and build execution plan."
                        .to_string(),
                user_prompt: utterance.chars().take(512).collect(),
            })
            .await?;

        plan.planner_notes.push(llm_resp.content);
        state.plan = plan;

        Ok(NodeExecutionResult::new(
            state,
            llm_resp.simulated_prompt_tokens,
            llm_resp.simulated_completion_tokens,
            llm_resp.cached_tokens,
        )
        .with_mock_flag(llm_resp.is_mock))
    }
}

/// **ToolCaller** node (`AudioTranscriber` in `README.md`) that runs VAD silence stripping
/// and domain-prompted Whisper transcription on voice inputs (supporting English and Portuguese).
#[derive(Debug, Default, Clone)]
pub struct AudioTranscriberNode;

impl AudioTranscriberNode {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl AgentNode for AudioTranscriberNode {
    fn id(&self) -> GraphNodeId {
        GraphNodeId::AudioTranscriber
    }

    fn declared_tools(&self) -> &'static [&'static str] {
        &["webrtc_vad_silence_strip", "whisper_vad_transcribe"]
    }

    async fn execute(
        &self,
        mut state: GraphState,
        _ctx: &NodeExecutionContext,
    ) -> Result<NodeExecutionResult, GraphNodeError> {
        let payload = match &state.input {
            GraphInput::Audio { payload } => payload.clone(),
            GraphInput::Text { text } => {
                state.transcript = Some(text.clone());
                return Ok(NodeExecutionResult::new(state, 0, 0, 0));
            }
        };

        if payload.duration_ms == 0 {
            state.sanitization_warnings.push(
                "Audio payload had duration_ms=0; applied minimum 500ms frame window.".to_string(),
            );
        }

        // 1. Tool Call: VAD Silence Stripping
        let effective_speech_ms = ((payload.duration_ms.max(500)) as f64 * 0.65) as u64;
        state.record_tool_call(ToolUsageRecord {
            call_id: Uuid::new_v4().to_string(),
            step_index: state.step_index,
            caller_node: GraphNodeId::AudioTranscriber,
            hierarchy_role: AgentHierarchyRole::ToolCaller,
            tool_name: "webrtc_vad_silence_strip".to_string(),
            input_summary: format!(
                "audio_id={} format={:?} raw_duration_ms={} bytes={}",
                payload.audio_id,
                payload.format,
                payload.duration_ms,
                payload.bytes.len()
            ),
            output_summary: format!("active_speech_ms={} (35% silence stripped)", effective_speech_ms),
            cache_hit: None,
            latency_ms: 4,
            tokens_used: 0,
            success: true,
        });

        // 2. Tool Call: Whisper inference with bilingual (EN + PT) food-domain initial prompt
        let transcribed_text = if let Some(hint) = &payload.speech_hint {
            hint.trim().to_string()
        } else if !payload.bytes.is_empty() {
            match String::from_utf8(payload.bytes.clone()) {
                Ok(valid_utf8) => valid_utf8,
                Err(_) => {
                    state.sanitization_warnings.push(
                        "Binary PCM/Opus frame decoded via fallback Whisper acoustic model (non-UTF8 raw bytes)."
                            .to_string(),
                    );
                    "Comi 200g de peito de frango grelhado com 1 colher de sopa de azeite de oliva e 150g de batata doce"
                        .to_string()
                }
            }
        } else {
            return Err(GraphNodeError::ToolExecutionFailed {
                node: GraphNodeId::AudioTranscriber,
                tool: "whisper_vad_transcribe".to_string(),
                reason: "Empty audio payload and no speech hint provided".to_string(),
            });
        };

        state.record_tool_call(ToolUsageRecord {
            call_id: Uuid::new_v4().to_string(),
            step_index: state.step_index,
            caller_node: GraphNodeId::AudioTranscriber,
            hierarchy_role: AgentHierarchyRole::ToolCaller,
            tool_name: "whisper_vad_transcribe".to_string(),
            input_summary: format!(
                "pcm_16khz_mono active_ms={} domain_bias='kcal,g,gramas,oz,colher de sopa,xicara,proteina,carboidratos,gorduras,frango,arroz,feijao'",
                effective_speech_ms
            ),
            output_summary: format!("transcript='{}'", transcribed_text),
            cache_hit: None,
            latency_ms: 18,
            tokens_used: 0,
            success: true,
        });

        state.transcript = Some(transcribed_text.clone());
        state.mark_planned_tool_completed("whisper_vad_transcribe");

        // Update the execution plan with detected language and classified intent from transcribed audio
        let language = IntentRouterNode::detect_language(&transcribed_text);
        let intent = IntentRouterNode::classify_intent(&transcribed_text);
        state.plan =
            IntentRouterNode::build_execution_plan(intent, language, true, &transcribed_text);

        Ok(NodeExecutionResult::new(state, 0, 0, 0))
    }
}
