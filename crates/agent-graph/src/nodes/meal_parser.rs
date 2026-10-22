//! `MealParserNode` (**Planner** / Decomposition tier) with full English & Portuguese (`pt-BR` / `pt-PT`)
//! entity extraction, unit conversion, and sanitization against weird/adversarial inputs.

use crate::nodes::{
    normalize_food_key, strip_portuguese_diacritics, AgentNode, GraphNodeError,
    ModelCompletionRequest, NodeExecutionContext, NodeExecutionResult,
};
use crate::state::{
    AgentHierarchyRole, FoodEntity, GraphNodeId, GraphState, PlannedToolCall, ToolUsageRecord,
};
use async_trait::async_trait;
use uuid::Uuid;

/// Maximum number of food items allowed in a single meal turn to guard against spam/flood inputs.
pub const MAX_ENTITIES_PER_MEAL: usize = 25;

/// Maximum gram weight allowed for a single food item before clamping.
pub const MAX_GRAMS_PER_ITEM: f32 = 5_000.0;

/// **Planner** node that parses raw meal descriptions (English or Portuguese) into structured
/// `FoodEntity` records, sanitizes weird inputs, and schedules cache/RAG tool calls.
#[derive(Debug, Default, Clone)]
pub struct MealParserNode;

impl MealParserNode {
    pub fn new() -> Self {
        Self
    }

    /// Parses a natural-language meal log (English or Portuguese) into structured `FoodEntity` items
    /// along with any sanitization warnings triggered by weird, negative, or adversarial inputs.
    pub fn parse_food_entities_with_warnings(transcript: &str) -> (Vec<FoodEntity>, Vec<String>) {
        let mut warnings = Vec::new();
        let sanitized = Self::sanitize_raw_input(transcript, &mut warnings);

        if sanitized.trim().is_empty() {
            warnings.push(
                "Input contained no parseable text after sanitization.".to_string(),
            );
            return (Vec::new(), warnings);
        }

        // Strip trailing question sentences if the user combined a meal log with a question (EN or PT)
        let meal_clause = sanitized
            .split('?')
            .next()
            .unwrap_or(&sanitized)
            .split(". How")
            .next()
            .unwrap_or(&sanitized)
            .split(". What")
            .next()
            .unwrap_or(&sanitized)
            .split(". Quant")
            .next()
            .unwrap_or(&sanitized)
            .split(". Qual")
            .next()
            .unwrap_or(&sanitized)
            .split(". Posso")
            .next()
            .unwrap_or(&sanitized)
            .split(". Devo")
            .next()
            .unwrap_or(&sanitized);

        // Pre-normalize compound Portuguese units before splitting on connectors
        let ascii_clause = strip_portuguese_diacritics(meal_clause);
        let pre_normalized = Self::normalize_compound_portuguese_phrases(&ascii_clause);

        // Normalize bilingual delimiters into commas for clause segmentation
        let normalized_delimiters = pre_normalized
            .replace(" accompanied by ", ", ")
            .replace(" acompanhado de ", ", ")
            .replace(" junto com ", ", ")
            .replace(" with ", ", ")
            .replace(" and ", ", ")
            .replace(" plus ", ", ")
            .replace(" com ", ", ")
            .replace(" mais ", ", ")
            .replace(" e ", ", ")
            .replace(" & ", ", ")
            .replace(';', ", ");

        let mut entities = Vec::new();
        for (idx, raw_segment) in normalized_delimiters.split(',').enumerate() {
            let trimmed = raw_segment.trim().trim_end_matches('.');
            if trimmed.is_empty() {
                continue;
            }
            if entities.len() >= MAX_ENTITIES_PER_MEAL {
                warnings.push(format!(
                    "Truncated meal entities at MAX_ENTITIES_PER_MEAL ({})",
                    MAX_ENTITIES_PER_MEAL
                ));
                break;
            }
            if let Some(entity) = Self::parse_single_segment(trimmed, idx, &mut warnings) {
                entities.push(entity);
            }
        }

        if entities.is_empty() && warnings.is_empty() {
            warnings.push(
                "No recognizable food entities could be extracted from input.".to_string(),
            );
        }

        (entities, warnings)
    }

    /// Backwards-compatible helper returning only the parsed `Vec<FoodEntity>`.
    pub fn parse_food_entities(transcript: &str) -> Vec<FoodEntity> {
        Self::parse_food_entities_with_warnings(transcript).0
    }

    /// Strips HTML/script tags, SQL injection keywords, and prompt-injection directives,
    /// recording warnings when suspicious patterns are encountered.
    fn sanitize_raw_input(raw: &str, warnings: &mut Vec<String>) -> String {
        let mut out = String::with_capacity(raw.len());
        let mut in_html_tag = false;
        let mut stripped_html = false;

        for ch in raw.chars() {
            if ch == '<' {
                in_html_tag = true;
                stripped_html = true;
            } else if ch == '>' && in_html_tag {
                in_html_tag = false;
                out.push(' ');
            } else if !in_html_tag && !ch.is_control() {
                out.push(ch);
            } else if ch == '\n' || ch == '\t' {
                out.push(' ');
            }
        }

        if stripped_html {
            warnings.push("Stripped HTML/script tags from user input.".to_string());
        }

        let adversarial_patterns = [
            "drop table",
            "delete from",
            "insert into",
            "ignore previous instructions",
            "ignore all instructions",
            "ignore as instrucoes anteriores",
            "alert(1)",
            "--",
        ];

        let mut lower_check = out.to_lowercase();
        for pat in adversarial_patterns {
            if lower_check.contains(pat) {
                warnings.push(format!(
                    "Sanitized adversarial/injection pattern '{}' from input.",
                    pat
                ));
                // Case-insensitive removal
                let mut rebuilt = String::new();
                let mut rest = out.as_str();
                while let Some(pos) = rest.to_lowercase().find(pat) {
                    rebuilt.push_str(&rest[..pos]);
                    rebuilt.push(' ');
                    rest = &rest[pos + pat.len()..];
                }
                rebuilt.push_str(rest);
                out = rebuilt;
                lower_check = out.to_lowercase();
            }
        }

        out
    }

    /// Replaces multi-word Portuguese units and compound food phrases with safe single tokens
    /// before clause splitting so `"colher de sopa"` or `"pao de queijo"` are kept intact.
    fn normalize_compound_portuguese_phrases(input: &str) -> String {
        let mut s = format!(" {} ", input.to_lowercase());
        let replacements = [
            (" colheres de sopa ", " tbsp "),
            (" colher de sopa ", " tbsp "),
            (" colheres de cha ", " tsp "),
            (" colher de cha ", " tsp "),
            (" xicaras de cha ", " cup "),
            (" xicara de cha ", " cup "),
            (" copos americanos ", " cup "),
            (" copo americano ", " cup "),
            (" no cafe da manha ", " "),
            (" cafe da manha ", " "),
        ];
        for (from, to) in replacements {
            s = s.replace(from, to);
        }
        s.trim().to_string()
    }

    fn parse_single_segment(
        segment: &str,
        idx: usize,
        warnings: &mut Vec<String>,
    ) -> Option<FoodEntity> {
        let lower = strip_portuguese_diacritics(segment).to_lowercase();
        let prefixes = [
            "eu acabei de comer ",
            "acabei de comer ",
            "i just ate ",
            "eu comi ",
            "i ate ",
            "i had ",
            "i drank ",
            "just ate ",
            "comi ",
            "almocei ",
            "jantei ",
            "lanchei ",
            "tomei ",
            "bebi ",
            "ingeri ",
            "consumi ",
            "adicione ",
            "adicionar ",
            "registra ",
            "registrar ",
            "anota ",
            "anotar ",
            "ate ",
            "had ",
            "add ",
            "register ",
            "logged ",
            "for lunch ",
            "for dinner ",
            "for breakfast ",
            "no almoco ",
            "no jantar ",
        ];
        let mut cleaned = lower.trim();
        for prefix in prefixes {
            if let Some(rest) = cleaned.strip_prefix(prefix) {
                cleaned = rest.trim();
            }
        }

        let tokens: Vec<&str> = cleaned.split_whitespace().collect();
        if tokens.is_empty() {
            return None;
        }

        let mut quantity = 1.0_f32;
        let mut unit = "serving".to_string();
        let mut consumed_tokens = 0_usize;

        // Detect leading number or article (English & Portuguese)
        let first = tokens[0];
        if matches!(first, "a" | "an" | "one" | "um" | "uma") {
            quantity = 1.0;
            consumed_tokens = 1;
        } else if matches!(first, "two" | "dois" | "duas") {
            quantity = 2.0;
            consumed_tokens = 1;
        } else if matches!(first, "three" | "tres") {
            quantity = 3.0;
            consumed_tokens = 1;
        } else if matches!(first, "four" | "quatro") {
            quantity = 4.0;
            consumed_tokens = 1;
        } else if matches!(first, "half" | "meio" | "meia") {
            quantity = 0.5;
            consumed_tokens = 1;
        } else if let Some((num, attached_unit)) = Self::split_number_and_unit(first) {
            if !num.is_finite() || num <= 0.0 {
                warnings.push(format!(
                    "Rejected non-positive or invalid quantity ({}) in segment '{}'.",
                    num, segment
                ));
                return None;
            }
            quantity = num;
            consumed_tokens = 1;
            if !attached_unit.is_empty() {
                unit = attached_unit;
            }
        }

        // Check if the next token is a unit (English or Portuguese)
        if consumed_tokens < tokens.len() && unit == "serving" {
            let candidate_unit = tokens[consumed_tokens];
            if Self::is_unit_word(candidate_unit) {
                unit = Self::normalize_unit(candidate_unit);
                consumed_tokens += 1;
            }
        } else if unit != "serving" {
            unit = Self::normalize_unit(&unit);
        }

        // Skip optional "of" / "de" / "do" / "da"
        if consumed_tokens < tokens.len()
            && matches!(tokens[consumed_tokens], "of" | "de" | "do" | "da")
        {
            consumed_tokens += 1;
        }

        let remaining_tokens = &tokens[consumed_tokens..];
        let mut food_words: Vec<&str> = if remaining_tokens.is_empty() {
            tokens.clone()
        } else {
            remaining_tokens.to_vec()
        };

        // Handle trailing Portuguese size adjectives (e.g., "1 batata doce media", "2 bananas pequenas")
        if unit == "serving" && food_words.len() > 1 {
            if let Some(&last_word) = food_words.last() {
                match last_word {
                    "media" | "medio" | "medias" | "medios" => {
                        unit = "medium".to_string();
                        food_words.pop();
                    }
                    "grande" | "grandes" => {
                        unit = "large".to_string();
                        food_words.pop();
                    }
                    "pequena" | "pequeno" | "pequenas" | "pequenos" => {
                        unit = "small".to_string();
                        food_words.pop();
                    }
                    _ => {}
                }
            }
        }

        // Extract optional EAN barcode token (8-13 digits)
        let mut barcode_ean: Option<String> = None;
        let mut name_words = Vec::new();
        for &w in &food_words {
            let stripped = w.trim_matches(|c: char| !c.is_ascii_alphanumeric());
            if stripped.len() >= 8
                && stripped.len() <= 13
                && stripped.chars().all(|c| c.is_ascii_digit())
            {
                barcode_ean = Some(stripped.to_string());
            } else if stripped
                .chars()
                .any(|c| c.is_ascii_alphabetic())
            {
                name_words.push(w);
            }
        }

        let display_name = name_words.join(" ").trim().to_string();
        if display_name.is_empty() {
            return None;
        }

        let canonical_key = normalize_food_key(&display_name);
        if canonical_key.is_empty() {
            return None;
        }

        let mut estimated_grams = Self::estimate_grams(quantity, &unit, &canonical_key);
        if estimated_grams > MAX_GRAMS_PER_ITEM {
            warnings.push(format!(
                "Clamped excessive gram weight ({:.1}g -> {:.1}g) for '{}'.",
                estimated_grams, MAX_GRAMS_PER_ITEM, display_name
            ));
            estimated_grams = MAX_GRAMS_PER_ITEM;
        }

        let preparation = [
            "grilled",
            "fried",
            "baked",
            "boiled",
            "steamed",
            "roasted",
            "raw",
            "grelhado",
            "grelhada",
            "frito",
            "frita",
            "assado",
            "assada",
            "cozido",
            "cozida",
            "refogado",
            "desfiado",
        ]
        .iter()
        .find(|&&prep| display_name.contains(prep))
        .map(|&s| s.to_string());

        Some(FoodEntity {
            entity_id: format!("food-{}-{}", idx + 1, &Uuid::new_v4().to_string()[..8]),
            raw_mention: segment.to_string(),
            canonical_key,
            display_name,
            brand: None,
            barcode_ean,
            quantity,
            unit,
            estimated_grams,
            preparation,
        })
    }

    fn split_number_and_unit(token: &str) -> Option<(f32, String)> {
        let mut num_str = String::new();
        let mut unit_str = String::new();
        for (i, ch) in token.chars().enumerate() {
            if ((ch == '-' && i == 0) || ch.is_ascii_digit() || ch == '.') && unit_str.is_empty() {
                num_str.push(ch);
            } else {
                unit_str.push(ch);
            }
        }
        if num_str.is_empty() || num_str == "-" || num_str == "." {
            return None;
        }
        let val = num_str.parse::<f32>().ok()?;
        Some((val, unit_str))
    }

    fn is_unit_word(word: &str) -> bool {
        matches!(
            word,
            "g" | "gram"
                | "grams"
                | "grama"
                | "gramas"
                | "kg"
                | "oz"
                | "ounce"
                | "ounces"
                | "ml"
                | "tbsp"
                | "tablespoon"
                | "tablespoons"
                | "colher"
                | "colheres"
                | "tsp"
                | "teaspoon"
                | "cup"
                | "cups"
                | "xicara"
                | "xicaras"
                | "copo"
                | "copos"
                | "medium"
                | "media"
                | "medio"
                | "large"
                | "grande"
                | "small"
                | "pequeno"
                | "pequena"
                | "slice"
                | "slices"
                | "fatia"
                | "fatias"
                | "scoop"
                | "scoops"
                | "bowl"
                | "tigela"
                | "prato"
                | "bar"
                | "barra"
                | "concha"
                | "conchas"
                | "escumadeira"
                | "escumadeiras"
                | "unidade"
                | "unidades"
                | "pedaco"
                | "pedacos"
        )
    }

    fn normalize_unit(unit: &str) -> String {
        match unit {
            "g" | "gram" | "grams" | "grama" | "gramas" => "g".to_string(),
            "kg" => "kg".to_string(),
            "oz" | "ounce" | "ounces" => "oz".to_string(),
            "tbsp" | "tablespoon" | "tablespoons" | "colher" | "colheres" => "tbsp".to_string(),
            "tsp" | "teaspoon" => "tsp".to_string(),
            "cup" | "cups" | "xicara" | "xicaras" | "copo" | "copos" => "cup".to_string(),
            "slice" | "slices" | "fatia" | "fatias" => "slice".to_string(),
            "scoop" | "scoops" => "scoop".to_string(),
            "medium" | "media" | "medio" => "medium".to_string(),
            "large" | "grande" => "large".to_string(),
            "small" | "pequena" | "pequeno" => "small".to_string(),
            "concha" | "conchas" => "concha".to_string(),
            "escumadeira" | "escumadeiras" => "escumadeira".to_string(),
            "unidade" | "unidades" => "unit".to_string(),
            "pedaco" | "pedacos" => "piece".to_string(),
            "prato" | "tigela" | "bowl" => "bowl".to_string(),
            other => other.to_string(),
        }
    }

    fn estimate_grams(quantity: f32, unit: &str, canonical_key: &str) -> f32 {
        let per_unit_grams = match unit {
            "g" => 1.0,
            "kg" => 1_000.0,
            "ml" => 1.0,
            "oz" => 28.35,
            "tbsp" => 14.0,
            "tsp" => 5.0,
            "cup" => 200.0,
            "slice" => 35.0,
            "scoop" => 32.0,
            "concha" => 100.0,
            "escumadeira" => 90.0,
            "piece" => 60.0,
            "bowl" => 250.0,
            "small" => 100.0,
            "medium" => {
                if canonical_key.contains("sweet_potato") || canonical_key.contains("potato") {
                    130.0
                } else if canonical_key.contains("banana") {
                    118.0
                } else if canonical_key.contains("apple") {
                    182.0
                } else if canonical_key.contains("egg") {
                    50.0
                } else {
                    120.0
                }
            }
            "large" => 180.0,
            "unit" | "serving" => {
                if canonical_key.contains("egg") {
                    50.0
                } else if canonical_key.contains("pao_de_queijo") {
                    40.0
                } else if canonical_key.contains("pao_frances") {
                    50.0
                } else if canonical_key.contains("tapioca") {
                    80.0
                } else if canonical_key.contains("banana") {
                    100.0
                } else {
                    100.0
                }
            }
            _ => 100.0,
        };
        (quantity * per_unit_grams * 10.0).round() / 10.0
    }
}

#[async_trait]
impl AgentNode for MealParserNode {
    fn id(&self) -> GraphNodeId {
        GraphNodeId::MealParser
    }

    fn declared_tools(&self) -> &'static [&'static str] {
        &["food_entity_extractor", "portion_gram_normalizer"]
    }

    async fn execute(
        &self,
        mut state: GraphState,
        ctx: &NodeExecutionContext,
    ) -> Result<NodeExecutionResult, GraphNodeError> {
        let transcript =
            state
                .transcript
                .clone()
                .ok_or_else(|| GraphNodeError::MissingTranscript {
                    node: GraphNodeId::MealParser,
                })?;

        let (entities, mut warnings) = Self::parse_food_entities_with_warnings(&transcript);
        state.sanitization_warnings.append(&mut warnings);

        let llm_resp = ctx
            .llm_provider
            .complete(ModelCompletionRequest {
                node_id: GraphNodeId::MealParser,
                model_tier: self.default_model_tier(),
                language: state.plan.language,
                system_prompt: "Extract food entities, quantities, and gram weights (EN/PT)."
                    .to_string(),
                user_prompt: transcript.chars().take(512).collect(),
            })
            .await?;

        state.record_tool_call(ToolUsageRecord {
            call_id: Uuid::new_v4().to_string(),
            step_index: state.step_index,
            caller_node: GraphNodeId::MealParser,
            hierarchy_role: AgentHierarchyRole::Planner,
            tool_name: "food_entity_extractor".to_string(),
            input_summary: format!(
                "lang={:?} transcript_len={}",
                state.plan.language,
                transcript.len()
            ),
            output_summary: format!(
                "extracted {} entities: [{}]",
                entities.len(),
                entities
                    .iter()
                    .map(|e| format!("{} ({}g)", e.canonical_key, e.estimated_grams))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            cache_hit: None,
            latency_ms: 6,
            tokens_used: llm_resp.simulated_prompt_tokens + llm_resp.simulated_completion_tokens,
            success: true,
        });

        // Update the Planner's ExecutionPlan with specific per-item cache lookup tasks for the ToolCaller tier
        for entity in &entities {
            state.plan.scheduled_tool_calls.push(PlannedToolCall {
                tool_name: "redis_l1_l2_rag_lookup".to_string(),
                assigned_node: GraphNodeId::CacheLookup,
                target_entity_or_query: entity.canonical_key.clone(),
                purpose: format!(
                    "Check Redis L1 exact/barcode & L2 semantic RAG cache for '{}' ({}g)",
                    entity.display_name, entity.estimated_grams
                ),
                completed: false,
            });
        }
        state.plan.current_stage = AgentHierarchyRole::ToolCaller;
        state.extracted_foods = entities;

        Ok(NodeExecutionResult::new(
            state,
            llm_resp.simulated_prompt_tokens,
            llm_resp.simulated_completion_tokens,
            llm_resp.cached_tokens,
        )
        .with_mock_flag(llm_resp.is_mock))
    }
}
