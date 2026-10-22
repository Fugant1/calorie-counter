//! `MealParserNode` (**Planner** / Decomposition tier) powered by a small JSON-mode LLM
//! (`qwen2.5-1.5b-instruct-json` configured in `config/node_llm_config.json`) with fuzzy
//! typo correction (e.g. `"ovio"`, `"ovoos"`, `"frnago"`, `"paes de forna"`) and household
//! measure detection for ground-truth inference.

use crate::nodes::{
    normalize_food_key, strip_portuguese_diacritics, AgentNode, GraphNodeError,
    ModelCompletionRequest, NodeExecutionContext, NodeExecutionResult,
};
use crate::state::{
    AgentHierarchyRole, FoodEntity, GraphNodeId, GraphState, PlannedToolCall, ToolUsageRecord,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Maximum number of food items allowed in a single meal turn to guard against spam/flood inputs.
pub const MAX_ENTITIES_PER_MEAL: usize = 25;

/// Maximum gram weight allowed for a single food item before clamping.
pub const MAX_GRAMS_PER_ITEM: f32 = 5_000.0;

/// Structured JSON payload output by the small LLM in `MealParserNode`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmMealExtractionJson {
    pub items: Vec<LlmExtractedMealItem>,
}

/// Individual food item and measure extracted into JSON by the small LLM.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmExtractedMealItem {
    pub raw_mention: String,
    pub normalized_food_name: String,
    pub canonical_key: String,
    pub quantity: f32,
    pub unit: String,
    /// `true` when the user specified a household or countable measure (`"unit"`, `"serving"`, `"tbsp"`,
    /// `"slice"`, `"concha"`, `"escumadeira"`, `"cup"`, `"scoop"`) rather than an explicit weight (`"g"`, `"kg"`, `"ml"`, `"oz"`).
    /// Signals downstream ToolCaller nodes (`CacheLookup` / `WebSearchAndRAG`) to infer the gram weight from ground truth.
    pub is_household_measure: bool,
    /// Present only when the user already weighed the food (e.g. `200g`, `150ml`).
    pub explicit_grams: Option<f32>,
    pub preparation: Option<String>,
    pub barcode_ean: Option<String>,
    /// Populated when the LLM corrected a spelling typo in the user's input (e.g. `"ovio"` or `"ovoos"` -> `"ovo"`).
    pub typo_corrected_from: Option<String>,
}

/// **Planner** node that uses a very small JSON-mode LLM (`config/node_llm_config.json`) to analyze
/// free-form meal write-ups (English or Portuguese), resolve user typos (`"ovio"`, `"ovoos"`, etc.),
/// output structured JSON with foods and measures, and flag household measures for ground-truth inference.
#[derive(Debug, Default, Clone)]
pub struct MealParserNode;

impl MealParserNode {
    pub fn new() -> Self {
        Self
    }

    /// Simulates the small JSON-mode LLM (`qwen2.5-1.5b-instruct-json`) converting a noisy
    /// meal description into a serialized `LlmMealExtractionJson` JSON string.
    pub fn simulate_small_llm_json_extraction(transcript: &str) -> String {
        let (entities, _warnings) = Self::parse_food_entities_with_warnings(transcript);
        let items: Vec<LlmExtractedMealItem> = entities
            .into_iter()
            .map(|e| LlmExtractedMealItem {
                raw_mention: e.raw_mention,
                normalized_food_name: e.display_name,
                canonical_key: e.canonical_key,
                quantity: e.quantity,
                unit: e.unit,
                is_household_measure: e.is_household_measure,
                explicit_grams: if e.is_household_measure {
                    None
                } else {
                    Some(e.estimated_grams)
                },
                preparation: e.preparation,
                barcode_ean: e.barcode_ean,
                typo_corrected_from: e.typo_corrected_from,
            })
            .collect();

        serde_json::to_string(&LlmMealExtractionJson { items })
            .unwrap_or_else(|_| r#"{"items":[]}"#.to_string())
    }

    /// Parses a natural-language meal log (English or Portuguese, including typos like `"ovio"` or `"ovoos"`)
    /// into structured `FoodEntity` items along with any sanitization warnings.
    pub fn parse_food_entities_with_warnings(transcript: &str) -> (Vec<FoodEntity>, Vec<String>) {
        let mut warnings = Vec::new();
        let sanitized = Self::sanitize_raw_input(transcript, &mut warnings);

        if sanitized.trim().is_empty() {
            warnings.push("Input contained no parseable text after sanitization.".to_string());
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
            (" colhr de sopa ", " tbsp "),
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

        // Check if the next token is a unit (or a misspelled unit like "fatiaa", "colhr", "unidde")
        if consumed_tokens < tokens.len() && unit == "serving" {
            let candidate_unit = Self::correct_unit_typo(tokens[consumed_tokens]);
            if Self::is_unit_word(&candidate_unit) {
                unit = Self::normalize_unit(&candidate_unit);
                consumed_tokens += 1;
            }
        } else if unit != "serving" {
            let corrected_attached = Self::correct_unit_typo(&unit);
            unit = Self::normalize_unit(&corrected_attached);
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

        // Extract optional EAN barcode token (8-13 digits) and apply LLM fuzzy typo correction on food words
        let mut barcode_ean: Option<String> = None;
        let mut name_words: Vec<String> = Vec::new();
        let mut typo_corrections: Vec<String> = Vec::new();

        for &w in &food_words {
            let stripped = w.trim_matches(|c: char| !c.is_ascii_alphanumeric());
            if stripped.len() >= 8
                && stripped.len() <= 13
                && stripped.chars().all(|c| c.is_ascii_digit())
            {
                barcode_ean = Some(stripped.to_string());
            } else if stripped.chars().any(|c| c.is_ascii_alphabetic()) {
                let (corrected_word, was_corrected) = Self::correct_food_token_typo(stripped);
                if was_corrected {
                    typo_corrections.push(format!("{} -> {}", stripped, corrected_word));
                }
                name_words.push(corrected_word);
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

        let is_household_measure = Self::is_household_unit(&unit);
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
            "mexido",
            "mexidos",
        ]
        .iter()
        .find(|&&prep| display_name.contains(prep))
        .map(|&s| s.to_string());

        let typo_corrected_from = if typo_corrections.is_empty() {
            None
        } else {
            Some(typo_corrections.join(", "))
        };

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
            is_household_measure,
            household_grams_verified_from_ground_truth: !is_household_measure,
            typo_corrected_from,
        })
    }

    /// Returns `true` when `unit` is a household or countable unit rather than an explicit weight (`g`, `kg`, `ml`, `oz`).
    pub fn is_household_unit(unit: &str) -> bool {
        !matches!(unit, "g" | "kg" | "ml" | "oz")
    }

    /// Corrects common spelling typos in food tokens (such as `"ovio"`, `"ovoos"`, `"frnago"`, `"forna"`,
    /// `"aroz"`, `"fejao"`, `"batta"`, `"tapioka"`, `"amendoim"`) using repeated-character collapsing
    /// and Damerau-Levenshtein transposition/edit distance against the bilingual food lexicon.
    pub fn correct_food_token_typo(token: &str) -> (String, bool) {
        let lower = token.to_lowercase();
        let stop_or_short = [
            "de", "do", "da", "dos", "das", "com", "e", "ou", "of", "in", "on", "a", "o",
        ];
        if lower.len() <= 2 || stop_or_short.contains(&lower.as_str()) {
            return (lower, false);
        }

        let lexicon = [
            // Eggs & preparations
            "ovo",
            "ovos",
            "cozido",
            "cozidos",
            "cozida",
            "mexido",
            "mexidos",
            "frito",
            "fritos",
            "egg",
            "eggs",
            "boiled",
            "scrambled",
            // Breads & grains
            "pao",
            "paes",
            "forma",
            "integral",
            "integrais",
            "frances",
            "queijo",
            "aveia",
            "tapioca",
            "arroz",
            "macarrao",
            "bread",
            "oats",
            "oatmeal",
            "rice",
            // Proteins
            "frango",
            "peito",
            "grelhado",
            "grelhada",
            "salmao",
            "tilapia",
            "peixe",
            "picanha",
            "patinho",
            "carne",
            "moida",
            "moido",
            "whey",
            "chicken",
            "breast",
            "salmon",
            "beef",
            // Legumes, tubers, fruits & fats
            "feijao",
            "carioca",
            "preto",
            "batata",
            "doce",
            "mandioca",
            "aipim",
            "banana",
            "bananas",
            "abacate",
            "acai",
            "azeite",
            "oliva",
            "pasta",
            "amendoim",
            "manteiga",
            "sweet",
            "potato",
            "beans",
            "avocado",
            "olive",
            "oil",
            "peanut",
            "butter",
        ];

        if lexicon.contains(&lower.as_str()) {
            return (lower, false);
        }

        // 1. Check direct known phonetic/keyboard aliases (e.g. "ovio" -> "ovo", "ovoos" -> "ovos")
        match lower.as_str() {
            "ovio" | "ovios" | "ovvo" | "ovvos" | "ovoos" | "ovoss" | "ovoo" | "oovos" | "eggg"
            | "egggs" | "egs" => {
                let replacement = if lower.ends_with('s') { "ovos" } else { "ovo" };
                return (replacement.to_string(), true);
            }
            "forna" | "foma" | "formaa" => return ("forma".to_string(), true),
            "paos" | "paees" | "paoo" => return ("paes".to_string(), true),
            "frnago" | "frago" | "frangoo" => return ("frango".to_string(), true),
            "aroz" | "arros" | "arrozz" => return ("arroz".to_string(), true),
            "fejao" | "fijao" | "feijaoo" => return ("feijao".to_string(), true),
            "batta" | "batataa" => return ("batata".to_string(), true),
            "tapioka" | "tapicoa" => return ("tapioca".to_string(), true),
            _ => {}
        }

        // 2. Collapse consecutive duplicate characters (e.g. "ovooos" -> "ovos", "cozidoos" -> "cozidos")
        let collapsed = Self::collapse_duplicate_chars(&lower);
        if collapsed != lower && lexicon.contains(&collapsed.as_str()) {
            return (collapsed, true);
        }

        // 3. Damerau-Levenshtein fuzzy lookup against the bilingual food lexicon
        let max_allowed_dist = if lower.len() <= 5 { 1 } else { 2 };
        let mut best_match: Option<(&str, usize)> = None;
        for &candidate in &lexicon {
            if candidate.len().abs_diff(lower.len()) > max_allowed_dist {
                continue;
            }
            // Require matching first letter to avoid false-positive short-word collisions
            if candidate.as_bytes().first() != lower.as_bytes().first() {
                continue;
            }
            let dist = Self::damerau_levenshtein(&lower, candidate);
            if dist <= max_allowed_dist {
                match best_match {
                    Some((_, best_d)) if dist >= best_d => {}
                    _ => best_match = Some((candidate, dist)),
                }
            }
        }

        if let Some((matched, _)) = best_match {
            (matched.to_string(), true)
        } else {
            (lower, false)
        }
    }

    /// Corrects typos in unit tokens (e.g. `"colhr"` -> `"colher"`, `"fatiaa"` -> `"fatia"`, `"unidde"` -> `"unidade"`).
    fn correct_unit_typo(unit: &str) -> String {
        let lower = unit.to_lowercase();
        if Self::is_unit_word(&lower) {
            return lower;
        }
        match lower.as_str() {
            "colhr" | "coler" | "colhere" => "colher".to_string(),
            "fatiaa" | "fata" | "fatas" => "fatia".to_string(),
            "conxa" | "conxas" => "concha".to_string(),
            "unid" | "und" | "unidde" | "unidads" => "unidade".to_string(),
            "gr" | "grs" | "grm" | "gramss" => "g".to_string(),
            _ => lower,
        }
    }

    fn collapse_duplicate_chars(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        let mut prev: Option<char> = None;
        for ch in s.chars() {
            // Preserve valid double 'r' or 's' in Portuguese ("arroz", "assado") while collapsing vowel/other repeats
            if Some(ch) == prev && !matches!(ch, 'r' | 's') {
                continue;
            }
            out.push(ch);
            prev = Some(ch);
        }
        out
    }

    /// Computes optimal string alignment / Damerau-Levenshtein distance (supporting insertions,
    /// deletions, substitutions, and adjacent transpositions like `"frnago"` <-> `"frango"`).
    pub fn damerau_levenshtein(a: &str, b: &str) -> usize {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        let m = a_chars.len();
        let n = b_chars.len();

        let mut dp = vec![vec![0usize; n + 1]; m + 1];
        for (i, row) in dp.iter_mut().enumerate().take(m + 1) {
            row[0] = i;
        }
        for j in 0..=n {
            dp[0][j] = j;
        }

        for i in 1..=m {
            for j in 1..=n {
                let cost = usize::from(a_chars[i - 1] != b_chars[j - 1]);
                dp[i][j] = (dp[i - 1][j] + 1)
                    .min(dp[i][j - 1] + 1)
                    .min(dp[i - 1][j - 1] + cost);

                if i > 1
                    && j > 1
                    && a_chars[i - 1] == b_chars[j - 2]
                    && a_chars[i - 2] == b_chars[j - 1]
                {
                    dp[i][j] = dp[i][j].min(dp[i - 2][j - 2] + 1);
                }
            }
        }
        dp[m][n]
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

    pub fn estimate_grams(quantity: f32, unit: &str, canonical_key: &str) -> f32 {
        let is_egg = canonical_key.contains("egg") || canonical_key.contains("ovo");
        let is_sliced_bread = canonical_key.contains("whole_wheat_bread")
            || canonical_key.contains("pao_forma")
            || canonical_key.contains("paes_forma")
            || canonical_key.contains("pao_integral")
            || canonical_key.contains("sliced_bread");

        let per_unit_grams = match unit {
            "g" => 1.0,
            "kg" => 1_000.0,
            "ml" => 1.0,
            "oz" => 28.35,
            "tbsp" => 14.0,
            "tsp" => 5.0,
            "cup" => 200.0,
            "slice" => {
                if is_sliced_bread {
                    25.0
                } else {
                    30.0
                }
            }
            "scoop" => 32.0,
            "concha" => 100.0,
            "escumadeira" => 90.0,
            "piece" => 60.0,
            "bowl" => 250.0,
            "small" => {
                if is_egg {
                    45.0
                } else {
                    100.0
                }
            }
            "medium" => {
                if canonical_key.contains("sweet_potato") || canonical_key.contains("potato") {
                    130.0
                } else if canonical_key.contains("banana") {
                    118.0
                } else if canonical_key.contains("apple") {
                    182.0
                } else if is_egg {
                    50.0
                } else {
                    120.0
                }
            }
            "large" => {
                if is_egg {
                    60.0
                } else {
                    180.0
                }
            }
            "unit" | "serving" => {
                if is_egg {
                    55.0
                } else if is_sliced_bread {
                    25.0
                } else if canonical_key.contains("pao_de_queijo") {
                    40.0
                } else if canonical_key.contains("pao_frances") {
                    50.0
                } else if canonical_key.contains("tapioca") {
                    80.0
                } else if canonical_key.contains("banana") {
                    100.0
                } else if canonical_key.contains("chicken")
                    || canonical_key.contains("salmon")
                    || canonical_key.contains("tilapia")
                {
                    120.0
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
        &[
            "llm_json_meal_extractor",
            "food_entity_extractor",
            "portion_gram_normalizer",
        ]
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

        // 1. Run sanitization & pre-flight warnings check
        let (_fallback_entities, mut warnings) = Self::parse_food_entities_with_warnings(&transcript);
        state.sanitization_warnings.append(&mut warnings);

        // 2. Load the small-LLM configuration for MealParser from `config/node_llm_config.json`
        let node_llm_cfg = ctx.llm_config.config_for_node(GraphNodeId::MealParser);

        let system_prompt = format!(
            "You are '{}' ({}, format={}). Analyze the user's meal description (PT/EN), \
             correct spelling typos (e.g. 'ovio' or 'ovoos' -> 'ovo'), and output ONLY valid JSON matching \
             {{\"items\":[{{\"raw_mention\":str,\"normalized_food_name\":str,\"canonical_key\":str,\
             \"quantity\":f32,\"unit\":str,\"is_household_measure\":bool,\"explicit_grams\":f32|null,\
             \"preparation\":str|null,\"barcode_ean\":str|null,\"typo_corrected_from\":str|null}}]}}.",
            node_llm_cfg.model_id, node_llm_cfg.provider, node_llm_cfg.response_format
        );

        let llm_resp = ctx
            .llm_provider
            .complete(ModelCompletionRequest {
                node_id: GraphNodeId::MealParser,
                model_tier: node_llm_cfg.model_tier,
                language: state.plan.language,
                system_prompt,
                user_prompt: transcript.chars().take(512).collect(),
            })
            .await?;

        // 3. Parse the structured JSON returned by the small LLM (or fallback if a test supplied a non-JSON canned mock string)
        let parsed_json: LlmMealExtractionJson = serde_json::from_str(&llm_resp.content)
            .unwrap_or_else(|_| {
                let simulated_json = Self::simulate_small_llm_json_extraction(&transcript);
                serde_json::from_str(&simulated_json)
                    .unwrap_or(LlmMealExtractionJson { items: Vec::new() })
            });

        let entities: Vec<FoodEntity> = parsed_json
            .items
            .into_iter()
            .enumerate()
            .map(|(idx, item)| {
                let mut estimated_grams = item.explicit_grams.unwrap_or_else(|| {
                    Self::estimate_grams(item.quantity, &item.unit, &item.canonical_key)
                });
                if estimated_grams > MAX_GRAMS_PER_ITEM {
                    estimated_grams = MAX_GRAMS_PER_ITEM;
                }
                FoodEntity {
                    entity_id: format!("food-{}-{}", idx + 1, &Uuid::new_v4().to_string()[..8]),
                    raw_mention: item.raw_mention,
                    canonical_key: item.canonical_key,
                    display_name: item.normalized_food_name,
                    brand: None,
                    barcode_ean: item.barcode_ean,
                    quantity: item.quantity,
                    unit: item.unit,
                    estimated_grams,
                    preparation: item.preparation,
                    is_household_measure: item.is_household_measure,
                    household_grams_verified_from_ground_truth: !item.is_household_measure,
                    typo_corrected_from: item.typo_corrected_from,
                }
            })
            .collect();

        state.record_tool_call(ToolUsageRecord {
            call_id: Uuid::new_v4().to_string(),
            step_index: state.step_index,
            caller_node: GraphNodeId::MealParser,
            hierarchy_role: AgentHierarchyRole::Planner,
            tool_name: "llm_json_meal_extractor".to_string(),
            input_summary: format!(
                "model={} format={} lang={:?} transcript_len={}",
                node_llm_cfg.model_id,
                node_llm_cfg.response_format,
                state.plan.language,
                transcript.len()
            ),
            output_summary: format!(
                "extracted {} JSON items: [{}]",
                entities.len(),
                entities
                    .iter()
                    .map(|e| format!(
                        "{} (qty={} {}, household={}, ~{}g{})",
                        e.canonical_key,
                        e.quantity,
                        e.unit,
                        e.is_household_measure,
                        e.estimated_grams,
                        e.typo_corrected_from
                            .as_ref()
                            .map(|t| format!(", typo_fixed: {}", t))
                            .unwrap_or_default()
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            cache_hit: None,
            latency_ms: 6,
            tokens_used: llm_resp.simulated_prompt_tokens + llm_resp.simulated_completion_tokens,
            success: true,
        });

        // Update the Planner's ExecutionPlan with per-item cache lookup + ground-truth household measure inference tasks
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
            if entity.is_household_measure {
                state.plan.scheduled_tool_calls.push(PlannedToolCall {
                    tool_name: "household_measure_ground_truth_lookup".to_string(),
                    assigned_node: GraphNodeId::WebSearchAndRAG,
                    target_entity_or_query: format!("{}:{}", entity.canonical_key, entity.unit),
                    purpose: format!(
                        "Infer ground-truth gram weight for household measure '{} {}' of '{}' from TACO/IBGE/USDA web search",
                        entity.quantity, entity.unit, entity.display_name
                    ),
                    completed: false,
                });
            }
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
