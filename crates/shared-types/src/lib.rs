//! Cross-crate and cross-platform domain contracts for Calorie-Counter AI,
//! including User Profile/Auth schemas, Chat Conversation & Message DTOs,
//! Nutrition Plan `.pdf` & `.txt` text readers (supporting compressed FlateDecode/Type3
//! PDFs like `Plano Alimentar Teste.pdf` as well as raw `%PDF-1.4` streams),
//! bilingual (Portuguese/English) nutrition plan parsing, probable daily foods extraction,
//! and daily calorie exchange calculations.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::process::{Command, Stdio};
use thiserror::Error;
use uuid::Uuid;

/// Errors returned during `.pdf` / `.txt` nutrition plan reading or user validation.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SharedDomainError {
    #[error("Unsupported file format '{0}'. Expected .pdf or .txt")]
    UnsupportedFileFormat(String),

    #[error("Document payload is empty or contains no readable text")]
    EmptyDocument,

    #[error("Invalid PDF byte stream: {0}")]
    InvalidPdfStream(String),

    #[error("Validation error on field '{field}': {message}")]
    ValidationError { field: String, message: String },
}

/// Supported file formats for nutritionist plan ingestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlanFileFormat {
    Pdf,
    Txt,
}

impl PlanFileFormat {
    /// Detects `PlanFileFormat` from a filename extension or magic header bytes (`%PDF-`).
    pub fn detect(filename: &str, bytes: &[u8]) -> Result<Self, SharedDomainError> {
        let lower = filename.trim().to_lowercase();
        if bytes.starts_with(b"%PDF-") || lower.ends_with(".pdf") {
            return Ok(Self::Pdf);
        }
        if lower.ends_with(".txt") || lower.ends_with(".md") || lower.ends_with(".plan") {
            return Ok(Self::Txt);
        }
        Err(SharedDomainError::UnsupportedFileFormat(
            filename.to_string(),
        ))
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pdf => "pdf",
            Self::Txt => "txt",
        }
    }
}

/// Macro and micronutrient breakdown used across crates.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MacroSummary {
    pub calories_kcal: f32,
    pub protein_g: f32,
    pub carbs_g: f32,
    pub fat_g: f32,
    pub fiber_g: f32,
}

impl MacroSummary {
    pub fn scaled_for_grams(&self, grams: f32) -> Self {
        let factor = if grams.is_finite() && grams > 0.0 {
            grams / 100.0
        } else {
            0.0
        };
        Self {
            calories_kcal: round1(self.calories_kcal * factor),
            protein_g: round1(self.protein_g * factor),
            carbs_g: round1(self.carbs_g * factor),
            fat_g: round1(self.fat_g * factor),
            fiber_g: round1(self.fiber_g * factor),
        }
    }

    pub fn add_assign(&mut self, other: &Self) {
        self.calories_kcal = round1(self.calories_kcal + other.calories_kcal);
        self.protein_g = round1(self.protein_g + other.protein_g);
        self.carbs_g = round1(self.carbs_g + other.carbs_g);
        self.fat_g = round1(self.fat_g + other.fat_g);
        self.fiber_g = round1(self.fiber_g + other.fiber_g);
    }
}

fn round1(v: f32) -> f32 {
    if !v.is_finite() {
        0.0
    } else {
        (v * 10.0).round() / 10.0
    }
}

/// User personal & authentication record stored in PostgreSQL (`users` table).
/// Includes CPF, phone, address, email, and hashed password.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserAccountRecord {
    pub id: String,
    pub cpf: String,
    pub phone: String,
    pub address: String,
    pub email: String,
    pub password_hash: String,
    pub full_name: Option<String>,
    pub created_at_iso: String,
}

/// Input payload to register or update a user with CPF, phone, address, email, and raw password.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateUserRegistration {
    pub cpf: String,
    pub phone: String,
    pub address: String,
    pub email: String,
    pub password: String,
    pub full_name: Option<String>,
}

/// Validates and normalizes a Brazilian CPF (`000.000.000-00` or 11 digits) into canonical `XXX.XXX.XXX-XX`.
pub fn validate_and_format_cpf(raw_cpf: &str) -> Result<String, SharedDomainError> {
    let digits: Vec<char> = raw_cpf.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.len() != 11 {
        return Err(SharedDomainError::ValidationError {
            field: "cpf".to_string(),
            message: format!("CPF must contain exactly 11 digits (got {})", digits.len()),
        });
    }
    let s: String = digits.iter().collect();
    Ok(format!(
        "{}.{}.{}-{}",
        &s[0..3],
        &s[3..6],
        &s[6..9],
        &s[9..11]
    ))
}

/// Deterministically hashes a user password with a per-user salt (email/CPF) using SHA-256 PBKDF-style rounds.
pub fn hash_user_password(password: &str, salt_seed: &str) -> Result<String, SharedDomainError> {
    if password.trim().len() < 6 {
        return Err(SharedDomainError::ValidationError {
            field: "password".to_string(),
            message: "Password must be at least 6 characters long".to_string(),
        });
    }
    let mut salt_hasher = Sha256::new();
    salt_hasher.update(b"calorie-counter-salt-v1:");
    salt_hasher.update(salt_seed.trim().to_lowercase().as_bytes());
    let salt_hex = hex::encode(salt_hasher.finalize());

    let mut digest = Vec::from(password.as_bytes());
    for round in 0..1024u32 {
        let mut h = Sha256::new();
        h.update(&salt_hex);
        h.update(round.to_le_bytes());
        h.update(&digest);
        digest = h.finalize().to_vec();
    }
    Ok(format!(
        "$sha256-1024${}${}",
        &salt_hex[..16],
        hex::encode(digest)
    ))
}

/// Verifies a raw password against a stored `password_hash`.
pub fn verify_user_password(password: &str, salt_seed: &str, expected_hash: &str) -> bool {
    match hash_user_password(password, salt_seed) {
        Ok(computed) => computed == expected_hash,
        Err(_) => false,
    }
}

/// Probable food item extracted from the nutritionist's plan to be suggested/added each day.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbableDailyFood {
    pub food_id: String,
    pub meal_name: String,
    pub scheduled_time: Option<String>,
    pub canonical_key: String,
    pub display_name: String,
    pub quantity: f32,
    pub unit: String,
    pub estimated_grams: f32,
    pub macros: MacroSummary,
    pub is_primary_choice: bool,
}

/// A calorie/food exchange option (`Lista de Substituição`) allowing a user to swap a primary
/// planned food with an equivalent alternative while tracking the exact calorie exchange delta.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalorieExchangeOption {
    pub exchange_id: String,
    pub meal_name: String,
    pub primary_food_key: String,
    pub primary_food_name: String,
    pub primary_grams: f32,
    pub primary_calories_kcal: f32,
    pub replacement_food_key: String,
    pub replacement_food_name: String,
    pub replacement_grams: f32,
    pub replacement_macros: MacroSummary,
    /// Difference in calories (`replacement_macros.calories_kcal - primary_calories_kcal`).
    pub calorie_delta_kcal: f32,
}

/// A single meal slot inside a parsed nutrition plan (e.g. `07:00 - Café da Manhã`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrescribedMealSlot {
    pub meal_slot_id: String,
    pub meal_name: String,
    pub scheduled_time: Option<String>,
    pub primary_foods: Vec<ProbableDailyFood>,
    pub calorie_exchanges: Vec<CalorieExchangeOption>,
    pub total_meal_macros: MacroSummary,
}

/// Extracted user profile hints embedded inside the nutritionist's PDF/TXT header.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractedPlanUserProfile {
    pub full_name: Option<String>,
    pub cpf: Option<String>,
    pub email: Option<String>,
    pub phone: Option<String>,
    pub address: Option<String>,
    pub goal: Option<String>,
}

/// Complete structured nutrition plan parsed from a `.pdf` or `.txt` file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParsedNutritionPlan {
    pub plan_id: String,
    pub user_id: String,
    pub source_filename: String,
    pub source_format: PlanFileFormat,
    pub file_sha256: String,
    pub title: String,
    pub nutritionist_name: Option<String>,
    pub nutritionist_crn: Option<String>,
    pub extracted_user_profile: ExtractedPlanUserProfile,
    pub target_daily_macros: MacroSummary,
    pub planned_daily_macros: MacroSummary,
    pub hydration_target_ml: u32,
    pub meals: Vec<PrescribedMealSlot>,
    /// Consolidated list of probable foods the user will eat/add each day.
    pub probable_daily_foods: Vec<ProbableDailyFood>,
    /// Consolidated list of daily calorie exchange / substitution options across all meals.
    pub calorie_exchange_options: Vec<CalorieExchangeOption>,
    pub general_guidelines: Vec<String>,
    pub raw_extracted_text: String,
}

/// Pure-Rust + Poppler (`pdftotext`) `.pdf` and `.txt` Nutrition Plan Reader & Bilingual Parser.
#[derive(Debug, Default, Clone)]
pub struct NutritionPlanDocumentReader;

impl NutritionPlanDocumentReader {
    /// Generates a valid `%PDF-1.4` binary document containing the given nutrition plan text.
    pub fn build_pdf_document_from_text(lines: &[&str]) -> Vec<u8> {
        let mut stream_body = String::from("BT\n/F1 11 Tf\n50 760 Td\n14 TL\n");
        for (idx, line) in lines.iter().enumerate() {
            let escaped = line
                .replace('\\', "\\\\")
                .replace('(', "\\(")
                .replace(')', "\\)");
            if idx == 0 {
                stream_body.push_str(&format!("({}) Tj\n", escaped));
            } else {
                stream_body.push_str(&format!("T* ({}) Tj\n", escaped));
            }
        }
        stream_body.push_str("ET\n");

        let mut pdf = String::new();
        pdf.push_str("%PDF-1.4\n");
        pdf.push_str("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
        pdf.push_str("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
        pdf.push_str("3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n");
        pdf.push_str(&format!(
            "4 0 obj\n<< /Length {} >>\nstream\n{}endstream\nendobj\n",
            stream_body.len(),
            stream_body
        ));
        pdf.push_str("xref\n0 5\n0000000000 65535 f \n");
        pdf.push_str("trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n");
        pdf.into_bytes()
    }

    /// Reads raw text from either a `.pdf` or `.txt` byte slice.
    pub fn extract_text_from_bytes(
        filename: &str,
        bytes: &[u8],
    ) -> Result<(PlanFileFormat, String), SharedDomainError> {
        if bytes.is_empty() {
            return Err(SharedDomainError::EmptyDocument);
        }
        let format = PlanFileFormat::detect(filename, bytes)?;
        let raw_text = match format {
            PlanFileFormat::Txt => Self::decode_txt_bytes(bytes)?,
            PlanFileFormat::Pdf => Self::decode_pdf_bytes(bytes)?,
        };

        let cleaned = raw_text.trim().to_string();
        if cleaned.is_empty() {
            return Err(SharedDomainError::EmptyDocument);
        }
        Ok((format, cleaned))
    }

    fn decode_txt_bytes(bytes: &[u8]) -> Result<String, SharedDomainError> {
        let s = match std::str::from_utf8(bytes) {
            Ok(valid) => valid.to_string(),
            Err(_) => bytes.iter().map(|&b| b as char).collect(),
        };
        Ok(s.replace("\r\n", "\n").replace('\r', "\n"))
    }

    /// Extracts text from PDF binary bytes.
    ///
    /// 1. For compressed FlateDecode / Type3 CID / ToUnicode CMap PDFs (such as Chromium/Skia PDFs
    ///    like `Plano Alimentar Teste.pdf`), invokes `pdftotext - -` via stdin/stdout to decode
    ///    all compressed streams and font CMaps accurately.
    /// 2. Falls back to our pure-Rust `BT ... ET` `(literal) Tj` stream parser for uncompressed
    ///    synthetic PDFs or environments without `pdftotext`.
    fn decode_pdf_bytes(bytes: &[u8]) -> Result<String, SharedDomainError> {
        let raw_str: String = match std::str::from_utf8(bytes) {
            Ok(utf8) => utf8.to_string(),
            Err(_) => bytes.iter().map(|&b| b as char).collect(),
        };

        // If the user uploaded a plain-text file renamed with `.pdf` extension, handle gracefully
        if !bytes.starts_with(b"%PDF-") && !raw_str.contains("endobj") {
            let trimmed = raw_str.trim();
            if trimmed.is_empty() {
                return Err(SharedDomainError::InvalidPdfStream(
                    "Missing %PDF header and no readable text content".to_string(),
                ));
            }
            return Ok(trimmed.replace("\r\n", "\n"));
        }

        // If the PDF uses FlateDecode compression or Type3/ToUnicode CMap fonts, decode via `pdftotext` first
        let uses_compressed_or_cmap_streams = raw_str.contains("/FlateDecode")
            || raw_str.contains("/ToUnicode")
            || raw_str.contains("begincmap");

        if uses_compressed_or_cmap_streams {
            if let Some(poppler_text) = Self::try_extract_with_pdftotext(bytes) {
                if !poppler_text.trim().is_empty() {
                    return Ok(poppler_text);
                }
            }
        }

        // Pure-Rust `BT ... ET` stream extractor for uncompressed literal PDF streams
        let mut extracted_lines: Vec<String> = Vec::new();
        let mut current_line = String::new();
        let mut search_slice = raw_str.as_str();
        let mut found_bt = false;

        while let Some(bt_pos) = search_slice.find("BT") {
            let after_bt = &search_slice[bt_pos + 2..];
            let et_pos = after_bt.find("ET").unwrap_or(after_bt.len());
            let bt_block = &after_bt[..et_pos];

            Self::extract_pdf_text_operators(bt_block, &mut current_line, &mut extracted_lines);
            if !current_line.trim().is_empty() {
                extracted_lines.push(std::mem::take(&mut current_line).trim().to_string());
                found_bt = true;
            } else if !extracted_lines.is_empty() {
                found_bt = true;
            }

            if et_pos + 2 <= after_bt.len() {
                search_slice = &after_bt[et_pos + 2..];
            } else {
                break;
            }
        }

        if !found_bt {
            if let Some(poppler_text) = Self::try_extract_with_pdftotext(bytes) {
                if !poppler_text.trim().is_empty() {
                    return Ok(poppler_text);
                }
            }
        }

        if extracted_lines.is_empty() {
            return Err(SharedDomainError::InvalidPdfStream(
                "No extractable text operators found in PDF stream".to_string(),
            ));
        }

        Ok(extracted_lines.join("\n"))
    }

    fn try_extract_with_pdftotext(bytes: &[u8]) -> Option<String> {
        let mut child = Command::new("pdftotext")
            .arg("-")
            .arg("-")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;

        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(bytes);
        }
        let output = child.wait_with_output().ok()?;
        if !output.status.success() {
            return None;
        }
        let text = String::from_utf8_lossy(&output.stdout)
            .replace("\r\n", "\n")
            .replace('\r', "\n")
            .replace('\u{000C}', "\n");
        Some(text)
    }

    fn extract_pdf_text_operators(
        stream: &str,
        current_line: &mut String,
        extracted_lines: &mut Vec<String>,
    ) {
        let chars: Vec<char> = stream.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            if i + 1 < chars.len()
                && chars[i] == 'T'
                && (chars[i + 1] == '*' || chars[i + 1] == 'd' || chars[i + 1] == 'D')
            {
                if !current_line.trim().is_empty() {
                    extracted_lines.push(std::mem::take(current_line).trim().to_string());
                }
                i += 2;
                continue;
            }

            if chars[i] == '(' {
                i += 1;
                let mut depth = 1;
                let mut literal = String::new();
                while i < chars.len() && depth > 0 {
                    let c = chars[i];
                    if c == '\\' && i + 1 < chars.len() {
                        let next_c = chars[i + 1];
                        match next_c {
                            'n' => literal.push('\n'),
                            'r' => literal.push('\r'),
                            't' => literal.push('\t'),
                            '(' => literal.push('('),
                            ')' => literal.push(')'),
                            '\\' => literal.push('\\'),
                            _ => literal.push(next_c),
                        }
                        i += 2;
                        continue;
                    } else if c == '(' {
                        depth += 1;
                        literal.push(c);
                    } else if c == ')' {
                        depth -= 1;
                        if depth > 0 {
                            literal.push(c);
                        }
                    } else {
                        literal.push(c);
                    }
                    i += 1;
                }
                for sub_line in literal.split('\n') {
                    if !current_line.is_empty() && !current_line.ends_with(' ') {
                        current_line.push(' ');
                    }
                    current_line.push_str(sub_line.trim());
                }
                continue;
            }

            i += 1;
        }
    }

    /// Pre-normalizes multi-column PDF table layouts (such as `Plano Alimentar Teste.pdf` where
    /// `Horário`, `Refeição`, `Alimentos / Preparações`, and `Quantidade` are in separate table columns)
    /// into unified meal sections with paired `{quantity} de {food}` lines.
    pub fn normalize_tabular_plan_text(raw_text: &str) -> String {
        let raw_lines: Vec<String> = raw_text
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();

        let mut normalized_lines: Vec<String> = Vec::new();
        let mut day_period_suffix = String::new();
        let mut i = 0;

        while i < raw_lines.len() {
            let line = &raw_lines[i];
            let lower = strip_diacritics(line).to_lowercase();

            // Skip page footers and table column header labels
            if lower.starts_with("pagina ")
                || lower.starts_with("plano alimentar -")
                || matches!(
                    lower.as_str(),
                    "horario"
                        | "refeicao"
                        | "alimentos / preparacoes"
                        | "quantidade"
                        | "horario refeicao alimentos / preparacoes quantidade"
                )
            {
                i += 1;
                continue;
            }

            // Detect Weekday vs Weekend section banners
            if lower.starts_with("dias de semana") {
                day_period_suffix = " (Seg-Sex)".to_string();
                i += 1;
                continue;
            }
            if lower.starts_with("final de semana") {
                day_period_suffix = " (Sáb-Dom)".to_string();
                i += 1;
                continue;
            }

            // Detect standalone `HH:MM` time row in a PDF table (e.g. "07:00", "13:30")
            if is_standalone_time_hhmm(line) && i + 1 < raw_lines.len() {
                let time_str = line.clone();
                i += 1;
                let mut meal_title = raw_lines[i].clone();
                i += 1;
                // Handle wrapped meal titles like "Almoço (Livre\nControlado)"
                if meal_title.contains('(') && !meal_title.contains(')') && i < raw_lines.len() {
                    meal_title = format!("{} {}", meal_title, raw_lines[i]);
                    i += 1;
                }

                normalized_lines.push(format!(
                    "{} - {}{}",
                    time_str, meal_title, day_period_suffix
                ));

                // Collect all lines belonging to this table row until the next `HH:MM` or section header
                let mut cell_lines: Vec<String> = Vec::new();
                while i < raw_lines.len() {
                    let peek = &raw_lines[i];
                    let peek_lower = strip_diacritics(peek).to_lowercase();
                    if is_standalone_time_hhmm(peek)
                        || peek_lower.starts_with("dias de semana")
                        || peek_lower.starts_with("final de semana")
                        || peek_lower.starts_with("orientacoes")
                        || peek_lower.starts_with("observacoes")
                        || peek_lower.starts_with("guidelines")
                    {
                        break;
                    }
                    if !peek_lower.starts_with("pagina ")
                        && !peek_lower.starts_with("plano alimentar -")
                        && !matches!(
                            peek_lower.as_str(),
                            "horario" | "refeicao" | "alimentos / preparacoes" | "quantidade"
                        )
                    {
                        cell_lines.push(peek.clone());
                    }
                    i += 1;
                }

                // Pair the food column lines with the quantity column lines
                let paired = Self::pair_tabular_foods_and_quantities(&cell_lines);
                normalized_lines.extend(paired);
                continue;
            }

            normalized_lines.push(line.clone());
            i += 1;
        }

        normalized_lines.join("\n")
    }

    /// Pairs a sequence of food names (Column 3) and wrapped quantity strings (Column 4)
    /// from a PDF table row into `- {quantity} de {food_name}` lines.
    fn pair_tabular_foods_and_quantities(cell_lines: &[String]) -> Vec<String> {
        if cell_lines.is_empty() {
            return Vec::new();
        }

        // Find the split point where the Quantity column starts
        let qty_start_idx = cell_lines
            .iter()
            .position(|l| is_quantity_column_start(l))
            .unwrap_or(cell_lines.len());

        // If the first line already starts with a quantity (inline bullet format), return as-is
        if qty_start_idx == 0 {
            return cell_lines.to_vec();
        }

        let food_names = &cell_lines[..qty_start_idx];
        let raw_qty_lines = &cell_lines[qty_start_idx..];

        // Merge wrapped quantity lines (e.g. `2 unidades` + `(100g)`, or `1,5` + `escumadeira` + `(150g)`)
        let mut merged_quantities: Vec<String> = Vec::new();
        for qline in raw_qty_lines {
            let trimmed = qline.trim_start_matches(&['•', '-', '*'][..]).trim();
            if trimmed.is_empty() {
                continue;
            }
            let lower = strip_diacritics(trimmed).to_lowercase();

            let is_continuation = if merged_quantities.is_empty() {
                false
            } else {
                let prev = merged_quantities.last().unwrap();
                let prev_lower = strip_diacritics(prev).to_lowercase();
                trimmed.starts_with('(')
                    || prev_lower.ends_with(" de")
                    || prev_lower.ends_with("col.")
                    || prev_lower.ends_with("col")
                    || prev_lower.ends_with('+')
                    || prev.chars().all(|c| c.is_ascii_digit() || c == ',' || c == '.')
                    || (!starts_with_digit_or_fraction(&lower) && !lower.starts_with("a vontade"))
            };

            if is_continuation {
                if let Some(last) = merged_quantities.last_mut() {
                    last.push(' ');
                    last.push_str(trimmed);
                }
            } else {
                merged_quantities.push(trimmed.to_string());
            }
        }

        let mut output = Vec::new();
        for (idx, food) in food_names.iter().enumerate() {
            let clean_food = food.trim_start_matches(&['•', '-', '*'][..]).trim();
            let qty_str = merged_quantities
                .get(idx)
                .map(|s| s.as_str())
                .unwrap_or("100g");

            let qty_lower = strip_diacritics(qty_str).to_lowercase();
            if qty_lower.contains("a vontade") {
                output.push(format!("- 100g de {}", clean_food));
            } else {
                output.push(format!("- {} de {}", qty_str, clean_food));
            }
        }
        output
    }

    /// Parses a `.pdf` or `.txt` nutrition plan byte payload into a structured `ParsedNutritionPlan`,
    /// extracting user profile metadata, meal schedule, probable daily foods, and calorie exchange options.
    pub fn parse_nutrition_plan(
        user_id: impl Into<String>,
        filename: &str,
        bytes: &[u8],
    ) -> Result<ParsedNutritionPlan, SharedDomainError> {
        let user_id_str = user_id.into();
        let (source_format, raw_extracted_text) = Self::extract_text_from_bytes(filename, bytes)?;
        let normalized_text = Self::normalize_tabular_plan_text(&raw_extracted_text);

        let mut hasher = Sha256::new();
        hasher.update(bytes);
        let file_sha256 = hex::encode(hasher.finalize());

        let mut title = "Plano Alimentar Personalizado / Custom Nutrition Plan".to_string();
        let mut nutritionist_name: Option<String> = None;
        let mut nutritionist_crn: Option<String> = None;
        let mut extracted_profile = ExtractedPlanUserProfile::default();
        let mut explicit_target_kcal: Option<f32> = None;
        let mut explicit_protein_g: Option<f32> = None;
        let mut explicit_carbs_g: Option<f32> = None;
        let mut explicit_fat_g: Option<f32> = None;
        let mut hydration_target_ml: u32 = 2500;
        let mut general_guidelines: Vec<String> = Vec::new();
        let mut meals: Vec<PrescribedMealSlot> = Vec::new();
        let mut current_meal: Option<PrescribedMealSlot> = None;
        let mut in_guidelines_section = false;

        for (line_idx, raw_line) in normalized_text.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            let lower = strip_diacritics(line).to_lowercase();

            // Check if the first heading line is a standalone plan title like "Plano Alimentar Individualizado"
            if line_idx == 0 && lower.starts_with("plano alimentar") && !lower.contains(':') {
                title = line.to_string();
                continue;
            }

            // 1. Check metadata headers (Patient, CPF, Email, Phone, Address, Nutritionist, CRN, Targets)
            if let Some(val) = extract_header_value(
                line,
                &lower,
                &["plano alimentar:", "nutrition plan:", "plano:", "plan:"],
            ) {
                if !val.is_empty() {
                    title = val;
                }
                continue;
            }
            if let Some(val) = extract_header_value(
                line,
                &lower,
                &["nutricionista:", "nutritionist:", "dra.", "dr."],
            ) {
                if lower.contains("crn") {
                    if let Some((name_part, crn_part)) = split_name_and_crn(&val) {
                        nutritionist_name = Some(name_part);
                        nutritionist_crn = Some(crn_part);
                    } else {
                        nutritionist_name = Some(val);
                    }
                } else {
                    let prefix_dr = if lower.starts_with("dra.") {
                        "Dra. "
                    } else if lower.starts_with("dr.") {
                        "Dr. "
                    } else {
                        ""
                    };
                    nutritionist_name = Some(format!("{}{}", prefix_dr, val));
                }
                continue;
            }
            if lower.starts_with("crn") {
                nutritionist_crn = Some(line.to_string());
                continue;
            }
            if let Some(val) = extract_header_value(
                line,
                &lower,
                &["paciente:", "patient:", "cliente:", "nome:", "name:"],
            ) {
                extracted_profile.full_name = Some(val);
                continue;
            }
            if let Some(val) = extract_header_value(line, &lower, &["cpf:"]) {
                extracted_profile.cpf = validate_and_format_cpf(&val).ok().or(Some(val));
                continue;
            }
            if let Some(val) = extract_header_value(line, &lower, &["email:", "e-mail:"]) {
                extracted_profile.email = Some(val);
                continue;
            }
            if let Some(val) = extract_header_value(
                line,
                &lower,
                &["telefone:", "celular:", "phone:", "whatsapp:"],
            ) {
                extracted_profile.phone = Some(val);
                continue;
            }
            if let Some(val) = extract_header_value(line, &lower, &["endereco:", "address:"]) {
                extracted_profile.address = Some(val);
                continue;
            }
            if let Some(val) = extract_header_value(line, &lower, &["objetivo:", "goal:"]) {
                extracted_profile.goal = Some(val.clone());
                if title.starts_with("Plano Alimentar Personalizado") {
                    title = format!("Plano Alimentar - {}", val);
                }
                continue;
            }

            // Daily calorie / macro / hydration targets (including Brazilian thousand separator `2.100 kcal`)
            if lower.starts_with("meta")
                || lower.starts_with("vet")
                || lower.starts_with("calorias")
                || lower.starts_with("target")
                || lower.starts_with("daily calories")
                || lower.starts_with("macros")
            {
                if let Some(kcal) = extract_number_before(&lower, "kcal") {
                    explicit_target_kcal = Some(kcal);
                }
                if let Some(p) = extract_macro_value(&lower, &["proteina", "protein", "p:"]) {
                    explicit_protein_g = Some(p);
                }
                if let Some(c) =
                    extract_macro_value(&lower, &["carboidrato", "carbs", "carb", "c:"])
                {
                    explicit_carbs_g = Some(c);
                }
                if let Some(f) = extract_macro_value(&lower, &["gordura", "fat", "lipidio", "f:"]) {
                    explicit_fat_g = Some(f);
                }
                continue;
            }

            if (lower.starts_with("hidratacao")
                || lower.starts_with("agua")
                || lower.starts_with("water")
                || lower.starts_with("hydration")
                || lower.contains("litros de agua"))
                && (lower.contains("ml") || lower.contains("litro") || lower.contains(" l"))
            {
                if let Some(ml) = extract_number_before(&lower, "ml") {
                    hydration_target_ml = ml.round() as u32;
                } else if let Some(liters) = extract_number_before(&lower, "litro")
                    .or_else(|| extract_number_before(&lower, " l"))
                {
                    hydration_target_ml = (liters * 1000.0).round() as u32;
                }
                if !in_guidelines_section {
                    continue;
                }
            }

            // Check if entering General Guidelines section
            if lower.starts_with("orientacoes")
                || lower.starts_with("observacoes")
                || lower.starts_with("guidelines")
                || lower.starts_with("notes:")
                || lower.starts_with("regras")
            {
                if let Some(meal) = current_meal.take() {
                    if !meal.primary_foods.is_empty() {
                        meals.push(meal);
                    }
                }
                in_guidelines_section = true;
                if let Some((_, rest)) = line.split_once(':') {
                    if !rest.trim().is_empty() {
                        general_guidelines.push(rest.trim().to_string());
                    }
                }
                continue;
            }

            // 2. Check if this line is a Meal Header (e.g. "07:00 - Café da Manhã", "12:30 - Almoço")
            if let Some((meal_name, scheduled_time)) = Self::detect_meal_header(line, &lower) {
                in_guidelines_section = false;
                if let Some(prev_meal) = current_meal.take() {
                    if !prev_meal.primary_foods.is_empty() {
                        meals.push(prev_meal);
                    }
                }
                current_meal = Some(PrescribedMealSlot {
                    meal_slot_id: format!("slot-{}", meals.len() + 1),
                    meal_name,
                    scheduled_time,
                    primary_foods: Vec::new(),
                    calorie_exchanges: Vec::new(),
                    total_meal_macros: MacroSummary::default(),
                });
                continue;
            }

            if in_guidelines_section {
                let cleaned_rule = line.trim_start_matches(&['-', '*', '•'][..]).trim();
                if !cleaned_rule.is_empty()
                    && !matches!(
                        lower.as_str(),
                        "hidratacao" | "preparo dos alimentos" | "substituicoes e observacoes"
                    )
                {
                    general_guidelines.push(cleaned_rule.to_string());
                }
                continue;
            }

            // 3. Parse food item line (with optional "OU" / "OR" calorie exchange alternatives)
            if let Some((primary, exchanges)) = Self::parse_plan_food_line(
                line,
                current_meal
                    .as_ref()
                    .map(|m| m.meal_name.as_str())
                    .unwrap_or("Refeição Diária / Daily Meal"),
                current_meal.as_ref().and_then(|m| m.scheduled_time.clone()),
            ) {
                let slot = current_meal.get_or_insert_with(|| PrescribedMealSlot {
                    meal_slot_id: "slot-1".to_string(),
                    meal_name: "Refeição Principal / Main Meal".to_string(),
                    scheduled_time: Some("12:00".to_string()),
                    primary_foods: Vec::new(),
                    calorie_exchanges: Vec::new(),
                    total_meal_macros: MacroSummary::default(),
                });
                slot.total_meal_macros.add_assign(&primary.macros);
                slot.primary_foods.push(primary);
                slot.calorie_exchanges.extend(exchanges);
            }
        }

        if let Some(last_meal) = current_meal.take() {
            if !last_meal.primary_foods.is_empty() {
                meals.push(last_meal);
            }
        }

        // If the plan has Weekday (Seg-Sex) and Weekend (Sáb-Dom) meals (like `Plano Alimentar Teste.pdf`),
        // link Weekday <-> Weekend equivalent slots as Calorie Exchange Options so the user can swap them!
        Self::populate_cross_day_and_default_exchanges(&mut meals);

        // Consolidate probable daily foods, calorie exchanges, and planned daily macros
        let mut planned_daily_macros = MacroSummary::default();
        let mut probable_daily_foods = Vec::new();
        let mut calorie_exchange_options = Vec::new();

        let has_weekday_tags = meals.iter().any(|m| m.meal_name.contains("(Seg-Sex)"));
        for meal in &meals {
            // Sum daily macros for primary day (or all meals if single-day plan)
            if !has_weekday_tags || meal.meal_name.contains("(Seg-Sex)") {
                planned_daily_macros.add_assign(&meal.total_meal_macros);
            }
            for food in &meal.primary_foods {
                probable_daily_foods.push(food.clone());
            }
            for ex in &meal.calorie_exchanges {
                calorie_exchange_options.push(ex.clone());
            }
        }

        let target_daily_macros = MacroSummary {
            calories_kcal: explicit_target_kcal.unwrap_or_else(|| {
                if planned_daily_macros.calories_kcal > 0.0 {
                    planned_daily_macros.calories_kcal
                } else {
                    2200.0
                }
            }),
            protein_g: explicit_protein_g.unwrap_or_else(|| {
                if planned_daily_macros.protein_g > 0.0 {
                    planned_daily_macros.protein_g
                } else {
                    165.0
                }
            }),
            carbs_g: explicit_carbs_g.unwrap_or_else(|| {
                if planned_daily_macros.carbs_g > 0.0 {
                    planned_daily_macros.carbs_g
                } else {
                    220.0
                }
            }),
            fat_g: explicit_fat_g.unwrap_or_else(|| {
                if planned_daily_macros.fat_g > 0.0 {
                    planned_daily_macros.fat_g
                } else {
                    65.0
                }
            }),
            fiber_g: if planned_daily_macros.fiber_g > 0.0 {
                planned_daily_macros.fiber_g
            } else {
                30.0
            },
        };

        Ok(ParsedNutritionPlan {
            plan_id: format!("plan-{}", &Uuid::new_v4().to_string()[..8]),
            user_id: user_id_str,
            source_filename: filename.to_string(),
            source_format,
            file_sha256,
            title,
            nutritionist_name,
            nutritionist_crn,
            extracted_user_profile: extracted_profile,
            target_daily_macros,
            planned_daily_macros,
            hydration_target_ml,
            meals,
            probable_daily_foods,
            calorie_exchange_options,
            general_guidelines,
            raw_extracted_text,
        })
    }

    /// Links Weekday (`(Seg-Sex)`) and Weekend (`(Sáb-Dom)`) meal equivalents as `CalorieExchangeOption`s
    /// when a plan separates Weekday and Weekend tables (such as `Plano Alimentar Teste.pdf`).
    fn populate_cross_day_and_default_exchanges(meals: &mut [PrescribedMealSlot]) {
        let weekend_meals: Vec<PrescribedMealSlot> = meals
            .iter()
            .filter(|m| m.meal_name.contains("(Sáb-Dom)"))
            .cloned()
            .collect();

        if weekend_meals.is_empty() {
            return;
        }

        for meal in meals.iter_mut() {
            if !meal.meal_name.contains("(Seg-Sex)") {
                continue;
            }
            let meal_lower = strip_diacritics(&meal.meal_name).to_lowercase();
            let matching_weekend = weekend_meals.iter().find(|wm| {
                let wm_lower = strip_diacritics(&wm.meal_name).to_lowercase();
                (meal_lower.contains("cafe da manha") && wm_lower.contains("cafe da manha"))
                    || (meal_lower.contains("almoco") && wm_lower.contains("almoco"))
                    || (meal_lower.contains("lanche da tarde")
                        && wm_lower.contains("lanche da tarde"))
                    || (meal_lower.contains("jantar") && wm_lower.contains("jantar"))
            });

            if let Some(wk_meal) = matching_weekend {
                let pairs = meal
                    .primary_foods
                    .len()
                    .min(wk_meal.primary_foods.len());
                for idx in 0..pairs {
                    let primary = &meal.primary_foods[idx];
                    let replacement = &wk_meal.primary_foods[idx];
                    if primary.canonical_key != replacement.canonical_key {
                        let delta =
                            round1(replacement.macros.calories_kcal - primary.macros.calories_kcal);
                        meal.calorie_exchanges.push(CalorieExchangeOption {
                            exchange_id: format!("ex-{}", &Uuid::new_v4().to_string()[..8]),
                            meal_name: meal.meal_name.clone(),
                            primary_food_key: primary.canonical_key.clone(),
                            primary_food_name: primary.display_name.clone(),
                            primary_grams: primary.estimated_grams,
                            primary_calories_kcal: primary.macros.calories_kcal,
                            replacement_food_key: replacement.canonical_key.clone(),
                            replacement_food_name: replacement.display_name.clone(),
                            replacement_grams: replacement.estimated_grams,
                            replacement_macros: replacement.macros.clone(),
                            calorie_delta_kcal: delta,
                        });
                    }
                }
            }
        }
    }

    fn detect_meal_header(line: &str, lower_ascii: &str) -> Option<(String, Option<String>)> {
        let trimmed = line.trim();
        if trimmed.starts_with("- ") || trimmed.starts_with("* ") || trimmed.starts_with("• ") {
            return None;
        }
        let cleaned = trimmed.trim_matches(&['#', '*', '-', ' '][..]);
        let meal_keywords = [
            "cafe da manha",
            "desjejum",
            "colacao",
            "lanche da manha",
            "almoco",
            "lanche da tarde",
            "pre-treino",
            "pre treino",
            "pos-treino",
            "pos treino",
            "jantar",
            "ceia",
            "breakfast",
            "morning snack",
            "lunch",
            "afternoon snack",
            "pre-workout",
            "post-workout",
            "dinner",
            "supper",
            "refeicao 1",
            "refeicao 2",
            "refeicao 3",
            "refeicao 4",
            "refeicao 5",
            "meal 1",
            "meal 2",
            "meal 3",
            "meal 4",
        ];

        let has_food_grams = lower_ascii.contains("0g ")
            || lower_ascii.contains("5g ")
            || lower_ascii.contains("0ml")
            || lower_ascii.contains(" ou ")
            || lower_ascii.contains(" or ");
        if has_food_grams {
            return None;
        }

        if meal_keywords.iter().any(|k| lower_ascii.contains(k)) && cleaned.len() <= 72 {
            let time_opt = extract_time_hhmm(cleaned);
            let title_clean = cleaned.trim_end_matches(':').trim().to_string();
            return Some((title_clean, time_opt));
        }
        None
    }

    fn parse_plan_food_line(
        line: &str,
        meal_name: &str,
        scheduled_time: Option<String>,
    ) -> Option<(ProbableDailyFood, Vec<CalorieExchangeOption>)> {
        let stripped = line
            .trim_start_matches(&['-', '*', '•'][..])
            .trim();
        if stripped.len() < 3 {
            return None;
        }

        // Split primary option vs substitution alternatives separated by " OU " / " ou " / " OR " / " or " / " | "
        let normalized_seps = stripped
            .replace(" OU ", " | ")
            .replace(" ou ", " | ")
            .replace(" OR ", " | ")
            .replace(" or ", " | ");

        let parts: Vec<&str> = normalized_seps
            .split('|')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        if parts.is_empty() {
            return None;
        }

        let primary = Self::parse_single_food_option(parts[0], meal_name, scheduled_time, true)?;
        let mut exchanges = Vec::new();

        for alt_raw in parts.iter().skip(1) {
            // If an inline alternative like "Maçã fuji ou gala" or "Chá de camomila ou erva-doce"
            // doesn't repeat the quantity, inherit the primary item's grams!
            let alt_food = if let Some(parsed_alt) =
                Self::parse_single_food_option(alt_raw, meal_name, None, false)
            {
                parsed_alt
            } else {
                let inherited_segment = format!(
                    "{:.0}g de {}",
                    primary.estimated_grams,
                    alt_raw.trim()
                );
                match Self::parse_single_food_option(&inherited_segment, meal_name, None, false) {
                    Some(p) => p,
                    None => continue,
                }
            };

            let delta = round1(alt_food.macros.calories_kcal - primary.macros.calories_kcal);
            exchanges.push(CalorieExchangeOption {
                exchange_id: format!("ex-{}", &Uuid::new_v4().to_string()[..8]),
                meal_name: meal_name.to_string(),
                primary_food_key: primary.canonical_key.clone(),
                primary_food_name: primary.display_name.clone(),
                primary_grams: primary.estimated_grams,
                primary_calories_kcal: primary.macros.calories_kcal,
                replacement_food_key: alt_food.canonical_key,
                replacement_food_name: alt_food.display_name,
                replacement_grams: alt_food.estimated_grams,
                replacement_macros: alt_food.macros,
                calorie_delta_kcal: delta,
            });
        }

        Some((primary, exchanges))
    }

    fn parse_single_food_option(
        segment: &str,
        meal_name: &str,
        scheduled_time: Option<String>,
        is_primary_choice: bool,
    ) -> Option<ProbableDailyFood> {
        let ascii = strip_diacritics(segment).to_lowercase();
        let (quantity, unit, estimated_grams, cleaned_name) =
            extract_quantity_unit_and_name(segment, &ascii)?;

        let canonical_key = canonicalize_plan_food_key(&cleaned_name);
        if canonical_key.is_empty() {
            return None;
        }
        let macros_100g = lookup_reference_macros_per_100g(&canonical_key);
        let scaled_macros = macros_100g.scaled_for_grams(estimated_grams);

        Some(ProbableDailyFood {
            food_id: format!("pfood-{}", &Uuid::new_v4().to_string()[..8]),
            meal_name: meal_name.to_string(),
            scheduled_time,
            canonical_key,
            display_name: cleaned_name,
            quantity,
            unit,
            estimated_grams,
            macros: scaled_macros,
            is_primary_choice,
        })
    }
}

fn is_standalone_time_hhmm(line: &str) -> bool {
    let b = line.trim().as_bytes();
    b.len() == 5
        && b[0].is_ascii_digit()
        && b[1].is_ascii_digit()
        && b[2] == b':'
        && b[3].is_ascii_digit()
        && b[4].is_ascii_digit()
}

fn starts_with_digit_or_fraction(lower: &str) -> bool {
    let trimmed = lower.trim_start_matches(&['•', '-', '*', ' '][..]);
    trimmed
        .chars()
        .next()
        .map(|c| c.is_ascii_digit())
        .unwrap_or(false)
}

fn is_quantity_column_start(line: &str) -> bool {
    let lower = strip_diacritics(line).to_lowercase();
    let trimmed = lower.trim_start_matches(&['•', '-', '*', ' '][..]);
    if trimmed.starts_with("a vontade") {
        return true;
    }
    if !starts_with_digit_or_fraction(trimmed) {
        return false;
    }
    let unit_markers = [
        "fatia",
        "unidade",
        "xicara",
        "colher",
        "col.",
        "concha",
        "file",
        "pote",
        "escumadeira",
        "copo",
        "g)",
        "ml)",
        "0g",
        "5g",
        "banana",
    ];
    unit_markers.iter().any(|m| trimmed.contains(m))
        || trimmed.chars().all(|c| c.is_ascii_digit() || c == ',' || c == '.')
}

fn strip_diacritics(input: &str) -> String {
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

fn extract_header_value(raw_line: &str, lower_ascii: &str, prefixes: &[&str]) -> Option<String> {
    for prefix in prefixes {
        if lower_ascii.starts_with(prefix) {
            if let Some((_, val)) = raw_line.split_once(':') {
                return Some(val.trim().to_string());
            }
            let rest = raw_line[prefix.len()..].trim();
            return Some(rest.to_string());
        }
    }
    None
}

fn split_name_and_crn(val: &str) -> Option<(String, String)> {
    let lower = val.to_lowercase();
    let pos = lower.find("crn")?;
    let name = val[..pos]
        .trim()
        .trim_end_matches(&['-', '|', '(', ','][..])
        .trim()
        .to_string();
    let crn = val[pos..]
        .trim()
        .trim_end_matches(')')
        .trim()
        .to_string();
    Some((name, crn))
}

fn extract_number_before(lower: &str, suffix: &str) -> Option<f32> {
    let pos = lower.find(suffix)?;
    let before = lower[..pos].trim_end();
    let num_str: String = before
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == ',')
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    if num_str.is_empty() {
        return None;
    }
    // Handle Brazilian thousand separator like `2.100 kcal` -> `2100.0`
    let normalized = if num_str.contains('.') && !num_str.contains(',') {
        let parts: Vec<&str> = num_str.split('.').collect();
        if parts.len() == 2 && parts[1].len() == 3 {
            format!("{}{}", parts[0], parts[1])
        } else {
            num_str
        }
    } else {
        num_str.replace('.', "").replace(',', ".")
    };
    normalized.parse::<f32>().ok()
}

fn extract_macro_value(lower: &str, labels: &[&str]) -> Option<f32> {
    for label in labels {
        if let Some(pos) = lower.find(label) {
            let after = &lower[pos + label.len()..];
            let num_str: String = after
                .chars()
                .skip_while(|c| !c.is_ascii_digit())
                .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == ',')
                .collect();
            if let Ok(val) = num_str.replace(',', ".").parse::<f32>() {
                return Some(val);
            }
        }
    }
    None
}

fn extract_time_hhmm(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    for i in 0..bytes.len().saturating_sub(4) {
        if bytes[i].is_ascii_digit()
            && bytes[i + 1].is_ascii_digit()
            && (bytes[i + 2] == b':' || bytes[i + 2] == b'h')
            && bytes[i + 3].is_ascii_digit()
            && bytes[i + 4].is_ascii_digit()
        {
            return Some(format!(
                "{}{}:{}{}",
                bytes[i] as char,
                bytes[i + 1] as char,
                bytes[i + 3] as char,
                bytes[i + 4] as char
            ));
        }
    }
    None
}

fn extract_quantity_unit_and_name(
    raw_segment: &str,
    lower_ascii: &str,
) -> Option<(f32, String, f32, String)> {
    // Extract parenthesized gram/ml weights like "(50g)", "(200ml)", "(8ml)"
    let mut explicit_parens_grams: Option<f32> = None;
    let mut search_slice = lower_ascii;
    while let Some(open) = search_slice.find('(') {
        if let Some(close_rel) = search_slice[open + 1..].find(')') {
            let inside = &search_slice[open + 1..open + 1 + close_rel];
            if let Some(g) = extract_number_before(inside, "g")
                .or_else(|| extract_number_before(inside, "ml"))
            {
                explicit_parens_grams = Some(g);
                break;
            }
            search_slice = &search_slice[open + 1 + close_rel + 1..];
        } else {
            break;
        }
    }

    // Remove only parenthesized segments that specify weights like `(50g)` or `(200ml)`,
    // preserving descriptive food parentheses like `(100% trigo)` or `(sem açúcar)`!
    let without_weight_parens = strip_weight_parentheses(raw_segment);

    // If segment contains " de " after a quantity expression (like `4 col. sopa de Arroz integral cozido`
    // or `1/2 banana + 1 fatia de Vitamina de banana com mamão`), split on the quantity-to-food boundary
    let tokens: Vec<&str> = without_weight_parens.split_whitespace().collect();
    if tokens.is_empty() {
        return None;
    }

    let first_lower = strip_diacritics(tokens[0]).to_lowercase();
    let mut quantity = 1.0_f32;
    let mut unit = "g".to_string();
    let mut consumed = 0_usize;

    let parsed_leading_num = if first_lower == "1/2" {
        Some((0.5_f32, String::new()))
    } else {
        let num_part: String = first_lower
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == ',')
            .collect();
        let unit_part: String = first_lower[num_part.len()..].to_string();
        if !num_part.is_empty() {
            num_part
                .replace(',', ".")
                .parse::<f32>()
                .ok()
                .map(|n| (n, unit_part))
        } else {
            None
        }
    };

    if let Some((n, unit_part)) = parsed_leading_num {
        if n <= 0.0 {
            return None;
        }
        quantity = n;
        consumed = 1;
        if !unit_part.is_empty() {
            unit = unit_part;
        } else if tokens.len() > 1 {
            let second_lower = strip_diacritics(tokens[1])
                .to_lowercase()
                .trim_end_matches('.')
                .to_string();
            if matches!(
                second_lower.as_str(),
                "g" | "gramas"
                    | "ml"
                    | "unidade"
                    | "unidades"
                    | "fatia"
                    | "fatias"
                    | "colher"
                    | "colheres"
                    | "col"
                    | "concha"
                    | "conchas"
                    | "scoop"
                    | "scoops"
                    | "tbsp"
                    | "xicara"
                    | "xicaras"
                    | "copo"
                    | "copos"
                    | "pote"
                    | "potes"
                    | "file"
                    | "files"
                    | "escumadeira"
                    | "banana"
            ) {
                unit = second_lower;
                consumed = 2;
            } else {
                unit = "unit".to_string();
            }
        }
    } else if explicit_parens_grams.is_none() {
        return None;
    }

    // Consume unit modifiers ("de", "do", "da", "sopa", "cha", "sobremesa", "media", "medio", "grande", "finas", "+ 1 fatia")
    while consumed < tokens.len() {
        let t = strip_diacritics(tokens[consumed])
            .to_lowercase()
            .trim_end_matches('.')
            .to_string();
        if matches!(
            t.as_str(),
            "de" | "do"
                | "da"
                | "of"
                | "sopa"
                | "cha"
                | "sobremesa"
                | "media"
                | "medio"
                | "grande"
                | "finas"
                | "fina"
                | "+"
                | "1"
                | "fatia"
        ) {
            consumed += 1;
        } else {
            break;
        }
    }

    let food_name = tokens[consumed..].join(" ").trim().to_string();
    if food_name.is_empty() {
        return None;
    }

    let estimated_grams = if let Some(pg) = explicit_parens_grams {
        pg
    } else {
        match unit.as_str() {
            "g" | "gramas" | "ml" => quantity,
            "unidade" | "unidades" | "unit" => {
                let lower_food = strip_diacritics(&food_name).to_lowercase();
                if lower_food.contains("ovo") || lower_food.contains("egg") {
                    quantity * 50.0
                } else if lower_food.contains("banana") {
                    quantity * 90.0
                } else if lower_food.contains("pao") {
                    quantity * 50.0
                } else {
                    quantity * 80.0
                }
            }
            "fatia" | "fatias" | "slice" => quantity * 30.0,
            "colher" | "colheres" | "col" | "tbsp" => quantity * 15.0,
            "concha" | "conchas" => quantity * 100.0,
            "escumadeira" => quantity * 100.0,
            "file" | "files" => quantity * 120.0,
            "pote" | "potes" => quantity * 170.0,
            "scoop" | "scoops" => quantity * 30.0,
            "xicara" | "xicaras" | "copo" | "copos" | "cup" => quantity * 200.0,
            "banana" => 120.0,
            _ => quantity,
        }
    };

    Some((quantity, unit, estimated_grams, food_name))
}

fn strip_weight_parentheses(raw: &str) -> String {
    let mut result = String::new();
    let chars: Vec<char> = raw.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '(' {
            let mut j = i + 1;
            while j < chars.len() && chars[j] != ')' {
                j += 1;
            }
            if j < chars.len() {
                let inside: String = chars[i + 1..j].iter().collect();
                let lower = inside.trim().to_lowercase();
                let is_weight = (lower.ends_with('g') || lower.ends_with("ml"))
                    && lower
                        .chars()
                        .next()
                        .map(|c| c.is_ascii_digit())
                        .unwrap_or(false);
                if is_weight {
                    i = j + 1;
                    continue;
                }
            }
        }
        result.push(chars[i]);
        i += 1;
    }
    result.trim().to_string()
}

/// Maps Portuguese and English food names in a nutrition plan to canonical keys.
pub fn canonicalize_plan_food_key(raw_name: &str) -> String {
    let lower = strip_diacritics(raw_name).to_lowercase();
    if lower.contains("frango") || lower.contains("chicken") {
        "grilled_chicken_breast".to_string()
    } else if lower.contains("tilapia") || lower.contains("peixe") || lower.contains("white fish") {
        "tilapia".to_string()
    } else if lower.contains("salmao") || lower.contains("salmon") {
        "salmon".to_string()
    } else if lower.contains("patinho")
        || lower.contains("carne moida")
        || lower.contains("ground beef")
    {
        "ground_beef".to_string()
    } else if lower.contains("picanha") {
        "picanha".to_string()
    } else if lower.contains("macarrao integral") || lower.contains("whole wheat pasta") {
        "whole_wheat_pasta".to_string()
    } else if lower.contains("arroz integral") || lower.contains("brown rice") {
        "brown_rice".to_string()
    } else if lower.contains("arroz") || lower.contains("rice") {
        "rice".to_string()
    } else if lower.contains("batata doce") || lower.contains("sweet potato") {
        "sweet_potato".to_string()
    } else if lower.contains("mandioca") || lower.contains("aipim") || lower.contains("cassava") {
        "mandioca".to_string()
    } else if lower.contains("feijao") || lower.contains("beans") {
        "feijao".to_string()
    } else if lower.contains("ovo") || lower.contains("omelete") || lower.contains("egg") {
        "egg".to_string()
    } else if lower.contains("aveia") || lower.contains("oats") || lower.contains("oatmeal") {
        "oats".to_string()
    } else if lower.contains("vitamina de banana") {
        "banana_papaya_smoothie".to_string()
    } else if lower.contains("banana") {
        "banana".to_string()
    } else if lower.contains("tapioca") {
        "tapioca".to_string()
    } else if lower.contains("queijo minas") {
        "minas_cheese".to_string()
    } else if lower.contains("pao de queijo") {
        "pao_de_queijo".to_string()
    } else if lower.contains("torrada integral") {
        "whole_wheat_toast".to_string()
    } else if lower.contains("pao integral") || lower.contains("whole wheat bread") {
        "whole_wheat_bread".to_string()
    } else if lower.contains("pao") || lower.contains("bread") {
        "pao_frances".to_string()
    } else if lower.contains("whey") || lower.contains("proteina") {
        "whey_protein".to_string()
    } else if lower.contains("azeite") || lower.contains("olive oil") {
        "olive_oil".to_string()
    } else if lower.contains("castanha do para") || lower.contains("brazil nut") {
        "brazil_nut".to_string()
    } else if lower.contains("pasta de amendoim") || lower.contains("peanut butter") {
        "peanut_butter".to_string()
    } else if lower.contains("abacate") || lower.contains("avocado") {
        "avocado".to_string()
    } else if lower.contains("iogurte") || lower.contains("yogurt") {
        "greek_yogurt".to_string()
    } else if lower.contains("leite desnatado") || lower.contains("skim milk") {
        "skim_milk".to_string()
    } else if lower.contains("suco de laranja") || lower.contains("orange juice") {
        "orange_juice".to_string()
    } else if lower.contains("cafe coado") || lower.contains("cafe sem acucar") {
        "black_coffee".to_string()
    } else if lower.contains("cha de") || lower.contains("erva-doce") || lower.contains("camomila")
    {
        "herbal_tea".to_string()
    } else if lower.contains("maracuja") {
        "passion_fruit".to_string()
    } else if lower.contains("brocolis") || lower.contains("broccoli") {
        "broccoli".to_string()
    } else if lower.contains("salada") {
        "green_salad".to_string()
    } else if lower.contains("tomate") {
        "tomato_onion".to_string()
    } else if lower.contains("mamao") || lower.contains("papaya") {
        "papaya".to_string()
    } else if lower.contains("maca") || lower.contains("gala") || lower.contains("apple") {
        "apple".to_string()
    } else {
        lower
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect::<String>()
            .split('_')
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("_")
    }
}

/// Reference nutritional table (TACO / USDA per 100g) for plan foods & substitution calculations.
pub fn lookup_reference_macros_per_100g(canonical_key: &str) -> MacroSummary {
    match canonical_key {
        "grilled_chicken_breast" => MacroSummary {
            calories_kcal: 165.0,
            protein_g: 31.0,
            carbs_g: 0.0,
            fat_g: 3.6,
            fiber_g: 0.0,
        },
        "tilapia" => MacroSummary {
            calories_kcal: 128.0,
            protein_g: 26.2,
            carbs_g: 0.0,
            fat_g: 2.7,
            fiber_g: 0.0,
        },
        "salmon" => MacroSummary {
            calories_kcal: 208.0,
            protein_g: 20.4,
            carbs_g: 0.0,
            fat_g: 13.4,
            fiber_g: 0.0,
        },
        "ground_beef" => MacroSummary {
            calories_kcal: 219.0,
            protein_g: 35.9,
            carbs_g: 0.0,
            fat_g: 7.3,
            fiber_g: 0.0,
        },
        "picanha" => MacroSummary {
            calories_kcal: 289.0,
            protein_g: 26.4,
            carbs_g: 0.0,
            fat_g: 19.5,
            fiber_g: 0.0,
        },
        "whole_wheat_pasta" => MacroSummary {
            calories_kcal: 124.0,
            protein_g: 5.3,
            carbs_g: 26.5,
            fat_g: 0.9,
            fiber_g: 3.9,
        },
        "brown_rice" => MacroSummary {
            calories_kcal: 124.0,
            protein_g: 2.6,
            carbs_g: 25.8,
            fat_g: 1.0,
            fiber_g: 2.7,
        },
        "rice" => MacroSummary {
            calories_kcal: 130.0,
            protein_g: 2.7,
            carbs_g: 28.2,
            fat_g: 0.3,
            fiber_g: 0.4,
        },
        "sweet_potato" => MacroSummary {
            calories_kcal: 86.0,
            protein_g: 1.6,
            carbs_g: 20.1,
            fat_g: 0.1,
            fiber_g: 3.0,
        },
        "mandioca" => MacroSummary {
            calories_kcal: 125.0,
            protein_g: 0.6,
            carbs_g: 30.1,
            fat_g: 0.3,
            fiber_g: 1.6,
        },
        "feijao" => MacroSummary {
            calories_kcal: 76.0,
            protein_g: 4.8,
            carbs_g: 13.6,
            fat_g: 0.5,
            fiber_g: 8.5,
        },
        "egg" => MacroSummary {
            calories_kcal: 155.0,
            protein_g: 12.6,
            carbs_g: 1.1,
            fat_g: 10.6,
            fiber_g: 0.0,
        },
        "oats" => MacroSummary {
            calories_kcal: 389.0,
            protein_g: 16.9,
            carbs_g: 66.3,
            fat_g: 6.9,
            fiber_g: 10.6,
        },
        "banana" => MacroSummary {
            calories_kcal: 89.0,
            protein_g: 1.1,
            carbs_g: 22.8,
            fat_g: 0.3,
            fiber_g: 2.6,
        },
        "banana_papaya_smoothie" => MacroSummary {
            calories_kcal: 66.0,
            protein_g: 0.8,
            carbs_g: 16.8,
            fat_g: 0.3,
            fiber_g: 2.1,
        },
        "papaya" => MacroSummary {
            calories_kcal: 43.0,
            protein_g: 0.5,
            carbs_g: 10.8,
            fat_g: 0.3,
            fiber_g: 1.7,
        },
        "apple" => MacroSummary {
            calories_kcal: 52.0,
            protein_g: 0.3,
            carbs_g: 13.8,
            fat_g: 0.2,
            fiber_g: 2.4,
        },
        "passion_fruit" => MacroSummary {
            calories_kcal: 68.0,
            protein_g: 2.0,
            carbs_g: 12.3,
            fat_g: 2.1,
            fiber_g: 1.5,
        },
        "tapioca" => MacroSummary {
            calories_kcal: 240.0,
            protein_g: 0.2,
            carbs_g: 59.8,
            fat_g: 0.0,
            fiber_g: 0.4,
        },
        "minas_cheese" => MacroSummary {
            calories_kcal: 264.0,
            protein_g: 17.4,
            carbs_g: 3.2,
            fat_g: 20.2,
            fiber_g: 0.0,
        },
        "whole_wheat_toast" => MacroSummary {
            calories_kcal: 377.0,
            protein_g: 13.4,
            carbs_g: 68.7,
            fat_g: 5.4,
            fiber_g: 7.4,
        },
        "whole_wheat_bread" => MacroSummary {
            calories_kcal: 247.0,
            protein_g: 13.0,
            carbs_g: 41.0,
            fat_g: 3.4,
            fiber_g: 7.0,
        },
        "pao_frances" => MacroSummary {
            calories_kcal: 300.0,
            protein_g: 8.0,
            carbs_g: 58.6,
            fat_g: 3.1,
            fiber_g: 2.3,
        },
        "whey_protein" => MacroSummary {
            calories_kcal: 375.0,
            protein_g: 78.0,
            carbs_g: 6.5,
            fat_g: 4.2,
            fiber_g: 0.5,
        },
        "olive_oil" => MacroSummary {
            calories_kcal: 884.0,
            protein_g: 0.0,
            carbs_g: 0.0,
            fat_g: 100.0,
            fiber_g: 0.0,
        },
        "brazil_nut" => MacroSummary {
            calories_kcal: 656.0,
            protein_g: 14.3,
            carbs_g: 12.3,
            fat_g: 66.4,
            fiber_g: 7.5,
        },
        "peanut_butter" => MacroSummary {
            calories_kcal: 588.0,
            protein_g: 25.1,
            carbs_g: 20.0,
            fat_g: 50.4,
            fiber_g: 6.0,
        },
        "greek_yogurt" => MacroSummary {
            calories_kcal: 56.0,
            protein_g: 5.8,
            carbs_g: 7.8,
            fat_g: 0.2,
            fiber_g: 0.0,
        },
        "skim_milk" => MacroSummary {
            calories_kcal: 35.0,
            protein_g: 3.4,
            carbs_g: 5.0,
            fat_g: 0.1,
            fiber_g: 0.0,
        },
        "orange_juice" => MacroSummary {
            calories_kcal: 45.0,
            protein_g: 0.7,
            carbs_g: 10.4,
            fat_g: 0.2,
            fiber_g: 0.2,
        },
        "black_coffee" => MacroSummary {
            calories_kcal: 2.0,
            protein_g: 0.1,
            carbs_g: 0.4,
            fat_g: 0.0,
            fiber_g: 0.0,
        },
        "herbal_tea" => MacroSummary {
            calories_kcal: 1.0,
            protein_g: 0.0,
            carbs_g: 0.2,
            fat_g: 0.0,
            fiber_g: 0.0,
        },
        "broccoli" => MacroSummary {
            calories_kcal: 35.0,
            protein_g: 2.4,
            carbs_g: 7.2,
            fat_g: 0.4,
            fiber_g: 3.3,
        },
        "green_salad" => MacroSummary {
            calories_kcal: 18.0,
            protein_g: 1.2,
            carbs_g: 3.3,
            fat_g: 0.2,
            fiber_g: 2.1,
        },
        "tomato_onion" => MacroSummary {
            calories_kcal: 24.0,
            protein_g: 1.0,
            carbs_g: 5.2,
            fat_g: 0.2,
            fiber_g: 1.4,
        },
        _ => MacroSummary {
            calories_kcal: 140.0,
            protein_g: 8.0,
            carbs_g: 18.0,
            fat_g: 4.0,
            fiber_g: 2.0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_plano_alimentar_teste_pdf_file_from_workspace() {
        let pdf_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../Plano Alimentar Teste.pdf");
        let pdf_bytes = std::fs::read(&pdf_path)
            .expect("Should read Plano Alimentar Teste.pdf from workspace root");

        let parsed = NutritionPlanDocumentReader::parse_nutrition_plan(
            "user-teste",
            "Plano Alimentar Teste.pdf",
            &pdf_bytes,
        )
        .expect("Should parse Plano Alimentar Teste.pdf without errors");

        assert_eq!(parsed.source_format, PlanFileFormat::Pdf);
        assert_eq!(parsed.title, "Plano Alimentar Individualizado");
        assert_eq!(
            parsed.nutritionist_name.as_deref(),
            Some("Dr. [Nome do Nutricionista]")
        );
        assert_eq!(parsed.nutritionist_crn.as_deref(), Some("CRN-3: 123456"));
        assert_eq!(
            parsed.extracted_user_profile.full_name.as_deref(),
            Some("[Nome do Paciente de Teste]")
        );
        assert_eq!(
            parsed.extracted_user_profile.goal.as_deref(),
            Some("Reeducação Alimentar e Manutenção")
        );
        // VET Calculado: 2.100 kcal/dia -> 2100.0 kcal
        assert_eq!(parsed.target_daily_macros.calories_kcal, 2100.0);
        // Consuma pelo menos 2,5 a 3 litros de água por dia -> 3000 ml
        assert_eq!(parsed.hydration_target_ml, 3000);

        // Verify all 10 meal slots (6 weekday + 4 weekend) and all 28 food items were extracted!
        assert_eq!(parsed.meals.len(), 10);
        assert_eq!(parsed.probable_daily_foods.len(), 28);
        assert!(!parsed.calorie_exchange_options.is_empty());
        assert!(parsed.general_guidelines.len() >= 5);
    }

    #[test]
    fn test_parse_txt_nutrition_plan_extracts_profile_meals_and_exchanges() {
        let txt_plan = r#"
Plano Alimentar: Hipertrofia Limpa & Definição
Nutricionista: Dra. Mariana Costa - CRN-3 45891
Paciente: Gabriel Silva
CPF: 12345678901
Telefone: (11) 99876-5432
Email: gabriel.silva@example.com
Endereço: Av. Paulista, 1000 - São Paulo, SP
Meta Calórica: 2350 kcal | Proteína: 180g | Carboidratos: 250g | Gordura: 65g
Hidratação: 3200 ml

07:30 - Café da Manhã
- 3 unidades de Ovos Mexidos (150g) OU 1 scoop de Whey Protein (30g)
- 2 fatias de Pão Integral (60g) OU 40g de Tapioca
- 1 unidade de Banana Prata (90g) OU 150g de Mamão

12:30 - Almoço
- 160g de Peito de Frango Grelhado OU 190g de Tilápia Grelhada OU 140g de Patinho Moído
- 150g de Arroz Integral Cozido OU 180g de Batata Doce Cozida
- 100g de Feijão Carioca Cozido
- 1 colher de sopa de Azeite de Oliva (10g)

16:30 - Lanche da Tarde
- 170g de Iogurte Grego Natural
- 40g de Aveia em Flocos
- 15g de Pasta de Amendoim

20:30 - Jantar
- 150g de Salmão Grelhado OU 160g de Peito de Frango Grelhado
- 130g de Mandioca Cozida OU 140g de Arroz Branco Cozido

Orientações Gerais:
- Pesar todos os alimentos já cozidos/prontos.
- Salada de folhas verdes à vontade no almoço e jantar.
"#;

        let parsed = NutritionPlanDocumentReader::parse_nutrition_plan(
            "user-gabriel",
            "plano_nutricionista.txt",
            txt_plan.as_bytes(),
        )
        .expect("Should parse .txt nutrition plan");

        assert_eq!(parsed.source_format, PlanFileFormat::Txt);
        assert_eq!(parsed.title, "Hipertrofia Limpa & Definição");
        assert_eq!(
            parsed.nutritionist_name.as_deref(),
            Some("Dra. Mariana Costa")
        );
        assert_eq!(parsed.nutritionist_crn.as_deref(), Some("CRN-3 45891"));
        assert_eq!(
            parsed.extracted_user_profile.full_name.as_deref(),
            Some("Gabriel Silva")
        );
        assert_eq!(
            parsed.extracted_user_profile.cpf.as_deref(),
            Some("123.456.789-01")
        );
        assert_eq!(
            parsed.extracted_user_profile.email.as_deref(),
            Some("gabriel.silva@example.com")
        );
        assert_eq!(parsed.target_daily_macros.calories_kcal, 2350.0);
        assert_eq!(parsed.target_daily_macros.protein_g, 180.0);
        assert_eq!(parsed.hydration_target_ml, 3200);
        assert_eq!(parsed.meals.len(), 4);
        assert_eq!(parsed.probable_daily_foods.len(), 12);
        assert!(parsed.calorie_exchange_options.len() >= 7);
        assert_eq!(parsed.general_guidelines.len(), 2);
    }

    #[test]
    fn test_parse_real_pdf_stream_nutrition_plan() {
        let pdf_lines = vec![
            "Plano Alimentar: Cutting Metabolico",
            "Nutricionista: Dr. Rafael Mendes - CRN-4 99120",
            "Paciente: Ana Beatriz",
            "CPF: 987.654.321-99",
            "Meta: 1850 kcal | Proteina: 150g | Carboidratos: 170g | Gordura: 55g",
            "Hidratacao: 2.8 L",
            "08:00 - Cafe da Manha",
            "- 2 unidades de Ovos Cozidos (100g) OU 30g de Whey Protein",
            "- 40g de Aveia OU 50g de Pao Integral",
            "13:00 - Almoco",
            "- 150g de Peito de Frango Grelhado OU 170g de Tilapia",
            "- 120g de Arroz Integral OU 150g de Batata Doce",
            "- 100g de Feijao",
        ];
        let pdf_bytes = NutritionPlanDocumentReader::build_pdf_document_from_text(&pdf_lines);
        assert!(pdf_bytes.starts_with(b"%PDF-1.4"));

        let parsed = NutritionPlanDocumentReader::parse_nutrition_plan(
            "user-ana",
            "plano_ana.pdf",
            &pdf_bytes,
        )
        .expect("Should parse binary .pdf nutrition plan");

        assert_eq!(parsed.source_format, PlanFileFormat::Pdf);
        assert_eq!(parsed.title, "Cutting Metabolico");
        assert_eq!(parsed.hydration_target_ml, 2800);
        assert_eq!(parsed.meals.len(), 2);
        assert_eq!(parsed.probable_daily_foods.len(), 5);
        assert_eq!(parsed.calorie_exchange_options.len(), 4);
        assert_eq!(
            parsed.extracted_user_profile.cpf.as_deref(),
            Some("987.654.321-99")
        );
    }

    #[test]
    fn test_cpf_validation_and_password_hashing() {
        let cpf = validate_and_format_cpf("11122233344").expect("Valid 11-digit CPF");
        assert_eq!(cpf, "111.222.333-44");
        assert!(validate_and_format_cpf("12345").is_err());

        let hash =
            hash_user_password("SuperSecret123!", "user@example.com").expect("Hash succeeds");
        assert!(hash.starts_with("$sha256-1024$"));
        assert!(verify_user_password(
            "SuperSecret123!",
            "user@example.com",
            &hash
        ));
        assert!(!verify_user_password(
            "WrongPass!",
            "user@example.com",
            &hash
        ));
    }
}
