//! Multi-Tier Conversation & User Profile Memory System (Working, Episodic, Semantic)
//! with `.pdf` and `.txt` Nutrition Plan ingestion, storing user profile facts,
//! probable daily foods to add each day, and chat conversation turns for agent recall.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use shared_types::{
    CalorieExchangeOption, NutritionPlanDocumentReader, ParsedNutritionPlan, ProbableDailyFood,
    SharedDomainError,
};
use std::collections::BTreeMap;

/// Snapshot of a user's consolidated memory profile (Working, Episodic, Semantic)
/// after ingesting a `.pdf`/`.txt` nutrition plan or conversing in the chat.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserMemoryProfileSnapshot {
    pub user_id: String,
    pub working_turns: Vec<String>,
    pub semantic_facts: BTreeMap<String, String>,
    pub episodic_events: Vec<String>,
    pub active_plan: Option<ParsedNutritionPlan>,
    pub probable_daily_foods: Vec<ProbableDailyFood>,
    pub calorie_exchange_options: Vec<CalorieExchangeOption>,
}

/// Multi-tier memory engine managing Working, Episodic, and Semantic user profile memory,
/// including `.pdf` and `.txt` nutritionist plan ingestion, probable daily food suggestions,
/// and chat conversation memory synchronization.
#[derive(Debug, Default)]
pub struct NutritionPlanMemoryEngine {
    working_turns: RwLock<BTreeMap<String, Vec<String>>>,
    episodic_events: RwLock<BTreeMap<String, Vec<String>>>,
    semantic_facts: RwLock<BTreeMap<String, BTreeMap<String, String>>>,
    active_plans: RwLock<BTreeMap<String, ParsedNutritionPlan>>,
    probable_daily_foods: RwLock<BTreeMap<String, Vec<ProbableDailyFood>>>,
}

impl NutritionPlanMemoryEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reads and parses a `.pdf` or `.txt` nutrition plan and saves all extracted user profile
    /// targets, nutritionist metadata, probable daily foods to add each day, and calorie
    /// exchange options into Semantic and Episodic memory.
    pub fn ingest_plan_bytes_into_memory(
        &self,
        user_id: &str,
        filename: &str,
        bytes: &[u8],
    ) -> Result<UserMemoryProfileSnapshot, SharedDomainError> {
        let parsed = NutritionPlanDocumentReader::parse_nutrition_plan(user_id, filename, bytes)?;
        Ok(self.save_parsed_plan_to_memory(parsed))
    }

    /// Persists an already-parsed `ParsedNutritionPlan` into Semantic, Episodic, and Probable Daily Foods memory.
    pub fn save_parsed_plan_to_memory(
        &self,
        plan: ParsedNutritionPlan,
    ) -> UserMemoryProfileSnapshot {
        let user_id = plan.user_id.clone();

        {
            let mut sem_guard = self.semantic_facts.write();
            let facts = sem_guard.entry(user_id.clone()).or_default();

            facts.insert("active_plan_id".to_string(), plan.plan_id.clone());
            facts.insert("active_plan_title".to_string(), plan.title.clone());
            facts.insert(
                "active_plan_format".to_string(),
                plan.source_format.as_str().to_string(),
            );
            facts.insert(
                "daily_calorie_target_kcal".to_string(),
                format!("{:.0}", plan.target_daily_macros.calories_kcal),
            );
            facts.insert(
                "daily_protein_target_g".to_string(),
                format!("{:.0}", plan.target_daily_macros.protein_g),
            );
            facts.insert(
                "daily_carbs_target_g".to_string(),
                format!("{:.0}", plan.target_daily_macros.carbs_g),
            );
            facts.insert(
                "daily_fat_target_g".to_string(),
                format!("{:.0}", plan.target_daily_macros.fat_g),
            );
            facts.insert(
                "hydration_target_ml".to_string(),
                plan.hydration_target_ml.to_string(),
            );

            if let Some(nutri) = &plan.nutritionist_name {
                facts.insert("nutritionist_name".to_string(), nutri.clone());
            }
            if let Some(crn) = &plan.nutritionist_crn {
                facts.insert("nutritionist_crn".to_string(), crn.clone());
            }
            if let Some(name) = &plan.extracted_user_profile.full_name {
                facts.insert("user_full_name".to_string(), name.clone());
            }
            if let Some(cpf) = &plan.extracted_user_profile.cpf {
                facts.insert("user_cpf".to_string(), cpf.clone());
            }
            if let Some(email) = &plan.extracted_user_profile.email {
                facts.insert("user_email".to_string(), email.clone());
            }
            if let Some(phone) = &plan.extracted_user_profile.phone {
                facts.insert("user_phone".to_string(), phone.clone());
            }
            if let Some(addr) = &plan.extracted_user_profile.address {
                facts.insert("user_address".to_string(), addr.clone());
            }
            if let Some(goal) = &plan.extracted_user_profile.goal {
                facts.insert("user_goal".to_string(), goal.clone());
            }

            let probable_summary = plan
                .probable_daily_foods
                .iter()
                .map(|f| {
                    format!(
                        "{}: {} ({:.0}g, {:.0} kcal)",
                        f.meal_name, f.display_name, f.estimated_grams, f.macros.calories_kcal
                    )
                })
                .collect::<Vec<_>>()
                .join(" | ");
            facts.insert("probable_daily_foods".to_string(), probable_summary);

            if let Ok(json) = serde_json::to_string(&plan.probable_daily_foods) {
                facts.insert("probable_daily_foods_json".to_string(), json);
            }
            if let Ok(ex_json) = serde_json::to_string(&plan.calorie_exchange_options) {
                facts.insert("calorie_exchange_options_json".to_string(), ex_json);
            }
        }

        let episode = format!(
            "Ingested {} nutrition plan '{}' ({} meals, {} probable daily foods, {} calorie exchanges, target {:.0} kcal)",
            plan.source_format.as_str().to_uppercase(),
            plan.title,
            plan.meals.len(),
            plan.probable_daily_foods.len(),
            plan.calorie_exchange_options.len(),
            plan.target_daily_macros.calories_kcal
        );
        self.episodic_events
            .write()
            .entry(user_id.clone())
            .or_default()
            .push(episode);

        self.probable_daily_foods
            .write()
            .insert(user_id.clone(), plan.probable_daily_foods.clone());
        self.active_plans
            .write()
            .insert(user_id.clone(), plan.clone());

        self.get_user_snapshot(&user_id)
    }

    /// Records a user <-> assistant chat exchange across Working, Episodic, and Semantic memory
    /// so agents can recall previous chat messages in future turns.
    pub fn record_chat_turn_in_memory(
        &self,
        user_id: &str,
        conversation_id: &str,
        user_message: &str,
        assistant_response: &str,
        detected_intent: &str,
    ) -> UserMemoryProfileSnapshot {
        let working_entry = format!(
            "[{}] User: {} => Assistant ({})",
            conversation_id,
            user_message,
            detected_intent
        );
        self.working_turns
            .write()
            .entry(user_id.to_string())
            .or_default()
            .push(working_entry);

        let episodic_entry = format!(
            "Chat [{}]: User asked/said '{}' -> Agent replied ({} chars)",
            detected_intent,
            user_message,
            assistant_response.len()
        );
        self.episodic_events
            .write()
            .entry(user_id.to_string())
            .or_default()
            .push(episodic_entry);

        {
            let mut sem_guard = self.semantic_facts.write();
            let facts = sem_guard.entry(user_id.to_string()).or_default();
            facts.insert("last_chat_user_message".to_string(), user_message.to_string());
            facts.insert(
                "last_chat_detected_intent".to_string(),
                detected_intent.to_string(),
            );
            let count: u32 = facts
                .get("total_chat_turns")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0)
                + 1;
            facts.insert("total_chat_turns".to_string(), count.to_string());
        }

        self.get_user_snapshot(user_id)
    }

    pub fn get_user_snapshot(&self, user_id: &str) -> UserMemoryProfileSnapshot {
        let working_turns = self
            .working_turns
            .read()
            .get(user_id)
            .cloned()
            .unwrap_or_default();
        let semantic_facts = self
            .semantic_facts
            .read()
            .get(user_id)
            .cloned()
            .unwrap_or_default();
        let episodic_events = self
            .episodic_events
            .read()
            .get(user_id)
            .cloned()
            .unwrap_or_default();
        let active_plan = self.active_plans.read().get(user_id).cloned();
        let probable_daily_foods = self
            .probable_daily_foods
            .read()
            .get(user_id)
            .cloned()
            .unwrap_or_default();
        let calorie_exchange_options = active_plan
            .as_ref()
            .map(|p| p.calorie_exchange_options.clone())
            .unwrap_or_default();

        UserMemoryProfileSnapshot {
            user_id: user_id.to_string(),
            working_turns,
            semantic_facts,
            episodic_events,
            active_plan,
            probable_daily_foods,
            calorie_exchange_options,
        }
    }

    pub fn append_working_turn(&self, user_id: &str, turn: String) {
        self.working_turns
            .write()
            .entry(user_id.to_string())
            .or_default()
            .push(turn);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ingest_pdf_and_txt_plans_and_chat_turns_in_memory() {
        let engine = NutritionPlanMemoryEngine::new();

        let pdf_bytes = NutritionPlanDocumentReader::build_pdf_document_from_text(&[
            "Plano Alimentar: Recomposicao Corporal",
            "Nutricionista: Dra. Helena - CRN-3 77889",
            "Paciente: Fernanda Lima",
            "CPF: 321.654.987-00",
            "Email: fernanda@calorie.ai",
            "Telefone: (21) 99111-2233",
            "Endereco: Rua das Laranjeiras, 120 - Rio de Janeiro",
            "Meta: 1950 kcal | Proteina: 155g | Carboidratos: 190g | Gordura: 58g",
            "Hidratacao: 2800 ml",
            "07:30 - Cafe da Manha",
            "- 2 unidades de Ovos Mexidos (100g) OU 30g de Whey Protein",
            "- 45g de Aveia OU 50g de Tapioca",
            "12:30 - Almoco",
            "- 150g de Peito de Frango Grelhado OU 180g de Tilapia",
            "- 130g de Arroz Integral OU 160g de Batata Doce",
        ]);

        let snapshot = engine
            .ingest_plan_bytes_into_memory("usr-fernanda", "plano_fernanda.pdf", &pdf_bytes)
            .expect("PDF plan ingestion into memory should succeed");

        assert_eq!(
            snapshot.semantic_facts.get("user_full_name").map(String::as_str),
            Some("Fernanda Lima")
        );
        assert_eq!(snapshot.probable_daily_foods.len(), 4);
        assert_eq!(snapshot.calorie_exchange_options.len(), 4);

        let after_chat = engine.record_chat_turn_in_memory(
            "usr-fernanda",
            "conv-fernanda-1",
            "Posso trocar os ovos por whey no cafe da manha?",
            "Sim, seu plano permite substituir 2 ovos mexidos (100g) por 30g de Whey Protein.",
            "AskNutrition",
        );
        assert_eq!(after_chat.working_turns.len(), 1);
        assert_eq!(after_chat.episodic_events.len(), 2);
        assert_eq!(
            after_chat.semantic_facts.get("total_chat_turns").map(String::as_str),
            Some("1")
        );
    }
}
