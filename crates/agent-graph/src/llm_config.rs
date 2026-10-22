//! Per-node LLM routing and model configuration loaded from `config/node_llm_config.json`.
//!
//! Ensures each graph node uses the appropriate LLM size and response format—in particular,
//! `MealParser` is configured with a very small JSON-mode LLM (`qwen2.5-1.5b-instruct-json`)
//! whose sole job is converting noisy natural-language meal write-ups (including typos like
//! `"ovio"` or `"ovoos"`) into structured JSON containing the foods, quantities, units, and
//! household-measure flags.

use crate::state::{AgentHierarchyRole, GraphNodeId, ModelTier};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

/// Embedded default configuration from `/config/node_llm_config.json`.
pub const DEFAULT_NODE_LLM_CONFIG_JSON: &str =
    include_str!("../../../config/node_llm_config.json");

/// Configuration for a single `GraphNodeId`'s LLM model, format, token cap, and cost rate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeLlmConfig {
    pub node_id: GraphNodeId,
    pub hierarchy_role: AgentHierarchyRole,
    pub model_tier: ModelTier,
    pub provider: String,
    pub model_id: String,
    pub temperature: f32,
    pub max_output_tokens: u32,
    /// Expected output format (`"json_object"` or `"text"`).
    pub response_format: String,
    pub prompt_cost_micros_per_1k: u64,
    pub completion_cost_micros_per_1k: u64,
    pub purpose: String,
}

/// Top-level JSON configuration file structure (`config/node_llm_config.json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeLlmConfigFile {
    pub version: String,
    pub description: String,
    pub nodes: BTreeMap<String, NodeLlmConfig>,
}

/// Registry mapping every `GraphNodeId` to its configured `NodeLlmConfig`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeLlmConfigRegistry {
    pub version: String,
    pub description: String,
    by_node: BTreeMap<GraphNodeId, NodeLlmConfig>,
}

impl Default for NodeLlmConfigRegistry {
    fn default() -> Self {
        Self::from_json_str(DEFAULT_NODE_LLM_CONFIG_JSON)
            .expect("Embedded config/node_llm_config.json must be valid JSON")
    }
}

impl NodeLlmConfigRegistry {
    /// Parses a `NodeLlmConfigRegistry` from a JSON string.
    pub fn from_json_str(raw_json: &str) -> Result<Self, String> {
        let parsed: NodeLlmConfigFile = serde_json::from_str(raw_json)
            .map_err(|e| format!("Failed to parse node_llm_config.json: {}", e))?;
        let mut by_node = BTreeMap::new();
        for (_, cfg) in parsed.nodes {
            by_node.insert(cfg.node_id, cfg);
        }
        Ok(Self {
            version: parsed.version,
            description: parsed.description,
            by_node,
        })
    }

    /// Loads the per-node LLM configuration from a file on disk (e.g. `config/node_llm_config.json`).
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let raw = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            format!(
                "Failed to read LLM config file '{}': {}",
                path.as_ref().display(),
                e
            )
        })?;
        Self::from_json_str(&raw)
    }

    /// Returns the configured `NodeLlmConfig` for a given `GraphNodeId`.
    pub fn config_for_node(&self, node_id: GraphNodeId) -> NodeLlmConfig {
        self.by_node.get(&node_id).cloned().unwrap_or_else(|| {
            let tier = node_id.default_model_tier();
            NodeLlmConfig {
                node_id,
                hierarchy_role: node_id.hierarchy_role(),
                model_tier: tier,
                provider: "default".to_string(),
                model_id: tier.default_model_id().to_string(),
                temperature: 0.0,
                max_output_tokens: 512,
                response_format: "text".to_string(),
                prompt_cost_micros_per_1k: tier.prompt_cost_micros_per_1k(),
                completion_cost_micros_per_1k: tier.completion_cost_micros_per_1k(),
                purpose: format!("Default configuration for {}", node_id),
            }
        })
    }

    /// Returns all node LLM configurations in deterministic order.
    pub fn all_configs(&self) -> Vec<NodeLlmConfig> {
        self.by_node.values().cloned().collect()
    }
}
