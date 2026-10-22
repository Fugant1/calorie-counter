//! Pre-built stateful multi-agent workflows (`UnifiedCalorieCounterFlow`, `MealLogFlow`, `CoachingFlow`).
//!
//! Organizes nodes according to the **Planner -> Tool Calling -> Executioner** hierarchy
//! and binds them to conditional routing & cyclic retry edges.

use crate::edges::{GraphEdgeDescriptor, GraphEdgeRouter};
use crate::nodes::{
    AgentNode, AudioTranscriberNode, CriticVerifierNode, FitnessCoachNode, IntentRouterNode,
    MacroCalculatorNode, MealParserNode, MemoryUpdaterNode, NutritionLookupNode,
    ResponseSynthesizerNode, WebResearcherNode,
};
use crate::state::{AgentHierarchyRole, GraphNodeId};
use indexmap::IndexMap;
use std::sync::Arc;

/// A compiled multi-agent workflow containing all registered `AgentNode` implementations
/// and conditional edge routing rules.
#[derive(Clone)]
pub struct AgentGraphWorkflow {
    pub name: String,
    pub entry_node: GraphNodeId,
    nodes: IndexMap<GraphNodeId, Arc<dyn AgentNode>>,
    edge_router: GraphEdgeRouter,
    edges: Vec<GraphEdgeDescriptor>,
}

impl Default for AgentGraphWorkflow {
    fn default() -> Self {
        Self::unified_calorie_counter()
    }
}

impl AgentGraphWorkflow {
    /// Builds the full multi-agent workflow described in `README.md` Section 5.1,
    /// handling both Audio/Text meal registration (with Redis L1/L2 RAG cache & WebSearch RAG warming)
    /// and nutrition/fitness coaching questions.
    pub fn unified_calorie_counter() -> Self {
        let mut nodes: IndexMap<GraphNodeId, Arc<dyn AgentNode>> = IndexMap::new();

        // 1. Planner Hierarchy Tier
        nodes.insert(
            GraphNodeId::IntentRouter,
            Arc::new(IntentRouterNode::new()),
        );
        nodes.insert(GraphNodeId::MealParser, Arc::new(MealParserNode::new()));
        nodes.insert(
            GraphNodeId::NutritionVerifier,
            Arc::new(CriticVerifierNode::new()),
        );

        // 2. Tool Calling Hierarchy Tier
        nodes.insert(
            GraphNodeId::AudioTranscriber,
            Arc::new(AudioTranscriberNode::new()),
        );
        nodes.insert(
            GraphNodeId::CacheLookup,
            Arc::new(NutritionLookupNode::new()),
        );
        nodes.insert(
            GraphNodeId::WebSearchAndRAG,
            Arc::new(WebResearcherNode::new()),
        );
        nodes.insert(
            GraphNodeId::CoachingAgent,
            Arc::new(FitnessCoachNode::new()),
        );

        // 3. Executioner Hierarchy Tier
        nodes.insert(
            GraphNodeId::MacroCalculator,
            Arc::new(MacroCalculatorNode::new()),
        );
        nodes.insert(
            GraphNodeId::MemoryUpdater,
            Arc::new(MemoryUpdaterNode::new()),
        );
        nodes.insert(
            GraphNodeId::ResponseSynthesizer,
            Arc::new(ResponseSynthesizerNode::new()),
        );

        Self {
            name: "UnifiedCalorieCounterFlow".to_string(),
            entry_node: GraphNodeId::IntentRouter,
            nodes,
            edge_router: GraphEdgeRouter::new(),
            edges: GraphEdgeRouter::canonical_edges(),
        }
    }

    /// Pre-built `MealLogFlow` workflow (`README.md` Section 3).
    pub fn meal_log_flow() -> Self {
        let mut wf = Self::unified_calorie_counter();
        wf.name = "MealLogFlow".to_string();
        wf
    }

    /// Pre-built `CoachingFlow` workflow (`README.md` Section 3).
    pub fn coaching_flow() -> Self {
        let mut wf = Self::unified_calorie_counter();
        wf.name = "CoachingFlow".to_string();
        wf
    }

    /// Replaces or registers a custom `AgentNode` implementation in the workflow.
    pub fn with_node(mut self, node: Arc<dyn AgentNode>) -> Self {
        self.nodes.insert(node.id(), node);
        self
    }

    /// Looks up a registered node by its `GraphNodeId`.
    pub fn get_node(&self, id: GraphNodeId) -> Option<&Arc<dyn AgentNode>> {
        self.nodes.get(&id)
    }

    /// Returns a reference to the conditional edge router.
    pub fn edge_router(&self) -> &GraphEdgeRouter {
        &self.edge_router
    }

    /// Returns the static edge descriptors of the workflow.
    pub fn edges(&self) -> &[GraphEdgeDescriptor] {
        &self.edges
    }

    /// Lists all registered node IDs belonging to a specific `AgentHierarchyRole`.
    pub fn nodes_by_role(&self, role: AgentHierarchyRole) -> Vec<GraphNodeId> {
        self.nodes
            .values()
            .filter(|n| n.hierarchy_role() == role)
            .map(|n| n.id())
            .collect()
    }
}
