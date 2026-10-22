-- Migration 0003: User Chat Conversations, Persisted Chat Messages, and Agent Chat Memories
-- enabling the multi-agent graph (Working, Episodic, and Semantic memory) to recall past chats.

CREATE TABLE IF NOT EXISTS chat_conversations (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title               TEXT NOT NULL DEFAULT 'Chat Nutricional & Calorias',
    language            VARCHAR(16) NOT NULL DEFAULT 'pt',
    message_count       INTEGER NOT NULL DEFAULT 0,
    total_tokens_used   INTEGER NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_conversations_user
    ON chat_conversations(user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS chat_messages (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id     UUID NOT NULL REFERENCES chat_conversations(id) ON DELETE CASCADE,
    user_id             UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    sender_role         VARCHAR(24) NOT NULL CHECK (sender_role IN ('user', 'assistant', 'system')),
    content             TEXT NOT NULL,
    detected_intent     VARCHAR(64),
    registered_meal_id  TEXT,
    node_decision_path  JSONB NOT NULL DEFAULT '[]'::jsonb,
    tokens_used         INTEGER NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_conv_created
    ON chat_messages(conversation_id, created_at ASC);

CREATE INDEX IF NOT EXISTS idx_chat_messages_user_created
    ON chat_messages(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS agent_chat_memories (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id     UUID REFERENCES chat_conversations(id) ON DELETE CASCADE,
    source_message_id   UUID REFERENCES chat_messages(id) ON DELETE SET NULL,
    memory_tier         VARCHAR(24) NOT NULL CHECK (memory_tier IN ('working', 'episodic', 'semantic')),
    memory_key          TEXT NOT NULL,
    memory_value        TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_chat_memories_user_tier
    ON agent_chat_memories(user_id, memory_tier);
