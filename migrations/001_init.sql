-- Initial schema for projects, channels, and messages

-- Projects
CREATE TABLE IF NOT EXISTS projects (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  owner TEXT NOT NULL,
  description TEXT,
  created_at BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS project_agents (
  pid TEXT NOT NULL,
  agent_id TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_project_agents ON project_agents (pid);

CREATE TABLE IF NOT EXISTS project_channels (
  pid TEXT NOT NULL,
  channel_id TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_project_channels ON project_channels (pid);

CREATE TABLE IF NOT EXISTS project_artifacts (
  pid TEXT NOT NULL,
  idx BIGSERIAL PRIMARY KEY,
  data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_project_artifacts ON project_artifacts (pid, idx);

-- Channels
CREATE TABLE IF NOT EXISTS channels (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  name TEXT NOT NULL,
  visibility TEXT NOT NULL,
  created_by TEXT NOT NULL,
  created_at BIGINT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_channels_project ON channels (project_id);

CREATE TABLE IF NOT EXISTS channel_members (
  channel_id TEXT NOT NULL,
  agent_id TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_channel_members ON channel_members (channel_id);

-- Messages
CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL,
  channel_id TEXT,
  members TEXT,
  root TEXT,
  sender TEXT NOT NULL,
  body TEXT NOT NULL,
  mentions TEXT,
  blocking BOOLEAN,
  resume_token TEXT,
  resolved BOOLEAN DEFAULT FALSE,
  resolved_ts BIGINT,
  ts BIGINT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_channel_ts ON messages (channel_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_messages_members_ts ON messages (members, ts DESC);
CREATE INDEX IF NOT EXISTS idx_messages_root_ts ON messages (root, ts DESC);
CREATE INDEX IF NOT EXISTS idx_messages_resume ON messages (resume_token);

