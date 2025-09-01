
from tools import (
    AgentAsyncTool,
    AgentBatchTool,
    AgentInboxTool,
    AgentRegistryTool,
    AgentSyncTool,
    ChatTool,
)


class TestToolAnnotations:
    def test_agent_tools_annotations(self):
        assert AgentRegistryTool().get_annotations()["category"] == "agent-orchestration"
        assert AgentRegistryTool().get_annotations()["readOnlyHint"] is True

        assert AgentInboxTool().get_annotations()["category"] == "agent-orchestration"
        assert AgentInboxTool().get_annotations()["readOnlyHint"] is True

        assert AgentSyncTool().get_annotations()["category"] == "agent-orchestration"
        assert AgentSyncTool().get_annotations()["destructiveHint"] is True
        assert AgentSyncTool().get_annotations()["readOnlyHint"] is False

        assert AgentAsyncTool().get_annotations()["category"] == "agent-orchestration"
        assert AgentAsyncTool().get_annotations()["destructiveHint"] is True
        assert AgentAsyncTool().get_annotations()["readOnlyHint"] is False

        assert AgentBatchTool().get_annotations()["category"] == "agent-orchestration"
        assert AgentBatchTool().get_annotations()["destructiveHint"] is True
        assert AgentBatchTool().get_annotations()["readOnlyHint"] is False

    def test_chat_tool_annotations(self):
        ann = ChatTool().get_annotations()
        assert ann["category"] == "conversation"
        assert ann["readOnlyHint"] is True
        assert "chat" in ann.get("tags", [])

