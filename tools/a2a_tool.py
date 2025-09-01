from __future__ import annotations

from typing import Any

from mcp.types import TextContent

from tools.shared.base_tool import BaseTool


class A2ATool(BaseTool):
    """Interact with A2A protocol: advertise, discover, send message.

    Actions:
      - advertise: agent_card{...}
      - discover: capability_filter?, organization_filter?, max_results?
      - message: payload{...}
    """

    def get_name(self) -> str:
        return "a2a"

    def get_description(self) -> str:
        return "Agent-to-agent protocol utilities: advertise, discover, send messages."

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["advertise", "discover", "message"]},
                "agent_card": {"type": "object"},
                "capability_filter": {"type": "string"},
                "organization_filter": {"type": "string"},
                "max_results": {"type": "integer"},
                "payload": {"type": "object"},
            },
            "required": ["action"],
        }

    def requires_model(self) -> bool:
        return False

    async def execute(self, arguments: dict[str, Any]) -> list[TextContent]:
        from utils.a2a_protocol import AgentCapability, AgentCard, get_a2a_manager
        action = arguments.get("action")
        mgr = get_a2a_manager()
        if action == "advertise":
            card_data = arguments.get("agent_card") or {}
            # Minimal card; fill required fields if missing
            import datetime
            import uuid
            if not card_data.get("agent_id"):
                card_data["agent_id"] = f"agent-{uuid.uuid4().hex[:6]}"
            if not card_data.get("name"):
                card_data["name"] = card_data["agent_id"]
            if not card_data.get("version"):
                card_data["version"] = "1.0.0"
            if not card_data.get("endpoint_url"):
                card_data["endpoint_url"] = "http://localhost:8080"
            if not card_data.get("capabilities"):
                card_data["capabilities"] = [AgentCapability(name="echo", description="echo", category="utility", input_schema={}, output_schema={}).model_dump()]
            card_data["last_seen"] = datetime.datetime.utcnow().isoformat() + "Z"
            card = AgentCard.model_validate(card_data)
            mgr.local_registry[card.agent_id] = card
            return [TextContent(type="text", text=f"advertised {card.agent_id}")]
        if action == "discover":
            agents = await mgr.discover_agents(arguments.get("capability_filter"), arguments.get("organization_filter"), int(arguments.get("max_results", 50)))
            return [TextContent(type="text", text=f"found {len(agents)} agents")]
        if action == "message":
            payload = arguments.get("payload") or {}
            resp = await mgr.handle_incoming_message(payload)
            return [TextContent(type="text", text=f"ok={bool(resp)}")]
        return [TextContent(type="text", text="unknown action")]

