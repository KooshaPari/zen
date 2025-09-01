from typing import Any, Optional

from mcp.types import TextContent
from pydantic import Field

from tools.shared.base_models import ToolRequest
from tools.simple.base import SimpleTool


class SemIngestRequest(ToolRequest):
    docs: list[dict[str, Any]] = Field(
        ..., description="List of documents to ingest: each item must include id (str), text (str), metadata (object)"
    )
    collection: Optional[str] = Field("code", description="Target collection: code, knowledge, or other")


class SemIngestTool(SimpleTool):
    def get_name(self) -> str:
        return "sem_ingest"

    def get_description(self) -> str:
        return "Ingest documents into the pgvector RAG store for the current scoped namespace (from work_dir and scope_context)."

    def requires_model(self) -> bool:
        return False

    def get_tool_fields(self) -> dict[str, dict[str, Any]]:
        return {
            "docs": {
                "type": "array",
                "description": "List of documents: {id: string, text: string, metadata: object}. Example: [{\"id\":\"d1\",\"text\":\"hello\",\"metadata\":{\"path\":\"README.md\"}}]",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "text": {"type": "string"},
                        "metadata": {"type": "object"},
                    },
                    "required": ["id", "text", "metadata"],
                },
            },
            "collection": {
                "type": "string",
                "description": "Target collection (code, knowledge, other)",
                "default": "code",
            },
        }

    def get_required_fields(self) -> list[str]:
        return ["docs"]

    async def execute(self, arguments: dict[str, Any]) -> list:
        from tools.models import ToolOutput
        from tools.semtools import ingest_documents

        request = SemIngestRequest(**arguments)
        scope = arguments.get("scope_context", {}) or {}
        ingest_documents([(d["id"], d["text"], d["metadata"]) for d in request.docs], scope, collection=request.collection)

        out = ToolOutput(status="success", content=f"Ingested {len(request.docs)} documents into collection '{request.collection}'", content_type="text")
        return [TextContent(type="text", text=out.model_dump_json())]

