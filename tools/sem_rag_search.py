from typing import Any, Optional

from mcp.types import TextContent
from pydantic import Field

from tools.shared.base_models import ToolRequest
from tools.simple.base import SimpleTool


class SemRAGSearchRequest(ToolRequest):
    query: str = Field(..., description="Search query text for vector retrieval")
    collection: Optional[str] = Field("code", description="Collection to search: code, knowledge, or other")
    top_k: Optional[int] = Field(8, ge=1, le=50, description="Number of results to return")
    use_bm25: Optional[bool] = Field(False, description="If true and OpenSearch configured, fuse BM25 with vector results")


class SemRAGSearchTool(SimpleTool):
    def get_name(self) -> str:
        return "sem_rag_search"

    def get_description(self) -> str:
        return "Perform vector search (pgvector) within the current scoped namespace and return top-k results."

    def requires_model(self) -> bool:
        return False

    def get_tool_fields(self) -> dict[str, dict[str, Any]]:
        return {
            "query": {"type": "string", "description": "Query text. Example: 'find hello world usage'"},
            "collection": {"type": "string", "description": "Target collection", "default": "code"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 8},
            "use_bm25": {"type": "boolean", "default": False, "description": "Enable BM25+vector hybrid if available"},
        }

    def get_required_fields(self) -> list[str]:
        return ["query"]

    async def execute(self, arguments: dict[str, Any]) -> list:
        from tools.models import ToolOutput
        from tools.semtools import rag_search

        request = SemRAGSearchRequest(**arguments)
        scope = arguments.get("scope_context", {}) or {}
        results = rag_search(request.query, scope, collection=request.collection, top_k=request.top_k)
        if arguments.get("use_bm25"):
            try:
                from tools.semtools_bm25 import bm25_search, rrf_fuse
                index = scope.get("org", "default") + "-" + scope.get("proj", "proj") + "-" + scope.get("repo", "repo")
                sparse = bm25_search(index=index, query=request.query, top_k=request.top_k)
                dense = [{"id": r.get("id") or str(i), "text": r.get("text"), "metadata": r.get("metadata", {})} for i, r in enumerate(results)]
                results = rrf_fuse(dense, sparse, top_k=request.top_k)
            except Exception:
                # Fallback silently to dense-only
                pass

        out = ToolOutput(status="success", content={"results": results}, content_type="json")
        return [TextContent(type="text", text=out.model_dump_json())]

