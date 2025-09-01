import json
import logging
from typing import Any, Optional

from mcp.types import TextContent

logger = logging.getLogger(__name__)


def _is_json_tooloutput(text: Optional[str]) -> bool:
    if not isinstance(text, str) or not text:
        return False
    try:
        parsed = json.loads(text)
        return isinstance(parsed, dict) and (
            "status" in parsed or "continuation_offer" in parsed or "metadata" in parsed
        )
    except Exception:
        return False


def _ensure_list_textcontent(result: Any) -> list[TextContent]:
    # Accept already-correct result
    if isinstance(result, list) and result and hasattr(result[0], "text"):
        return result  # type: ignore[return-value]
    # If single TextContent
    if hasattr(result, "text"):
        return [result]  # type: ignore[list-item]
    # If dict or list (non-TextContent), encode as JSON text
    if isinstance(result, (dict, list)):
        return [TextContent(type="text", text=json.dumps(result))]
    # If string
    if isinstance(result, str):
        return [TextContent(type="text", text=result)]
    # Fallback to string repr
    return [TextContent(type="text", text=str(result) if result is not None else "")]


def normalize_tool_result(tool_name: str, arguments: dict[str, Any], raw_result: Any) -> list[TextContent]:
    """
    Normalize a tool's raw result into a list[TextContent] that the MCP simulator can parse.
    Always ensures a JSON ToolOutput with continuation_id is present for initial conversations
    (when result isn't already a JSON ToolOutput).
    """
    from tools.models import ToolOutput
    from utils.conversation_memory import create_thread

    result_list = _ensure_list_textcontent(raw_result)

    # If empty result, synthesize a minimal continuation offer
    if not result_list:
        cont_id = arguments.get("continuation_id") or create_thread(tool_name=tool_name, initial_request=arguments)
        wrapper = ToolOutput(
            status="continuation_available" if not arguments.get("continuation_id") else "success",
            content="",
            content_type="text",
            continuation_offer=(
                {"continuation_id": cont_id, "note": "Conversation can continue", "remaining_turns": 9}
                if not arguments.get("continuation_id")
                else None
            ),
            metadata={"tool_name": tool_name},
        )
        return [TextContent(type="text", text=wrapper.model_dump_json())]

    # Use first content as basis
    content0 = result_list[0]
    text0 = getattr(content0, "text", "") if hasattr(content0, "text") else ""

    # Helper: unwrap nested JSON strings in known ToolOutput fields
    def _unwrap_if_nested_json(s: Optional[str]) -> Optional[dict]:
        if not isinstance(s, str) or not s:
            return None
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    # If already JSON ToolOutput, ensure top-level continuation_id for initial calls
    if _is_json_tooloutput(text0):
        try:
            parsed = json.loads(text0)
            # Only inject/ensure continuation_id at top level for initial calls; do not alter content encoding
            if not arguments.get("continuation_id") and isinstance(parsed, dict):
                cont_id = parsed.get("continuation_id")
                if not cont_id:
                    offer = parsed.get("continuation_offer") or {}
                    cont_id = offer.get("continuation_id") if isinstance(offer, dict) else None
                if not cont_id:
                    cont_id = create_thread(tool_name=tool_name, initial_request=arguments)
                    parsed["continuation_offer"] = {
                        "continuation_id": cont_id,
                        "note": "Conversation can continue",
                        "remaining_turns": 9,
                    }
                parsed["continuation_id"] = cont_id
            # Save back normalized JSON
            result_list[0] = TextContent(type="text", text=json.dumps(parsed))
        except Exception:
            logger.debug("[MCP_NORMALIZE] Failed to normalize existing JSON ToolOutput", exc_info=True)
        return result_list

    # Otherwise, wrap into ToolOutput and return single-element list
    try:
        cont_id = arguments.get("continuation_id")
        if not cont_id:
            cont_id = create_thread(tool_name=tool_name, initial_request=arguments)
        # If text0 itself is nested JSON for ToolOutput content, unwrap just content for consistency
        inner_unwrapped = _unwrap_if_nested_json(text0)
        content_payload = inner_unwrapped if inner_unwrapped else text0
        wrapper = ToolOutput(
            status="continuation_available" if not arguments.get("continuation_id") else "success",
            content=content_payload,
            content_type="text",
            continuation_offer=(
                {"continuation_id": cont_id, "note": "Conversation can continue", "remaining_turns": 9}
                if not arguments.get("continuation_id")
                else None
            ),
            metadata={"tool_name": tool_name},
        )
        # Add top-level continuation_id for simulator compatibility
        wrapper_dict = wrapper.model_dump()
        if cont_id and not arguments.get("continuation_id"):
            wrapper_dict["continuation_id"] = cont_id
        normalized = [TextContent(type="text", text=json.dumps(wrapper_dict))]
        logger.debug(
            f"[MCP_NORMALIZE] Wrapped tool '{tool_name}' response into JSON ToolOutput; continuation_id={cont_id}"
        )
        return normalized
    except Exception:
        # If ToolOutput validation failed (e.g., content type), fall back to a minimal JSON wrapper
        try:
            cont_id = arguments.get("continuation_id") or create_thread(tool_name=tool_name, initial_request=arguments)
            fallback = {
                "status": "continuation_available" if not arguments.get("continuation_id") else "success",
                "content": text0 if isinstance(text0, str) else str(text0),
                "content_type": "text",
                "metadata": {"tool_name": tool_name},
            }
            if not arguments.get("continuation_id"):
                fallback["continuation_offer"] = {
                    "continuation_id": cont_id,
                    "note": "Conversation can continue",
                    "remaining_turns": 9,
                }
                fallback["continuation_id"] = cont_id
            return [TextContent(type="text", text=json.dumps(fallback))]
        except Exception:
            # On error, at least ensure a string text return
            logger.debug("[MCP_NORMALIZE] Failed to wrap ToolOutput; returning original result", exc_info=True)
            return result_list

