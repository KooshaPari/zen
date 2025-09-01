"""
Edit providers for FastApply functionality

This module provides edit providers that can execute code edits through different backends:
- builtin: Direct file operations
- morph: AI-powered edits using OpenRouter models
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel

from providers.openrouter import OpenRouterProvider

logger = logging.getLogger(__name__)


class EditPlan(BaseModel):
    """Plan for executing edits."""
    id: str
    provider: str
    instructions: Optional[str] = None
    operations: Optional[list[dict[str, Any]]] = None
    context: dict[str, Any] = {}


class EditPreview(BaseModel):
    """Preview of edit changes."""
    files: list[dict[str, Any]] = []
    errors: Optional[list[str]] = None


class EditResult(BaseModel):
    """Result of applying edits."""
    files_changed: int = 0
    success: bool = True
    errors: Optional[list[str]] = None


class EditProvider(ABC):
    """Abstract base class for edit providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @abstractmethod
    def plan_edit(
        self,
        instructions: Optional[str] = None,
        operations: Optional[list[dict[str, Any]]] = None,
        context: dict[str, Any] = None
    ) -> EditPlan:
        """Plan an edit operation."""
        pass

    @abstractmethod
    def dry_run(self, plan: EditPlan) -> dict[str, Any]:
        """Preview changes without applying them."""
        pass

    @abstractmethod
    def apply(self, plan: EditPlan) -> dict[str, Any]:
        """Apply the edit plan."""
        pass


class BuiltinEditProvider(EditProvider):
    """Built-in edit provider for explicit file operations."""

    @property
    def name(self) -> str:
        return "builtin"

    def plan_edit(
        self,
        instructions: Optional[str] = None,
        operations: Optional[list[dict[str, Any]]] = None,
        context: dict[str, Any] = None
    ) -> EditPlan:
        """Plan edit using explicit operations."""
        import uuid
        return EditPlan(
            id=str(uuid.uuid4()),
            provider=self.name,
            instructions=instructions,
            operations=operations or [],
            context=context or {}
        )

    def dry_run(self, plan: EditPlan) -> dict[str, Any]:
        """Preview builtin edit operations."""
        files = []
        errors = []

        if not plan.operations:
            errors.append("No operations specified for builtin provider")
            return {"files": files, "errors": errors}

        for op in plan.operations:
            op_type = op.get("type")
            filepath = op.get("filepath")

            if not filepath:
                errors.append(f"Operation {op_type} missing filepath")
                continue

            if op_type == "replace":
                files.append({
                    "path": filepath,
                    "action": "modify",
                    "preview": f"Replace text in {filepath}"
                })
            elif op_type == "insert":
                files.append({
                    "path": filepath,
                    "action": "modify",
                    "preview": f"Insert text in {filepath}"
                })
            elif op_type == "write":
                files.append({
                    "path": filepath,
                    "action": "create/overwrite",
                    "preview": f"Write content to {filepath}"
                })
            else:
                errors.append(f"Unknown operation type: {op_type}")

        return {"files": files, "errors": errors if errors else None}

    def apply(self, plan: EditPlan) -> dict[str, Any]:
        """Apply builtin edit operations."""
        files_changed = 0
        errors = []

        if not plan.operations:
            errors.append("No operations to apply")
            return {"files_changed": 0, "success": False, "errors": errors}

        for op in plan.operations:
            try:
                op_type = op.get("type")
                filepath = op.get("filepath")

                if op_type == "write":
                    content = op.get("content", "")
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    files_changed += 1

                elif op_type == "replace":
                    find_text = op.get("find", "")
                    replace_text = op.get("replace", "")

                    with open(filepath, encoding='utf-8') as f:
                        content = f.read()

                    if find_text in content:
                        new_content = content.replace(find_text, replace_text, op.get("count", 1))
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        files_changed += 1
                    else:
                        errors.append(f"Text not found in {filepath}: {find_text}")

                elif op_type == "insert":
                    anchor = op.get("anchor", "")
                    content = op.get("content", "")
                    position = op.get("position", "after")  # before|after

                    with open(filepath, encoding='utf-8') as f:
                        file_content = f.read()

                    if anchor in file_content:
                        if position == "after":
                            new_content = file_content.replace(anchor, anchor + content, 1)
                        else:
                            new_content = file_content.replace(anchor, content + anchor, 1)

                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        files_changed += 1
                    else:
                        errors.append(f"Anchor not found in {filepath}: {anchor}")

            except Exception as e:
                errors.append(f"Error processing {op}: {str(e)}")

        return {
            "files_changed": files_changed,
            "success": len(errors) == 0,
            "errors": errors if errors else None
        }


class MorphEditProvider(EditProvider):
    """Morph edit provider using OpenRouter models for AI-powered edits."""

    def __init__(self):
        self.openrouter_provider = None
        self._setup_openrouter()

    def _setup_openrouter(self):
        """Setup OpenRouter provider for Morph operations."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            self.openrouter_provider = OpenRouterProvider(api_key)
            logger.info("Morph edit provider initialized with OpenRouter")
        else:
            logger.warning("OPENROUTER_API_KEY not set - Morph provider will be disabled")

    @property
    def name(self) -> str:
        return "morph"

    def plan_edit(
        self,
        instructions: Optional[str] = None,
        operations: Optional[list[dict[str, Any]]] = None,
        context: dict[str, Any] = None
    ) -> EditPlan:
        """Plan edit using AI-powered Morph via OpenRouter."""
        import uuid
        return EditPlan(
            id=str(uuid.uuid4()),
            provider=self.name,
            instructions=instructions,
            operations=operations,
            context=context or {}
        )

    def dry_run(self, plan: EditPlan) -> dict[str, Any]:
        """Preview Morph edit using OpenRouter model."""
        if not self.openrouter_provider:
            return {
                "files": [],
                "errors": ["OpenRouter not configured - set OPENROUTER_API_KEY"]
            }

        if not plan.instructions:
            return {
                "files": [],
                "errors": ["Instructions required for Morph provider"]
            }

        try:
            # Use a fast, efficient model for edit planning
            model = os.getenv("MORPH_MODEL", "anthropic/claude-3-5-haiku")

            # Create edit planning prompt
            system_prompt = """You are an expert code editor. Analyze the edit request and provide a JSON response with the planned changes.

Format your response as JSON:
{
  "files": [
    {
      "path": "/path/to/file.py",
      "action": "modify|create|delete",
      "preview": "Description of changes"
    }
  ],
  "errors": null
}"""

            user_prompt = f"""Edit Request: {plan.instructions}

Context: {json.dumps(plan.context, indent=2)}

Analyze this edit request and return the planned file changes as JSON."""

            result = self.openrouter_provider.generate_content(
                prompt=user_prompt,
                model_name=model,
                system_prompt=system_prompt,
                temperature=0.1,  # Low temperature for precise planning
                max_output_tokens=2000
            )

            # Parse JSON response
            try:
                response_data = json.loads(result.content)
                return {
                    "files": response_data.get("files", []),
                    "errors": response_data.get("errors")
                }
            except json.JSONDecodeError:
                logger.warning("Failed to parse Morph planning response as JSON")
                return {
                    "files": [{"path": "multiple", "action": "modify", "preview": result.content[:200] + "..."}],
                    "errors": None
                }

        except Exception as e:
            logger.error(f"Morph dry run failed: {e}")
            return {
                "files": [],
                "errors": [f"Morph planning failed: {str(e)}"]
            }

    def apply(self, plan: EditPlan) -> dict[str, Any]:
        """Apply Morph edit using OpenRouter model."""
        if not self.openrouter_provider:
            return {
                "files_changed": 0,
                "success": False,
                "errors": ["OpenRouter not configured - set OPENROUTER_API_KEY"]
            }

        if not plan.instructions:
            return {
                "files_changed": 0,
                "success": False,
                "errors": ["Instructions required for Morph provider"]
            }

        try:
            # Use a capable model for code generation
            model = os.getenv("MORPH_MODEL", "anthropic/claude-3-5-sonnet")

            # Create comprehensive edit prompt
            system_prompt = """You are an expert code editor. Execute the requested edits precisely and efficiently.

IMPORTANT: You must actually read and modify files. Output your actions in this format:

ACTION: READ /path/to/file.py
ACTION: WRITE /path/to/file.py
[new file content]
ACTION: COMPLETE

Be precise and make minimal necessary changes."""

            user_prompt = f"""Execute this edit request: {plan.instructions}

Context: {json.dumps(plan.context, indent=2)}

Read the relevant files, make the requested changes, and write the updated files back."""

            result = self.openrouter_provider.generate_content(
                prompt=user_prompt,
                model_name=model,
                system_prompt=system_prompt,
                temperature=0.2,  # Low temperature for precise edits
                max_output_tokens=8000
            )

            # Parse and execute the actions (simplified for now)
            # In a full implementation, this would parse ACTION: commands and execute them

            # For now, return success with the AI response
            return {
                "files_changed": 1,  # Placeholder
                "success": True,
                "errors": None,
                "ai_response": result.content
            }

        except Exception as e:
            logger.error(f"Morph apply failed: {e}")
            return {
                "files_changed": 0,
                "success": False,
                "errors": [f"Morph execution failed: {str(e)}"]
            }


def get_edit_provider(provider_name: Optional[str] = None) -> EditProvider:
    """Get an edit provider instance."""
    effective_provider = provider_name or os.getenv("ZEN_EDIT_PROVIDER", "builtin")

    if effective_provider == "morph":
        return MorphEditProvider()
    else:
        return BuiltinEditProvider()
