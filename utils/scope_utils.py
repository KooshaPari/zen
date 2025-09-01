"""
Scope Utilities for Work Directory Validation and Context Injection

This module provides core functionality for enforcing work_dir boundaries and
injecting scope context into all tool requests, enabling secure multi-tenant
shared data access.

Key Features:
- Work directory validation and normalization
- Scope context generation from identity and work_dir
- Path traversal protection
- Tenant isolation enforcement
- Integration with shared data systems
"""

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class WorkDirError(ValueError):
    """Exception raised for work directory validation errors."""
    pass


class WorkDirScope(BaseModel):
    """Represents a validated work directory scope."""

    work_dir: str = Field(..., description="Repository-relative work directory path")
    repo_root: str = Field(..., description="Absolute path to repository root")
    absolute_path: str = Field(..., description="Absolute path to work directory")
    allowed_subdirs: list[str] = Field(default_factory=list, description="Allowed subdirectories")
    is_shared: bool = Field(default=False, description="Whether this is a shared interface directory")

    @field_validator('work_dir')
    @classmethod
    def validate_work_dir(cls, v: str):
        """Validate work_dir doesn't contain path traversal attempts."""
        if not v or v.strip() == '':
            return ""  # Empty work_dir is valid (means repo root)

        # Normalize path separators
        v = v.replace('\\', '/')

        # Check for path traversal attempts
        if '..' in v or v.startswith('~'):
            raise ValueError(f"Invalid work_dir: {v} - no path traversal allowed")

        # Ensure no double slashes
        v = re.sub(r'/+', '/', v)

        return v.strip('/')


@dataclass
class ScopeContext:
    """
    Complete scope context for a tool request.

    This context is injected by the server and propagated to all
    downstream services (vector store, knowledge graph, LSP, etc.)
    """

    # Identity
    agent_id: str
    org_id: Optional[str] = None
    team_id: Optional[str] = None
    project_id: Optional[str] = None
    user_id: Optional[str] = None

    # Scope
    work_dir: str = ""  # Repository-relative path
    repo_root: Optional[str] = None  # Absolute repo path
    allowed_work_dirs: list[str] = None  # List of permitted work_dirs

    # Permissions
    roles: list[str] = None
    permissions: set[str] = None
    can_read_shared: bool = True  # Can read shared interfaces
    can_write_shared: bool = False  # Can write to shared interfaces

    # Session
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    # Metadata
    source_tool: Optional[str] = None
    parent_context: Optional['ScopeContext'] = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.allowed_work_dirs is None:
            self.allowed_work_dirs = []
        if self.roles is None:
            self.roles = []
        if self.permissions is None:
            self.permissions = set()
        if self.metadata is None:
            self.metadata = {}
        if self.request_id is None:
            self.request_id = str(uuid4())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'agent_id': self.agent_id,
            'org_id': self.org_id,
            'team_id': self.team_id,
            'project_id': self.project_id,
            'user_id': self.user_id,
            'work_dir': self.work_dir,
            'repo_root': self.repo_root,
            'allowed_work_dirs': self.allowed_work_dirs,
            'roles': self.roles,
            'permissions': list(self.permissions) if self.permissions else [],
            'can_read_shared': self.can_read_shared,
            'can_write_shared': self.can_write_shared,
            'session_id': self.session_id,
            'request_id': self.request_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'source_tool': self.source_tool,
            'metadata': self.metadata
        }

    def get_namespace_key(self) -> str:
        """
        Generate namespace key for shared data systems.
        Format: {org}/{project}/{repo}/{work_dir}
        """
        parts = []
        if self.org_id:
            parts.append(self.org_id)
        if self.project_id:
            parts.append(self.project_id)
        if self.repo_root:
            repo_name = Path(self.repo_root).name
            parts.append(repo_name)
        if self.work_dir:
            parts.append(self.work_dir.replace('/', '_'))

        return '/'.join(parts) if parts else 'default'

    def get_cache_key(self) -> str:
        """Generate cache key for this scope context."""
        key_parts = [
            self.agent_id,
            self.org_id or 'no-org',
            self.project_id or 'no-project',
            self.work_dir or 'root'
        ]
        key_str = ':'.join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]


class ScopeValidator:
    """Validates and enforces scope boundaries."""

    # Shared interface directories (allowed for cross-boundary access)
    SHARED_INTERFACES = {
        'api',
        'contracts',
        'interfaces',
        'proto',
        'schema',
        'shared',
        'types'
    }

    # System directories that should never be accessed
    FORBIDDEN_DIRS = {
        '.git',
        '.env',
        'node_modules',
        '__pycache__',
        '.venv',
        'venv',
        '.zen_venv',
        'dist',
        'build',
        'target'
    }

    @classmethod
    def validate_work_dir(
        cls,
        work_dir: str,
        repo_root: Optional[str] = None,
        allowed_work_dirs: Optional[list[str]] = None
    ) -> tuple[bool, str, Optional[WorkDirScope]]:
        """
        Validate a work_dir against security policies.

        Returns:
            Tuple of (is_valid, error_message, validated_scope)
        """
        try:
            # Basic validation
            if not work_dir:
                return True, "", WorkDirScope(
                    work_dir="",
                    repo_root=repo_root or os.getcwd(),
                    absolute_path=repo_root or os.getcwd()
                )

            # Create WorkDirScope for validation
            scope = WorkDirScope(
                work_dir=work_dir,
                repo_root=repo_root or os.getcwd(),
                absolute_path=""
            )

            # Check against forbidden directories
            work_parts = scope.work_dir.split('/')
            for forbidden in cls.FORBIDDEN_DIRS:
                if forbidden in work_parts:
                    return False, f"Access to {forbidden} directory is forbidden", None

            # Check if it's a shared interface
            first_part = work_parts[0] if work_parts else ""
            scope.is_shared = first_part in cls.SHARED_INTERFACES

            # Validate against allowed list if provided
            if allowed_work_dirs is not None and len(allowed_work_dirs) > 0:
                # Check exact match or parent directory match
                is_allowed = False
                for allowed in allowed_work_dirs:
                    if scope.work_dir == allowed or scope.work_dir.startswith(f"{allowed}/"):
                        is_allowed = True
                        break

                if not is_allowed and not scope.is_shared:
                    return False, f"work_dir '{work_dir}' not in allowed list", None

            # Calculate absolute path
            if repo_root:
                abs_path = os.path.join(repo_root, scope.work_dir) if scope.work_dir else repo_root
                abs_path = os.path.normpath(abs_path)

                # Ensure the path doesn't escape repo_root
                if not abs_path.startswith(os.path.normpath(repo_root)):
                    return False, "work_dir escapes repository root", None

                scope.absolute_path = abs_path
            else:
                scope.absolute_path = scope.work_dir

            return True, "", scope

        except Exception as e:
            logger.error(f"Error validating work_dir: {e}")
            return False, str(e), None

    @classmethod
    def check_path_access(
        cls,
        file_path: str,
        scope_context: ScopeContext
    ) -> tuple[bool, str]:
        """
        Check if a file path is accessible within the given scope context.

        Returns:
            Tuple of (is_allowed, reason)
        """
        if not scope_context.work_dir:
            # No work_dir restriction, allow access
            return True, "No work_dir restriction"

        try:
            # Normalize the file path
            file_path = os.path.normpath(file_path)

            # If it's an absolute path, check if it's within repo_root
            if os.path.isabs(file_path):
                if scope_context.repo_root:
                    repo_root = os.path.normpath(scope_context.repo_root)
                    if not file_path.startswith(repo_root):
                        return False, "File path outside repository root"

                    # Convert to repo-relative path
                    rel_path = os.path.relpath(file_path, repo_root)
                else:
                    # Can't validate absolute paths without repo_root
                    return False, "Cannot validate absolute path without repo_root"
            else:
                rel_path = file_path

            # Normalize path separators
            rel_path = rel_path.replace('\\', '/')

            # Check if path is within work_dir
            work_dir = scope_context.work_dir.replace('\\', '/')

            if rel_path.startswith(work_dir):
                return True, "Path within work_dir scope"

            # Check if it's a shared interface that can be read
            path_parts = rel_path.split('/')
            if path_parts[0] in cls.SHARED_INTERFACES and scope_context.can_read_shared:
                return True, "Shared interface access allowed"

            # Check parent directories in allowed_work_dirs
            for allowed in scope_context.allowed_work_dirs:
                if rel_path.startswith(allowed):
                    return True, f"Path within allowed work_dir: {allowed}"

            return False, f"Path outside work_dir scope: {work_dir}"

        except Exception as e:
            logger.error(f"Error checking path access: {e}")
            return False, str(e)

    @classmethod
    def filter_file_list(
        cls,
        files: list[str],
        scope_context: ScopeContext
    ) -> list[str]:
        """Filter a list of files to only include those accessible in scope."""
        filtered = []
        for file_path in files:
            allowed, _ = cls.check_path_access(file_path, scope_context)
            if allowed:
                filtered.append(file_path)
            else:
                logger.debug(f"Filtered out file outside scope: {file_path}")

        return filtered


def get_repo_root() -> str:
    """Get repository root directory."""
    # Assumes process CWD is repo root in this environment
    return os.getcwd()


def _is_subpath(path: str, base: str) -> bool:
    """Check if 'path' is inside 'base' (or equal to it)."""
    try:
        path = os.path.realpath(path)
        base = os.path.realpath(base)
        return os.path.commonpath([path, base]) == base
    except Exception:
        return False


def validate_and_normalize_work_dir(work_dir: str) -> tuple[str, str]:
    """
    Validate that work_dir is a repo-relative directory within the repository root.

    Returns: (normalized_relative, absolute_path)
    Raises: WorkDirError on validation failure.
    """
    if not work_dir or not isinstance(work_dir, str):
        # Empty work_dir means repo root
        repo_root = get_repo_root()
        return "", repo_root

    # Normalize separators and remove leading ./
    rel = os.path.normpath(work_dir).lstrip(os.sep)

    # Disallow absolute paths or parent traversal
    if rel.startswith("..") or os.path.isabs(work_dir):
        raise WorkDirError("'work_dir' must be a repo-relative path within the repository")

    repo_root = get_repo_root()
    abs_path = os.path.realpath(os.path.join(repo_root, rel))

    if not _is_subpath(abs_path, repo_root):
        raise WorkDirError("'work_dir' must resolve inside the repository root")

    # Require the directory to already exist; server handles warn-only fallback
    if not os.path.exists(abs_path):
        raise WorkDirError(f"'work_dir' directory does not exist: {rel}")
    if not os.path.isdir(abs_path):
        raise WorkDirError(f"'work_dir' exists but is not a directory: {rel}")

    return rel, abs_path


def inject_scope_context(
    tool_args: dict[str, Any],
    scope_context: ScopeContext
) -> dict[str, Any]:
    """
    Inject scope context into tool arguments.

    This function is called by the server before passing arguments to tools.
    """
    # Create a copy to avoid modifying original
    args = tool_args.copy()

    # Add work_dir if not present
    if 'work_dir' not in args and scope_context.work_dir:
        args['work_dir'] = scope_context.work_dir

    # Validate and potentially override work_dir
    if 'work_dir' in args:
        is_valid, error, validated_scope = ScopeValidator.validate_work_dir(
            args['work_dir'],
            scope_context.repo_root,
            scope_context.allowed_work_dirs
        )

        if not is_valid:
            logger.warning(f"Invalid work_dir in tool args: {error}")
            # Fall back to scope context work_dir
            args['work_dir'] = scope_context.work_dir

    # Add scope context metadata (for tools that need it)
    args['_scope_context'] = scope_context.to_dict()

    # Filter file lists if present
    if 'files' in args and isinstance(args['files'], list):
        args['files'] = ScopeValidator.filter_file_list(args['files'], scope_context)

    return args


def extract_work_dir_from_args(args: dict[str, Any]) -> Optional[str]:
    """Extract work_dir from tool arguments."""
    # Direct work_dir parameter
    if 'work_dir' in args:
        return args['work_dir']

    # Check common aliases
    for key in ['working_directory', 'workdir', 'directory', 'dir', 'path']:
        if key in args:
            value = args[key]
            if isinstance(value, str) and not os.path.isabs(value):
                return value

    # Try to extract from file paths
    if 'file' in args or 'file_path' in args:
        file_path = args.get('file') or args.get('file_path')
        if file_path and not os.path.isabs(file_path):
            # Get directory part of file path
            dir_part = os.path.dirname(file_path)
            if dir_part:
                return dir_part

    return None


def create_default_scope_context(
    agent_id: str = "default-agent",
    work_dir: str = "",
    session_id: Optional[str] = None
) -> ScopeContext:
    """Create a default scope context for testing or fallback."""
    return ScopeContext(
        agent_id=agent_id,
        work_dir=work_dir,
        repo_root=os.getcwd(),
        allowed_work_dirs=[work_dir] if work_dir else [],
        roles=["user"],
        permissions={"read", "write"},
        session_id=session_id or str(uuid4())
    )
