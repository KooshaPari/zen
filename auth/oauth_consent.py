"""
OAuth 2.0 Consent and Scope Management System
Advanced consent handling with granular scope control
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ConsentStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


class ScopeCategory(Enum):
    BASIC = "basic"
    SENSITIVE = "sensitive"
    ADMINISTRATIVE = "administrative"


@dataclass
class ScopeDefinition:
    """Enhanced scope definition with consent metadata"""
    name: str
    display_name: str
    description: str
    category: ScopeCategory
    required: bool = False
    risk_level: str = "low"  # low, medium, high
    examples: list[str] = field(default_factory=list)
    dependencies: set[str] = field(default_factory=set)  # Other scopes this depends on

    def to_dict(self) -> dict:
        """Convert to dictionary for client display"""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "category": self.category.value,
            "required": self.required,
            "risk_level": self.risk_level,
            "examples": self.examples,
        }


@dataclass
class ConsentRecord:
    """Record of user consent for specific client and scopes"""
    user_id: str
    client_id: str
    granted_scopes: set[str]
    denied_scopes: set[str] = field(default_factory=set)
    consent_time: float = field(default_factory=time.time)
    expires_at: Optional[float] = None  # None = no expiration
    device_info: dict = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    def is_expired(self) -> bool:
        """Check if consent has expired"""
        return self.expires_at is not None and time.time() > self.expires_at

    def has_scope(self, scope: str) -> bool:
        """Check if consent includes a specific scope"""
        return scope in self.granted_scopes and not self.is_expired()

    def covers_scopes(self, requested_scopes: set[str]) -> bool:
        """Check if consent covers all requested scopes"""
        return requested_scopes.issubset(self.granted_scopes) and not self.is_expired()


class ConsentManager:
    """Manages OAuth consent and scope approval"""

    def __init__(self):
        self.consent_records: dict[str, ConsentRecord] = {}  # key: f"{user_id}:{client_id}"
        self.scope_definitions = self._initialize_scope_definitions()

        # Consent settings
        self.require_explicit_consent = True
        self.consent_expiry_days = 90  # Default consent expiration
        self.remember_consent = True

    def _initialize_scope_definitions(self) -> dict[str, ScopeDefinition]:
        """Initialize comprehensive scope definitions"""
        scopes = {
            # Basic access scopes
            "read": ScopeDefinition(
                name="read",
                display_name="Read Access",
                description="Read-only access to your data and resources",
                category=ScopeCategory.BASIC,
                required=True,
                risk_level="low",
                examples=["View your profile", "Read your files", "Access tool results"],
            ),

            "write": ScopeDefinition(
                name="write",
                display_name="Write Access",
                description="Create, modify, and delete your data",
                category=ScopeCategory.SENSITIVE,
                risk_level="medium",
                examples=["Create new files", "Modify existing data", "Save tool outputs"],
                dependencies={"read"},
            ),

            # Tool execution scopes
            "tools": ScopeDefinition(
                name="tools",
                display_name="Execute Tools",
                description="Execute MCP tools and services on your behalf",
                category=ScopeCategory.BASIC,
                risk_level="medium",
                examples=["Run code analysis", "Generate documentation", "Execute refactoring"],
            ),

            "tools:code": ScopeDefinition(
                name="tools:code",
                display_name="Code Tools",
                description="Execute code-related tools (analyze, refactor, debug)",
                category=ScopeCategory.SENSITIVE,
                risk_level="high",
                examples=["Analyze your code", "Suggest refactoring", "Debug applications"],
                dependencies={"tools", "read"},
            ),

            "tools:files": ScopeDefinition(
                name="tools:files",
                display_name="File Operations",
                description="Perform file system operations",
                category=ScopeCategory.SENSITIVE,
                risk_level="high",
                examples=["Create files", "Modify file contents", "Delete files"],
                dependencies={"tools", "write"},
            ),

            # System and admin scopes
            "admin": ScopeDefinition(
                name="admin",
                display_name="Administrative Access",
                description="Administrative access to server configuration",
                category=ScopeCategory.ADMINISTRATIVE,
                risk_level="high",
                examples=["Modify server settings", "Manage user accounts", "Access system logs"],
            ),

            "profile": ScopeDefinition(
                name="profile",
                display_name="Profile Information",
                description="Access to your basic profile information",
                category=ScopeCategory.BASIC,
                risk_level="low",
                examples=["Your user ID", "Account creation date", "Authentication methods"],
            ),

            # Session and device scopes
            "devices": ScopeDefinition(
                name="devices",
                display_name="Device Management",
                description="View and manage your registered devices",
                category=ScopeCategory.SENSITIVE,
                risk_level="medium",
                examples=["List your devices", "Revoke device access", "See device activity"],
                dependencies={"profile"},
            ),

            "sessions": ScopeDefinition(
                name="sessions",
                display_name="Session Management",
                description="View and manage your active sessions",
                category=ScopeCategory.SENSITIVE,
                risk_level="medium",
                examples=["List active sessions", "Revoke sessions", "See session history"],
                dependencies={"profile"},
            ),

            # Offline access
            "offline_access": ScopeDefinition(
                name="offline_access",
                display_name="Offline Access",
                description="Maintain access when you're not actively using the app",
                category=ScopeCategory.SENSITIVE,
                risk_level="medium",
                examples=["Long-running operations", "Scheduled tasks", "Background processing"],
            ),
        }

        return scopes

    def get_scope_info(self, scope_name: str) -> Optional[ScopeDefinition]:
        """Get detailed information about a scope"""
        return self.scope_definitions.get(scope_name)

    def validate_scopes(self, requested_scopes: set[str]) -> tuple[set[str], set[str]]:
        """Validate requested scopes and return (valid, invalid)"""
        valid_scopes: set[str] = set()
        invalid_scopes: set[str] = set()

        for scope in requested_scopes:
            if scope in self.scope_definitions:
                valid_scopes.add(scope)
            else:
                invalid_scopes.add(scope)

        return valid_scopes, invalid_scopes

    def resolve_scope_dependencies(self, requested_scopes: set[str]) -> set[str]:
        """Resolve scope dependencies and return all required scopes"""
        resolved_scopes = set(requested_scopes)

        # Keep adding dependencies until no new ones are found
        while True:
            new_scopes: set[str] = set()
            for scope in resolved_scopes:
                scope_def = self.scope_definitions.get(scope)
                if scope_def and scope_def.dependencies:
                    new_scopes.update(scope_def.dependencies)

            if new_scopes.issubset(resolved_scopes):
                break  # No new dependencies found

            resolved_scopes.update(new_scopes)

        return resolved_scopes

    def group_scopes_by_category(self, scopes: set[str]) -> dict[ScopeCategory, list[ScopeDefinition]]:
        """Group scopes by category for display"""
        categories: dict[ScopeCategory, list[ScopeDefinition]] = {
            ScopeCategory.BASIC: [],
            ScopeCategory.SENSITIVE: [],
            ScopeCategory.ADMINISTRATIVE: [],
        }

        for scope_name in scopes:
            scope_def = self.scope_definitions.get(scope_name)
            if scope_def:
                categories[scope_def.category].append(scope_def)

        # Remove empty categories
        return {cat: scopes for cat, scopes in categories.items() if scopes}

    def calculate_risk_score(self, scopes: set[str]) -> tuple[int, str]:
        """Calculate overall risk score for scope combination"""
        risk_scores = {"low": 1, "medium": 3, "high": 5}
        total_score = 0
        scope_count = 0

        for scope_name in scopes:
            scope_def = self.scope_definitions.get(scope_name)
            if scope_def:
                total_score += risk_scores.get(scope_def.risk_level, 1)
                scope_count += 1

        if scope_count == 0:
            return 0, "none"

        avg_score = total_score / scope_count

        if avg_score <= 1.5:
            return int(avg_score * scope_count), "low"
        elif avg_score <= 3.5:
            return int(avg_score * scope_count), "medium"
        else:
            return int(avg_score * scope_count), "high"

    def check_existing_consent(self, user_id: str, client_id: str, requested_scopes: set[str]) -> Optional[ConsentRecord]:
        """Check if user has existing valid consent for requested scopes"""
        consent_key = f"{user_id}:{client_id}"
        consent_record = self.consent_records.get(consent_key)

        if consent_record and consent_record.covers_scopes(requested_scopes):
            return consent_record
        return None

    def needs_consent(self, user_id: str, client_id: str, requested_scopes: set[str], client_trusted: bool = False) -> bool:
        """Determine if explicit consent is needed"""
        # Trusted clients may skip consent for basic scopes
        if client_trusted and requested_scopes.issubset({"read", "profile", "tools"}):
            return False

        # Check existing consent
        existing_consent = self.check_existing_consent(user_id, client_id, requested_scopes)
        if existing_consent:
            return False

        # Always require consent for high-risk scopes
        high_risk_scopes = {
            scope for scope in requested_scopes
            if self.scope_definitions.get(scope, ScopeDefinition("", "", "", ScopeCategory.BASIC)).risk_level == "high"
        }
        if high_risk_scopes:
            return True

        # Require consent if explicitly configured
        return self.require_explicit_consent

    def create_consent_context(self, user_id: str, client_id: str, client_name: str, requested_scopes: set[str], **context) -> dict:
        """Create consent context for UI display"""

        # Resolve dependencies
        resolved_scopes = self.resolve_scope_dependencies(requested_scopes)

        # Validate scopes
        valid_scopes, invalid_scopes = self.validate_scopes(resolved_scopes)

        if invalid_scopes:
            raise ValueError(f"Invalid scopes requested: {invalid_scopes}")

        # Group by category
        scope_groups = self.group_scopes_by_category(valid_scopes)

        # Calculate risk
        risk_score, risk_level = self.calculate_risk_score(valid_scopes)

        # Check existing consent
        existing_consent = self.check_existing_consent(user_id, client_id, requested_scopes)

        consent_context = {
            "user_id": user_id,
            "client_id": client_id,
            "client_name": client_name,
            "requested_scopes": list(requested_scopes),
            "resolved_scopes": list(resolved_scopes),
            "scope_groups": {
                category.value: [scope.to_dict() for scope in scopes]
                for category, scopes in scope_groups.items()
            },
            "risk_score": risk_score,
            "risk_level": risk_level,
            "existing_consent": existing_consent is not None,
            "consent_expires": self.consent_expiry_days,
            "remember_choice": self.remember_consent,
            **context,
        }

        return consent_context

    def record_consent(
        self,
        user_id: str,
        client_id: str,
        granted_scopes: set[str],
        denied_scopes: Optional[set[str]] = None,
        device_info: Optional[dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        custom_expiry: Optional[int] = None,
    ) -> ConsentRecord:
        """Record user consent decision"""

        denied_scopes = denied_scopes or set()
        device_info = device_info or {}

        # Calculate expiry
        expires_at = None
        if self.consent_expiry_days and custom_expiry is None:
            expires_at = time.time() + (self.consent_expiry_days * 24 * 3600)
        elif custom_expiry:
            expires_at = time.time() + (custom_expiry * 24 * 3600)

        consent_record = ConsentRecord(
            user_id=user_id,
            client_id=client_id,
            granted_scopes=granted_scopes,
            denied_scopes=denied_scopes,
            expires_at=expires_at,
            device_info=device_info,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        consent_key = f"{user_id}:{client_id}"
        self.consent_records[consent_key] = consent_record

        return consent_record

    def revoke_consent(self, user_id: str, client_id: str) -> bool:
        """Revoke user consent for a specific client"""
        consent_key = f"{user_id}:{client_id}"
        if consent_key in self.consent_records:
            del self.consent_records[consent_key]
            return True
        return False

    def get_user_consents(self, user_id: str) -> list[ConsentRecord]:
        """Get all consent records for a user"""
        return [
            consent for consent in self.consent_records.values()
            if consent.user_id == user_id and not consent.is_expired()
        ]

    def cleanup_expired_consents(self) -> int:
        """Clean up expired consent records"""
        expired_keys = [
            key for key, consent in self.consent_records.items()
            if consent.is_expired()
        ]

        for key in expired_keys:
            del self.consent_records[key]

        return len(expired_keys)

