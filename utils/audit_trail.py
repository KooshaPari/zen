"""
Compliance and Audit Trail System for Agent Orchestration

This module provides comprehensive audit logging and compliance capabilities
for agent orchestration systems, ensuring regulatory compliance and
providing immutable, tamper-evident audit trails.

Key Features:
- Immutable audit log entries with cryptographic integrity
- Compliance framework support (SOX, GDPR, HIPAA, PCI-DSS)
- Tamper-evident audit trails with chain verification
- Advanced audit querying and reporting
- Real-time compliance monitoring and alerting
- Audit event correlation and forensic analysis
- Retention policy management
- Export capabilities for compliance reporting
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable

try:
    import base64

    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

from pydantic import BaseModel, Field

from .kafka_events import AgentEvent, EventType, KafkaEventPublisher

logger = logging.getLogger(__name__)


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks."""

    SOX = "sox"              # Sarbanes-Oxley Act
    GDPR = "gdpr"            # General Data Protection Regulation
    HIPAA = "hipaa"          # Health Insurance Portability and Accountability Act
    PCI_DSS = "pci_dss"      # Payment Card Industry Data Security Standard
    SOC2 = "soc2"            # Service Organization Control 2
    ISO27001 = "iso27001"    # ISO 27001 Information Security Management
    NIST = "nist"            # NIST Cybersecurity Framework
    CUSTOM = "custom"        # Custom compliance requirements


class AuditEventCategory(str, Enum):
    """Categories of audit events."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    SYSTEM_ACCESS = "system_access"
    CONFIGURATION_CHANGE = "configuration_change"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SECURITY_INCIDENT = "security_incident"
    COMPLIANCE_VIOLATION = "compliance_violation"
    AGENT_LIFECYCLE = "agent_lifecycle"
    TASK_EXECUTION = "task_execution"
    WORKFLOW_EXECUTION = "workflow_execution"
    RESOURCE_USAGE = "resource_usage"
    ERROR_CONDITION = "error_condition"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ComplianceRule:
    """Compliance rule definition."""

    rule_id: str
    framework: ComplianceFramework
    title: str
    description: str
    category: AuditEventCategory
    severity: AuditSeverity
    required_fields: set[str] = field(default_factory=set)
    retention_days: int = 2555  # 7 years default
    encryption_required: bool = False
    real_time_monitoring: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert rule to dictionary."""
        return {
            "rule_id": self.rule_id,
            "framework": self.framework.value,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "severity": self.severity.value,
            "required_fields": list(self.required_fields),
            "retention_days": self.retention_days,
            "encryption_required": self.encryption_required,
            "real_time_monitoring": self.real_time_monitoring
        }


class AuditEntry(BaseModel):
    """Immutable audit log entry."""

    audit_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: EventType
    category: AuditEventCategory
    severity: AuditSeverity

    # Core audit fields
    user_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    task_id: str | None = None
    workflow_id: str | None = None

    # Event details
    action: str
    resource: str | None = None
    outcome: str  # "success", "failure", "partial"
    description: str

    # Context and metadata
    source_ip: str | None = None
    user_agent: str | None = None
    correlation_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    # Compliance and integrity
    compliance_frameworks: set[ComplianceFramework] = Field(default_factory=set)
    previous_hash: str | None = None
    current_hash: str | None = None
    digital_signature: str | None = None

    # Retention and lifecycle
    retention_date: datetime | None = None
    encrypted: bool = False

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            EventType: lambda v: v.value,
            AuditEventCategory: lambda v: v.value,
            AuditSeverity: lambda v: v.value,
            ComplianceFramework: lambda v: v.value,
            set: lambda v: list(v)
        }

    def calculate_hash(self, secret_key: str) -> str:
        """Calculate cryptographic hash for integrity verification."""

        # Create deterministic representation
        hash_data = {
            "audit_id": self.audit_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "category": self.category.value,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "action": self.action,
            "resource": self.resource,
            "outcome": self.outcome,
            "description": self.description,
            "payload": json.dumps(self.payload, sort_keys=True),
            "previous_hash": self.previous_hash
        }

        # Create HMAC signature
        data_string = json.dumps(hash_data, sort_keys=True)
        signature = hmac.new(
            secret_key.encode(),
            data_string.encode(),
            hashlib.sha256
        ).hexdigest()

        return signature

    def verify_integrity(self, secret_key: str) -> bool:
        """Verify the integrity of the audit entry."""

        if not self.current_hash:
            return False

        calculated_hash = self.calculate_hash(secret_key)
        return hmac.compare_digest(calculated_hash, self.current_hash)

    def to_kafka_message(self) -> dict[str, Any]:
        """Convert to Kafka message format."""
        return {
            "audit_id": self.audit_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "category": self.category.value,
            "severity": self.severity.value,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "workflow_id": self.workflow_id,
            "action": self.action,
            "resource": self.resource,
            "outcome": self.outcome,
            "description": self.description,
            "source_ip": self.source_ip,
            "user_agent": self.user_agent,
            "correlation_id": self.correlation_id,
            "payload": self.payload,
            "compliance_frameworks": list(self.compliance_frameworks),
            "previous_hash": self.previous_hash,
            "current_hash": self.current_hash,
            "digital_signature": self.digital_signature,
            "retention_date": self.retention_date.isoformat() if self.retention_date else None,
            "encrypted": self.encrypted
        }


class AuditChain:
    """Manages cryptographic chain of audit entries."""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.last_hash: str | None = None
        self._lock = asyncio.Lock()

    async def add_entry(self, entry: AuditEntry) -> AuditEntry:
        """Add entry to audit chain with integrity protection."""

        async with self._lock:
            # Set previous hash
            entry.previous_hash = self.last_hash

            # Calculate current hash
            entry.current_hash = entry.calculate_hash(self.secret_key)

            # Update chain state
            self.last_hash = entry.current_hash

            return entry

    async def verify_chain(self, entries: list[AuditEntry]) -> dict[str, Any]:
        """Verify the integrity of an audit chain."""

        verification_result = {
            "valid": True,
            "total_entries": len(entries),
            "verified_entries": 0,
            "broken_links": [],
            "integrity_failures": []
        }

        previous_hash = None

        for i, entry in enumerate(entries):
            # Verify entry integrity
            if not entry.verify_integrity(self.secret_key):
                verification_result["valid"] = False
                verification_result["integrity_failures"].append({
                    "index": i,
                    "audit_id": entry.audit_id,
                    "timestamp": entry.timestamp
                })
                continue

            # Verify chain linkage
            if entry.previous_hash != previous_hash:
                verification_result["valid"] = False
                verification_result["broken_links"].append({
                    "index": i,
                    "audit_id": entry.audit_id,
                    "expected_previous": previous_hash,
                    "actual_previous": entry.previous_hash
                })
            else:
                verification_result["verified_entries"] += 1

            previous_hash = entry.current_hash

        return verification_result


class ComplianceRuleEngine:
    """Engine for managing and evaluating compliance rules."""

    def __init__(self):
        self.rules: dict[str, ComplianceRule] = {}
        self._load_default_rules()

    def _load_default_rules(self):
        """Load default compliance rules for common frameworks."""

        # SOX Rules
        self.add_rule(ComplianceRule(
            rule_id="SOX-001",
            framework=ComplianceFramework.SOX,
            title="Financial Data Access Logging",
            description="All access to financial data must be logged",
            category=AuditEventCategory.DATA_ACCESS,
            severity=AuditSeverity.HIGH,
            required_fields={"user_id", "resource", "timestamp"},
            retention_days=2555,  # 7 years
            real_time_monitoring=True
        ))

        self.add_rule(ComplianceRule(
            rule_id="SOX-002",
            framework=ComplianceFramework.SOX,
            title="Privileged Access Monitoring",
            description="All administrative actions must be audited",
            category=AuditEventCategory.PRIVILEGE_ESCALATION,
            severity=AuditSeverity.CRITICAL,
            required_fields={"user_id", "action", "timestamp", "outcome"},
            retention_days=2555,
            real_time_monitoring=True
        ))

        # GDPR Rules
        self.add_rule(ComplianceRule(
            rule_id="GDPR-001",
            framework=ComplianceFramework.GDPR,
            title="Personal Data Processing Log",
            description="All processing of personal data must be logged",
            category=AuditEventCategory.DATA_MODIFICATION,
            severity=AuditSeverity.HIGH,
            required_fields={"user_id", "resource", "timestamp", "description"},
            retention_days=2190,  # 6 years
            encryption_required=True,
            real_time_monitoring=True
        ))

        self.add_rule(ComplianceRule(
            rule_id="GDPR-002",
            framework=ComplianceFramework.GDPR,
            title="Data Subject Rights Exercise",
            description="Requests for data subject rights must be logged",
            category=AuditEventCategory.DATA_ACCESS,
            severity=AuditSeverity.HIGH,
            required_fields={"user_id", "action", "timestamp", "outcome"},
            retention_days=2190,
            encryption_required=True
        ))

        # HIPAA Rules
        self.add_rule(ComplianceRule(
            rule_id="HIPAA-001",
            framework=ComplianceFramework.HIPAA,
            title="PHI Access Logging",
            description="All access to Protected Health Information must be logged",
            category=AuditEventCategory.DATA_ACCESS,
            severity=AuditSeverity.CRITICAL,
            required_fields={"user_id", "resource", "timestamp", "source_ip"},
            retention_days=2190,  # 6 years
            encryption_required=True,
            real_time_monitoring=True
        ))

        # PCI-DSS Rules
        self.add_rule(ComplianceRule(
            rule_id="PCI-001",
            framework=ComplianceFramework.PCI_DSS,
            title="Cardholder Data Access",
            description="All access to cardholder data must be logged and monitored",
            category=AuditEventCategory.DATA_ACCESS,
            severity=AuditSeverity.CRITICAL,
            required_fields={"user_id", "resource", "timestamp", "source_ip", "outcome"},
            retention_days=365,  # 1 year minimum
            encryption_required=True,
            real_time_monitoring=True
        ))

    def add_rule(self, rule: ComplianceRule):
        """Add a compliance rule."""
        self.rules[rule.rule_id] = rule

    def get_applicable_rules(self,
                           category: AuditEventCategory,
                           frameworks: set[ComplianceFramework] = None) -> list[ComplianceRule]:
        """Get rules applicable to an event category and frameworks."""

        applicable_rules = []

        for rule in self.rules.values():
            if rule.category == category:
                if not frameworks or rule.framework in frameworks:
                    applicable_rules.append(rule)

        return applicable_rules

    def validate_entry(self,
                      entry: AuditEntry,
                      frameworks: set[ComplianceFramework] = None) -> dict[str, Any]:
        """Validate audit entry against compliance rules."""

        validation_result = {
            "valid": True,
            "violations": [],
            "warnings": [],
            "applicable_rules": []
        }

        applicable_rules = self.get_applicable_rules(entry.category, frameworks)

        for rule in applicable_rules:
            validation_result["applicable_rules"].append(rule.rule_id)

            # Check required fields
            entry_dict = entry.dict()
            missing_fields = rule.required_fields - set(entry_dict.keys())

            if missing_fields:
                validation_result["valid"] = False
                validation_result["violations"].append({
                    "rule_id": rule.rule_id,
                    "type": "missing_required_fields",
                    "details": list(missing_fields)
                })

            # Check encryption requirement
            if rule.encryption_required and not entry.encrypted:
                validation_result["warnings"].append({
                    "rule_id": rule.rule_id,
                    "type": "encryption_recommended",
                    "details": "Entry should be encrypted per compliance requirements"
                })

            # Set retention date
            if entry.retention_date is None:
                retention_date = entry.timestamp + timedelta(days=rule.retention_days)
                entry.retention_date = retention_date

        return validation_result


class AuditTrailManager:
    """Comprehensive audit trail management system."""

    def __init__(self,
                 kafka_publisher: KafkaEventPublisher | None = None,
                 secret_key: str | None = None,
                 enable_encryption: bool = False):

        self.kafka_publisher = kafka_publisher
        self.secret_key = secret_key or os.getenv("AUDIT_SECRET_KEY", self._generate_secret_key())
        self.enable_encryption = enable_encryption and CRYPTO_AVAILABLE

        # Initialize components
        self.audit_chain = AuditChain(self.secret_key)
        self.rule_engine = ComplianceRuleEngine()

        # Encryption setup
        if self.enable_encryption and CRYPTO_AVAILABLE:
            self.encryption_key = self._derive_encryption_key(self.secret_key)
            self.cipher = Fernet(self.encryption_key)
        else:
            self.encryption_key = None
            self.cipher = None

        # Audit storage
        self.audit_entries: list[AuditEntry] = []
        self._lock = asyncio.Lock()

        # Compliance monitoring
        self.violation_handlers: list[Callable] = []
        self.real_time_monitors: dict[str, Callable] = {}

    def _generate_secret_key(self) -> str:
        """Generate a secret key for HMAC operations."""
        return base64.b64encode(os.urandom(32)).decode()

    def _derive_encryption_key(self, secret: str) -> bytes:
        """Derive encryption key from secret."""
        if not CRYPTO_AVAILABLE:
            return b""

        salt = b"audit_trail_salt"  # In production, use random salt per installation
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(secret.encode()))
        return key

    def _encrypt_payload(self, data: dict[str, Any]) -> str:
        """Encrypt sensitive audit data."""
        if not self.cipher:
            return json.dumps(data)

        plaintext = json.dumps(data).encode()
        encrypted = self.cipher.encrypt(plaintext)
        return base64.b64encode(encrypted).decode()

    def _decrypt_payload(self, encrypted_data: str) -> dict[str, Any]:
        """Decrypt audit data."""
        if not self.cipher:
            return json.loads(encrypted_data)

        encrypted = base64.b64decode(encrypted_data.encode())
        decrypted = self.cipher.decrypt(encrypted)
        return json.loads(decrypted.decode())

    async def create_audit_entry(self,
                               event_type: EventType,
                               category: AuditEventCategory,
                               action: str,
                               description: str,
                               severity: AuditSeverity = AuditSeverity.INFO,
                               user_id: str | None = None,
                               agent_id: str | None = None,
                               task_id: str | None = None,
                               resource: str | None = None,
                               outcome: str = "success",
                               payload: dict[str, Any] = None,
                               compliance_frameworks: set[ComplianceFramework] = None,
                               **kwargs) -> AuditEntry:
        """Create a new audit entry."""

        # Create base entry
        entry = AuditEntry(
            event_type=event_type,
            category=category,
            severity=severity,
            user_id=user_id,
            agent_id=agent_id,
            task_id=task_id,
            action=action,
            resource=resource,
            outcome=outcome,
            description=description,
            payload=payload or {},
            compliance_frameworks=compliance_frameworks or set(),
            **kwargs
        )

        # Validate against compliance rules
        frameworks = compliance_frameworks or {ComplianceFramework.CUSTOM}
        validation_result = self.rule_engine.validate_entry(entry, frameworks)

        if not validation_result["valid"]:
            logger.warning(f"Audit entry validation failed: {validation_result['violations']}")

            # Notify violation handlers
            for handler in self.violation_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(entry, validation_result)
                    else:
                        handler(entry, validation_result)
                except Exception as e:
                    logger.error(f"Error in violation handler: {e}")

        # Encrypt sensitive data if required
        applicable_rules = self.rule_engine.get_applicable_rules(category, frameworks)
        encryption_required = any(rule.encryption_required for rule in applicable_rules)

        if encryption_required and self.enable_encryption:
            entry.payload = {"encrypted": self._encrypt_payload(entry.payload)}
            entry.encrypted = True

        # Add to audit chain for integrity
        entry = await self.audit_chain.add_entry(entry)

        # Store entry
        async with self._lock:
            self.audit_entries.append(entry)

        # Publish to Kafka for persistence and downstream processing
        if self.kafka_publisher:
            await self._publish_audit_event(entry)

        # Real-time monitoring
        await self._process_real_time_monitoring(entry, applicable_rules)

        logger.info(f"Created audit entry {entry.audit_id} for {action}")
        return entry

    async def _publish_audit_event(self, entry: AuditEntry):
        """Publish audit entry as Kafka event."""

        from .kafka_events import EventMetadata

        audit_event = AgentEvent(
            event_type=EventType.AUTHENTICATION_SUCCESS,  # Map to appropriate event type
            aggregate_id=entry.agent_id or entry.audit_id,
            aggregate_type="audit",
            payload=entry.to_kafka_message(),
            metadata=EventMetadata(
                correlation_id=entry.correlation_id,
                session_id=entry.session_id,
                user_id=entry.user_id
            )
        )

        await self.kafka_publisher.publish_event(audit_event, topic="audit-events")

    async def _process_real_time_monitoring(self,
                                          entry: AuditEntry,
                                          applicable_rules: list[ComplianceRule]):
        """Process real-time monitoring for compliance rules."""

        for rule in applicable_rules:
            if rule.real_time_monitoring:
                monitor = self.real_time_monitors.get(rule.rule_id)
                if monitor:
                    try:
                        if asyncio.iscoroutinefunction(monitor):
                            await monitor(entry, rule)
                        else:
                            monitor(entry, rule)
                    except Exception as e:
                        logger.error(f"Error in real-time monitor {rule.rule_id}: {e}")

    async def query_audit_trail(self,
                              start_time: datetime | None = None,
                              end_time: datetime | None = None,
                              user_id: str | None = None,
                              agent_id: str | None = None,
                              task_id: str | None = None,
                              category: AuditEventCategory | None = None,
                              severity: AuditSeverity | None = None,
                              outcome: str | None = None,
                              compliance_framework: ComplianceFramework | None = None,
                              limit: int = 1000,
                              offset: int = 0) -> dict[str, Any]:
        """Query audit trail with filtering and pagination."""

        # Default time range (last 24 hours)
        if not end_time:
            end_time = datetime.now(timezone.utc)
        if not start_time:
            start_time = end_time - timedelta(days=1)

        # Filter entries
        filtered_entries = []

        async with self._lock:
            for entry in self.audit_entries:
                # Time range filter
                if not (start_time <= entry.timestamp <= end_time):
                    continue

                # Field filters
                if user_id and entry.user_id != user_id:
                    continue
                if agent_id and entry.agent_id != agent_id:
                    continue
                if task_id and entry.task_id != task_id:
                    continue
                if category and entry.category != category:
                    continue
                if severity and entry.severity != severity:
                    continue
                if outcome and entry.outcome != outcome:
                    continue
                if compliance_framework and compliance_framework not in entry.compliance_frameworks:
                    continue

                filtered_entries.append(entry)

        # Sort by timestamp (newest first)
        filtered_entries.sort(key=lambda e: e.timestamp, reverse=True)

        # Apply pagination
        total_count = len(filtered_entries)
        paginated_entries = filtered_entries[offset:offset + limit]

        # Decrypt entries if needed
        decrypted_entries = []
        for entry in paginated_entries:
            entry_dict = entry.dict()
            if entry.encrypted and self.cipher and "encrypted" in entry.payload:
                try:
                    entry_dict["payload"] = self._decrypt_payload(entry.payload["encrypted"])
                except Exception as e:
                    logger.error(f"Failed to decrypt audit entry {entry.audit_id}: {e}")
                    entry_dict["payload"] = {"error": "decryption_failed"}

            decrypted_entries.append(entry_dict)

        return {
            "total_count": total_count,
            "returned_count": len(decrypted_entries),
            "offset": offset,
            "limit": limit,
            "entries": decrypted_entries,
            "query_timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def verify_audit_integrity(self,
                                   start_time: datetime | None = None,
                                   end_time: datetime | None = None) -> dict[str, Any]:
        """Verify the integrity of audit trail."""

        # Get entries for verification
        query_result = await self.query_audit_trail(
            start_time=start_time,
            end_time=end_time,
            limit=10000  # Large limit for integrity check
        )

        entries = [AuditEntry(**entry_data) for entry_data in query_result["entries"]]

        # Verify chain integrity
        verification_result = await self.audit_chain.verify_chain(entries)

        verification_result.update({
            "query_period": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None
            },
            "verification_timestamp": datetime.now(timezone.utc).isoformat()
        })

        return verification_result

    async def generate_compliance_report(self,
                                       framework: ComplianceFramework,
                                       start_time: datetime | None = None,
                                       end_time: datetime | None = None) -> dict[str, Any]:
        """Generate compliance report for a specific framework."""

        # Get relevant rules
        relevant_rules = [
            rule for rule in self.rule_engine.rules.values()
            if rule.framework == framework
        ]

        # Query audit entries
        query_result = await self.query_audit_trail(
            start_time=start_time,
            end_time=end_time,
            compliance_framework=framework,
            limit=10000
        )

        # Analyze compliance
        report = {
            "framework": framework.value,
            "report_period": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None
            },
            "total_events": query_result["total_count"],
            "rules_analyzed": len(relevant_rules),
            "compliance_summary": {},
            "violations": [],
            "recommendations": [],
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

        # Analyze each rule
        for rule in relevant_rules:
            rule_events = [
                entry for entry in query_result["entries"]
                if entry.get("category") == rule.category.value
            ]

            rule_analysis = {
                "rule_id": rule.rule_id,
                "title": rule.title,
                "events_count": len(rule_events),
                "compliance_status": "compliant",
                "issues": []
            }

            # Check for violations
            for entry in rule_events:
                validation = self.rule_engine.validate_entry(
                    AuditEntry(**entry),
                    {framework}
                )

                if not validation["valid"]:
                    rule_analysis["compliance_status"] = "non_compliant"
                    rule_analysis["issues"].extend(validation["violations"])

            report["compliance_summary"][rule.rule_id] = rule_analysis

        # Generate recommendations
        non_compliant_rules = [
            rule_id for rule_id, analysis in report["compliance_summary"].items()
            if analysis["compliance_status"] == "non_compliant"
        ]

        if non_compliant_rules:
            report["recommendations"].append({
                "type": "compliance_violation",
                "description": f"Address violations in rules: {', '.join(non_compliant_rules)}",
                "priority": "high"
            })

        return report

    def add_violation_handler(self, handler: Callable):
        """Add handler for compliance violations."""
        self.violation_handlers.append(handler)

    def add_real_time_monitor(self, rule_id: str, monitor: Callable):
        """Add real-time monitor for a compliance rule."""
        self.real_time_monitors[rule_id] = monitor

    async def cleanup_expired_entries(self):
        """Clean up expired audit entries based on retention policies."""

        now = datetime.now(timezone.utc)

        async with self._lock:
            # Find expired entries
            expired_entries = []
            active_entries = []

            for entry in self.audit_entries:
                if entry.retention_date and now > entry.retention_date:
                    expired_entries.append(entry)
                else:
                    active_entries.append(entry)

            # Update active entries list
            self.audit_entries = active_entries

        if expired_entries:
            logger.info(f"Cleaned up {len(expired_entries)} expired audit entries")

        return len(expired_entries)


# Global audit trail manager
_audit_manager: AuditTrailManager | None = None


async def get_audit_manager() -> AuditTrailManager:
    """Get global audit trail manager instance."""
    global _audit_manager

    if _audit_manager is None:
        from .kafka_events import get_event_publisher
        publisher = await get_event_publisher()
        _audit_manager = AuditTrailManager(
            kafka_publisher=publisher,
            enable_encryption=os.getenv("AUDIT_ENABLE_ENCRYPTION", "false").lower() == "true"
        )

    return _audit_manager


async def create_audit_log(event_type: EventType,
                         category: AuditEventCategory,
                         action: str,
                         description: str,
                         **kwargs) -> AuditEntry:
    """Convenience function to create audit log entry."""

    manager = await get_audit_manager()
    return await manager.create_audit_entry(
        event_type=event_type,
        category=category,
        action=action,
        description=description,
        **kwargs
    )
