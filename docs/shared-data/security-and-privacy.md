# Security and Privacy

## Principles
- Zero-trust, least privilege, strong identity
- Tenant isolation by default; explicit shared interfaces only
- Comprehensive auditability and data lifecycle controls

## Authentication
- OAuth2/OIDC with rotating keys; mTLS for service-to-service
- Token lifetimes short; refresh via secure channel
- Device-bound tokens for high-sensitivity agents (hardware-bound)

## Authorization
- RBAC + ABAC policy engine; deny by default
- Inputs: identity claims, tool, resource type, scope_context, operation
- Policy bundles versioned and signed

## Data Protection
- At rest: per-tenant encryption keys (KMS) for vector DB, KG, caches
- In transit: TLS everywhere
- Secret management: environment + vault; never log secrets

## Privacy Controls
- PII tagging in knowledge corpora; masked or excluded from RAG by default
- Redaction at ingestion; reversible only with elevated permissions
- Memory retention windows configurable; right-to-forget workflows

## Auditing
- Structured logs with request_id, session_id, principal, scope_context
- Event bus for security analytics and anomaly detection
- Immutable append-only audit store with retention policies

## Threats & Mitigations
- Data exfiltration across work_dirs -> strict path guards and namespace filters
- Prompt injection via shared stores -> retrieval sanitization and citation check (CAG)
- Token leakage -> short TTL, rotation, mTLS pinning
- Supply chain -> SBOM verification, signature checks, allowlists

## Compliance Posture
- Map controls to SOC2/ISO domains; document evidence from logs and tests

## Incident Response
- Kill-switch to revoke sessions by tenant
- Quarantine namespaces in vector/KG
- Forensics via audit trail + snapshot backups

