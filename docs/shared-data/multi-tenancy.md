# Multi-Tenancy Model

## Tenant Hierarchy
- org -> project -> team -> agent/user
- Each level contributes to namespace and policy resolution

## Isolation Strategies
- Hard isolation: DB-per-tenant (KG), collection-per-tenant (VectorDB), cache prefixing (LSP/QA)
- Soft isolation: label/attribute scoping with query guards
- Hybrid: shared infra + strong namespace keys

## Namespacing Keys
- `{org}/{proj}/{repo}/{work_dir}` for vector/LSP
- Node/edge attributes for KG with mandatory filters
- Memory keys prefixed by tenant + agent

## Resource Quotas
- Per-tenant limits: QPS, storage, concurrent jobs
- Backpressure and fair scheduling at service edges

## Onboarding/Offboarding
- Create namespaces, keys, and initial policies
- Revoke keys, archive namespaces, export data on offboarding

## Cross-Tenant Sharing
- Allow only via explicit shared resources (e.g., public API specs)
- Read-only with signed access grants; time-bound

## Migration & Versioning
- Policy and schema versioning with rolling upgrades
- Data migration jobs are idempotent and resumable

