"""
Console pairing service for passwordless WebAuthn device registration.

Provides a secure flow:
1) Console starts pairing and receives pairing_id + short display code
2) User opens /auth/register?pairing=<id>, enters code
3) Server validates and returns WebAuthn create() options
4) User completes WebAuthn, server binds credential and marks completed

Storage uses Redis if available via utils.redis_manager, otherwise in-memory with TTL.
Codes are stored as salted SHA-256 digests (no plaintext at rest).
"""
from __future__ import annotations

import base64
import os
import secrets
import time
from dataclasses import asdict, dataclass
from typing import Any

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore

from fastapi import HTTPException

from auth.webauthn_flow import WebAuthnDeviceAuth
from utils import ratelimit


def _now() -> float:
    return time.time()


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _sha256_hex(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


@dataclass
class PairingRecord:
    pairing_id: str
    code_hash: str
    salt_b64: str
    status: str  # pending | claimed | completed | expired
    created_at: float
    expires_at: float
    attempts: int = 0
    user_id: str | None = None
    created_by_ip: str | None = None
    claim_token: str | None = None
    # Operator approval gating
    operator_required: bool = False
    operator_token: str | None = None
    operator_approved: bool = False

    def is_expired(self) -> bool:
        return _now() > self.expires_at


class PairingService:
    def __init__(
        self,
        webauthn: WebAuthnDeviceAuth,
        ttl_seconds: int = 600,
        code_length: int = 6,
        max_attempts: int = 5,
        redis_enabled: bool = False,
        require_operator_approval: bool = False,
    ):
        self.webauthn = webauthn
        self.ttl_seconds = max(60, int(ttl_seconds))
        self.code_length = min(8, max(6, int(code_length)))
        self.max_attempts = max(1, int(max_attempts))
        self._mem: dict[str, PairingRecord] = {}
        self.require_operator_approval = bool(require_operator_approval)

        self._redis = None
        if redis_enabled and redis is not None:
            try:
                from utils.redis_manager import RedisDB, get_redis_manager  # type: ignore

                self._redis = get_redis_manager().get_connection(RedisDB.STATE)
            except Exception:
                self._redis = None

    # --------------------------- helpers ---------------------------
    def _key(self, pairing_id: str) -> str:
        return f"pairing:{pairing_id}"

    def _store(self, rec: PairingRecord) -> None:
        if self._redis is not None:
            try:
                pipe = self._redis.pipeline()
                pipe.hset(self._key(rec.pairing_id), mapping={k: str(v) for k, v in asdict(rec).items()})
                pipe.expire(self._key(rec.pairing_id), int(rec.expires_at - _now()))
                pipe.execute()
                return
            except Exception:
                pass
        self._mem[rec.pairing_id] = rec

    def _load(self, pairing_id: str) -> PairingRecord | None:
        if self._redis is not None:
            try:
                data = self._redis.hgetall(self._key(pairing_id))
                if data:
                    # deserialize
                    rec = PairingRecord(
                        pairing_id=data.get("pairing_id", pairing_id),
                        code_hash=data.get("code_hash", ""),
                        salt_b64=data.get("salt_b64", ""),
                        status=data.get("status", "pending"),
                        created_at=float(data.get("created_at", "0")),
                        expires_at=float(data.get("expires_at", "0")),
                        attempts=int(data.get("attempts", "0")),
                        user_id=data.get("user_id") or None,
                        created_by_ip=data.get("created_by_ip") or None,
                        claim_token=data.get("claim_token") or None,
                    )
                    return rec
            except Exception:
                pass
        rec = self._mem.get(pairing_id)
        if rec and rec.is_expired():
            rec.status = "expired"
        return rec

    def _delete(self, pairing_id: str) -> None:
        if self._redis is not None:
            try:
                self._redis.delete(self._key(pairing_id))
            except Exception:
                pass
        self._mem.pop(pairing_id, None)

    def _gen_code(self) -> str:
        # 6–8 digit numeric code
        n = 10 ** self.code_length
        val = secrets.randbelow(n)
        return f"{val:0{self.code_length}d}"

    def _hash_code(self, code: str, salt_b64: str | None = None) -> tuple[str, str]:
        if salt_b64 is None:
            salt = os.urandom(16)
            salt_b64 = _b64url(salt)
        else:
            # decode
            pad = "=" * (-len(salt_b64) % 4)
            salt = base64.urlsafe_b64decode(salt_b64 + pad)
        digest = _sha256_hex(salt + code.encode())
        return digest, salt_b64

    # --------------------------- API ---------------------------
    def start_pairing(self, created_by_ip: str | None = None) -> dict[str, Any]:
        pairing_id = secrets.token_urlsafe(24)
        code = self._gen_code()
        code_hash, salt_b64 = self._hash_code(code)
        now = _now()
        rec = PairingRecord(
            pairing_id=pairing_id,
            code_hash=code_hash,
            salt_b64=salt_b64,
            status="pending",
            created_at=now,
            expires_at=now + self.ttl_seconds,
            attempts=0,
            user_id=None,
            created_by_ip=created_by_ip,
        )
        self._store(rec)
        qr_url = f"/auth/register?pairing={pairing_id}"
        return {
            "pairing_id": pairing_id,
            "display_code": code,
            "expires_at": rec.expires_at,
            "qr_url": qr_url,
        }

    def claim_pairing(self, pairing_id: str, code: str, user_id: str) -> dict[str, Any]:
        scope = f"pair_claim:{pairing_id}"
        if not ratelimit.allow(scope, max_per_window=10, window_seconds=60):
            raise HTTPException(status_code=429, detail="Too many attempts. Please wait and try again.")

        rec = self._load(pairing_id)
        if not rec or rec.status not in ("pending", "claimed") or rec.is_expired():
            raise HTTPException(status_code=400, detail="Invalid or expired pairing session")

        if rec.attempts >= self.max_attempts:
            rec.status = "expired"
            self._store(rec)
            raise HTTPException(status_code=400, detail="Too many attempts — pairing expired")

        rec.attempts += 1
        calc_hash, _ = self._hash_code(code, rec.salt_b64)
        if calc_hash != rec.code_hash:
            self._store(rec)
            raise HTTPException(status_code=401, detail="Invalid code")

        # Good code
        rec.status = "claimed"
        rec.user_id = user_id
        # issue a short claim token (not stored hashed; pairing_id still required)
        rec.claim_token = secrets.token_urlsafe(24)
        # if operator approval required, generate an operator_token (not returned)
        if self.require_operator_approval:
            rec.operator_required = True
            rec.operator_token = secrets.token_urlsafe(16)
            rec.operator_approved = False
        self._store(rec)

        # Do NOT return options here; endpoints decide what to return.
        return {
            "status": "ok",
            "claim_token": rec.claim_token,
            "user_id": user_id,
            "operator_required": rec.operator_required,
        }

    def complete_pairing(self, pairing_id: str, claim_token: str) -> PairingRecord:
        rec = self._load(pairing_id)
        if not rec or rec.is_expired() or rec.status not in ("claimed",):
            raise HTTPException(status_code=400, detail="Invalid or expired pairing session")
        if not rec.claim_token or claim_token != rec.claim_token:
            raise HTTPException(status_code=401, detail="Invalid claim token")
        rec.status = "completed"
        # Invalidate claim token
        rec.claim_token = None
        self._store(rec)
        return rec

    def get_status(self, pairing_id: str) -> dict[str, Any]:
        rec = self._load(pairing_id)
        if not rec:
            raise HTTPException(status_code=404, detail="Not found")
        return {
            "pairing_id": rec.pairing_id,
            "status": rec.status,
            "expires_at": rec.expires_at,
            "user_id": rec.user_id,
            "attempts": rec.attempts,
            "operator_required": rec.operator_required,
            "operator_approved": rec.operator_approved,
        }

    # Operator approval flow
    def get_operator_token_for_logging(self, pairing_id: str) -> str | None:
        rec = self._load(pairing_id)
        if not rec:
            return None
        return rec.operator_token

    def approve_pairing(self, pairing_id: str, operator_token: str) -> bool:
        rec = self._load(pairing_id)
        if not rec or rec.is_expired() or not rec.operator_required:
            raise HTTPException(status_code=400, detail="Invalid approval request")
        if not rec.operator_token or operator_token != rec.operator_token:
            raise HTTPException(status_code=401, detail="Invalid operator token")
        rec.operator_approved = True
        # one-time use
        rec.operator_token = None
        self._store(rec)
        return True
