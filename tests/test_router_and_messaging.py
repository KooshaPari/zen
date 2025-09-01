

def test_router_cache_decide(monkeypatch):
    """RouterService caches decisions in Redis CACHE DB; when Redis unavailable, falls back gracefully.

    We assert:
    - First decide() returns cache_hit False
    - Second decide() with same input returns cache_hit True
    """
    from utils.router_service import RouterInput, RouterService
    # Force in-memory Redis manager to None so RouterService handles gracefully
    monkeypatch.setenv("ZEN_ROUTER_CACHE_TTL", "60")

    rin = RouterInput(task_type="quick_qa", prompt="Hello world")
    svc = RouterService(cache_ttl_seconds=60)

    d1 = svc.decide(rin)
    assert isinstance(d1, dict)
    assert d1.get("chosen_model")
    assert d1.get("cache_hit") is False

    d2 = svc.decide(rin)
    assert isinstance(d2, dict)
    assert d2.get("cache_hit") is True


def test_messaging_block_and_resume(monkeypatch):
    """Posting a blocking channel message then resuming marks it resolved and reduces unread count."""
    # Ensure memory storage mode
    monkeypatch.delenv("ZEN_STORAGE", raising=False)
    monkeypatch.setenv("ZEN_STORAGE_MODE", "memory")

    from utils.messaging_store import (
        create_channel,
        get_message,
        get_unread_summary,
        mark_block_resolved,
        mark_read,
        post_channel_message,
    )

    ch = create_channel("proj:test", "general", created_by="tester")
    cid = ch["id"]

    # Post a blocking message mentioning agent "alice"
    msg = post_channel_message(cid, from_id="system", body="Please review", mentions=["alice"], blocking=True)
    mid = msg["id"]

    unread_before = get_unread_summary("alice").get("total", 0)
    assert unread_before >= 1

    # Mark resolved and read
    ok = mark_block_resolved(mid, agent_id="alice")
    assert ok is True
    mark_read("alice", mid)

    # Verify message fields updated
    m = get_message(mid)
    assert m is not None
    assert m.get("resolved") in (True, "True")

    unread_after = get_unread_summary("alice").get("total", 0)
    assert unread_after <= unread_before - 1


def test_threads_resume_by_token_store(monkeypatch):
    """Blocking channel message should generate a resume_token; resolving by token works and can post a reply."""
    monkeypatch.setenv("ZEN_STORAGE_MODE", "memory")
    from utils.messaging_store import create_channel, get_thread_history, post_channel_message, resolve_by_token

    ch = create_channel("proj:t2", "general", created_by="tester")
    msg = post_channel_message(ch["id"], from_id="system", body="Block with token", mentions=["bob"], blocking=True)
    token = msg.get("resume_token")
    assert token

    # Resolve by token and add a reply
    res = resolve_by_token(token, reply_body="ack", from_id="bob")
    assert res.get("ok") is True
    rid = res.get("reply_id")
    assert rid
    # Thread history should now include the reply
    items = get_thread_history(msg["id"], limit=10, offset=0)
    assert any(m.get("id") == rid for m in items)
