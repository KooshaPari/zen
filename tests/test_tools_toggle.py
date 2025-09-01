

def test_disable_legacy_tools(monkeypatch):
    from tools import get_all_tools
    # Ensure legacy tools shown by default
    monkeypatch.delenv('DISABLE_LEGACY_TOOLS', raising=False)
    names = set(get_all_tools().keys())
    assert 'deploy' in names
    assert 'chat' in names  # legacy present by default

    # Now disable legacy
    monkeypatch.setenv('DISABLE_LEGACY_TOOLS', '1')
    names2 = set(get_all_tools().keys())
    assert 'deploy' in names2
    assert 'messaging' in names2 and 'project' in names2 and 'a2a' in names2
    assert 'chat' not in names2

