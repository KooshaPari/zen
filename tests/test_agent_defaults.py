import os

from tools.shared.agent_models import AgentType
from utils.agent_defaults import build_effective_path_env, get_default_working_directory


def test_default_working_directory_prefers_provided():
    assert get_default_working_directory(AgentType.CLAUDE, "/tmp/foo") == "/tmp/foo"

def test_default_working_directory_client_friendly():
    # We can't easily stub client info here, but we can ensure it returns a path under ~/agents/
    wd = get_default_working_directory(AgentType.CLAUDE, None)
    assert os.path.expanduser("~/agents/") in wd

def test_build_effective_path_env_prepends():
    os.environ["ZEN_AGENT_PATHS"] = "/opt/homebrew/bin:/usr/local/bin"
    eff = build_effective_path_env()
    assert eff.startswith("/opt/homebrew/bin:/usr/local/bin")

