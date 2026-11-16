"""Model profiles: put any OpenAI-compatible model in a seat by name or by URL.

Profiles live in ~/.config/frontier/profiles.json (or FRONTIER_PROFILES):

    {"qwen":    {"base_url": "http://127.0.0.1:8001/v1", "model": "Qwen3.6Cyber", "context": 65536},
     "mistral": {"base_url": "https://integrate.api.nvidia.com/v1", "model": "mistralai/mistral-large-4-1t-a52b",
                 "keychain": "https://integrate.api.nvidia.com/v1"}}

Optional fields: api_key_env (name of an env var), keychain (macOS keychain internet-password
service), context (tokens), max_output (tokens), tool_mode ("auto" or "text").
local_runtime selects a trusted plugin factory, pinned model digest and optional nested bridge.
A seat can also be given as "<base_url>|<model>" with no profile.
"""

from __future__ import annotations

import json
from copy import deepcopy
import os
import pathlib
import subprocess
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import agent  # noqa: E402

PROFILES_PATH = pathlib.Path(os.environ.get("FRONTIER_PROFILES", "~/.config/frontier/profiles.json")).expanduser()
DEFAULT_CONTEXT = 32768
CONTEXT_KEYS = ("context_length", "max_model_len", "context_window", "max_context_length", "n_ctx")


def load_profiles() -> dict:
    try:
        return json.loads(PROFILES_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def resolve_local_runtime(seat: str | None):
    """Optional local tensor runtime chosen by this driver profile alone."""
    from frontier.runtime_registry import load_runtime
    seat = seat or os.environ.get("FRONTIER_DRIVER", "")
    profiles = load_profiles()
    if seat not in profiles:
        return None  # URL/API seats retain the shared text-memory path
    return load_runtime(profiles[seat])


def _api_key(profile: dict) -> str:
    if profile.get("api_key_env") and os.environ.get(profile["api_key_env"]):
        return os.environ[profile["api_key_env"]]
    if profile.get("keychain"):
        found = subprocess.run(["security", "find-internet-password", "-s", profile["keychain"], "-w"],
                               capture_output=True, text=True)
        if found.returncode == 0 and found.stdout.strip():
            return found.stdout.strip()
    return os.environ.get("FRONTIER_API_KEY") or "none"


def probe_context(base_url: str, model: str, api_key: str) -> int | None:
    """Asks the server for the model's context window. Returns None if it does not say."""
    try:
        request = urllib.request.Request(base_url.rstrip("/") + "/models",
                                         headers={"Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(request, timeout=8) as response:
            data = json.loads(response.read(2_000_000))
    except Exception:
        return None
    for item in data.get("data", []) if isinstance(data, dict) else []:
        if item.get("id") == model:
            for key in CONTEXT_KEYS:
                if isinstance(item.get(key), int) and item[key] > 0:
                    return item[key]
    return None


def resolve(seat: str | None, max_steps: int = 60) -> agent.Config:
    """Turns a profile name or "<base_url>|<model>" into an agent Config sized to the model."""
    if not seat:
        seat = os.environ.get("FRONTIER_DRIVER", "")
    profiles = load_profiles()
    if seat in profiles:
        profile = dict(profiles[seat])
    elif "|" in seat:
        base_url, model = seat.split("|", 1)
        profile = {"base_url": base_url, "model": model}
    elif os.environ.get("FRONTIER_BASE_URL") and os.environ.get("FRONTIER_MODEL") and not seat:
        profile = {"base_url": os.environ["FRONTIER_BASE_URL"], "model": os.environ["FRONTIER_MODEL"]}
    else:
        sys.exit(f"unknown model seat {seat!r}: use a profile name from {PROFILES_PATH} "
                 f"(have: {sorted(profiles) or 'none'}) or '<base_url>|<model>'")
    key = _api_key(profile)
    runtime = deepcopy(profile.get("local_runtime"))
    if runtime is not None:
        context = profile.get("context") or DEFAULT_CONTEXT
    else:
        context = profile.get("context") or probe_context(profile["base_url"], profile["model"], key) or DEFAULT_CONTEXT
    return agent.Config(
        model=profile["model"], base_url=profile.get("base_url"), api_key=key, max_steps=max_steps,
        max_output_tokens=int(profile.get("max_output") or min(4096 if runtime is not None else 8000, context // 6)),
        compact_at_tokens=int(context * 0.6), checkpoint_every=0,
        tool_mode=profile.get("tool_mode", "auto"), local_runtime=runtime,
        carve_high=int(profile.get("carve_high") or 0))
