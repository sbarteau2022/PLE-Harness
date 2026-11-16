"""Load a driver-selected local runtime without choosing a model or serving wire."""

import importlib
import re

from frontier.kv_contract import (ExactPageSet, ExactStateSet, RuntimeIdentity,
                                  _token_bytes, continue_exact)


class BoundRuntime:
    """Pin the selected runtime's full identity across export and continuation."""

    def __init__(self, runtime, expected_model_digest):
        if not callable(getattr(runtime, "identity", None)) or not callable(getattr(runtime, "continue_from_pages", None)):
            raise ValueError("runtime does not implement LocalPageRuntime")
        identity = runtime.identity()
        if not isinstance(identity, RuntimeIdentity):
            raise ValueError("runtime must report a typed identity and tensor shape")
        identity.validate()
        if identity.model_digest != expected_model_digest:
            raise ValueError("runtime artifact differs from selected driver profile")
        self._runtime, self._pinned = runtime, identity

    def identity(self):
        live = self._runtime.identity()
        if live != self._pinned:
            raise ValueError("selected runtime identity changed")
        return live

    def call(self, name, *args):
        """Invoke a trusted plugin capability with the selected identity pinned."""
        self.identity()
        method = getattr(self._runtime, name, None)
        if not callable(method):
            raise ValueError(f"selected runtime does not support {name}")
        result = method(*args)
        self.identity()
        return result

    def require(self, *names):
        for name in names:
            if not callable(getattr(self._runtime, name, None)):
                raise ValueError(f"selected runtime does not support {name}")

    def export_prefix(self, prefix_ids):
        _token_bytes(prefix_ids)
        self.identity()
        pages = self.call("export_prefix", prefix_ids)
        if not isinstance(pages, (ExactPageSet, ExactStateSet)):
            raise ValueError("runtime export must be a typed exact checkpoint")
        pages.admit(self.identity(), prefix_ids)
        return pages

    def continue_from_pages(self, pages, continuation_ids, max_tokens):
        if not isinstance(pages, (ExactPageSet, ExactStateSet)):
            raise ValueError("runtime continuation requires a typed checkpoint")
        self.identity()
        output = continue_exact(self._runtime, pages, pages.prefix_ids, continuation_ids, max_tokens)
        self.identity()
        return output


def load_runtime(profile):
    """Trusted local profile owns plugin selection; no built-in adapter fallback."""
    config = profile.get("local_runtime")
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError("local_runtime must be an object")
    factory = config.get("factory", "")
    expected = config.get("model_digest", "")
    options = config.get("options", {})
    if not isinstance(factory, str) or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", factory):
        raise ValueError("runtime factory must be module:callable")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("runtime profile needs a pinned model SHA-256 digest")
    if not isinstance(options, dict) or "model" in options:
        raise ValueError("runtime options cannot override the driver model")
    module, name = factory.split(":")
    create = getattr(importlib.import_module(module), name)
    if not callable(create):
        raise ValueError("runtime factory is not callable")
    runtime = create(model=profile["model"], **options)
    return BoundRuntime(runtime, expected)
