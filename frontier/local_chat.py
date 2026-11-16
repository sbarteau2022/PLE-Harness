"""Driver-owned chat codecs with PLE-owned admission of nested tensor prefixes."""

from copy import deepcopy
from dataclasses import dataclass
from threading import Lock
import time
import uuid

from openai.types.chat import ChatCompletion, ChatCompletionMessage

from frontier.kv_contract import MAX_COMPONENTS, RuntimeIdentity, _token_bytes
from frontier.runtime_registry import load_runtime
from frontier.checkpoints import NativeCacheContext, eligible_anchors, discover, publish


@dataclass(frozen=True)
class NestedPrefix:
    identity: RuntimeIdentity
    prefix_ids: tuple[int, ...]
    witnesses: tuple[dict, ...]
    recalled_addresses: frozenset[str]


@dataclass(frozen=True)
class PreparedChat:
    prompt_ids: tuple[int, ...]
    cache: NestedPrefix | None = None
    context: NativeCacheContext | None = None


class LocalChatClient:
    def __init__(self, config):
        self.model = config.model
        self.limit = config.max_output_tokens
        if type(self.limit) is not int or not 1 <= self.limit <= 4096:
            raise ValueError("local generation limit must be between 1 and 4096")
        self.runtime = load_runtime({"model": config.model, "local_runtime": config.local_runtime})
        if self.runtime is None:
            raise ValueError("local chat requires an explicitly configured runtime")
        self.runtime.require("prepare_chat", "generate_tokens", "decode_chat")
        self.bridge = config.local_runtime.get("bridge")
        self.mind = None
        self._lock = Lock()

    def bind_mind(self, mind):
        with self._lock:
            self.mind = mind

    def complete(self, messages, tools):
        with self._lock:
            return self._complete(messages, tools)

    def _complete(self, messages, tools):
        prepared = self.runtime.call("prepare_chat", deepcopy(messages), deepcopy(tools), self.mind)
        if not isinstance(prepared, PreparedChat):
            raise ValueError("runtime preparation must return PreparedChat")
        _token_bytes(prepared.prompt_ids)
        cache = prepared.cache
        anchors = ()
        if prepared.context is not None:
            if cache is not None:
                raise ValueError("chat must choose automatic context or an explicit prefix receipt")
            if not isinstance(prepared.context, NativeCacheContext):
                raise ValueError("automatic checkpoint context must be typed")
            prepared.context.validate(self.runtime.identity())
            if self.mind is None or not isinstance(self.bridge, str) or not self.bridge:
                raise ValueError("automatic checkpoints require the shared Mind and bridge")
            self.runtime.require("export_prefix")
            anchors = eligible_anchors(self.runtime.identity(), prepared.prompt_ids)
            for length, _ in reversed(anchors):
                prefix = prepared.prompt_ids[:length]
                roster = discover(self.mind, self.runtime.identity(), prefix, prepared.context)
                if roster is not None:
                    cache = NestedPrefix(self.runtime.identity(), prefix, roster, frozenset({prepared.context.address}))
                    break
        cached = 0
        if cache is None:
            output = self.runtime.call("generate_tokens", prepared.prompt_ids, self.limit)
        else:
            if not isinstance(cache, NestedPrefix):
                raise ValueError("runtime cache candidate must be a NestedPrefix")
            _token_bytes(cache.prefix_ids)
            cached = len(cache.prefix_ids)
            if cached >= len(prepared.prompt_ids) or prepared.prompt_ids[:cached] != cache.prefix_ids:
                raise ValueError("cache candidate is not a strict causal prefix of this chat")
            if cache.identity != self.runtime.identity():
                raise ValueError("cache candidate runtime identity changed")
            if (type(cache.witnesses) is not tuple or not 1 <= len(cache.witnesses) <= MAX_COMPONENTS
                    or any(type(q) is not dict for q in cache.witnesses)
                    or type(cache.recalled_addresses) is not frozenset or not 1 <= len(cache.recalled_addresses) <= MAX_COMPONENTS
                    or any(not isinstance(a, str) or not a for a in cache.recalled_addresses)):
                raise ValueError("cache candidate has invalid native roster or selected addresses")
            if self.mind is None or not isinstance(self.bridge, str) or not self.bridge:
                raise ValueError("nested chat continuation requires the shared Mind and configured bridge")
            output = self.mind.continue_tensor_cache(
                self.runtime, cache.prefix_ids, deepcopy(cache.witnesses), prepared.prompt_ids[cached:],
                self.limit, self.bridge, recalled_addresses=set(cache.recalled_addresses))
        _token_bytes(output)
        if len(output) > self.limit:
            raise ValueError("runtime output exceeded generation limit")
        decoded = self.runtime.call("decode_chat", output)
        if type(decoded) is not dict:
            raise ValueError("runtime decoder must return an assistant message object")
        message = ChatCompletionMessage.model_validate(decoded)
        if message.role != "assistant":
            raise ValueError("runtime decoder returned a non-assistant message")
        if anchors and cached != anchors[-1][0]:
            checkpoint = self.runtime.export_prefix(prepared.prompt_ids[:anchors[-1][0]])
            publish(self.mind, checkpoint, prepared.context, self.bridge)
            self.runtime.identity()
        return ChatCompletion.model_validate({
            "id": "ple-local-" + uuid.uuid4().hex, "object": "chat.completion",
            "created": int(time.time()), "model": self.model,
            "choices": [{"index": 0, "finish_reason": "tool_calls" if message.tool_calls else "stop",
                         "message": message.model_dump(exclude_none=True)}],
            "usage": {"prompt_tokens": len(prepared.prompt_ids), "completion_tokens": len(output),
                      "total_tokens": len(prepared.prompt_ids) + len(output),
                      "prompt_tokens_details": {"cached_tokens": cached}},
        })
