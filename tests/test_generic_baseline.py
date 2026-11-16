import ast
import asyncio
import pathlib
from types import SimpleNamespace

from ablations.generic import agent


def test_generic_agent_has_no_ple_mind_imports():
    path = pathlib.Path(agent.__file__)
    tree = ast.parse(path.read_text())
    roots = {n.names[0].name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
    roots |= {str(n.module).split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not ({"frontier"} & roots)


def test_generic_server_does_not_import_ple_normalizer():
    source = (pathlib.Path(agent.__file__).parents[2] / "serve_generic.py").read_text()
    assert "ablations.normalize" not in source


def test_finish_beside_tool_is_refused(monkeypatch):
    calls = [SimpleNamespace(id="1", function=SimpleNamespace(name="write_file", arguments='{"path":"/tmp/x","content":"x"}')),
             SimpleNamespace(id="2", function=SimpleNamespace(name="finish", arguments='{"summary":"done"}'))]
    response = SimpleNamespace(id="live", usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=calls))])
    class Client:
        def __init__(self, **_): self.chat = SimpleNamespace(completions=self)
        async def create(self, **_): return response
    monkeypatch.setattr(agent.openai, "AsyncOpenAI", Client)
    class Session:
        mcp_servers=[]; workdir="/work"; instruction="do it"
        def time_left(self): return 1000
        async def write_file(self, *_): return 1
        async def log(self, *_): pass
        async def usage(self, *_args, **_kwargs): pass
    meter = {}
    cfg = agent.Config("m", None, "k", max_steps=1)
    asyncio.run(agent.run(Session(), cfg, meter))
    assert meter["finish_refusals"] == 1
    assert meter["by_name"] == {"write_file": 1, "finish": 1}
