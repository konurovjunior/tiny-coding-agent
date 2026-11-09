"""Traces for the agent in OpenTelemetry's GenAI shape, sent to Langfuse, with eval scores on them (lesson 14/24).

One trace per run (or per job attempt):

    invoke_agent  repair leap-year            type agent: task, job, attempt, prompt version, outcome; eval scores
      chat  gpt-6-luna                         type generation: model, tokens, cost, what was sent, what came back
      execute_tool  run_tests                  type tool: arguments, result; a refused or failed action is WARNING
      chat  gpt-6-luna
      execute_tool  read_file
      ...

Attribute names follow the OTel GenAI semantic conventions (gen_ai.operation.name, gen_ai.request.model,
gen_ai.usage.input_tokens, ...), so any OTel backend can read them, plus Langfuse's own langfuse.* names where
Langfuse needs them (observation type, input/output, level, usage and cost details, session, tags, version).

Why each piece (the lesson's three warnings):
  - tracing without evaluation is expensive logging: the eval verdict (resolved, overfit, steps over gold) is posted as
    Langfuse scores on the run's trace, so "show me the overfit runs" is a filter, and one click opens the steps.
  - prompt versions not tied to traces: every span carries langfuse.version = a hash of the system prompt, the rules
    and the tool schemas; when results regress, filter by version to find the change.
  - opaque background work: queued jobs are traced too, one trace per attempt, with the job id and attempt number.

How it attaches without touching the graded loop (main.drive): the model client and the planner are wrapped.
  TracedClient    wraps client.responses.create: one generation span per model call (a timeout becomes an ERROR span)
  TracedPlanner   a step's span opens when the planner hands drive() an action, and closes when the planner is called
                  again and sees that action's result in the trace (or when the run ends)

Modes (setup()):  off (default: a no-op tracer, nothing leaves the process)
                  langfuse (OTLP/HTTP to LANGFUSE_HOST with the project keys; the same three values work for Langfuse
                  Cloud as for the self-hosted one in observability/)
                  memory (tests: spans and scores kept in lists)
"""

import base64
import hashlib
import json
import os
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

from llm_planner import MEMORY_FIRST_RULE, MEMORY_RULE, PRICES, SYSTEM, UNTRUSTED_RULE, openai_tools

HERE = Path(__file__).resolve().parent
LANGFUSE_ENV = HERE / "observability" / ".env"   # written by observability/setup.sh
TEXT_LIMIT = 4000                                 # characters of any input/output kept on a span
TOOL_NAMES = {"read": "read_file", "patch": "apply_patch", "test": "run_tests"}

_tracer = trace.NoOpTracer()
_mode = "off"
_provider: TracerProvider | None = None
_langfuse: dict | None = None     # host + keys when mode is langfuse
memory_spans: InMemorySpanExporter | None = None
memory_scores: list[dict] = []


def prompt_version() -> str:
    """What the model is told and offered, hashed: a change to any prompt rule or tool schema is a new version."""
    text = SYSTEM + MEMORY_RULE + MEMORY_FIRST_RULE + UNTRUSTED_RULE + json.dumps(openai_tools(True), sort_keys=True)
    return "prompt-" + hashlib.sha256(text.encode()).hexdigest()[:10]


def _langfuse_config() -> dict:
    """LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY from the environment (e.g. Langfuse Cloud), else the
    self-hosted project's keys from observability/.env."""
    values = {}
    if LANGFUSE_ENV.is_file():
        for line in LANGFUSE_ENV.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
    host = os.environ.get("LANGFUSE_HOST") or values.get("LANGFUSE_HOST", "http://localhost:3000")
    public = os.environ.get("LANGFUSE_PUBLIC_KEY") or values.get("LANGFUSE_INIT_PROJECT_PUBLIC_KEY")
    secret = os.environ.get("LANGFUSE_SECRET_KEY") or values.get("LANGFUSE_INIT_PROJECT_SECRET_KEY")
    if not (public and secret):
        raise SystemExit("no Langfuse keys: run observability/setup.sh, or set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY")
    auth = base64.b64encode(f"{public}:{secret}".encode()).decode()
    return {"host": host.rstrip("/"), "auth": f"Basic {auth}"}


def setup(mode: str = "off") -> None:
    global _tracer, _mode, _provider, _langfuse, memory_spans
    _mode = mode
    if mode == "off":
        _tracer = trace.NoOpTracer()
        return
    _provider = TracerProvider(resource=Resource.create({"service.name": "tiny-coding-agent"}))
    if mode == "memory":
        memory_spans = InMemorySpanExporter()
        memory_scores.clear()
        _provider.add_span_processor(SimpleSpanProcessor(memory_spans))
    elif mode == "langfuse":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        _langfuse = _langfuse_config()
        exporter = OTLPSpanExporter(endpoint=f"{_langfuse['host']}/api/public/otel/v1/traces",
                                    headers={"Authorization": _langfuse["auth"],
                                             "x-langfuse-ingestion-version": "4"})  # without it: up to 10 min delay
        _provider.add_span_processor(BatchSpanProcessor(exporter))  # exports in the background, off the agent's path
    else:
        raise ValueError(f"unknown tracing mode {mode!r}: off, langfuse or memory")
    _tracer = _provider.get_tracer("tiny-coding-agent")


def flush() -> None:
    if _provider is not None:
        _provider.force_flush()


def enabled() -> bool:
    return _mode != "off"


def _text(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= TEXT_LIMIT else text[:TEXT_LIMIT] + f"... [{len(text)} chars]"


# ----------------------------------------------------------------------------------------------------- the run span

class Run:
    """The root span of one run and what hangs off it: the session/tags every child copies, and the score sink."""

    def __init__(self, span, common: dict):
        self.span, self.common = span, common
        self.planners: list[TracedPlanner] = []
        self.trace_id = format(span.get_span_context().trace_id, "032x")  # Langfuse uses the OTel trace id as is

    def instrument(self, planner):
        """Wrap a model planner for this run: its OpenAI client (generation spans) and itself (tool spans)."""
        planner.client = TracedClient(planner.client, self.common)
        traced = TracedPlanner(planner, self.common)
        self.planners.append(traced)
        return traced

    def close_steps(self) -> None:
        for p in self.planners:
            p.close()

    def finish(self, outcome: dict) -> None:
        self.close_steps()
        self.span.set_attribute("langfuse.observation.output", _text(outcome))
        if outcome.get("error"):
            self.span.set_attribute("langfuse.observation.level", "ERROR")
            self.span.set_attribute("langfuse.observation.status_message", _text(outcome["error"]))

    def score(self, name: str, value, data_type: str, comment: str = "") -> None:
        """A Langfuse score on this run's trace (POST /api/public/scores). Scores may arrive before the trace."""
        body = {"traceId": self.trace_id, "name": name, "value": value, "dataType": data_type}
        if comment:
            body["comment"] = comment[:500]
        if _mode == "memory":
            memory_scores.append(body)
        elif _mode == "langfuse":
            request = urllib.request.Request(f"{_langfuse['host']}/api/public/scores", data=json.dumps(body).encode(),
                                             headers={"Authorization": _langfuse["auth"],
                                                      "Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=10) as response:
                response.read()

    def score_eval(self, graded: dict, steps: int, gold: int) -> None:
        self.score("resolved", int(graded["resolved"]), "BOOLEAN")
        self.score("overfit", int(graded["overfit"]), "BOOLEAN",
                   "visible tests pass, hidden tests fail" if graded["overfit"] else "")
        self.score("steps_over_gold", round(steps / gold, 2), "NUMERIC")


@contextmanager
def agent_run(name: str, session: str, task: str, issue: str | None, tags: list[str] | None = None,
              metadata: dict | None = None):
    """with agent_run(...) as run: planner = run.instrument(planner), run the loop, then run.finish(outcome) and
    run.score_eval(...)."""
    common = {"langfuse.session.id": session, "langfuse.trace.tags": tags or [], "langfuse.version": prompt_version()}
    for key, value in (metadata or {}).items():
        common[f"langfuse.trace.metadata.{key}"] = str(value)
    with _tracer.start_as_current_span(f"invoke_agent {name}") as span:
        span.set_attributes({**common, "langfuse.trace.name": name, "langfuse.observation.type": "agent",
                             "gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "tiny-coding-agent",
                             "langfuse.trace.metadata.task": task,
                             "langfuse.observation.input": _text({"task": task, "issue": issue})})
        run = Run(span, common)
        try:
            yield run
        except BaseException as e:  # the run's own span shows why it ended; OTel records the exception itself
            run.close_steps()
            span.set_attributes({"langfuse.observation.level": "ERROR",
                                 "langfuse.observation.status_message": f"{type(e).__name__}: {e}"[:500]})
            raise


# ------------------------------------------------------------------------------------------ model calls and steps

class TracedClient:
    """Stands in for the OpenAI client: same .responses.create, one generation span around each call."""

    def __init__(self, inner, common: dict):
        self.inner, self.common = inner, common
        self.responses = self  # client.responses.create(...) → self.create(...)

    def create(self, **kwargs):
        model = kwargs.get("model", "")
        with _tracer.start_as_current_span(f"chat {model}", record_exception=True) as span:
            span.set_attributes({**self.common, "langfuse.observation.type": "generation",
                                 "gen_ai.operation.name": "chat", "gen_ai.provider.name": "openai",
                                 "gen_ai.request.model": model,
                                 "langfuse.observation.input": _text(kwargs.get("input"))})
            try:
                response = self.inner.responses.create(**kwargs)
            except Exception as e:  # a timeout or an API error: visible as a red step instead of a missing one
                span.set_attributes({"langfuse.observation.level": "ERROR",
                                     "langfuse.observation.status_message": f"{type(e).__name__}: {e}"})
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            used_in, used_out = response.usage.input_tokens, response.usage.output_tokens
            price_in, price_out = PRICES.get(model, (0.0, 0.0))
            cost_in, cost_out = used_in * price_in / 1e6, used_out * price_out / 1e6
            calls = [{"name": i.name, "arguments": i.arguments} for i in response.output if i.type == "function_call"]
            span.set_attributes({
                "gen_ai.usage.input_tokens": used_in, "gen_ai.usage.output_tokens": used_out,
                "langfuse.observation.usage_details": json.dumps({"input": used_in, "output": used_out,
                                                                   "total": used_in + used_out}),
                # our own prices: Langfuse's model table doesn't know gpt-6-luna, so the cost is sent, not inferred
                "langfuse.observation.cost_details": json.dumps({"input": cost_in, "output": cost_out,
                                                                  "total": cost_in + cost_out}),
                "langfuse.observation.output": _text(calls or response.output_text),
            })
            return response


class TracedPlanner:
    """Wraps a planner: one tool span per action, from the moment the planner hands it over until its result is seen.
    Everything else (state(), restore(), model_calls, cost_usd...) is the inner planner's, so durable runs and
    the eval suite use it unchanged."""

    def __init__(self, inner, common: dict):
        self._inner, self._common = inner, common
        self._open = None      # (span, index in the trace the action's result will get)
        self._trace = None     # the trace list drive() passes in (the same list object every call)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __call__(self, trace_):
        self._trace = trace_
        self._close_step()
        action = self._inner(trace_)
        if action is not None:
            tool = TOOL_NAMES.get(action.get("tool"), str(action.get("tool")))
            span = _tracer.start_span(f"execute_tool {tool}")  # a child of the run span (the current context)
            args = {k: v for k, v in action.items() if k != "tool"}
            span.set_attributes({**self._common, "langfuse.observation.type": "tool",
                                 "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": tool,
                                 "langfuse.observation.input": _text(args)})
            self._open = (span, len(trace_))
        return action

    def close(self) -> None:
        """At the end of a run: the last action's result is in the trace, but no planner call came after it."""
        self._close_step()

    def _close_step(self) -> None:
        if self._open is None:
            return
        span, index = self._open
        self._open = None
        result = self._trace[index]["result"] if self._trace is not None and len(self._trace) > index else None
        if result is None:  # the run stopped before the action ran (budget, or an approval gate)
            span.set_attribute("langfuse.observation.status_message", "not run")
        else:
            span.set_attribute("langfuse.observation.output", _text(result))
            if "error" in result:
                span.set_attributes({"langfuse.observation.level": "WARNING",
                                     "langfuse.observation.status_message": _text(result["error"])})
                if result.get("guard"):
                    span.add_event("guard_refused", {"reason": _text(result["error"])})
            elif result.get("passed") is False:
                span.set_attribute("langfuse.observation.metadata.tests_failed", "true")
        span.end()
