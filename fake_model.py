"""A scripted stand-in for the OpenAI Responses API: tests and failure-mode scenarios run the real planner and loop
against it, with no API calls and no cost, and the same result every time."""

import json
from types import SimpleNamespace


def call(call_id, name, **args):
    return SimpleNamespace(type="function_call", call_id=call_id, name=name, arguments=json.dumps(args))


def reply(*calls, text=None):
    """A Responses API reply: output items (reasoning + function calls), the text if any, and token usage."""
    return SimpleNamespace(id=f"resp_{id(calls)}", output=[SimpleNamespace(type="reasoning"), *calls],
                           output_text=text or "", usage=SimpleNamespace(input_tokens=1000, output_tokens=100))


class FakeModel:
    """Plays back scripted turns and keeps every request, so tests can read what the model was sent."""

    def __init__(self, *turns):
        self.turns, self.requests = list(turns), []
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs)))  # a snapshot, not a live reference
        return self.turns.pop(0)
