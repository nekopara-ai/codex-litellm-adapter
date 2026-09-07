import json
import re

from codex_litellm_adapter import context_compat as compat


def test_context_retry_requires_model_opt_in(monkeypatch):
    body = json.dumps({"model": "example-model", "input": "hello"}).encode()
    error = json.dumps(
        {
            "error": {
                "message": "requested 8192 output tokens but prompt contains 125000 input tokens"
            }
        }
    ).encode()
    assert compat.context_window_retry_body(body, 400, error) is None
    monkeypatch.setattr(compat, "MODEL_PATTERN", re.compile(r"^example-model$"))
    retry, budget = compat.context_window_retry_body(body, 400, error)
    assert 0 < budget < 8192
    assert json.loads(retry)["max_output_tokens"] == budget
    assert compat.context_window_retry_body(body, 503, error) is None


def test_context_retry_rejects_unrelated_errors(monkeypatch):
    monkeypatch.setattr(compat, "MODEL_PATTERN", re.compile(r"^example-model$"))
    body = b'{"model":"example-model"}'
    assert compat.context_window_retry_body(body, 400, b'{"error":"invalid tools"}') is None
    assert compat.context_window_retry_body(body, 400, b"not json") is None
