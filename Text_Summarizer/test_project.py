import pytest
import sys
import os

# Import your functions directly
from project import summarize_text, get_input_file

def test_summarize_text_returns_summary(monkeypatch):
    # Monkeypatch the API token
    monkeypatch.setenv("HF_TOKEN", "fake_token")
    # Monkeypatch requests.post to fake the API response
    import requests
    class MockResponse:
        def raise_for_status(self): pass
        def json(self): return [{"summary_text": "Test Summary"}]
    monkeypatch.setattr(requests, "post", lambda *a, **k: MockResponse())

    # Call your function and check output
    assert summarize_text("Some input text.", "briefly") == "Test Summary"

def test_summarize_text_no_token(monkeypatch):
    # Remove HF_TOKEN if set
    monkeypatch.delenv("HF_TOKEN", raising=False)
    # Check that sys.exit is called
    with pytest.raises(SystemExit):
        summarize_text("Some input text.", "briefly")

def test_get_input_file(monkeypatch, tmp_path):
    # Create a fake input file
    test_file = tmp_path / "input.txt"
    test_file.write_text("content here")

    # Monkeypatch input to return the file path
    monkeypatch.setattr("builtins.input", lambda _: str(test_file))
    # Should return the filename (no sys.exit)
    assert get_input_file() == str(test_file)

def test_get_input_file_default(monkeypatch, tmp_path):
    # Create default file "input.txt"
    test_file = tmp_path / "input.txt"
    test_file.write_text("hello")
    # Change working dir to tmp_path
    monkeypatch.chdir(tmp_path)
    # Monkeypatch input to simulate user pressing Enter
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert get_input_file() == "input.txt"

def test_get_input_file_not_found(monkeypatch):
    # Monkeypatch input to return a file that doesn't exist
    monkeypatch.setattr("builtins.input", lambda _: "doesnotexist.txt")
    with pytest.raises(SystemExit):
        get_input_file()
