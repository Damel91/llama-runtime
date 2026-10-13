from runtime_llama.errors import sanitize_diagnostic


def test_diagnostic_redacts_user_home_and_bounds_message():
    diagnostic = sanitize_diagnostic(
        'Assertion failed in "/Users/alice/project/src/runtime.cpp" ' + "x" * 800,
        limit=120,
    )
    assert "/Users/alice" not in diagnostic
    assert "<abs_path_redacted>" in diagnostic
    assert len(diagnostic) == 120
