from vpm2.translate_prompt import build_translation_prompt


def test_prompt_contains_segment_and_context():
    p = build_translation_prompt("Let's paint a tree.",
                                 prev_text="Hello there.",
                                 next_text="A happy little tree.")
    assert "Let's paint a tree." in p
    assert "Hello there." in p
    assert "A happy little tree." in p


def test_prompt_handles_missing_neighbors():
    p = build_translation_prompt("Hi.", prev_text=None, next_text=None)
    assert "Hi." in p
    # must not crash and must still ask for PT-BR
    assert "português" in p.lower() or "pt-br" in p.lower()


def test_prompt_requests_only_translation():
    p = build_translation_prompt("Hi.", None, None)
    assert "apenas" in p.lower() or "only" in p.lower()