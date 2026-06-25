def build_translation_prompt(segment_text, prev_text=None, next_text=None):
    context_lines = []
    if prev_text:
        context_lines.append(f"Frase anterior (contexto): {prev_text}")
    if next_text:
        context_lines.append(f"Próxima frase (contexto): {next_text}")
    context = "\n".join(context_lines)
    context_block = f"{context}\n\n" if context else ""

    return (
        "Você é um tradutor de legendas para dublagem. Traduza para "
        "PORTUGUÊS (PT-BR) de forma natural e falada, como uma narração — "
        "não literal, mas fiel ao sentido. Mantenha o comprimento parecido "
        "com o original quando possível.\n\n"
        f"{context_block}"
        "Traduza APENAS a frase abaixo. Responda somente com a tradução, "
        "sem aspas, sem comentários, sem o texto original.\n\n"
        f"Frase: {segment_text}"
    )