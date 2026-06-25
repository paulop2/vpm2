def pick_reference_window(speech_spans, target=10.0, min_len=6.0):
    if not speech_spans:
        return None
    longest = max(speech_spans, key=lambda s: s[1] - s[0])
    span_start, span_end = longest
    if (span_end - span_start) < min_len:
        return None
    start = min(span_start + 0.25, span_end - min_len)
    start = max(start, span_start)
    end = min(start + target, span_end)
    return (start, end)