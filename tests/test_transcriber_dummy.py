from voxtral_wyoming.transcriber.dummy import DummyTranscriber


def test_dummy_transcriber_returns_fixed_text():
    t = DummyTranscriber(text="hello", language="en-US")
    result = t.transcribe(b"1234", sample_rate=16000)
    assert result.text == "hello"
    assert result.language == "en-US"
    assert result.duration_sec is not None
    assert result.duration_sec > 0


def test_dummy_transcriber_empty_audio():
    t = DummyTranscriber(text="hello", language="en-US")
    result = t.transcribe(b"", sample_rate=16000)
    assert result.text == ""
    assert result.language == "en-US"
    assert result.duration_sec == 0.0
