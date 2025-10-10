from voxtral_wyoming.audio import AudioSpec, expected_bytes_per_second, clamp_audio_size


def test_expected_bytes_per_second():
    spec = AudioSpec(sample_rate=16000, channels=1, sample_width_bytes=2)
    assert expected_bytes_per_second(spec) == 16000 * 1 * 2


def test_clamp_audio_size():
    spec = AudioSpec(sample_rate=16000, channels=1, sample_width_bytes=2)
    one_second = b"0" * expected_bytes_per_second(spec)
    three_seconds = one_second * 3

    # Clamp to 2 seconds
    clamped = clamp_audio_size(three_seconds, spec, max_seconds=2)
    assert len(clamped) == 2 * expected_bytes_per_second(spec)

    # No clamp when under limit
    clamped2 = clamp_audio_size(one_second, spec, max_seconds=2)
    assert len(clamped2) == len(one_second)
