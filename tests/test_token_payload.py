import pytest

from token_kv_agent_system.models import (
    TokenPayload,
    bits_to_bytes,
    theoretical_bits_per_token,
)


@pytest.mark.parametrize(
    ("vocab_size", "expected_bits"),
    [(1, 0), (2, 1), (3, 2), (256, 8), (257, 9), (65_537, 17)],
)
def test_theoretical_bits_per_token_is_ceil_log2(vocab_size, expected_bits):
    assert theoretical_bits_per_token(vocab_size) == expected_bits


def test_token_payload_rounds_total_bits_only_after_aggregation():
    payload = TokenPayload.from_token_ids([1, 2, 3], vocab_size=5)

    assert payload.token_count == 3
    assert payload.theoretical_bits_per_token == 3
    assert payload.theoretical_payload_bits == 9
    assert payload.theoretical_payload_bytes == 2
    assert payload.estimated_size_bytes == 2
    assert payload.estimated_size_bytes != payload.token_count * 1


def test_mock_style_payload_remains_backward_compatible():
    payload = TokenPayload((1, 2, 3), 12)

    assert payload.token_count == 3
    assert payload.estimated_size_bytes == 12
    assert payload.theoretical_payload_bits is None
    assert payload.theoretical_payload_bytes is None


def test_invalid_payload_sizes_are_rejected():
    with pytest.raises(ValueError):
        theoretical_bits_per_token(0)
    with pytest.raises(ValueError):
        bits_to_bytes(-1)
    with pytest.raises(ValueError, match="must equal"):
        TokenPayload((1,), 1, theoretical_bits_per_token=3, theoretical_payload_bits=2)
