import numpy as np
import pytest
from scipy.ndimage import binary_dilation, binary_erosion


def clean_mask(mask):
    return binary_dilation(
        binary_erosion(mask, structure=[1, 1, 1, 1]),
        structure=[1, 1, 1, 1],
    )


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        ([0, 1, 0, 0, 0], [0, 0, 0, 0, 0]),
        ([0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0]),
        ([0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 0]),
        ([0, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 0]),
    ],
)
def test_opening_keeps_only_true_runs_at_least_four_samples(mask, expected):
    result = clean_mask(np.array(mask, dtype=bool))

    np.testing.assert_array_equal(result, np.array(expected, dtype=bool))


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        ([1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0]),
        ([0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]),
        ([1, 1, 1, 0, 0], [0, 0, 0, 0, 0]),
        ([0, 0, 1, 1, 1], [0, 0, 0, 0, 0]),
    ],
)
def test_opening_applies_same_run_length_rule_at_edges(mask, expected):
    result = clean_mask(np.array(mask, dtype=bool))

    np.testing.assert_array_equal(result, np.array(expected, dtype=bool))


def test_opening_does_not_bridge_false_gaps_between_valid_runs():
    mask = np.array([0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0], dtype=bool)

    result = clean_mask(mask)

    np.testing.assert_array_equal(result, mask)
