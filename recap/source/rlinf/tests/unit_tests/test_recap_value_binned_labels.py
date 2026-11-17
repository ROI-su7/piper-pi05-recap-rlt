"""Value-binned RECAP advantage labels (rlinf.data.process.advantage)."""

import numpy as np
import pytest

from rlinf.data.process.advantage import (
    apply_value_binned_label,
    quantile_threshold,
    value_binned_thresholds,
)


def _bin_rates(labels, values, edges):
    assignment = np.searchsorted(edges, values, side="right")
    return np.array([labels[assignment == i].mean() for i in range(len(edges) + 1)])


def test_each_bin_keeps_the_positive_fraction():
    rng = np.random.default_rng(0)
    values = rng.uniform(-1.0, 0.0, 40000)
    # A large stage-dependent offset defeats one global threshold.
    scores = rng.normal(size=values.size) * 0.05 + 2.0 * values
    edges, thresholds = value_binned_thresholds(scores, values, 0.4, 4)
    labels = apply_value_binned_label(scores, values, edges, thresholds)
    assert len(edges) == 3 and len(thresholds) == 4
    np.testing.assert_allclose(_bin_rates(labels, values, edges), 0.4, atol=0.01)
    global_labels = scores >= quantile_threshold(scores, 0.4)
    assert _bin_rates(global_labels, values, edges).min() < 0.01


def test_labels_apply_to_new_frames_with_fitted_bins():
    edges, thresholds = np.array([-0.5]), np.array([-1.0, 1.0])
    labels = apply_value_binned_label(
        np.array([-0.9, -1.1, 1.1, 0.9]),
        np.array([-0.8, -0.8, -0.2, -0.2]),
        edges,
        thresholds,
    )
    assert labels.tolist() == [True, False, True, False]


def test_threshold_is_inclusive():
    labels = apply_value_binned_label(
        np.array([1.0]), np.array([0.0]), np.array([]), np.array([1.0])
    )
    assert labels.tolist() == [True]


def test_constant_values_fall_back_to_one_bin():
    scores = np.arange(10.0)
    edges, thresholds = value_binned_thresholds(scores, np.zeros(10), 0.4, 4)
    assert len(edges) == 1 and len(thresholds) == 2
    labels = apply_value_binned_label(scores, np.zeros(10), edges, thresholds)
    assert labels.sum() == 4


@pytest.mark.parametrize("bins", [0, 1])
def test_rejects_too_few_bins(bins):
    with pytest.raises(ValueError):
        value_binned_thresholds(np.arange(4.0), np.arange(4.0), 0.4, bins)
