"""Tests for evaluation metrics."""

import pytest

from src.evaluation.metrics import (
    numeric_accuracy,
    categorical_accuracy,
    directional_accuracy,
    check_weight_usage,
)


class TestNumericAccuracy:
    def test_exact_match(self):
        assert numeric_accuracy("The answer is 42.5", 42.5) == 1.0

    def test_within_tolerance(self):
        # 44.0 is within 5% of 42.5
        score = numeric_accuracy("The value is 44.0", 42.5)
        assert score > 0.5

    def test_no_number(self):
        assert numeric_accuracy("No numbers here", 42.5) == 0.0

    def test_way_off(self):
        assert numeric_accuracy("The answer is 100", 42.5) == 0.0

    def test_multiple_numbers_picks_best(self):
        score = numeric_accuracy("Between 40 and 42.5 we found", 42.5)
        assert score == 1.0


class TestCategoricalAccuracy:
    def test_match(self):
        assert categorical_accuracy("The top country is Egypt", "Egypt") == 1.0

    def test_case_insensitive(self):
        assert categorical_accuracy("married is most common", "Married") == 1.0

    def test_no_match(self):
        assert categorical_accuracy("The top country is Iraq", "Egypt") == 0.0


class TestDirectionalAccuracy:
    def test_positive(self):
        assert directional_accuracy("There is a positive correlation", "positive") == 1.0

    def test_negative(self):
        assert directional_accuracy("Income decreases with age", "negative") == 1.0

    def test_none(self):
        assert directional_accuracy("No significant relationship found", "none") == 1.0

    def test_wrong_direction(self):
        assert directional_accuracy("There is a positive correlation", "negative") == 0.0


class TestWeightUsage:
    def test_weight_found(self):
        code = "np.average(df['age'], weights=df['WTSSPS'])"
        assert check_weight_usage(code, "WTSSPS") is True

    def test_weight_missing(self):
        assert check_weight_usage("df['age'].mean()", "WTSSPS") is False

    def test_empty_code(self):
        assert check_weight_usage("", "WT") is False

    def test_case_insensitive(self):
        assert check_weight_usage("weights=df['wtssps']", "WTSSPS") is True
