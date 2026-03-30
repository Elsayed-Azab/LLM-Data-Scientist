"""Tests for the dataset registry."""

from src.data.registry import list_datasets, get_dataset_info, get_all_dataset_info


def test_list_datasets():
    datasets = list_datasets()
    assert "gss" in datasets
    assert "arab_barometer" in datasets
    assert "wvs" in datasets


def test_get_dataset_info():
    info = get_dataset_info("gss")
    assert info.name == "gss"
    assert info.weight_column == "wtssps"
    assert info.format == "stata"


def test_get_dataset_info_arab_barometer():
    info = get_dataset_info("arab_barometer")
    assert info.weight_column == "WT"


def test_get_dataset_info_wvs():
    info = get_dataset_info("wvs")
    assert info.weight_column == "W_WEIGHT"


def test_get_all_dataset_info():
    all_info = get_all_dataset_info()
    assert len(all_info) == 3
    assert "gss" in all_info


def test_unknown_dataset():
    import pytest
    with pytest.raises(KeyError):
        get_dataset_info("nonexistent")
