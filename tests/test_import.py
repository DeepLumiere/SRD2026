"""
Basic import tests for SRD2026
"""

import pytest


def test_import_srd():
    """Test that the main package can be imported"""
    import srd
    assert srd.__version__ == "0.1.0"


def test_import_models():
    """Test that models module can be imported"""
    from srd import models


def test_import_data():
    """Test that data module can be imported"""
    from srd import data


def test_import_utils():
    """Test that utils module can be imported"""
    from srd import utils


if __name__ == "__main__":
    pytest.main([__file__])
