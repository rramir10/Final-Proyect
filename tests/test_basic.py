"""
Tests básicos del proyecto
"""
import pytest
import os


def test_project_structure():
    """Test que existe la estructura básica"""
    assert os.path.exists('src/')
    assert os.path.exists('tests/')
    assert os.path.exists('requirements.txt')


def test_imports():
    """Test que los imports básicos funcionan"""
    try:
        import pandas
        import numpy
        import sklearn
        import xgboost
        import mlflow
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_directories_created():
    """Test que se pueden crear directorios"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("mlruns", exist_ok=True)
    assert os.path.exists("models")
    assert os.path.exists("mlruns")