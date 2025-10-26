"""
Tests Básicos del Proyecto
"""
import pytest
import os
import sys


def test_project_structure():
    """Verificar que existe la estructura básica del proyecto"""
    assert os.path.exists('src/'), "Carpeta src/ no existe"
    assert os.path.exists('tests/'), "Carpeta tests/ no existe"
    assert os.path.exists('requirements.txt'), "requirements.txt no existe"
    assert os.path.exists('src/train.py'), "src/train.py no existe"


def test_python_version():
    """Verificar versión de Python"""
    assert sys.version_info >= (3, 9), "Python 3.9+ es requerido"


def test_imports_core():
    """Test que los imports core funcionan"""
    try:
        import pandas
        import numpy
        import sklearn
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_imports_ml():
    """Test que los imports de ML funcionan"""
    try:
        import xgboost
        import mlflow
        assert True
    except ImportError as e:
        pytest.fail(f"ML import failed: {e}")


def test_directories_can_be_created():
    """Test que se pueden crear directorios necesarios"""
    test_dirs = ["models", "mlruns", "plots", "data/processed"]
    
    for directory in test_dirs:
        os.makedirs(directory, exist_ok=True)
        assert os.path.exists(directory), f"No se pudo crear {directory}"


def test_sklearn_version():
    """Test versión de scikit-learn"""
    import sklearn
    version = tuple(map(int, sklearn.__version__.split('.')[:2]))
    assert version >= (1, 3), "scikit-learn 1.3+ es requerido"


def test_xgboost_version():
    """Test versión de XGBoost"""
    import xgboost
    version = tuple(map(int, xgboost.__version__.split('.')[:2]))
    assert version >= (2, 0), "XGBoost 2.0+ es requerido"


def test_data_loading_simulation():
    """Test simulación de carga de datos"""
    import pandas as pd
    import numpy as np
    
    # Crear datos de prueba
    df = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'target': np.random.randint(0, 2, 100)
    })
    
    assert len(df) == 100
    assert 'target' in df.columns