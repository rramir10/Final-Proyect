# =============================================================================
# Makefile - Wine Quality MLOps Project
# =============================================================================

.PHONY: help install setup download-data train test lint format clean all

# Variables
PYTHON := python3
PIP := pip3

# =============================================================================
# HELP
# =============================================================================

help:
	@echo "=========================================="
	@echo "Wine Quality MLOps - Comandos Disponibles"
	@echo "=========================================="
	@echo ""
	@echo "📦 SETUP:"
	@echo "  make install       - Instalar dependencias"
	@echo "  make setup         - Setup completo del proyecto"
	@echo "  make download-data - Descargar dataset"
	@echo ""
	@echo "🤖 DESARROLLO:"
	@echo "  make train         - Entrenar modelo"
	@echo "  make test          - Ejecutar tests"
	@echo "  make lint          - Verificar código"
	@echo "  make format        - Formatear código"
	@echo ""
	@echo "🚀 PIPELINE:"
	@echo "  make all           - Ejecutar pipeline completo"
	@echo ""
	@echo "🧹 LIMPIEZA:"
	@echo "  make clean         - Limpiar archivos generados"
	@echo "  make clean-all     - Limpieza completa"
	@echo ""
	@echo "📊 UTILIDADES:"
	@echo "  make mlflow-ui     - Abrir MLflow UI"
	@echo "  make info          - Información del proyecto"
	@echo ""

# =============================================================================
# INSTALACIÓN
# =============================================================================

install:
	@echo "📦 Instalando dependencias..."
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencias instaladas"

setup: install
	@echo "⚙️  Configurando proyecto..."
	mkdir -p data/raw data/processed models mlruns plots logs
	@echo "✅ Directorios creados"
	@$(MAKE) download-data

download-data:
	@echo "📥 Descargando dataset..."
	@if [ ! -f "data/raw/winequality-red.csv" ]; then \
		curl -o data/raw/winequality-red.csv \
			"https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"; \
		echo "✅ Dataset descargado"; \
	else \
		echo "⚠️  Dataset ya existe"; \
	fi

# =============================================================================
# ENTRENAMIENTO
# =============================================================================

train:
	@echo "=========================================="
	@echo "🤖 ENTRENANDO MODELO"
	@echo "=========================================="
	$(PYTHON) src/train.py
	@echo ""
	@echo "✅ Entrenamiento completado"

# =============================================================================
# TESTING
# =============================================================================

test:
	@echo "🧪 Ejecutando tests..."
	pytest tests/ -v --cov=src --cov-report=term-missing
	@echo "✅ Tests completados"

test-quick:
	@echo "🧪 Tests rápidos..."
	pytest tests/ -v
	@echo "✅ Tests completados"

# =============================================================================
# CODE QUALITY
# =============================================================================

lint:
	@echo "🔍 Verificando código..."
	flake8 src/ tests/ --max-line-length=100 --exclude=__pycache__
	@echo "✅ Linting completado"

format:
	@echo "🎨 Formateando código..."
	black src/ tests/ --line-length=100
	@echo "✅ Código formateado"

# =============================================================================
# PIPELINE COMPLETO
# =============================================================================

all: lint test train
	@echo ""
	@echo "=========================================="
	@echo "✅ PIPELINE COMPLETO EJECUTADO"
	@echo "=========================================="

# =============================================================================
# MLFLOW
# =============================================================================

mlflow-ui:
	@echo "📊 Iniciando MLflow UI..."
	@echo "🌐 Abre tu navegador en: http://localhost:5000"
	mlflow ui

# =============================================================================
# LIMPIEZA
# =============================================================================

clean:
	@echo "🧹 Limpiando archivos generados..."
	rm -rf models/*.pkl
	rm -rf plots/*.png
	rm -rf data/processed/*
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	rm -rf .pytest_cache
	rm -rf .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Limpieza completada"

clean-all: clean
	@echo "🧹 Limpieza profunda..."
	rm -rf mlruns/
	rm -rf data/raw/*.csv
	@echo "✅ Limpieza completa"

# =============================================================================
# INFORMACIÓN
# =============================================================================

info:
	@echo "=========================================="
	@echo "📊 INFORMACIÓN DEL PROYECTO"
	@echo "=========================================="
	@echo ""
	@echo "📁 Estructura:"
	@ls -R | head -30
	@echo ""
	@echo "📦 Dependencias instaladas:"
	@$(PIP) list | grep -E "(pandas|numpy|scikit|xgboost|mlflow)"
	@echo ""
	@echo "📊 Estado:"
	@test -f "data/raw/winequality-red.csv" && echo "  ✅ Dataset descargado" || echo "  ❌ Dataset NO descargado"
	@test -f "models/best_model.pkl" && echo "  ✅ Modelo entrenado" || echo "  ⚠️  Modelo NO entrenado"
	@test -d "mlruns" && echo "  ✅ MLflow configurado" || echo "  ⚠️  MLflow NO configurado"
	@echo ""