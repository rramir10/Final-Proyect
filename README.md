# 🍷 Wine Quality MLOps Project

Pipeline automatizado de Machine Learning para clasificación de calidad de vino con CI/CD, MLflow tracking y GitHub Actions.

[![ML Pipeline](https://github.com/TU_USUARIO/wine-quality-mlops/actions/workflows/ml.yml/badge.svg)](https://github.com/TU_USUARIO/wine-quality-mlops/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Descripción

Este proyecto implementa un pipeline completo de MLOps que:
- ✅ Entrena un modelo de clasificación binaria para calidad de vino
- ✅ Automatiza el flujo con GitHub Actions (CI/CD)
- ✅ Registra experimentos con MLflow
- ✅ Genera visualizaciones automáticas
- ✅ Incluye tests automatizados

### 🎯 Objetivo

Predecir si un vino es "bueno" (≥7) o "malo" (<7) basándose en 11 características fisicoquímicas.

### 📊 Dataset

**Wine Quality Dataset** de UCI ML Repository
- **Fuente**: [UCI Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Muestras**: ~1,600 vinos tintos
- **Features**: 11 características (alcohol, acidez, pH, etc.)
- **Target**: Calidad (convertida a binario)

---

## 🚀 Quick Start

### Requisitos

- Python 3.9+
- Git
- (Opcional) Make

### Instalación

```bash
# 1. Clonar repositorio
git clone https://github.com/TU_USUARIO/wine-quality-mlops.git
cd wine-quality-mlops

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
make install
# O manualmente: pip install -r requirements.txt

# 4. Descargar dataset
make download-data

# 5. Entrenar modelo
make train

# 6. Ver resultados en MLflow
make mlflow-ui
# Abre http://localhost:5000
```

---

## 💻 Uso

### Comandos Principales

```bash
# Ver todos los comandos
make help

# Setup completo
make setup

# Entrenar modelo
make train

# Ejecutar tests
make test

# Verificar código
make lint

# Pipeline completo
make all
```

### Ejecución Manual

```bash
# Entrenar
python src/train.py

# Tests
pytest tests/ -v

# Linting
flake8 src/ tests/
```

---

## 📁 Estructura del Proyecto

```
wine-quality-mlops/
├── .github/workflows/
│   └── ml.yml              # Pipeline CI/CD
├── src/
│   ├── __init__.py
│   └── train.py            # Script principal
├── tests/
│   ├── __init__.py
│   └── test_basic.py       # Tests unitarios
├── data/
│   ├── raw/                # Datos originales
│   └── processed/          # Datos procesados
├── models/                 # Modelos entrenados
├── mlruns/                 # MLflow tracking
├── plots/                  # Visualizaciones
├── requirements.txt        # Dependencias
├── Makefile                # Comandos automatizados
├── .gitignore
└── README.md
```

---

## 🔄 Pipeline CI/CD

El pipeline se ejecuta automáticamente en cada push:

### Fases

1. **Lint**: Verificación de calidad de código (Flake8, Black)
2. **Test**: Tests unitarios con cobertura
3. **Train**: Entrenamiento del modelo
4. **Summary**: Resumen de resultados

### Ver Resultados

1. Ve a **Actions** en GitHub
2. Click en el workflow más reciente
3. Descarga artefactos generados

---

## 📊 Resultados

### Métricas Esperadas

| Métrica | Valor |
|---------|-------|
| Accuracy | ~0.82 |
| Precision | ~0.78 |
| Recall | ~0.75 |
| F1-Score | ~0.76 |
| ROC-AUC | ~0.88 |

### Visualizaciones

- Matriz de Confusión
- Curva ROC
- Feature Importance

---

## 🧪 Tests

```bash
# Todos los tests
make test

# Solo tests básicos
pytest tests/test_basic.py -v

# Con cobertura
pytest tests/ --cov=src --cov-report=html
```

---

## 📊 MLflow Tracking

### Ver Experimentos

```bash
make mlflow-ui
# Abre http://localhost:5000
```

### Información Registrada

- ✅ Hiperparámetros del modelo
- ✅ Métricas de evaluación
- ✅ Modelo entrenado
- ✅ Visualizaciones
- ✅ Feature importance

---

## 🛠️ Desarrollo

### Agregar Features

1. Edita `src/train.py`
2. Añade tests en `tests/`
3. Ejecuta `make all`
4. Commit y push

### Cambiar Algoritmo

Modifica los hiperparámetros en `src/train.py`:

```python
PARAMS = {
    'n_estimators': 300,  # Cambiar aquí
    'max_depth': 8,       # Ajustar
    # ...
}
```

---

## 📝 Documentación

- [Guía de Instalación](docs/INSTALLATION.md)
- [Guía de Contribución](docs/CONTRIBUTING.md)
- [Changelog](docs/CHANGELOG.md)

---

## 🤝 Contribuir

Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea tu feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📄 Licencia

Distribuido bajo licencia MIT. Ver `LICENSE` para más información.

---

## 👥 Autor

**Tu Nombre**
- GitHub: [@rramir10](https://github.com/rramir10)
- Email: rramire82391@universidadean.edu.co

---

## 🙏 Agradecimientos

- [UCI ML Repository](https://archive.ics.uci.edu/ml/) por el dataset
- [MLflow](https://mlflow.org/) por el tracking
- [GitHub Actions](https://github.com/features/actions) por CI/CD

---

## 📞 Soporte

¿Problemas o preguntas?
- [Abrir un Issue](https://github.com/TU_USUARIO/wine-quality-mlops/issues)
- [Documentación](https://github.com/TU_USUARIO/wine-quality-mlops/wiki)

---
