#!/usr/bin/env python3
"""
Wine Quality Classification - Training Script
==============================================
Pipeline completo de entrenamiento con MLflow tracking
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Imports básicos
import pandas as pd
import numpy as np
import joblib

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import xgboost as xgb

# MLflow
import mlflow
import mlflow.sklearn

# Visualization
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para servidor
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("🍷 WINE QUALITY CLASSIFICATION - ML PIPELINE")
print("=" * 70)

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

# Crear directorios necesarios
for directory in ['models', 'mlruns', 'plots', 'data/processed', 'logs']:
    os.makedirs(directory, exist_ok=True)

# Configuración MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("wine-quality-classification")

# Hiperparámetros
PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}

# Configuración general
TEST_SIZE = 0.2
RANDOM_STATE = 42
QUALITY_THRESHOLD = 7

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def log_step(step_number, message):
    """Helper para logging consistente"""
    print(f"\n{'='*70}")
    print(f"PASO {step_number}: {message}")
    print(f"{'='*70}")


def save_plot(fig, filename):
    """Guarda una figura de matplotlib"""
    try:
        filepath = f"plots/{filename}"
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"   ✓ Plot guardado: {filepath}")
        return filepath
    except Exception as e:
        print(f"   ⚠ Error guardando plot: {e}")
        return None


# =============================================================================
# PASO 1: CARGAR DATOS
# =============================================================================

log_step(1, "CARGANDO DATOS")

try:
    df = pd.read_csv('data/raw/winequality-red.csv', sep=';')
    print(f"   ✓ Dataset cargado exitosamente")
    print(f"   ✓ Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
    print(f"   ✓ Columnas: {list(df.columns)}")
except FileNotFoundError:
    print("   ✗ ERROR: No se encontró el archivo de datos")
    print("   ✗ Ruta esperada: data/raw/winequality-red.csv")
    sys.exit(1)
except Exception as e:
    print(f"   ✗ ERROR cargando datos: {e}")
    sys.exit(1)

# Verificar datos
print(f"\n   📊 Información del dataset:")
print(f"      - Valores nulos: {df.isnull().sum().sum()}")
print(f"      - Duplicados: {df.duplicated().sum()}")
print(f"      - Rango de calidad: {df['quality'].min()} - {df['quality'].max()}")

# =============================================================================
# PASO 2: PREPROCESAMIENTO
# =============================================================================

log_step(2, "PREPROCESAMIENTO")

# Eliminar duplicados
initial_rows = len(df)
df = df.drop_duplicates()
removed = initial_rows - len(df)
print(f"   ✓ Duplicados eliminados: {removed}")

# Convertir a clasificación binaria
df['quality_binary'] = (df['quality'] >= QUALITY_THRESHOLD).astype(int)

# Mostrar distribución
class_counts = df['quality_binary'].value_counts()
print(f"\n   📊 Distribución de clases:")
print(f"      - Clase 0 (Malo):  {class_counts[0]:4d} ({class_counts[0]/len(df)*100:5.1f}%)")
print(f"      - Clase 1 (Bueno): {class_counts[1]:4d} ({class_counts[1]/len(df)*100:5.1f}%)")

# Separar features y target
X = df.drop(['quality', 'quality_binary'], axis=1)
y = df['quality_binary']

print(f"\n   ✓ Features preparadas: {X.shape[1]} columnas")
print(f"   ✓ Features: {list(X.columns)}")

# Split train/test con estratificación
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE, 
    stratify=y
)

print(f"\n   ✓ Split completado:")
print(f"      - Train: {len(X_train)} muestras ({len(X_train)/len(X)*100:.1f}%)")
print(f"      - Test:  {len(X_test)} muestras ({len(X_test)/len(X)*100:.1f}%)")

# Verificar distribución en splits
train_dist = y_train.value_counts(normalize=True)
test_dist = y_test.value_counts(normalize=True)
print(f"\n   ✓ Distribución balanceada:")
print(f"      - Train: {train_dist[1]:.2%} buenos")
print(f"      - Test:  {test_dist[1]:.2%} buenos")

# Escalamiento
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n   ✓ Features escaladas (StandardScaler)")
print(f"      - Media train: ~{np.mean(X_train_scaled):.3f}")
print(f"      - Std train:   ~{np.std(X_train_scaled):.3f}")

# Guardar scaler
scaler_path = 'models/scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"   ✓ Scaler guardado: {scaler_path}")

# =============================================================================
# PASO 3: ENTRENAMIENTO
# =============================================================================

log_step(3, "ENTRENAMIENTO DEL MODELO")

print(f"   🤖 Algoritmo: XGBoost Classifier")
print(f"   📋 Hiperparámetros:")
for key, value in PARAMS.items():
    print(f"      - {key}: {value}")

# Crear modelo
model = xgb.XGBClassifier(**PARAMS)

# Entrenar con early stopping
print(f"\n   🎓 Entrenando modelo...")
model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=False
)

print(f"   ✓ Entrenamiento completado")
print(f"   ✓ Número de árboles: {model.n_estimators}")

# =============================================================================
# PASO 4: EVALUACIÓN
# =============================================================================

log_step(4, "EVALUACIÓN DEL MODELO")

# Predicciones
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Calcular métricas
metrics = {
    'train_accuracy': accuracy_score(y_train, y_pred_train),
    'test_accuracy': accuracy_score(y_test, y_pred_test),
    'precision': precision_score(y_test, y_pred_test, zero_division=0),
    'recall': recall_score(y_test, y_pred_test, zero_division=0),
    'f1_score': f1_score(y_test, y_pred_test, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba)
}

print(f"\n   📊 Métricas de rendimiento:")
print(f"      - Train Accuracy: {metrics['train_accuracy']:.4f}")
print(f"      - Test Accuracy:  {metrics['test_accuracy']:.4f}")
print(f"      - Precision:      {metrics['precision']:.4f}")
print(f"      - Recall:         {metrics['recall']:.4f}")
print(f"      - F1-Score:       {metrics['f1_score']:.4f}")
print(f"      - ROC-AUC:        {metrics['roc_auc']:.4f}")

# Calcular overfitting
overfitting = metrics['train_accuracy'] - metrics['test_accuracy']
print(f"\n   📉 Overfitting: {overfitting:.4f}")
if overfitting > 0.05:
    print(f"      ⚠ Advertencia: Posible overfitting detectado")
else:
    print(f"      ✓ Generalización adecuada")

# =============================================================================
# PASO 5: GUARDAR MODELO
# =============================================================================

log_step(5, "GUARDANDO MODELO")

# Guardar modelo principal
model_path = 'models/best_model.pkl'
joblib.dump(model, model_path)

# Verificar que se guardó
if os.path.exists(model_path):
    size_kb = os.path.getsize(model_path) / 1024
    print(f"   ✓ Modelo guardado: {model_path}")
    print(f"   ✓ Tamaño: {size_kb:.2f} KB")
else:
    print(f"   ✗ ERROR: El modelo no se guardó correctamente")
    sys.exit(1)

# Guardar feature importance
if hasattr(model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv('models/feature_importance.csv', index=False)
    print(f"   ✓ Feature importance guardada")
    
    print(f"\n   🎯 Top 5 features más importantes:")
    for idx, row in importance_df.head().iterrows():
        print(f"      {row['feature']:20s}: {row['importance']:.4f}")

# =============================================================================
# PASO 6: VISUALIZACIONES
# =============================================================================

log_step(6, "CREANDO VISUALIZACIONES")

try:
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_xticklabels(['Malo (0)', 'Bueno (1)'])
    ax.set_yticklabels(['Malo (0)', 'Bueno (1)'])
    save_plot(fig, 'confusion_matrix.png')
    
    # 2. Feature Importance (si existe)
    if hasattr(model, 'feature_importances_'):
        fig, ax = plt.subplots(figsize=(10, 6))
        top_10 = importance_df.head(10)
        ax.barh(range(len(top_10)), top_10['importance'], color='steelblue')
        ax.set_yticks(range(len(top_10)))
        ax.set_yticklabels(top_10['feature'])
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title('Top 10 Feature Importances', fontsize=16, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        save_plot(fig, 'feature_importance.png')
    
    # 3. ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {metrics["roc_auc"]:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=16, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    save_plot(fig, 'roc_curve.png')
    
    print(f"   ✓ Visualizaciones completadas")
    
except Exception as e:
    print(f"   ⚠ Error en visualizaciones: {e}")

# =============================================================================
# PASO 7: MLFLOW TRACKING
# =============================================================================

log_step(7, "REGISTRANDO EN MLFLOW")

try:
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
        print(f"   🆔 Run ID: {run_id}")
        
        # Registrar parámetros
        mlflow.log_params(PARAMS)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("quality_threshold", QUALITY_THRESHOLD)
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("n_features", X.shape[1])
        print(f"   ✓ Parámetros registrados")
        
        # Registrar métricas
        mlflow.log_metrics(metrics)
        mlflow.log_metric("overfitting", overfitting)
        print(f"   ✓ Métricas registradas")
        
        # Registrar modelo
        mlflow.sklearn.log_model(model, "model")
        print(f"   ✓ Modelo registrado")
        
        # Registrar artifacts
        if os.path.exists('models/feature_importance.csv'):
            mlflow.log_artifact('models/feature_importance.csv')
        
        for plot_file in os.listdir('plots'):
            if plot_file.endswith('.png'):
                mlflow.log_artifact(f'plots/{plot_file}')
        
        print(f"   ✓ Artifacts registrados")
        
except Exception as e:
    print(f"   ⚠ Advertencia MLflow: {e}")
    print(f"   ⚠ Continuando sin tracking...")

# =============================================================================
# RESUMEN FINAL
# =============================================================================

print("\n" + "=" * 70)
print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
print("=" * 70)
print(f"\n📊 Resumen de Resultados:")
print(f"   • Algoritmo:     XGBoost Classifier")
print(f"   • Muestras:      {len(df)} (Train: {len(X_train)}, Test: {len(X_test)})")
print(f"   • Features:      {X.shape[1]}")
print(f"   • Accuracy:      {metrics['test_accuracy']:.4f}")
print(f"   • F1-Score:      {metrics['f1_score']:.4f}")
print(f"   • ROC-AUC:       {metrics['roc_auc']:.4f}")

print(f"\n📁 Archivos Generados:")
print(f"   • Modelo:        models/best_model.pkl")
print(f"   • Scaler:        models/scaler.pkl")
print(f"   • Importance:    models/feature_importance.csv")
print(f"   • Plots:         plots/*.png")

print(f"\n🎯 Próximos Pasos:")
print(f"   1. Revisar métricas en MLflow UI: mlflow ui")
print(f"   2. Verificar visualizaciones en: plots/")
print(f"   3. Probar predicciones con el modelo guardado")

print("\n" + "=" * 70)

sys.exit(0)