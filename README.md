# Tumor Type Prediction

Классификация типа опухоли по матрице экспрессии генов (RNA-seq / gene expression) с использованием:
- classical ML (LogReg, LinearSVC, ExtraTrees, HistGradientBoosting)
- SMOTE внутри CV-пайплайна
- 1D-CNN (PyTorch) для табличных векторов экспрессии

## Данные

В проекте используются файлы:
- `data.csv` — матрица экспрессии (samples x genes)
- `labels.csv` — тип опухоли (`Class`) для каждого sample

> `data.csv` большой и хранится через Git LFS.

## Что реализовано

### 1) Формирование признаков

В проект добавлен модуль [`feature_utils.py`](feature_utils.py) с тремя типами признаков:

1. `build_morgan_fingerprints(...)`
- для молекул (SMILES)
- Morgan fingerprints (RDKit)

2. `build_kmer_features(...)`
- для ДНК/белков
- k-mer частоты (нормированные или абсолютные)

3. `build_expression_features(...)`
- для gene-expression
- численные векторы признаков из табличных данных

### 2) Метрики и финальная оценка моделей

В ноутбуке считается единый набор метрик для всех моделей (включая CNN):
- **Top-1 Accuracy**
- **Precision (macro)**
- **Recall (macro)**
- **F1 (macro)**
- **Balanced Accuracy**
- MCC (дополнительно)

### 3) Визуализации

В ноутбуке строятся:
- Confusion matrix (counts)
- Confusion matrix, нормированная по строкам
- График per-class precision/recall/f1
- Топ ошибок классов (`true -> pred`)
- Анализ confidence/reliability
- График важности признаков (для моделей с `coef_` или `feature_importances_`)

### 4) Таблица сравнения моделей

Формируется сводная таблица `metrics_df` и сохраняется в:
- `artifacts/model_metrics_comparison.csv`

## Структура проекта

- `tumor_type_prediction_smote_pipeline.ipynb` — основной pipeline и анализ
- `feature_utils.py` — feature engineering + feature importance helpers
- `data.csv` — матрица экспрессии
- `labels.csv` — метки классов
- `artifacts/` — сохранённые модели и метрики

## Установка

```bash
pip install -U pandas numpy scikit-learn imbalanced-learn matplotlib joblib torch
```

Опционально для molecular fingerprints:

```bash
pip install rdkit
```

## Запуск

1. Откройте `tumor_type_prediction_smote_pipeline.ipynb`.
2. Выполните ячейки сверху вниз.
3. В блоке финальной оценки получите:
- таблицу сравнения всех моделей
- confusion matrices для каждой модели
- лучшую модель по `f1_macro`

## Работа с Git LFS

Если клонируете репозиторий:

```bash
git lfs install
git clone <repo_url>
cd Tumor_type_prediction
git lfs pull
```

## Артефакты

После выполнения ноутбука сохраняются:
- `artifacts/best_classical_model.joblib`
- `artifacts/label_encoder.joblib`
- `artifacts/model_metrics_comparison.csv`
- `artifacts/cnn_1d_state_dict.pt` (если CNN обучалась)
- `artifacts/cnn_pre_var_pipeline.joblib`
- `artifacts/cnn_kbest.joblib`
