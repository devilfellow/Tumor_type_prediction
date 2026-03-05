# Tumor Type Prediction

Классификация типа опухоли по матрице экспрессии генов (RNA-seq / gene expression).

Текущая версия проекта использует **classical ML**:
- `LogisticRegression`
- `LinearSVC`
- `ExtraTreesClassifier`
- `HistGradientBoostingClassifier`

С дисбалансом классов работаем через **SMOTE внутри CV-пайплайна**.

## Данные

- `data.csv` — матрица экспрессии (`samples x genes`)
- `labels.csv` — целевые метки (`Class`) для образцов

## Что есть в ноутбуке

Файл: `tumor_type_prediction_smote_pipeline.ipynb`

### 1) Подготовка данных
- чтение `data.csv` и `labels.csv`
- маппинг по `sample_id`
- формирование `X`/`y` и train/test split

### 2) Препроцессинг и отбор признаков
- `SimpleImputer(strategy="median")`
- `StandardScaler()`
- `VarianceThreshold`
- `SelectKBest(f_classif)`
- подбор `kbest__k` через `GridSearchCV`

### 3) Обучение моделей
- единый `imblearn.Pipeline`: `preprocess -> var -> kbest -> smote -> model`
- 5-fold `StratifiedKFold`
- метрика подбора: `f1_macro`

### 4) Финальная оценка моделей
Для каждой модели на test считаются:
- `Accuracy (Top-1)`
- `Precision (macro)`
- `Recall (macro)`
- `F1 (macro)`
- `Balanced Accuracy`
- `MCC`

Плюс визуализации:
- confusion matrix (counts)
- confusion matrix, нормированная по строкам

Также формируется таблица сравнения `metrics_df`.

### 5) Важность признаков
Секция важности признаков использует `feature_utils.py`.

## Модуль `feature_utils.py`

Текущая версия содержит функции:
- `get_feature_importance(estimator, feature_names)`
- `plot_top_feature_importance(estimator, feature_names, top_n=25, title=...)`

Поддерживаются модели с:
- `coef_` (линейные)
- `feature_importances_` (деревья/ансамбли)

## Установка

```bash
pip install -U pandas numpy scikit-learn imbalanced-learn matplotlib joblib
```

## Запуск

1. Откройте `tumor_type_prediction_smote_pipeline.ipynb`.
2. Выполните ячейки сверху вниз.
3. В финальных ячейках получите:
- таблицу сравнения моделей (`metrics_df`)
- confusion matrix (обычную и нормированную) для каждой модели
- график топ-важных признаков для выбранной модели

## Работа с Git LFS

```bash
git lfs install
git clone <repo_url>
cd Tumor_type_prediction
git lfs pull
```
