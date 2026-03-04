from __future__ import annotations

import itertools
from typing import Iterable, List, Literal, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_feature_importance(
    estimator,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Extract feature importance from linear or tree models.

    Supports:
    - `coef_` (linear models, including multiclass)
    - `feature_importances_` (tree ensembles)
    """
    feature_names = list(feature_names)

    if hasattr(estimator, "coef_"):
        coefs = np.asarray(estimator.coef_)
        if coefs.ndim == 1:
            scores = np.abs(coefs)
        else:
            scores = np.mean(np.abs(coefs), axis=0)
    elif hasattr(estimator, "feature_importances_"):
        scores = np.asarray(estimator.feature_importances_)
    else:
        raise ValueError(
            "Estimator has no coef_ or feature_importances_. "
            "Use a linear or tree-based model for direct importance."
        )

    if len(scores) != len(feature_names):
        raise ValueError(
            f"Length mismatch: {len(scores)} importances vs {len(feature_names)} feature names"
        )

    imp_df = pd.DataFrame({"feature": feature_names, "importance": scores})
    return imp_df.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_top_feature_importance(
    estimator,
    feature_names: Sequence[str],
    top_n: int = 25,
    title: str = "Top feature importance",
) -> pd.DataFrame:
    """Plot top-N important features and return the full importance table."""
    imp_df = get_feature_importance(estimator, feature_names)
    top = imp_df.head(top_n).iloc[::-1]

    plt.figure(figsize=(10, max(4, 0.35 * len(top))))
    plt.barh(top["feature"], top["importance"], color="steelblue")
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return imp_df
