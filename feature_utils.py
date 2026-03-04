"""Utilities for feature engineering and feature-importance visualization.

This module supports three common bio/cheminformatics modalities:
- molecular SMILES -> Morgan fingerprints
- DNA/protein sequence -> k-mer frequency vectors
- gene-expression table -> numeric matrix
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Literal, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_expression_features(
    expression_df: pd.DataFrame,
) -> Tuple[np.ndarray, List[str]]:
    """Build numeric feature matrix for gene-expression data.

    Args:
        expression_df: DataFrame of samples x expression features.

    Returns:
        X: np.ndarray of shape (n_samples, n_features)
        feature_names: list of feature names
    """
    numeric_df = expression_df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        raise ValueError("No numeric columns found for expression features.")

    # Median fill makes the function robust to sparse missing values.
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))
    return numeric_df.to_numpy(dtype=np.float32), numeric_df.columns.tolist()


def _alphabet_for_modality(modality: Literal["dna", "protein"]) -> str:
    if modality == "dna":
        return "ACGT"
    # 20 canonical amino acids
    return "ACDEFGHIKLMNPQRSTVWY"


def build_kmer_features(
    sequences: Sequence[str],
    k: int = 3,
    modality: Literal["dna", "protein"] = "dna",
    normalize: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Convert DNA/protein sequences into k-mer frequency vectors.

    Args:
        sequences: list of sequence strings.
        k: k-mer length.
        modality: "dna" or "protein".
        normalize: if True, returns relative frequencies.

    Returns:
        X: np.ndarray (n_sequences, n_kmers)
        feature_names: k-mer names in deterministic lexical order.
    """
    if k <= 0:
        raise ValueError("k must be >= 1")

    alphabet = _alphabet_for_modality(modality)
    kmers = ["".join(chars) for chars in itertools.product(alphabet, repeat=k)]
    kmer_to_idx = {kmer: idx for idx, kmer in enumerate(kmers)}

    X = np.zeros((len(sequences), len(kmers)), dtype=np.float32)

    for row_idx, seq_raw in enumerate(sequences):
        seq = (seq_raw or "").upper()
        if len(seq) < k:
            continue

        valid_count = 0
        for i in range(len(seq) - k + 1):
            kmer = seq[i : i + k]
            if kmer in kmer_to_idx:
                X[row_idx, kmer_to_idx[kmer]] += 1.0
                valid_count += 1

        if normalize and valid_count > 0:
            X[row_idx, :] /= valid_count

    return X, kmers


def build_morgan_fingerprints(
    smiles_list: Sequence[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """Convert SMILES strings into Morgan fingerprints.

    Requires RDKit.

    Returns:
        X: np.ndarray (n_molecules, n_bits)
        feature_names: ["fp_0", ..., "fp_{n_bits-1}"]
        invalid_indices: indices of invalid/failed SMILES
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for molecular fingerprints. Install via conda/pip."
        ) from exc

    X = np.zeros((len(smiles_list), n_bits), dtype=np.int8)
    invalid_indices: List[int] = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            invalid_indices.append(i)
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[i, :] = arr

    feature_names = [f"fp_{i}" for i in range(n_bits)]
    return X, feature_names, invalid_indices


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
            # Multiclass: aggregate absolute coefficients across classes.
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
