"""Data loading, domain-specific cleaning, encoding and the SEAF split protocol.

Split protocol (paper Section 3.6)::

    full data --80/20--> train | test
    test      --50/50--> baseline | evaluation

The *baseline* half is the only data used to estimate attribution scalers, the
isolation forest, the stability reference and the normalisation percentiles.
The *evaluation* half supplies instances that are scored and attacked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, DOMAINS, HEALTHCARE_N, SEED, TEST_SIZE, DomainSpec

# A column with more distinct values than this (and not categorical) is treated
# as continuous: it receives stability jitter and adversarial perturbation.
NUMERIC_MIN_UNIQUE = 10


@dataclass
class Schema:
    """Everything needed to rebuild, validate and display a feature vector."""

    feature_names: list[str]
    numeric_features: list[str]                     # jittered / attacked
    integer_features: list[str]                     # rounded after perturbation
    categorical_maps: dict[str, list[str]]          # encoded col -> class list
    ranges: dict[str, tuple[float, float]]          # observed min / max (train)
    choices: dict[str, list[float]]                 # low-cardinality value sets
    target: str
    class_names: tuple[str, str]
    positive_label: str
    feature_labels: dict[str, str] = field(default_factory=dict)
    value_labels: dict[str, dict] = field(default_factory=dict)
    display_units: dict[str, tuple[float, str]] = field(default_factory=dict)

    @property
    def numeric_idx(self) -> np.ndarray:
        return np.array([self.feature_names.index(c) for c in self.numeric_features], dtype=int)

    @property
    def integer_idx(self) -> np.ndarray:
        return np.array([self.feature_names.index(c) for c in self.integer_features], dtype=int)

    def label(self, col: str) -> str:
        return self.feature_labels.get(col, col)

    def display_value(self, col: str, value: float) -> str:
        """Human-readable value for a (possibly encoded) feature."""
        if col in self.categorical_maps:
            cats = self.categorical_maps[col]
            i = int(round(value))
            return cats[i] if 0 <= i < len(cats) else str(value)
        if col in self.value_labels:
            lab = self.value_labels[col].get(int(round(value)))
            if lab is not None:
                return lab
        if col in self.display_units:
            factor, unit = self.display_units[col]
            return f"{value * factor:,.1f} {unit}"
        if col in self.integer_features:
            return f"{int(round(value)):,}"
        return f"{value:,.2f}"


@dataclass
class Splits:
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_base: pd.DataFrame
    y_base: np.ndarray
    X_eval: pd.DataFrame
    y_eval: np.ndarray

    @property
    def X_test(self) -> pd.DataFrame:
        return pd.concat([self.X_base, self.X_eval])

    @property
    def y_test(self) -> np.ndarray:
        return np.concatenate([self.y_base, self.y_eval])


# --------------------------------------------------------------------------- #
# Loading and domain-specific cleaning
# --------------------------------------------------------------------------- #
def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()
    return df


def load_raw(spec: DomainSpec, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = Path(data_dir) / spec.csv
    df = pd.read_csv(path, sep=spec.sep, encoding="utf-8-sig", skipinitialspace=True)
    return _strip_strings(df)


def clean_healthcare(df: pd.DataFrame, n: int = HEALTHCARE_N, seed: int = SEED) -> pd.DataFrame:
    """Remove physiologically inadmissible records, then downsample to ``n``.

    Bounds from the paper: systolic 80-200, diastolic 40-130, systolic >
    diastolic, height 130-210 cm, weight 35-180 kg.  Age stays in days, as in
    the source data; the UI converts it to years for display only.
    """
    keep = (
        df["ap_hi"].between(80, 200)
        & df["ap_lo"].between(40, 130)
        & (df["ap_hi"] > df["ap_lo"])
        & df["height"].between(130, 210)
        & df["weight"].between(35, 180)
    )
    df = df.loc[keep].copy()
    if n and len(df) > n:
        df = df.sample(n=n, random_state=seed)
    return df.reset_index(drop=True)


def domain_frame(key: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Raw frame after the domain's cleaning rule (before encoding)."""
    spec = DOMAINS[key]
    df = load_raw(spec, data_dir)
    if key == "healthcare":
        df = clean_healthcare(df)
    return df


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #
def encode_frame(
    df: pd.DataFrame,
    target: str,
    positive_label: str,
    drop_columns: tuple[str, ...] = (),
    class_names: tuple[str, str] | None = None,
    feature_labels: dict | None = None,
    value_labels: dict | None = None,
    display_units: dict | None = None,
) -> tuple[pd.DataFrame, np.ndarray, Schema]:
    """Drop ids/constant columns, label-encode categoricals, binarise target.

    Label encoding uses the sorted category list (identical to
    ``sklearn.preprocessing.LabelEncoder``); the list is stored in the schema
    so the UI can decode values back to their original strings.
    """
    df = df.drop(columns=[c for c in drop_columns if c in df.columns])
    df = df.dropna(axis=0, how="any").reset_index(drop=True)
    y = (df[target].astype(str) == str(positive_label)).astype(int).to_numpy()
    X = df.drop(columns=[target])

    constant = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    X = X.drop(columns=constant)

    categorical_maps: dict[str, list[str]] = {}
    out = pd.DataFrame(index=X.index)
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            out[c] = X[c].astype(float)
        else:
            cats = sorted(X[c].astype(str).unique().tolist())
            categorical_maps[c] = cats
            out[c] = X[c].astype(str).map({v: i for i, v in enumerate(cats)}).astype(float)

    integer_features = [
        c for c in X.columns
        if c in categorical_maps or pd.api.types.is_integer_dtype(X[c])
    ]
    numeric_features = [
        c for c in X.columns
        if c not in categorical_maps and out[c].nunique() > NUMERIC_MIN_UNIQUE
    ]
    choices = {
        c: sorted(out[c].unique().tolist())
        for c in out.columns
        if c not in numeric_features
    }
    ranges = {c: (float(out[c].min()), float(out[c].max())) for c in out.columns}
    neg_name = class_names[0] if class_names else f"not {positive_label}"
    pos_name = class_names[1] if class_names else str(positive_label)
    schema = Schema(
        feature_names=list(out.columns),
        numeric_features=numeric_features,
        integer_features=integer_features,
        categorical_maps=categorical_maps,
        ranges=ranges,
        choices=choices,
        target=target,
        class_names=(neg_name, pos_name),
        positive_label=str(positive_label),
        feature_labels=dict(feature_labels or {}),
        value_labels=dict(value_labels or {}),
        display_units=dict(display_units or {}),
    )
    return out, y, schema


def prepare_domain(key: str, data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, np.ndarray, Schema]:
    spec = DOMAINS[key]
    df = domain_frame(key, data_dir)
    return encode_frame(
        df,
        target=spec.target,
        positive_label=spec.positive_label,
        drop_columns=spec.id_columns,
        class_names=spec.class_names,
        feature_labels=spec.feature_labels,
        value_labels=spec.value_labels,
        display_units=spec.display_units,
    )


def make_splits(X: pd.DataFrame, y: np.ndarray, seed: int = SEED, test_size: float = TEST_SIZE) -> Splits:
    """80/20 stratified train/test, then halve the test set into baseline/eval."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    X_b, X_e, y_b, y_e = train_test_split(X_te, y_te, test_size=0.5, random_state=seed, stratify=y_te)
    return Splits(
        X_tr.reset_index(drop=True), np.asarray(y_tr),
        X_b.reset_index(drop=True), np.asarray(y_b),
        X_e.reset_index(drop=True), np.asarray(y_e),
    )


def id_like_columns(df: pd.DataFrame) -> list[str]:
    """Heuristic for uploaded data: integer columns unique on every row, or named *id."""
    cols = []
    for c in df.columns:
        name = str(c).lower()
        unique_int = pd.api.types.is_integer_dtype(df[c]) and df[c].is_unique and len(df) > 20
        if unique_int or name in {"id", "index"} or name.endswith("_id"):
            cols.append(c)
        elif not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 0.5 * len(df):
            cols.append(c)  # free-text / identifier strings
    return cols
