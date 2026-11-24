""" Тестирования корр.функции """

from __future__ import annotations

import pytest
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from gtm_detector.ml_freq.tests.conftest import make_well
from gtm_detector.ml_freq.features.builder import FeatureBuilder


def test_corr_func():
    test_well = make_well(step=30, effect=25, seed=42)

    f_bulder = FeatureBuilder()

    features = \
    [
        "liq_missing",
        "pzab_vs_30d",
        "wc_vs_30d",
        "oil_vs_30d",
        "liq_vs_30d",
        "f_below_max30",
        "f_above_min30",
        "f_diff",
        "f_rel",
        "working"
    ]

    test_df = f_bulder._channels(test_well)
    corr = test_df[features].corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(8,6))
    sns.heatmap\
        (
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.5
        )
    plt.title("Раноговая корреляция признаков")
    plt.tight_layout()
    plt.show(block=True)