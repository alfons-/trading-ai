"""Tests ligeros para RegimeAgent."""

from __future__ import annotations

import pandas as pd
import pytest

from src.agents.regime_agent import (
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_SIDEWAYS,
    RegimeAgent,
)


def test_bull_bear_sideways_rules():
    agent = RegimeAgent(adx_trending_min=20.0)
    df = pd.DataFrame(
        {
            "weekly_trend": [1, 0, 1, 0, 1],
            "weekly_adx": [25.0, 25.0, 10.0, 10.0, float("nan")],
        }
    )
    out = agent.assign_regime(df)
    assert list(out["regime"]) == [
        REGIME_BULL,
        REGIME_BEAR,
        REGIME_SIDEWAYS,
        REGIME_SIDEWAYS,
        REGIME_SIDEWAYS,
    ]


def test_missing_columns_raises():
    agent = RegimeAgent()
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(KeyError, match="weekly_trend"):
        agent.assign_regime(df)
