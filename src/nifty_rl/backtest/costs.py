"""Transaction cost models.

The notebook applied a flat 10 bps + 5 bps to every fill. That is retained as
``FlatCostModel`` so old results stay reproducible, but it understates Indian delivery
equity costs, where STT alone is 10 bps on each side.

Both models expose the same interface: given a side, price and quantity, return the
*effective* per-share fill price. Callers never apply costs themselves.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..config import CostConfig

BUY = "buy"
SELL = "sell"


class CostModel(ABC):
    """Maps a desired trade to an effective fill price."""

    @abstractmethod
    def cost_rate(self, side: str, price: float, quantity: float, adv_value: Optional[float] = None) -> float:
        """Total round-of-one-side cost as a fraction of notional."""

    def fill_price(self, side: str, price: float, quantity: float, adv_value: Optional[float] = None) -> float:
        rate = self.cost_rate(side, price, quantity, adv_value)
        if side == BUY:
            return price * (1.0 + rate)
        if side == SELL:
            return price * (1.0 - rate)
        raise ValueError(f"side must be {BUY!r} or {SELL!r}, got {side!r}")


class FlatCostModel(CostModel):
    """Notebook-compatible flat cost: ``transaction_cost + slippage`` on each side."""

    def __init__(self, cfg: CostConfig):
        self.cfg = cfg
        self._base = cfg.transaction_cost + cfg.slippage

    def cost_rate(self, side: str, price: float, quantity: float, adv_value: Optional[float] = None) -> float:
        return self._base + _impact_rate(self.cfg, price, quantity, adv_value)


class IndiaEquityCostModel(CostModel):
    """Delivery-equity charge stack for a retail Indian investor.

    Components (each as a fraction of turnover unless noted):

    ==================  ==========  ==========  ==================================
    Charge              Buy         Sell        Notes
    ==================  ==========  ==========  ==================================
    Brokerage           0.03%       0.03%       capped at Rs20 per order
    STT                 0.10%       0.10%       delivery equity, both sides
    Stamp duty          0.015%      --          buy side only
    Exchange txn        0.00297%    0.00297%    NSE equity
    SEBI turnover       0.0001%     0.0001%
    GST                 18%         18%         on brokerage + exchange + SEBI
    ==================  ==========  ==========  ==================================

    Slippage and square-root market impact are layered on top.
    """

    def __init__(self, cfg: CostConfig):
        self.cfg = cfg

    def cost_rate(self, side: str, price: float, quantity: float, adv_value: Optional[float] = None) -> float:
        cfg = self.cfg
        notional = max(price * quantity, 1e-12)

        brokerage_value = min(cfg.brokerage_rate * notional, cfg.brokerage_cap)
        brokerage = brokerage_value / notional

        exchange = cfg.exchange_txn_rate
        sebi = cfg.sebi_turnover_rate
        gst = cfg.gst_rate * (brokerage + exchange + sebi)

        if side == BUY:
            statutory = cfg.stt_buy + cfg.stamp_duty_buy
        elif side == SELL:
            statutory = cfg.stt_sell
        else:
            raise ValueError(f"side must be {BUY!r} or {SELL!r}, got {side!r}")

        return (
            brokerage
            + exchange
            + sebi
            + gst
            + statutory
            + cfg.slippage
            + _impact_rate(cfg, price, quantity, adv_value)
        )


def _impact_rate(cfg: CostConfig, price: float, quantity: float, adv_value: Optional[float]) -> float:
    """Square-root market impact: ``coef * sqrt(order_value / ADV)``.

    Disabled by default (``impact_coef=0``). This is the term that should discipline the
    RL agent's turnover once enabled -- a flat per-trade cost does not penalise size.
    """
    if cfg.impact_coef <= 0 or not adv_value or adv_value <= 0:
        return 0.0
    participation = (price * abs(quantity)) / adv_value
    return float(cfg.impact_coef * np.sqrt(max(participation, 0.0)))


def build_cost_model(cfg: CostConfig) -> CostModel:
    if cfg.model == "flat":
        return FlatCostModel(cfg)
    if cfg.model == "india":
        return IndiaEquityCostModel(cfg)
    raise ValueError(f"Unknown cost model {cfg.model!r}; expected 'flat' or 'india'.")
