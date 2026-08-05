

from __future__ import annotations

import abc
from typing import Optional

# adjust to your real Phase 2 filename
from execution_replay import Order, OrderSide, OrderType


class TradingAgent(abc.ABC):

    @abc.abstractmethod
    def decide(self, snapshot: dict) -> list[Order]:
      
        raise NotImplementedError

    def reset(self) -> None:
        pass

class SmaCrossoverAgent(TradingAgent):


    def __init__(self, order_qty: float = 10.0):
        self.order_qty = order_qty
        self._position_open = False

    def reset(self) -> None:
        self._position_open = False

    def decide(self, snapshot: dict) -> list[Order]:
        features = snapshot.get("derived_features") or {}
        r5, r21 = features.get("return_5d"), features.get("return_21d")
        if r5 is None or r21 is None:
            return []  # not enough history yet this early in the episode

        ticker = snapshot["ticker"]
        as_of = snapshot["as_of"]
        want_long = r5 > r21

        orders: list[Order] = []
        if want_long and not self._position_open:
            orders.append(Order(
                order_id=f"BUY_{ticker}_{as_of.date()}",
                timestamp=as_of.isoformat(),
                symbol=ticker,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=self.order_qty,
            ))
            self._position_open = True
        elif not want_long and self._position_open:
            orders.append(Order(
                order_id=f"SELL_{ticker}_{as_of.date()}",
                timestamp=as_of.isoformat(),
                symbol=ticker,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=self.order_qty,
            ))
            self._position_open = False

        return orders


