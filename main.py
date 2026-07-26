import dataclasses
from enum import Enum
from typing import List, Dict, Optional
import math
import json


# =====================================================================
# 1. ENUMS & CORE DATA STRUCTURES
# =====================================================================

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    POST_ONLY = "POST_ONLY"  # Enforces liquidity maker execution


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class MarketRegime(Enum):
    TRENDING = "TRENDING"
    MEAN_REVERTING = "MEAN_REVERTING"
    CRISIS_HIGH_VOLATILITY = "CRISIS_HIGH_VOLATILITY"


@dataclasses.dataclass
class Order:
    order_id: str
    timestamp: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    limit_price: Optional[float] = None


@dataclasses.dataclass
class MarketDataSnapshot:
    """Point-in-Time (PiT) market snapshot received from Phase 1."""
    timestamp: str
    symbol: str
    bid_price: float
    ask_price: float
    bid_depth: float  # Available liquidity volume at best bid
    ask_depth: float  # Available liquidity volume at best ask
    daily_volatility: float  # Asset volatility (sigma)
    average_daily_volume: float  # ADV for impact modeling
    regime: MarketRegime


@dataclasses.dataclass
class ExecutionFill:
    order_id: str
    timestamp: str
    symbol: str
    side: OrderSide
    requested_qty: float
    executed_qty: float
    execution_price: float
    base_price: float
    slippage: float
    fee_paid: float
    is_maker: bool
    regime: MarketRegime


@dataclasses.dataclass
class ReplayConfig:
    maker_fee_bps: float = -1.0  # Liquidity rebate (-1 bps)
    taker_fee_bps: float = 5.0    # Liquidity removal fee (+5 bps)
    impact_gamma: float = 0.1     # Almgren-Chriss market impact scale factor
    initial_cash: float = 100_000.0


# =====================================================================
# 2. EXECUTION REPLAY ENGINE
# =====================================================================

class ExecutionReplayEngine:
    """
    Simulates realistic order execution against Point-in-Time order book states,
    accounting for spreads, order book depth, maker/taker fees, and market impact.
    """

    def __init__(self, config: ReplayConfig):
        self.config = config
        self.cash_balance: float = config.initial_cash
        self.positions: Dict[str, float] = {}  # Symbol -> Quantity
        self.execution_log: List[ExecutionFill] = []

    def _calculate_market_impact(self, quantity: float, snapshot: MarketDataSnapshot) -> float:
        """
        Calculates non-linear price impact using the Square-Root Law:
        Impact % = gamma * sigma * sqrt(Quantity / ADV)
        """
        if snapshot.average_daily_volume <= 0:
            return 0.0

        volume_fraction = quantity / snapshot.average_daily_volume
        impact_pct = self.config.impact_gamma * snapshot.daily_volatility * math.sqrt(volume_fraction)
        return impact_pct

    def process_order(self, order: Order, snapshot: MarketDataSnapshot) -> ExecutionFill:
        """Processes an incoming agent order against current PiT market snapshot."""
        assert order.symbol == snapshot.symbol, "Symbol mismatch between order and market snapshot."

        # 1. Determine baseline quote and book depth based on side
        if order.side == OrderSide.BUY:
            base_price = snapshot.ask_price
            depth_available = snapshot.ask_depth
        else:
            base_price = snapshot.bid_price
            depth_available = snapshot.bid_depth

        # 2. Enforce top-of-book depth liquidity constraint
        fill_qty = min(order.quantity, depth_available)

        # 3. Determine Maker vs Taker semantics
        is_maker = False
        if order.order_type == OrderType.POST_ONLY:
            is_maker = True
            base_price = snapshot.bid_price if order.side == OrderSide.BUY else snapshot.ask_price
        elif order.order_type == OrderType.LIMIT and order.limit_price is not None:
            if order.side == OrderSide.BUY and order.limit_price < snapshot.ask_price:
                is_maker = True
                base_price = order.limit_price
            elif order.side == OrderSide.SELL and order.limit_price > snapshot.bid_price:
                is_maker = True
                base_price = order.limit_price

        # 4. Compute Non-Linear Market Impact
        impact_pct = self._calculate_market_impact(fill_qty, snapshot)

        if order.side == OrderSide.BUY:
            execution_price = base_price * (1.0 + impact_pct)
            slippage = execution_price - snapshot.ask_price
        else:
            execution_price = base_price * (1.0 - impact_pct)
            slippage = snapshot.bid_price - execution_price

        # 5. Fee Accounting
        fee_bps = self.config.maker_fee_bps if is_maker else self.config.taker_fee_bps
        notional_value = fill_qty * execution_price
        fee_paid = notional_value * (fee_bps / 10000.0)

        # 6. Construct Fill Record
        fill = ExecutionFill(
            order_id=order.order_id,
            timestamp=snapshot.timestamp,
            symbol=order.symbol,
            side=order.side,
            requested_qty=order.quantity,
            executed_qty=fill_qty,
            execution_price=execution_price,
            base_price=base_price,
            slippage=slippage,
            fee_paid=fee_paid,
            is_maker=is_maker,
            regime=snapshot.regime
        )

        # 7. Update Internal State
        self._update_portfolio(fill)
        self.execution_log.append(fill)

        return fill

    def _update_portfolio(self, fill: ExecutionFill):
        """Updates internal cash balance and position inventory."""
        current_pos = self.positions.get(fill.symbol, 0.0)

        if fill.side == OrderSide.BUY:
            self.positions[fill.symbol] = current_pos + fill.executed_qty
            self.cash_balance -= (fill.executed_qty * fill.execution_price + fill.fee_paid)
        else:
            self.positions[fill.symbol] = current_pos - fill.executed_qty
            self.cash_balance += (fill.executed_qty * fill.execution_price - fill.fee_paid)

    def compute_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculates total mark-to-market portfolio value."""
        positions_value = sum(
            qty * current_prices.get(symbol, 0.0)
            for symbol, qty in self.positions.items()
        )
        return self.cash_balance + positions_value

    def calculate_breakeven_threshold(self, step_bps: float = 1.0, max_bps: float = 100.0) -> float:
        """Re-evaluates logged trades across increasing fee steps to find zero-PnL threshold."""
        if not self.execution_log:
            return 0.0

        current_bps = 0.0
        while current_bps <= max_bps:
            simulated_net_pnl = 0.0

            for fill in self.execution_log:
                notional = fill.executed_qty * fill.execution_price
                simulated_fee = notional * (current_bps / 10000.0)

                direction = 1.0 if fill.side == OrderSide.SELL else -1.0
                simulated_net_pnl += (direction * notional) - simulated_fee

            if simulated_net_pnl <= 0:
                return current_bps

            current_bps += step_bps

        return max_bps

    def export_logs_to_json(self, filepath: str):
        """Exports execution logs to a JSON file using standard Python library."""
        log_dicts = []
        for fill in self.execution_log:
            record = dataclasses.asdict(fill)
            record['side'] = fill.side.value
            record['regime'] = fill.regime.value
            log_dicts.append(record)

        with open(filepath, 'w') as f:
            json.dump(log_dicts, f, indent=4)

        print(f"Successfully exported {len(log_dicts)} execution log record(s) to '{filepath}'")


# =====================================================================
# 3. SIMULATION DEMO PIPELINE
# =====================================================================

def run_simulation_demo():
    print("============================================================")
    print("   STARTING PHASE 2 EXECUTION REPLAY ENGINE SIMULATION     ")
    print("============================================================\n")

    # Initialize Engine
    config = ReplayConfig(maker_fee_bps=-1.0, taker_fee_bps=5.0, initial_cash=100000.0)
    engine = ExecutionReplayEngine(config)

    # Simulated Market Snapshots
    mock_market_stream = [
        MarketDataSnapshot(
            timestamp="2026-03-15T09:30:00Z",
            symbol="AAPL",
            bid_price=180.00,
            ask_price=180.10,
            bid_depth=1000.0,
            ask_depth=500.0,
            daily_volatility=0.02,
            average_daily_volume=1_000_000.0,
            regime=MarketRegime.TRENDING
        ),
        MarketDataSnapshot(
            timestamp="2026-03-15T10:00:00Z",
            symbol="AAPL",
            bid_price=182.50,
            ask_price=182.60,
            bid_depth=800.0,
            ask_depth=600.0,
            daily_volatility=0.025,
            average_daily_volume=1_000_000.0,
            regime=MarketRegime.CRISIS_HIGH_VOLATILITY
        )
    ]

    # Simulated Orders sent by Agent
    mock_agent_orders = [
        Order(
            order_id="ORD_001",
            timestamp="2026-03-15T09:30:00Z",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=200.0
        ),
        Order(
            order_id="ORD_002",
            timestamp="2026-03-15T10:00:00Z",
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.POST_ONLY,
            quantity=200.0,
            limit_price=182.50
        )
    ]

    # Process Market Stream & Orders
    for order, snapshot in zip(mock_agent_orders, mock_market_stream):
        fill = engine.process_order(order, snapshot)

        print(f"[Fill Generated] Order ID: {fill.order_id}")
        print(f"  Side:            {fill.side.value}")
        print(f"  Executed Qty:    {fill.executed_qty} / Requested: {fill.requested_qty}")
        print(f"  Base Price:      ${fill.base_price:.2f}")
        print(f"  Execution Price: ${fill.execution_price:.4f} (Slippage: ${fill.slippage:.4f})")
        print(f"  Fee Paid:        ${fill.fee_paid:.4f} (Is Maker: {fill.is_maker})")
        print(f"  Regime Tag:      {fill.regime.value}\n")

    # Mark-to-Market Valuation
    latest_prices = {"AAPL": 182.50}
    total_val = engine.compute_portfolio_value(latest_prices)
    net_pnl = total_val - config.initial_cash

    print("------------------------------------------------------------")
    print("PORTFOLIO SUMMARY")
    print("------------------------------------------------------------")
    print(f"Ending Cash Balance:    ${engine.cash_balance:.2f}")
    print(f"Positions:              {engine.positions}")
    print(f"Mark-to-Market Value:   ${total_val:.2f}")
    print(f"Net Realized Profit:    ${net_pnl:.2f}\n")

    # Breakeven Sensitivity Analysis
    breakeven_bps = engine.calculate_breakeven_threshold()
    print("------------------------------------------------------------")
    print("COST SENSITIVITY & BREAKEVEN ANALYSIS")
    print("------------------------------------------------------------")
    print(f"Execution Breakeven Fee Threshold: {breakeven_bps:.1f} bps\n")

    # Export Data
    print("------------------------------------------------------------")
    print("PHASE 3 DATA EXPORT")
    print("------------------------------------------------------------")
    engine.export_logs_to_json("execution_log.json")


if __name__ == "__main__":
    run_simulation_demo()