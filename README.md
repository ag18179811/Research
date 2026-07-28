## Phase 2: Execution Replay Engine

### Overview
Phase 2 acts as a realistic market physics engine. It processes trading orders from evaluated AI agents against Point-in-Time (PiT) market snapshots provided by Phase 1. 

By modeling real-world trading frictions, Phase 2 ensures agents are evaluated on realistic execution costs rather than assuming frictionless trades.

#### Key Capabilities
* **Bid-Ask Spread Matching:** Matches buy orders to ask prices and sell orders to bid prices.
* **Liquidity Depth Constraints:** Caps order fills at top-of-book volume limits.
* **Maker/Taker Fee Modeling:** Rewards passive liquidity (`POST_ONLY`, passive limit orders) with maker rebates while charging aggressive execution (`MARKET` orders) taker fees.
* **Non-Linear Market Impact:** Uses the Almgren-Chriss square-root law to model price slippage driven by order size relative to Average Daily Volume (ADV) and market volatility.
* **Breakeven Cost Sensitivity:** Iteratively tests logged trades under increasing fee assumptions to find the exact fee threshold where strategy returns drop to zero.

---

### How to Run Phase 2

#### Prerequisites
Phase 2 relies strictly on standard Python libraries (`dataclasses`, `enum`, `typing`, `math`, `json`). No external installations are required.

#### Execution
To run the simulation and export trade logs:

```bash
python3 main.py
