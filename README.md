# DeFi Liquidation Risk Simulator

An agent-based cascade simulator for DeFi lending protocols, grounded in the equilibrium flash crash model of Mishricky (2025). Stress-test Aave V3-equivalent pools under collateral price shocks, with endogenous bot participation feedback derived from the theoretical equilibrium.

## Overview

DeFi lending protocols allow users to borrow against crypto collateral. When collateral prices fall below a threshold, positions are automatically liquidated — bots repay the debt and seize discounted collateral. The danger is self-reinforcing cascades: liquidation selling pushes prices lower, triggering further liquidations in a doom loop that destroyed hundreds of millions in protocols like Venus and Compound during the 2021 and 2022 crashes.

This simulator models that process on either a **synthetic Aave V3-equivalent pool** (~$1.8B collateral, ~$762M debt, 1,000 positions) or a **live position pool** generated from real-time Aave V3 Ethereum reserve data. Choose a price drop, configure market conditions, and watch the cascade unfold round by round through an interactive Plotly Dash dashboard.

## Key Contribution

Most liquidation simulators count bad debt and stop. This project integrates the **flash crash probability F** — the per-cycle likelihood that no competitive liquidation quote is posted — directly into the cascade engine as an endogenous feedback mechanism.

F is derived from the equilibrium in Mishricky (2025), where bots decide whether to post quotes by weighing expected profit against participation cost (gas). As the cascade drains liquidity and gas costs rise, fewer bots post competitive quotes. In the limit, the liquidation market collapses entirely.

**The feedback loop works as follows.** At each round, the simulator computes a compound participation rate `(1 − F)^1000` across 1,000 quote cycles. Only that fraction of liquidatable positions are actually cleared. The remainder stay underwater, accumulating bad debt and keeping liquidity depressed — which raises F further in the next round:

```
cascade drains liquidity → φᵐ falls → F rises → bots exit → liquidity drains further
```

The gas spike scenario isolates a distinct failure channel where the cascade mechanism operates through costs rather than liquidity:

```
gas spike → κ rises → θ = ln(φᵐΓ/κ) falls → F = e^(−θ) rises → bots exit
    ↓                                                                  ↓
bad debt stays low ← positions remain unliquidated ← no competitive quotes posted
    ↓
risk monitors show "healthy" protocol ← but liquidation market is non-functional
```

The dashboard's Bot Participation Model toggle lets you compare this endogenous mode against open-loop behaviour (bots always participate) to observe the difference directly. Bot-absent rounds are highlighted in orange on the cascade chart.

## The Conservation Law

The central theoretical result (Mishricky 2025, Footnote 28) is:

```
MSE · F = (κ / φᵐ)²
```

The product of price dispersion (MSE) and flash crash probability (F) is determined entirely by the ratio of posting costs to liquidity. The practical implication: **a protocol can generate minimal bad debt while F is already elevated and the liquidation market is near collapse**. Low bad debt does not mean the market mechanism that keeps the protocol solvent is functioning — it may simply mean that the cascade has not yet reached the positions where losses concentrate, while the infrastructure required to clear those positions is already breaking down. The gas spike preset in the simulator reproduces exactly this pattern: F enters the ELEVATED RISK range while bad debt remains negligible. Standard risk monitors, which track bad debt alone, would not flag the scenario.

## Theoretical Framework

The model implements the equilibrium from Mishricky (2025), which embeds a price posting game among liquidation bots within a monetary economy to derive closed-form expressions for flash crash probability, bid/ask distributions, and price dispersion.

**Parameter mapping:**

| Theoretical Parameter | DeFi Observable |
|---|---|
| κ (quote posting cost) | Gas cost per liquidation (USD), normalised by liquidity depth |
| φᵐ (real value of money) | Stablecoin liquidity depth / total protocol debt |
| Γ (book width) | Daily volatility of collateral asset |

**Implemented results:** flash crash probability `F = e^(−θ)` where `θ = ln(φᵐΓ/κ)` (Proposition 1); equilibrium bid/ask distributions A(p) and B(p); monotonicity of F in κ and φᵐ (Proposition 11); speculative premium (Proposition 12); and the conservation law `MSE · F = (κ/φᵐ)²` (Footnote 28).

**Interpreting F:** F is a per-cycle probability. At 1,000 quote cycles per hour, the daily flash crash probability `1 − (1−F)^24000` reaches ~70% at the STABLE boundary (F = 0.00005) and effectively 100% at the CRITICAL boundary (F = 0.00050).

| Status | F Threshold | Interpretation |
|---|---|---|
| STABLE | F < 0.00005 | Competitive liquidation market, low flash crash risk |
| ELEVATED RISK | 0.00005 ≤ F < 0.00050 | Liquidity thinning, market quality degrading |
| CRITICAL | F ≥ 0.00050 | Near-certain daily flash crash, protocol solvency at risk |

## Live Data

The simulator can run against **live Aave V3 Ethereum data** fetched from the official Aave GraphQL API (no API key required). When live mode is selected, `fetch_live.py` queries current reserve parameters — liquidation thresholds and bonuses, total supply and debt, and available stablecoin liquidity — across all active Ethereum markets. From these, it generates 1,000 synthetic positions weighted by each reserve's share of total protocol debt, with health factors drawn from a calibrated log-normal distribution.

If the API is unreachable, the dashboard automatically falls back to the synthetic pool (calibrated to Aave V3 Ethereum statistics as of March 2026 via DeFiLlama and the Aave app) and displays a status message.

## Crisis Scenario Presets

| Preset | Price Drop | Liquidity | Gas | Scenario |
|---|---|---|---|---|
| Normal market | 30% | 40% | $80 | Baseline — starts STABLE, deteriorates to ELEVATED RISK as the cascade drains liquidity |
| Liquidity crisis | 30% | 3% | $80 | Capital flight before the shock; φᵐ near zero from round one |
| Gas spike | 30% | 40% | $450 | κ shock — demonstrates how gas cost elevates F even without meaningful bad debt |
| Combined shock | 45% | 5% | $400 | Simultaneous liquidity and gas failure (March 2020 / November 2022 analogue) |

## Getting Started

**Requirements:** Python 3.9+

```bash
pip install dash plotly pandas numpy scipy requests
```

**Launch the dashboard:**

```bash
python dashboard.py
```

Then open http://127.0.0.1:8050 in your browser.

**Run the simulation from the command line:**

```bash
python simulate.py
```

This executes five benchmark scenarios (10%–50% price drops) and prints round-by-round cascade results with theoretical scores.

**Test the live data feed:**

```bash
python fetch_live.py
```

**Run theory tests:**

```bash
python test_theory.py           # Stress-test F across gas/liquidity parameter space
python test_distributions.py    # Plot bid/ask distributions under each crisis preset
python test_speculation.py      # Speculative premium analysis (Proposition 12)
```

## Dashboard Features

The interactive dashboard provides a data source toggle (live Aave V3 vs. synthetic pool), a bot participation model toggle (endogenous feedback vs. open-loop), four crisis scenario presets with one-click switching, and interactive sliders for price drop (5–60%), liquidity (1–80%), and gas cost ($20–500). It displays eight summary statistics — including bad debt, cascade rounds, rounds with bot absence, initial and final F, and market status — alongside four charts: the liquidation cascade by round (with bot-absent rounds highlighted), θ and F evolution, equilibrium bid/ask distributions, and a full stress test surface.

## Project Structure

```
defi-liquidation-sim/
├── dashboard.py            Plotly Dash interactive dashboard
├── simulate.py             Cascade engine with Bernoulli(F) bot feedback
├── theory.py               BurdettJuddDeFi class implementing Mishricky (2025)
├── agents.py               BorrowerAgent class with health factor calculation
├── fetch_aave.py           Synthetic pool calibrated to Aave V3 Ethereum
├── fetch_live.py           Live data via Aave V3 GraphQL API (no key required)
├── test_theory.py          Gas/liquidity stress tests
├── test_distributions.py   Bid/ask distribution plots under stress
├── test_speculation.py     Speculative premium analysis
└── test_agents.py          Unit tests for BorrowerAgent
```

## Limitations

The live data feed provides current reserve parameters, but full quantitative validation of F against empirical liquidation gap frequency requires historical time-series of on-chain liquidation events (March 2020, November 2022) which are not yet integrated. The position pool is single-asset by construction — real Aave positions are often multi-collateral, which affects both health factor dynamics and liquidation incentive calculations.

Planned extensions include historical backtesting against on-chain liquidation gaps, multi-collateral position modelling, cross-protocol contagion (Aave ↔ Compound ↔ Morpho), and endogenous gas pricing during network congestion.

## License

MIT

## Reference

Mishricky, S. (2025). *Asset Price Dispersion, Monetary Policy and Macroprudential Regulation*. Working paper, Australian National University.
