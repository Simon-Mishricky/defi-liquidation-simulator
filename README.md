# DeFi Liquidation Risk Simulator

A theory-driven simulator that stress-tests DeFi lending protocols under collateral price shocks, combining agent-based cascade simulation with the equilibrium flash crash model of **Mishricky (2025)**.

---

## What is this?

DeFi lending protocols like Aave let users borrow against crypto collateral. If the collateral price falls enough, the protocol automatically liquidates the position — a bot repays the debt and seizes the collateral at a discount.

The danger is that liquidations can feed on themselves. Imagine ETH drops 30%. Thousands of positions are liquidated simultaneously. The liquidation bots sell the seized ETH to repay stablecoin debt, pushing ETH down further. That triggers more liquidations. This self-reinforcing loop is called a **liquidation cascade**, and it is what destroyed hundreds of millions of dollars in protocols like Venus and Compound during the May 2021 and November 2022 crashes.

This simulator models exactly that process on either a synthetic Aave V3-equivalent pool ($1.8B collateral, $762M debt, 1,000 borrower positions) or a live position pool generated from real-time Aave V3 Ethereum reserve data fetched directly from the Aave API. You choose a price drop, hit run, and watch the cascade unfold round by round.

---

## What makes it different?

Most liquidation simulators stop at counting how many positions get liquidated and how much bad debt accumulates. That is useful, but it misses a deeper question: **as the cascade drains liquidity and drives up gas costs, does the liquidation market remain competitive?**

In the Burdett-Judd equilibrium, bots decide whether to post quotes by weighing the expected profit against the cost of participation (gas). When that trade-off deteriorates — because φᵐ falls as liquidity is consumed, or because κ rises during network congestion — fewer bots post competitive quotes. In the limit, no bot posts at all. This is a flash crash: not just bad debt, but a complete breakdown of the price discovery mechanism.

This simulator tracks **F**, the per-cycle probability that no competitive quote is posted, derived from the equilibrium in Mishricky (2025). F rises endogenously as the cascade progresses. It is calibrated from three observable quantities:

- **Gas cost per liquidation** (κ in the model) — high gas makes liquidation unprofitable, bots exit
- **Stablecoin liquidity depth as a fraction of total debt** (φᵐ) — as liquidations consume liquidity, this falls
- **Collateral volatility** (Γ) — wider price swings widen the book but also increase execution risk

As the cascade progresses, liquidity drains and F rises. The dashboard shows this transition in real time.

---

## Live data

The dashboard can run against **live Aave V3 Ethereum data** fetched directly from the official Aave API — no API key required.

When live mode is selected, `fetch_live.py` queries the Aave V3 GraphQL endpoint for current reserve parameters across all active Ethereum markets:

| Parameter | Source |
|---|---|
| Liquidation threshold (per asset) | `supplyInfo.liquidationThreshold` |
| Liquidation bonus (per asset) | `supplyInfo.liquidationBonus` |
| Total supply and debt (USD) | `supplyInfo.total`, `borrowInfo.total` |
| Available stablecoin liquidity | `borrowInfo.availableLiquidity` |

From these, `fetch_live.py` generates 1,000 synthetic positions weighted by each reserve's share of total protocol debt. Health factors are drawn from a log-normal calibrated to match Aave's published risk dashboard. This means the cascade simulation runs on the actual current distribution of protocol fragility rather than a fixed historical snapshot.

**Fallback behaviour:** if the API is unreachable (network timeout, endpoint change), the dashboard automatically falls back to the synthetic pool and displays a status message. The synthetic pool is independently calibrated to Aave V3 Ethereum statistics as of March 2026 (DeFiLlama, Aave app).

To toggle between data sources, use the **Data Source** radio button at the top of the dashboard.

---

## The conservation law

The central theoretical result is a **conservation law** (Mishricky 2025, Footnote 28):

```
MSE · F = (κ / φᵐ)²
```

This says that the product of price dispersion (MSE) and flash crash probability (F) is determined entirely by the ratio of posting costs to liquidity. The practical implication: **a protocol can show low F while generating massive bad debt**, because the bad debt raises MSE even as F appears controlled. The 40% and 50% drop scenarios in this simulator reproduce exactly that pattern — $48M and $174M in bad debt respectively, both with F remaining in the STABLE range. Standard risk monitors would not flag either scenario as critical.

---

## Theoretical framework

The model implements the equilibrium from Mishricky (2025), which combines the Burdett-Judd (1983) price posting framework with the Lagos-Wright (2005) monetary model:

| Theoretical parameter | DeFi observable |
|---|---|
| κ (quote posting cost) | Gas cost per liquidation (USD) |
| φᵐ (real value of money) | Stablecoin liquidity depth / total protocol debt |
| Γ (book width) | Daily volatility of collateral asset |

**Core results implemented:**

- Flash crash probability: `F = e^(-θ)` where `θ = ln(φᵐ Γ / κ)`
- Proposition 1: Equilibrium bid/ask distributions A(p) and B(p)
- Proposition 11: F is increasing in κ and decreasing in φᵐ
- Proposition 12: Speculative premium
- Conservation law: `MSE · F = (κ / φᵐ)²`

**Interpreting F:** F is a *per-cycle* probability — the chance that no competitive quote is posted in a single auction round. Small per-cycle values compound rapidly across the thousands of quote cycles that occur each day. At 1,000 quote cycles per hour, the daily flash crash probability `1 - (1-F)^24000` is approximately 70% at the STABLE boundary (F = 0.00005) and effectively 100% at the CRITICAL boundary (F = 0.00050). The thresholds therefore reflect meaningfully distinct risk regimes despite the small absolute values of F.

Market status thresholds calibrated to a $1.8B Aave V3-equivalent pool:

| Status | F threshold | What it means |
|---|---|---|
| STABLE | F < 0.00005 | Competitive liquidation market, low flash crash risk |
| ELEVATED RISK | 0.00005 ≤ F < 0.00050 | Liquidity thinning, market quality degrading |
| CRITICAL | F ≥ 0.00050 | Near-certain daily flash crash, protocol solvency at risk |

---

## Crisis scenario presets

| Preset | Price drop | Liquidity | Gas | What it models |
|---|---|---|---|---|
| Normal market | 30% | 40% | $80 | Baseline conditions. Starts STABLE, deteriorates to ELEVATED RISK as cascade drains liquidity — this is the hidden fragility the model detects. |
| Liquidity crisis | 30% | 3% | $80 | Capital has fled the protocol before the shock hits. φᵐ is already near zero, so the market is fragile from round one. |
| Gas spike | 30% | 40% | $450 | κ shock. Gas cost raised to $450, raising F and degrading market quality. Large positions remain profitable to liquidate — the scenario demonstrates how κ elevates flash crash risk even without generating meaningful bad debt. |
| Combined shock | 45% | 5% | $400 | Both liquidity and gas fail simultaneously. Analogue of March 2020 (COVID) or November 2022 (FTX collapse). |

---

## Project structure

```
defi-liquidation-sim/
├── fetch_aave.py          # Synthetic pool calibrated to Aave V3 Ethereum (offline)
├── fetch_live.py          # Live Aave V3 data via official GraphQL API (no key required)
├── agents.py              # BorrowerAgent class with health factor calculation
├── simulate.py            # Cascade loop with bad debt accounting and theory scoring
├── theory.py              # BurdettJuddDeFi class implementing Mishricky (2025)
├── dashboard.py           # Plotly Dash interactive dashboard
├── test_theory.py         # Gas/liquidity stress tests
├── test_distributions.py  # Bid/ask distribution plots under stress
└── test_speculation.py    # Speculative premium analysis
```

---

## Getting started

**Requirements:** Python 3.9+

```bash
pip install dash plotly pandas numpy scipy requests
```

### Run the dashboard

```bash
python dashboard.py
```

Open [http://127.0.0.1:8050](http://127.0.0.1:8050) in your browser.

Use the **Data Source** toggle to switch between synthetic and live Aave V3 data. Live mode fetches current reserve parameters from the Aave API; if the connection fails it falls back to synthetic automatically.

### Run the simulation directly

```bash
python simulate.py
```

Runs five benchmark scenarios (10%–50% price drops) and prints round-by-round cascade results with theoretical scores.

### Test the live data feed

```bash
python fetch_live.py
```

Fetches current Aave V3 Ethereum reserve data and prints a summary of active reserves, total protocol TVL, and a sample of the generated position pool.

### Run theory tests

```bash
python test_theory.py        # Stress test F across gas/liquidity parameter space
python test_distributions.py # Plot bid/ask distributions under each crisis preset
python test_speculation.py   # Analyse speculative premium (Proposition 12)
```

---

## Dashboard features

- **Data source toggle** — switch between live Aave V3 API data and synthetic offline pool
- **4 crisis scenario presets** with one-click switching
- **Interactive sliders** for price drop (5–60%), liquidity (1–80%), and gas cost ($20–500)
- **10 summary statistics** including bad debt, cascade rounds, initial/final F, and market status
- **4 charts:** liquidation cascade by round, θ and F evolution, equilibrium bid/ask distributions, full stress test

---

## Limitations and extensions

The conservation law result — that bad debt and flash crash risk can decouple, with F elevated and the liquidation market near-collapse even when bad debt is minimal — is reproduced consistently across scenarios. This is the failure mode that standard risk monitors miss: a protocol can look solvent while the market mechanism that keeps it solvent is breaking down. The live data feed provides current reserve parameters, but full quantitative validation of F against empirical liquidation gap frequency requires historical time-series of on-chain liquidation events (March 2020, November 2022) which are not yet integrated.

The position pool is also single-asset by construction — each simulated borrower is assigned to one reserve. Real Aave positions are often multi-collateral, which affects both health factor dynamics and the liquidation incentive calculation.

Natural extensions:
- Historical backtesting of F against on-chain liquidation gap events
- Multi-collateral positions (mixed ETH/wBTC/stablecoin collateral)
- Cross-protocol contagion (Aave ↔ Compound ↔ Morpho)
- Endogenous gas pricing during network congestion

---

## Reference

Mishricky, S. (2025). *Asset Price Dispersion, Monetary Policy and Macroprudential Regulation*. Working paper, Australian National University.
