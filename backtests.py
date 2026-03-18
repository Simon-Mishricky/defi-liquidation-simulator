"""
backtests.py — Multi-Event Crisis Backtest Registry
=====================================================
Mishricky (2025) — Asset Price Dispersion, Monetary Policy and Macroprudential Regulation

Provides reconstructed pool states and F timelines for every major DeFi flash crash / 
liquidation cascade event in the Ethereum ecosystem since 2020.

Events covered
--------------
1. BLACK THURSDAY (Mar 2020)  — COVID crash, MakerDAO / early Compound V2
2. FTX COLLAPSE (Nov 2022)    — Exchange contagion, Aave V2
3. USDC DEPEG (Mar 2023)      — Silicon Valley Bank, Aave V3 (first V3 event)

Note: Only events with genuine sourced daily data are included. LUNA/UST (May 2022)
and Celsius/3AC (Jun 2022) are excluded because granular daily stablecoin depth,
utilisation, and gas figures are not publicly available without paid API access
(DeFiLlama Pro, Dune Analytics). The FTX event retains its original backtest_ftx.py
sourcing; Black Thursday and USDC Depeg are reconstructed from primary sources.

Protocol notes
--------------
- Aave V2 Ethereum launched December 2020. Events before this date use approximate
  Compound V2 / MakerDAO parameters recalibrated to the Mishricky (2025) framework
  (φᵐ, κ, Γ are protocol-agnostic).
- Aave V3 Ethereum launched January 27, 2023. The USDC depeg (Mar 2023) is the
  first event that runs natively on V3 parameters.
- All figures are reconstructed from: DeFiLlama TVL series, Dune Analytics,
  Etherscan gas oracle historical exports, CoinGecko OHLCV, Messari Research,
  Rekt.news post-mortems, and archived risk dashboard snapshots.

Usage
-----
    from backtests import EVENTS, get_event, build_timeline, build_positions

    timeline = build_timeline("ftx_2022")
    positions = build_positions("ftx_2022")
"""

import numpy as np
import pandas as pd

# ── Event registry ─────────────────────────────────────────────────────────────
#
# Each entry defines the reconstructed pool state at the START of the crisis window
# (i.e., before the worst price action, to test whether F signalled early).

EVENTS = {

    # ── 1. Black Thursday — March 2020 ────────────────────────────────────────
    "black_thursday_2020": {
        "label":       "Black Thursday (Mar 2020)",
        "short_label": "Mar 2020",
        "date":        "2020-03-12",
        "protocol":    "MakerDAO / Compound V2",
        "description": (
            "COVID-19 market panic. ETH dropped ~43% in 24 hours on March 12, 2020. "
            "MakerDAO liquidation auctions cleared at near-zero bids as gas spiked and "
            "keeper bots failed to submit competitive quotes — the textbook flash crash. "
            "$5.7M in bad debt was created in MakerDAO's ETH-A vault. Compound V2 "
            "simultaneously experienced ~$3M in protocol losses. This is the historical "
            "prototype for the F=1 regime. Note: Aave V2 had not yet launched; "
            "parameters are calibrated to MakerDAO / Compound V2 conditions."
        ),
        "eth_price_open_usd":    194.0,
        "eth_price_close_usd":   112.0,
        "eth_price_low_usd":      90.0,
        "realised_price_drop_pct": 0.430,
        "total_supplied_usd":    820_000_000,
        "total_borrowed_usd":    210_000_000,
        "utilisation_rate":      0.256,
        "stablecoin_depth_usd":   8_000_000,   # near zero — keepers couldn't source DAI
        "gas_usd":                25.0,           # ~200 gwei mean * 350k gas * ~$0.00036/gwei; keepers paid $25–$140/liq
        "daily_volatility":      0.145,          # 14.5% daily vol — extreme
        "hf_pct_below_1_2":      0.180,          # severe tail risk pre-crash
        "hf_median":             1.55,
        "n_borrowers_estimated": 2_400,
        "key_event_date":        "2020-03-12",
        "key_event_label":       "COVID panic — ETH −43%",
        "actual_outcomes": {
            "total_liquidations_usd": 8_320_000,   # Blocknative: $8.32M liquidated at zero-bid
            "bad_debt_created_usd":   4_500_000,   # MakerDAO post-mortem: $4.5M DAI uncollateralised
            "peak_gas_gwei":           400,        # Glassnode: mean peaked ~200 gwei Mar 12; to 400 gwei Mar 13
            "source": "MakerDAO governance post-mortem / DeFiPulse",
        },
    },

    # ── 2. FTX Collapse — November 2022 ────────────────────────────────────────
    "ftx_2022": {
        "label":       "FTX Collapse (Nov 2022)",
        "short_label": "Nov 2022",
        "date":        "2022-11-08",
        "protocol":    "Aave V2 Ethereum",
        "description": (
            "Binance signed a tentative letter of intent to acquire FTX on November 8, "
            "2022 (later withdrawn Nov 9). ETH fell ~20% from $1,358 (Nov 2) to $1,080 "
            "(Nov 12). Gas spiked as panic trading hit the network. Stablecoin liquidity "
            "drained as borrowers rushed to deleverage. Aave V2 experienced ~$210M in "
            "liquidations with $4.6M bad debt. Note: Aave V3 did not deploy on Ethereum "
            "until January 2023; the FTX-era market ran V2."
        ),
        "eth_price_open_usd":    1_358.0,
        "eth_price_close_usd":   1_245.0,
        "eth_price_low_usd":     1_080.0,
        "realised_price_drop_pct": 0.205,
        "total_supplied_usd":    6_200_000_000,
        "total_borrowed_usd":    2_900_000_000,
        "utilisation_rate":      0.468,
        "stablecoin_depth_usd":  180_000_000,
        "gas_usd":                75.0,
        "daily_volatility":       0.055,
        "hf_pct_below_1_2":       0.082,
        "hf_median":              1.48,
        "n_borrowers_estimated":  12_000,
        "key_event_date":         "2022-11-08",
        "key_event_label":        "Binance no-rescue announcement",
        "actual_outcomes": {
            "total_liquidations_usd": 210_000_000,
            "bad_debt_created_usd":     4_600_000,
            "peak_gas_gwei":          118,
            "source": "Messari Research / Dune Analytics / Rekt.news",
        },
    },

    # ── 5. USDC Depeg — March 2023 ────────────────────────────────────────────
    "usdc_depeg_2023": {
        "label":       "USDC Depeg / SVB (Mar 2023)",
        "short_label": "Mar 2023",
        "date":        "2023-03-10",
        "protocol":    "Aave V3 Ethereum",
        "description": (
            "Silicon Valley Bank collapsed on March 10, 2023. Circle held $3.3B of "
            "USDC reserves at SVB, triggering a USDC depeg to $0.87. ETH fell ~14% "
            "as panic spread. The depeg created an unusual φᵐ shock: nominally 'stable' "
            "collateral was repriced, compressing liquidation thresholds for USDC-backed "
            "positions. Aave V3 (launched Jan 2023) experienced ~$80M in liquidations. "
            "The event is notable for the speed of recovery once the Fed backstop was "
            "announced (Mar 12): φᵐ recovered within 48 hours, and F returned to "
            "STABLE without a sustained cascade."
        ),
        "eth_price_open_usd":    1_640.0,
        "eth_price_close_usd":   1_440.0,
        "eth_price_low_usd":     1_385.0,
        "realised_price_drop_pct": 0.156,
        "total_supplied_usd":    5_800_000_000,
        "total_borrowed_usd":    2_100_000_000,
        "utilisation_rate":      0.362,
        "stablecoin_depth_usd":  145_000_000,    # USDC liquidity severely impaired
        "gas_usd":                55.0,
        "daily_volatility":       0.048,
        "hf_pct_below_1_2":       0.058,
        "hf_median":              1.55,
        "n_borrowers_estimated":  9_500,
        "key_event_date":         "2023-03-10",
        "key_event_label":        "SVB collapse / USDC depeg",
        "actual_outcomes": {
            "total_liquidations_usd":  24_000_000,  # Kraken/Aave: ~3,400 liquidations, $24M collateral
            "bad_debt_created_usd":       300_000,   # Chaos Labs: small bad debt, quick FDIC recovery
            "peak_gas_gwei":           231,          # Nansen/CoinDesk: median peaked ~231 gwei Mar 11
            "source": "Messari Research / Aave V3 risk dashboard / DeFiLlama",
        },
    },
}


# ── Per-event timeline data ────────────────────────────────────────────────────
#
# Each timeline covers a 15-day window around the crisis.
# Columns: date, eth, gas_gwei, total_supplied_B, utilisation, stablecoin_depth_usd, note

_TIMELINES_RAW = {

    "black_thursday_2020": [
        # date,         eth,   gas_gwei, supplied_B, util,  depth,       note
        ("2020-03-06",  253,    8,  0.82, 0.230,  22e6,  "Pre-crisis"),
        ("2020-03-07",  248,    8,  0.82, 0.232,  21e6,  ""),
        ("2020-03-08",  233,   10,  0.81, 0.238,  20e6,  "Risk-off begins"),
        ("2020-03-09",  215,   14,  0.80, 0.244,  18e6,  "WHO declares pandemic"),
        ("2020-03-10",  210,   18,  0.79, 0.248,  16e6,  ""),
        ("2020-03-11",  200,   25,  0.78, 0.252,  13e6,  ""),
        ("2020-03-12",  112,  200, 0.75, 0.256,   8e6,  "Black Thursday — ETH −43%"),  # Glassnode: mean ~200 gwei
        ("2020-03-13",   90,  400, 0.68, 0.310,   3e6,  "Peak liquidations — keeper failure"),  # Glassnode: peaked ~400 gwei Mar 13
        ("2020-03-14",  125,  420,  0.70, 0.280,   6e6,  "Partial recovery"),
        ("2020-03-15",  135,  180,  0.71, 0.265,   9e6,  ""),
        ("2020-03-16",  130,  120,  0.72, 0.260,  11e6,  ""),
        ("2020-03-17",  140,   90,  0.73, 0.255,  13e6,  ""),
        ("2020-03-18",  155,   65,  0.74, 0.248,  15e6,  ""),
        ("2020-03-19",  148,   55,  0.75, 0.244,  16e6,  ""),
        ("2020-03-20",  150,   45,  0.76, 0.240,  17e6,  "Stabilising"),
    ],

    "ftx_2022": [
        # date,         eth,   gas_gwei, supplied_B, util,  depth,         note
        ("2022-11-01", 1580,   22,  6.8, 0.430, 380e6, "Pre-CoinDesk"),
        ("2022-11-02", 1570,   24,  6.8, 0.435, 360e6, "CoinDesk story drops"),
        ("2022-11-03", 1580,   21,  6.7, 0.438, 355e6, ""),
        ("2022-11-04", 1640,   20,  6.8, 0.432, 370e6, "FTT recovery attempt"),
        ("2022-11-05", 1600,   22,  6.7, 0.440, 350e6, ""),
        ("2022-11-06", 1545,   35,  6.5, 0.455, 290e6, "Binance sells FTT"),
        ("2022-11-07", 1450,   42,  6.3, 0.462, 240e6, "Tension building"),
        ("2022-11-08", 1245,   85,  6.2, 0.468, 180e6, "Binance no rescue — cascade begins"),
        ("2022-11-09", 1195,  118,  5.8, 0.491, 120e6, "Peak liquidations"),
        ("2022-11-10", 1220,   95,  5.6, 0.480, 150e6, "Partial recovery"),
        ("2022-11-11", 1120,   72,  5.4, 0.470, 170e6, "FTX files bankruptcy"),
        ("2022-11-12", 1080,   58,  5.2, 0.460, 190e6, "ETH local bottom"),
        ("2022-11-13", 1150,   45,  5.3, 0.452, 220e6, ""),
        ("2022-11-14", 1230,   38,  5.5, 0.445, 260e6, "Recovery begins"),
        ("2022-11-15", 1260,   32,  5.6, 0.440, 300e6, ""),
    ],

    "usdc_depeg_2023": [
        # date,         eth,   gas_gwei, supplied_B, util,  depth,         note
        ("2023-03-05", 1660,   22,  5.9, 0.352, 210e6, "Pre-crisis"),
        ("2023-03-06", 1645,   24,  5.9, 0.354, 205e6, ""),
        ("2023-03-07", 1635,   26,  5.8, 0.356, 200e6, ""),
        ("2023-03-08", 1620,   28,  5.8, 0.358, 195e6, "SVB news breaks"),
        ("2023-03-09", 1580,   35,  5.8, 0.360, 182e6, "Bank run accelerates"),
        ("2023-03-10", 1520,   55,  5.7, 0.362, 165e6, "SVB collapses — USDC depeg"),
        ("2023-03-11", 1440,  231,  5.6, 0.365, 145e6, "USDC hits $0.87 — peak pressure"),  # Nansen: 231 gwei median
        ("2023-03-12", 1385,   95,  5.5, 0.368, 135e6, "Fed backstop announced"),  # Gas elevated but declining
        ("2023-03-13", 1460,   52,  5.6, 0.364, 155e6, "USDC re-pegs to $0.98"),
        ("2023-03-14", 1540,   42,  5.7, 0.360, 170e6, "Relief rally"),
        ("2023-03-15", 1600,   35,  5.8, 0.356, 182e6, ""),
        ("2023-03-16", 1620,   30,  5.8, 0.354, 188e6, ""),
        ("2023-03-17", 1660,   28,  5.9, 0.352, 194e6, ""),
        ("2023-03-18", 1700,   25,  5.9, 0.350, 200e6, ""),
        ("2023-03-19", 1740,   24,  6.0, 0.348, 205e6, "Full recovery"),
    ],
}


# ── Core builder functions ──────────────────────────────────────────────────────

def build_timeline(event_key: str) -> pd.DataFrame:
    """
    Build the daily F time series for the given event.

    Returns a DataFrame with columns:
        date, eth_price_usd, gas_gwei, gas_usd, total_supplied_usd,
        total_debt_usd, stablecoin_depth_usd, Gamma, phi_m, kappa,
        theta, F, market_status, note
    """
    raw = _TIMELINES_RAW[event_key]
    GAS_UNITS = 350_000
    WINDOW = 5

    eth_series = [d[1] for d in raw]
    log_rets = [np.nan] + [
        np.log(eth_series[i] / eth_series[i - 1]) for i in range(1, len(eth_series))
    ]
    gammas = []
    for i in range(len(eth_series)):
        start = max(0, i - WINDOW + 1)
        window_rets = [log_rets[j] for j in range(start, i + 1) if not np.isnan(log_rets[j])]
        gammas.append(float(np.std(window_rets, ddof=1)) if len(window_rets) >= 2 else 0.055)

    rows = []
    for i, (date, eth, gas_gwei, supplied_B, util, stablecoin_depth, note) in enumerate(raw):
        total_supplied = supplied_B * 1e9
        total_debt = total_supplied * util
        gas_usd = gas_gwei * 1e-9 * GAS_UNITS * eth
        Gamma = gammas[i]

        phi_m = stablecoin_depth / total_debt
        kappa = gas_usd / stablecoin_depth

        ratio = (phi_m * Gamma) / kappa
        if ratio > 1:
            theta = np.log(ratio)
            F = np.exp(-theta)
        else:
            theta = 0.0
            F = 1.0

        # Status thresholds based on daily flash-crash probability
        # P(fc|24h) = 1 - (1-F)^24000. These thresholds are chosen for the
        # backtest's daily-resolution timeline charts, where the y-axis shows
        # P(fc|24h) and horizontal reference lines sit at 15% and 80%.
        # The simulator tab uses per-cycle F thresholds (theory.py summary());
        # both are valid — they simply present the same F at different time scales.
        p_daily = 1 - (1 - F) ** 24000
        status = (
            "STABLE"        if p_daily < 0.15 else
            "ELEVATED RISK" if p_daily < 0.80 else
            "CRITICAL"
        )

        rows.append({
            "date":                 date,
            "eth_price_usd":        eth,
            "gas_gwei":             gas_gwei,
            "gas_usd":              round(gas_usd, 2),
            "total_supplied_usd":   total_supplied,
            "total_debt_usd":       total_debt,
            "stablecoin_depth_usd": stablecoin_depth,
            "Gamma":                round(Gamma, 6),
            "phi_m":                round(phi_m, 6),
            "kappa":                round(kappa, 12),
            "theta":                round(theta, 4),
            "F":                    round(F, 8),
            "market_status":        status,
            "note":                 note,
        })

    return pd.DataFrame(rows)


def build_positions(event_key: str, n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic position pool calibrated to the event's pool state.
    """
    state = EVENTS[event_key]
    rng = np.random.default_rng(seed)

    avg_debt = state["total_borrowed_usd"] / state["n_borrowers_estimated"]
    debt_vals = rng.lognormal(mean=np.log(avg_debt), sigma=1.5, size=n)
    debt_vals = np.clip(debt_vals, 500, avg_debt * 100)

    n_bulk = int(n * (1 - state["hf_pct_below_1_2"]))
    n_tail = n - n_bulk

    hf_bulk = rng.lognormal(mean=np.log(state["hf_median"]), sigma=0.35, size=n_bulk)
    hf_bulk = np.clip(hf_bulk, 1.21, 8.0)

    hf_tail = rng.lognormal(mean=np.log(1.08), sigma=0.08, size=n_tail)
    hf_tail = np.clip(hf_tail, 1.001, 1.2)

    hf_vals = np.concatenate([hf_bulk, hf_tail])
    rng.shuffle(hf_vals)

    LT = 0.825
    LB = 0.05
    collateral_vals = (hf_vals * debt_vals) / LT

    actual_util = debt_vals.sum() / collateral_vals.sum()
    if actual_util > 0:
        debt_vals = debt_vals * (state["utilisation_rate"] / actual_util)

    return pd.DataFrame({
        "collateral_usd":        collateral_vals,
        "debt_usd":              debt_vals,
        "health_factor":         hf_vals,
        "liquidation_threshold": LT,
        "liq_bonus":             LB,
    })


def run_backtest(event_key: str, n_positions: int = 1000, verbose: bool = False):
    """
    Run the cascade simulation for a given event.

    Returns (results_df, agents, pre_crash_dict, summary_dict)
    """
    from theory import calibrate_from_positions
    from simulate import run_cascade

    state = EVENTS[event_key]
    positions = build_positions(event_key, n=n_positions)

    metrics = calibrate_from_positions(
        positions,
        gas_usd=state["gas_usd"],
        stablecoin_depth_usd=state["stablecoin_depth_usd"],
        daily_volatility=state["daily_volatility"],
    )
    pre_crash = metrics.summary()
    pre_crash["hf_tail_pct"] = round((positions["health_factor"] < 1.2).mean(), 4)

    results, agents = run_cascade(
        price_drop_pct=state["realised_price_drop_pct"],
        n_positions=n_positions,
        gas_usd=state["gas_usd"],
        initial_liquidity_pct=(
            state["stablecoin_depth_usd"] / state["total_borrowed_usd"]
        ),
        daily_volatility=state["daily_volatility"],
        use_feedback=True,
        rng_seed=42,
    )

    total_liquidated = sum(1 for a in agents if a.liquidated)
    total_bad_debt = results["bad_debt_usd"].sum() if len(results) > 0 else 0
    final_F = results["F (crash prob)"].iloc[-1] if len(results) > 0 else pre_crash["flash crash prob (F)"]
    final_status = results["market_status"].iloc[-1] if len(results) > 0 else "STABLE"

    scale_factor = state["total_borrowed_usd"] / positions["debt_usd"].sum()
    sim_liquidations_usd = (results["liquidation_vol_usd"].sum() if len(results) > 0 else 0) * scale_factor
    sim_bad_debt_usd = total_bad_debt * scale_factor

    summary = {
        "event_key":                event_key,
        "event_label":              state["label"],
        "backtest_date":            state["date"],
        "realised_price_drop":      state["realised_price_drop_pct"],
        "pre_cascade_F":            pre_crash["flash crash prob (F)"],
        "pre_cascade_status":       pre_crash["market status"],
        "pre_cascade_theta":        pre_crash["theta (quote intensity)"],
        "final_F":                  final_F,
        "final_status":             final_status,
        "cascade_rounds":           len(results),
        "positions_liquidated_sim": total_liquidated,
        "sim_liquidation_vol_usd":  round(sim_liquidations_usd, 0),
        "sim_bad_debt_usd":         round(sim_bad_debt_usd, 0),
        "actual_liquidations_usd":  state["actual_outcomes"]["total_liquidations_usd"],
        "actual_bad_debt_usd":      state["actual_outcomes"]["bad_debt_created_usd"],
    }

    return results, agents, pre_crash, summary


def get_event(event_key: str) -> dict:
    return EVENTS[event_key]


# ── Convenience: dropdown options for Dash ─────────────────────────────────────

DROPDOWN_OPTIONS = [
    {"label": v["label"], "value": k}
    for k, v in EVENTS.items()
]
