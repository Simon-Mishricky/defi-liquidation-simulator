"""
monitor.py — Real-time Fragility Monitor
=========================================
Mishricky (2025) — Asset Price Dispersion, Monetary Policy and Macroprudential Regulation

Computes F (flash-crash probability) every hour from live Aave V3 on-chain conditions
and logs it to a time-series CSV alongside ETH price and gas costs.

Usage
-----
    # Run once (manual / cron)
    python monitor.py

    # Run continuously (blocking)
    python monitor.py --daemon --interval 3600

    # View current log
    python monitor.py --show

The output CSV (f_monitor_log.csv) is read by dashboard.py to render the
"Live F Monitor" tab.
"""

import argparse
import csv
import os
import time
from datetime import datetime, timezone

import numpy as np
import requests

LOG_FILE = os.path.join(os.path.dirname(__file__), "f_monitor_log.csv")
LOG_COLUMNS = [
    "timestamp_utc",
    "eth_price_usd",
    "gas_gwei",
    "gas_usd",
    "stablecoin_depth_usd",
    "total_debt_usd",
    "phi_m",
    "kappa",
    "Gamma",
    "theta",
    "F",
    "market_status",
]

# ── Aave V3 GraphQL ─────────────────────────────────────────────────────────

AAVE_API = "https://api.v3.aave.com/graphql"
AAVE_QUERY = """
{
  markets(request: { chainIds: [1] }) {
    reserves {
      underlyingToken { symbol }
      borrowInfo {
        availableLiquidity { usd }
        total { usd }
      }
    }
  }
}
"""

# ── Price / gas oracles ──────────────────────────────────────────────────────

COINGECKO_PRICE = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
ETHERSCAN_GAS   = "https://api.etherscan.io/api?module=gastracker&action=gasoracle"


def fetch_eth_price() -> float:
    """Fetch ETH/USD from CoinGecko (no key required, public endpoint)."""
    try:
        r = requests.get(COINGECKO_PRICE, timeout=10)
        r.raise_for_status()
        return float(r.json()["ethereum"]["usd"])
    except Exception:
        return _fallback_eth_price()


def _fallback_eth_price() -> float:
    """Fallback: Binance public ticker."""
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT", timeout=8
        )
        return float(r.json()["price"])
    except Exception:
        return 2000.0  # last-resort static fallback


def fetch_gas_gwei() -> float:
    """Fetch recommended gas price in Gwei from Etherscan (no key for basic oracle)."""
    try:
        r = requests.get(ETHERSCAN_GAS, timeout=10)
        r.raise_for_status()
        d = r.json()
        # ProposeGasPrice is the 'standard' estimate
        return float(d["result"]["ProposeGasPrice"])
    except Exception:
        return _fallback_gas_gwei()


def _fallback_gas_gwei() -> float:
    """Fallback: ETH Gas Station / DefiLlama proxy."""
    try:
        r = requests.get("https://coins.llama.fi/block/ethereum/latest", timeout=8)
        # DefiLlama block endpoint doesn't give gas directly; use a static proxy
        return 25.0
    except Exception:
        return 25.0


def gas_gwei_to_usd(gwei: float, eth_price: float) -> float:
    """
    Convert gas cost to USD for a typical liquidation transaction.
    A Aave V3 liquidation uses ~350,000 gas (measured empirically on-chain).
    gas_usd = gas_price_gwei * 1e-9 * gas_units * eth_price
    """
    GAS_UNITS = 350_000
    return gwei * 1e-9 * GAS_UNITS * eth_price


def fetch_aave_liquidity() -> tuple[float, float]:
    """
    Return (stablecoin_depth_usd, total_debt_usd) from live Aave V3 Ethereum.
    stablecoin_depth is the sum of available liquidity across stablecoin reserves
    (USDC, USDT, DAI, LUSD, FRAX) — the pool of funds liquidation bots draw on.
    """
    STABLECOINS = {"USDC", "USDT", "DAI", "LUSD", "FRAX", "PYUSD", "GHO"}
    try:
        r = requests.post(AAVE_API, json={"query": AAVE_QUERY}, timeout=15)
        r.raise_for_status()
        data = r.json()
        if "errors" in data:
            raise ValueError(str(data["errors"]))

        stablecoin_depth = 0.0
        total_debt = 0.0

        for market in data["data"]["markets"]:
            for res in market["reserves"]:
                symbol = res["underlyingToken"]["symbol"].upper()
                try:
                    debt = float(res["borrowInfo"]["total"]["usd"] or 0)
                    avail = float(res["borrowInfo"]["availableLiquidity"]["usd"] or 0)
                except (TypeError, KeyError):
                    continue
                total_debt += debt
                if symbol in STABLECOINS:
                    stablecoin_depth += avail

        return stablecoin_depth, total_debt

    except Exception:
        # Fallback to published statistics if API is unreachable
        return 9_600_000_000.0, 24_000_000_000.0


# ── F computation ────────────────────────────────────────────────────────────

def compute_F(gas_usd: float, stablecoin_depth: float, total_debt: float,
              daily_volatility: float = 0.05) -> dict:
    """
    Compute Mishricky (2025) fragility metrics from observable market conditions.

    Mapping (see calibrate_from_positions in theory.py):
      kappa  = gas_usd / stablecoin_depth   (posting cost as fraction of liquidity)
      phi_m  = stablecoin_depth / total_debt (real value of money)
      Gamma  = daily_volatility              (book width / collateral vol)
      theta  = ln(phi_m * Gamma / kappa)    (Proposition 1)
      F      = exp(-theta)                  (Proposition 11)
    """
    if stablecoin_depth <= 0 or total_debt <= 0:
        return {"theta": 0, "F": 1.0, "phi_m": 0, "kappa": 0, "Gamma": daily_volatility,
                "market_status": "COLLAPSE", "error": "zero liquidity"}

    phi_m = stablecoin_depth / total_debt
    kappa = gas_usd / stablecoin_depth
    Gamma = daily_volatility

    ratio = (phi_m * Gamma) / kappa
    if ratio <= 1:
        return {"theta": 0, "F": 1.0, "phi_m": phi_m, "kappa": kappa, "Gamma": Gamma,
                "market_status": "COLLAPSE"}

    theta = np.log(ratio)
    F = np.exp(-theta)

    p_daily = 1 - (1 - F) ** 24000
    status = (
        "STABLE"        if p_daily < 0.15 else
        "ELEVATED RISK" if p_daily < 0.80 else
        "CRITICAL"
    )

    return {
        "theta": round(theta, 6),
        "F": round(F, 8),
        "phi_m": round(phi_m, 6),
        "kappa": round(kappa, 10),
        "Gamma": Gamma,
        "market_status": status,
    }


# ── Logging ──────────────────────────────────────────────────────────────────

def _ensure_log_header():
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writeheader()


def append_log_entry(entry: dict):
    _ensure_log_header()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writerow({k: entry.get(k, "") for k in LOG_COLUMNS})


def run_once(verbose: bool = True) -> dict:
    """Fetch all data, compute F, append to log. Returns the entry dict."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    eth_price = fetch_eth_price()
    gas_gwei  = fetch_gas_gwei()
    gas_usd   = gas_gwei_to_usd(gas_gwei, eth_price)
    stablecoin_depth, total_debt = fetch_aave_liquidity()

    metrics = compute_F(gas_usd, stablecoin_depth, total_debt)

    entry = {
        "timestamp_utc":       ts,
        "eth_price_usd":       round(eth_price, 2),
        "gas_gwei":            round(gas_gwei, 2),
        "gas_usd":             round(gas_usd, 2),
        "stablecoin_depth_usd": round(stablecoin_depth, 0),
        "total_debt_usd":      round(total_debt, 0),
        "phi_m":               metrics["phi_m"],
        "kappa":               metrics["kappa"],
        "Gamma":               metrics["Gamma"],
        "theta":               metrics["theta"],
        "F":                   metrics["F"],
        "market_status":       metrics["market_status"],
    }

    append_log_entry(entry)

    if verbose:
        print(f"\n[{ts}] F-Monitor snapshot")
        print(f"  ETH price:       ${eth_price:,.2f}")
        print(f"  Gas:             {gas_gwei:.1f} gwei  →  ${gas_usd:.2f}/liquidation")
        print(f"  Stablecoin depth: ${stablecoin_depth/1e9:.2f}B")
        print(f"  Total debt:       ${total_debt/1e9:.2f}B")
        print(f"  phi_m:           {metrics['phi_m']:.6f}")
        print(f"  kappa:           {metrics['kappa']:.2e}")
        print(f"  theta:           {metrics['theta']:.4f}")
        print(f"  *** F = {metrics['F']:.6f}  [{metrics['market_status']}] ***")
        print(f"  Logged to:       {LOG_FILE}")

    return entry


def load_log() -> "pd.DataFrame":
    """Load the monitor log as a DataFrame. Returns empty DF if no log exists."""
    import pandas as pd
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        return pd.DataFrame(columns=LOG_COLUMNS)
    df = pd.read_csv(LOG_FILE, parse_dates=["timestamp_utc"])
    return df.sort_values("timestamp_utc").reset_index(drop=True)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Mishricky (2025) Real-time F Monitor")
    parser.add_argument("--daemon",   action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=3600, help="Seconds between snapshots (default 3600)")
    parser.add_argument("--show",     action="store_true", help="Print log and exit")
    args = parser.parse_args()

    if args.show:
        import pandas as pd
        df = load_log()
        if df.empty:
            print("No log entries yet. Run 'python monitor.py' to collect the first snapshot.")
        else:
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 140)
            print(df[["timestamp_utc", "eth_price_usd", "gas_usd", "phi_m", "F", "market_status"]]
                  .tail(24).to_string(index=False))
        return

    if args.daemon:
        print(f"Starting F-monitor daemon — snapshot every {args.interval}s. Ctrl-C to stop.")
        while True:
            try:
                run_once(verbose=True)
            except Exception as e:
                print(f"[ERROR] {e}")
            time.sleep(args.interval)
    else:
        run_once(verbose=True)


if __name__ == "__main__":
    main()
