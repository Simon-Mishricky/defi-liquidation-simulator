import os
import requests
import pandas as pd
import numpy as np

AAVE_API = "https://api.v3.aave.com/graphql"

QUERY = """
{
  markets(request: { chainIds: [1] }) {
    name
    reserves {
      underlyingToken { symbol decimals }
      supplyInfo {
        liquidationThreshold { value }
        liquidationBonus { value }
        maxLTV { value }
        total { value }
      }
      borrowInfo {
        availableLiquidity { usd }
        total { usd }
      }
    }
  }
}
"""


def fetch_live_positions(n_sample: int = 1000) -> pd.DataFrame:
    """
    Fetch live Aave V3 Ethereum reserve data from the official Aave API
    (no API key required) and generate a representative synthetic position
    pool calibrated to real on-chain parameters.

    Returns a DataFrame with columns:
        collateral_usd, debt_usd, health_factor, liquidation_threshold, liq_bonus
    matching the format of generate_aave_positions().
    """
    r = requests.post(AAVE_API, json={"query": QUERY}, timeout=15)
    r.raise_for_status()
    data = r.json()

    if "errors" in data:
        raise ValueError(f"Aave API error: {data['errors']}")

    # Parse reserves across all Ethereum markets
    reserves = []
    for market in data["data"]["markets"]:
        for res in market["reserves"]:
            try:
                liq_threshold = float(res["supplyInfo"]["liquidationThreshold"]["value"])
                liq_bonus     = float(res["supplyInfo"]["liquidationBonus"]["value"])
                total_supply  = float(res["supplyInfo"]["total"]["value"] or 0)
                total_debt    = float(res["borrowInfo"]["total"]["usd"] or 0)
                avail_liq     = float(res["borrowInfo"]["availableLiquidity"]["usd"] or 0)
                symbol        = res["underlyingToken"]["symbol"]

                if liq_threshold <= 0 or total_debt <= 0:
                    continue

                reserves.append({
                    "symbol":            symbol,
                    "liq_threshold":     liq_threshold,
                    "liq_bonus":         liq_bonus,
                    "total_supply_usd":  total_supply,
                    "total_debt_usd":    total_debt,
                    "avail_liq_usd":     avail_liq,
                })
            except (KeyError, TypeError, ValueError):
                continue

    if not reserves:
        raise RuntimeError("No valid reserves returned from Aave API")

    res_df = pd.DataFrame(reserves)
    total_debt_pool = res_df["total_debt_usd"].sum()

    print(f"Live Aave V3 Ethereum — {len(res_df)} active reserves")
    print(f"  Total debt across pool:     ${total_debt_pool/1e9:.2f}B")
    print(f"  Total supply across pool:   ${res_df['total_supply_usd'].sum()/1e9:.2f}B")

    # Allocate n_sample positions weighted by each reserve's share of total debt
    res_df["weight"] = res_df["total_debt_usd"] / total_debt_pool
    res_df["n_positions"] = (res_df["weight"] * n_sample).round().astype(int)
    # Fix rounding so total = n_sample exactly
    diff = n_sample - res_df["n_positions"].sum()
    res_df.iloc[res_df["weight"].idxmax(), res_df.columns.get_loc("n_positions")] += diff

    rng = np.random.default_rng(42)
    records = []

    for _, row in res_df.iterrows():
        n = int(row["n_positions"])
        if n == 0:
            continue

        lt = row["liq_threshold"]
        lb = row["liq_bonus"]

        # Average debt per position: total debt / estimated 5000 borrowers per reserve
        avg_debt = max(row["total_debt_usd"] / 5000, 500)

        # Draw debt from a log-normal distribution
        debt_vals = rng.lognormal(
            mean=np.log(avg_debt),
            sigma=1.2,
            size=n
        )
        debt_vals = np.clip(debt_vals, 100, avg_debt * 200)

        # Draw health factors calibrated to Aave V3 risk dashboard:
        # lognormal(0.8, 0.3) → median HF ≈ 2.23, P(HF<1.0) ≈ 0.4%, P(HF<1.2) ≈ 2%
        # Clip at 1.01 — sub-1.0 positions are cleared by bots continuously in
        # the real protocol and should not appear in a realistic snapshot.
        hf_vals = rng.lognormal(mean=0.8, sigma=0.3, size=n)
        hf_vals = np.clip(hf_vals, 1.01, 10.0)

        # Collateral implied by HF: collateral * lt / debt = hf
        collateral_vals = (hf_vals * debt_vals) / lt

        for debt, collateral, hf in zip(debt_vals, collateral_vals, hf_vals):
            records.append({
                "collateral_usd":        round(collateral, 2),
                "debt_usd":              round(debt, 2),
                "health_factor":         round(hf, 4),
                "liquidation_threshold": round(lt, 4),
                "liq_bonus":             round(lb, 4),
                "symbol":                row["symbol"],
            })

    df = pd.DataFrame(records).head(n_sample)
    print(f"  Generated {len(df)} synthetic positions from live parameters")
    print(f"  Median HF: {df['health_factor'].median():.3f}")
    print(f"  Positions near liquidation (HF<1.1): {(df['health_factor']<1.1).sum()}")
    return df


def check_connection() -> bool:
    try:
        r = requests.post(AAVE_API, json={"query": "{ chains { name } }"}, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


if __name__ == "__main__":
    print("Testing live Aave V3 data fetch...")
    df = fetch_live_positions(n_sample=1000)
    print(df[["symbol", "collateral_usd", "debt_usd", "health_factor", "liquidation_threshold"]].head(10))
    print(f"\nTotal collateral: ${df['collateral_usd'].sum()/1e6:.1f}M")
    print(f"Total debt:       ${df['debt_usd'].sum()/1e6:.1f}M")
