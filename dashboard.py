import dash
from dash import dcc, html, Input, Output, ctx
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from simulate import run_cascade
from theory import calibrate_from_positions
from fetch_aave import generate_aave_positions
try:
    from fetch_live import fetch_live_positions, check_connection
    LIVE_AVAILABLE = True
except (ImportError, Exception):
    LIVE_AVAILABLE = False

try:
    from monitor import load_log, run_once as monitor_run_once, compute_F
    MONITOR_AVAILABLE = True
except (ImportError, Exception):
    MONITOR_AVAILABLE = False

try:
    from backtest_ftx import run_ftx_backtest, build_f_timeline, FTX_BACKTEST_STATE, ACTUAL_OUTCOMES
    BACKTEST_AVAILABLE = True
except (ImportError, Exception):
    BACKTEST_AVAILABLE = False

try:
    from fetch_positions_dune import fetch_real_positions, generate_calibrated_positions, compare_distributions
    DUNE_AVAILABLE = True
except (ImportError, Exception):
    DUNE_AVAILABLE = False

app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([

    html.H1("DeFi Liquidation Risk Monitor",
            style={"textAlign": "center", "fontFamily": "Arial", "marginBottom": "4px"}),

    html.P("Mishricky (2025) — Asset Price Dispersion, Monetary Policy and Macroprudential Regulation",
           style={"textAlign": "center", "fontFamily": "Arial", "color": "grey", "marginBottom": "16px"}),

    dcc.Tabs(id="main-tabs", value="simulator", children=[

        # ── TAB 1: Original Simulator ─────────────────────────────────────
        dcc.Tab(label="🔬 Simulator", value="simulator", children=[

    html.Div([
        html.Label("Data Source: ", style={"fontFamily": "Arial", "fontSize": "14px", "marginRight": "10px"}),
        dcc.RadioItems(
            id="data-source",
            options=[
                {"label": "  Synthetic (offline)", "value": "synthetic"},
                {"label": "  Live Aave V3", "value": "live"},
            ],
            value="synthetic",
            inline=True,
            style={"fontFamily": "Arial", "fontSize": "14px", "display": "inline-block"}
        ),
        html.Span(id="data-source-status",
                  style={"fontFamily": "Arial", "fontSize": "13px",
                         "color": "#555", "marginLeft": "20px", "fontStyle": "italic"}),
    ], style={"textAlign": "center", "marginBottom": "20px"}),

    html.Div([
        html.Div([
            html.Label("Crisis Scenario Preset"),
            dcc.RadioItems(
                id="scenario-preset",
                options=[
                    {"label": "  Normal market", "value": "normal"},
                    {"label": "  Liquidity crisis", "value": "liquidity"},
                    {"label": "  Gas spike", "value": "gas"},
                    {"label": "  Combined shock (2022-style)", "value": "combined"},
                ],
                value="normal",
                inline=True,
                style={"fontFamily": "Arial", "fontSize": "14px"}
            ),
        ], style={"marginBottom": "10px"}),

        html.P(id="preset-description",
               style={"fontFamily": "Arial", "fontSize": "13px",
                      "color": "#555", "fontStyle": "italic", "marginBottom": "20px"}),

        html.Div([
            html.Label("Price Drop (%)"),
            dcc.Slider(id="price-drop", min=5, max=60, step=5, value=30,
                       marks={i: f"{i}%" for i in range(5, 65, 5)}),
        ], style={"marginBottom": "30px"}),

        html.Div([
            html.Label("Initial Liquidity (% of total debt)"),
            dcc.Slider(id="liquidity-pct", min=1, max=80, step=1, value=40,
                       marks={i: f"{i}%" for i in [1, 5, 10, 20, 40, 60, 80]}),
        ], style={"marginBottom": "30px"}),

        html.Div([
            html.Label("Gas Cost per Liquidation (USD)"),
            dcc.Slider(id="gas-cost", min=20, max=500, step=20, value=80,
                       marks={i: f"${i}" for i in range(20, 520, 80)}),
        ], style={"marginBottom": "10px"}),

        # Feedback toggle
        html.Div([
            html.Label("Bot Participation Model: ",
                       style={"fontFamily": "Arial", "fontSize": "14px", "marginRight": "10px"}),
            dcc.RadioItems(
                id="feedback-mode",
                options=[
                    {"label": "  Endogenous (Bernoulli F feedback)", "value": "on"},
                    {"label": "  Open-loop (bots always participate)", "value": "off"},
                ],
                value="on",
                inline=True,
                style={"fontFamily": "Arial", "fontSize": "14px", "display": "inline-block"}
            ),
        ], style={"marginTop": "10px", "marginBottom": "5px"}),

        html.P(
            "Endogenous: each round, bots participate with probability (1 − F). "
            "When they don't, positions stay underwater and liquidity continues to drain, "
            "raising F further — the doom loop runs in the simulation, not just the README.",
            style={"fontFamily": "Arial", "fontSize": "12px",
                   "color": "#777", "fontStyle": "italic", "marginBottom": "5px"}
        ),

    ], style={"padding": "20px", "backgroundColor": "#f9f9f9",
              "borderRadius": "8px", "marginBottom": "30px"}),

    html.Div(id="summary-stats",
             style={"display": "flex", "justifyContent": "space-around",
                    "flexWrap": "wrap", "gap": "10px",
                    "marginBottom": "30px", "fontFamily": "Arial"}),

    html.Div([
        dcc.Graph(id="cascade-chart"),
        dcc.Graph(id="theory-chart"),
    ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}),

    html.Div([
        dcc.Graph(id="distributions-chart"),
        dcc.Graph(id="stress-test-chart"),
    ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
              "gap": "20px", "marginTop": "20px"}),

        ]),  # end Tab 1

        # ── TAB 2: Live F Monitor ─────────────────────────────────────────
        dcc.Tab(label="📡 Live F Monitor", value="monitor", children=[
            html.Div([
                html.Div([
                    html.H3("Real-Time Fragility Signal", style={"fontFamily": "Arial"}),
                    html.P(
                        "F (flash-crash probability) computed from live Aave V3 on-chain conditions "
                        "every hour. Logged to f_monitor_log.csv. F rises when gas costs spike, "
                        "stablecoin depth drains, or utilisation increases — before bad debt materialises.",
                        style={"fontFamily": "Arial", "color": "#555", "fontSize": "13px"}
                    ),
                    html.Div([
                        html.Button("🔄 Fetch Snapshot Now", id="monitor-refresh-btn",
                                    style={"marginRight": "15px", "padding": "8px 18px",
                                           "backgroundColor": "#2c7be5", "color": "white",
                                           "border": "none", "borderRadius": "4px",
                                           "cursor": "pointer", "fontSize": "14px"}),
                        html.Span(id="monitor-status", style={"fontFamily": "Arial",
                                                               "fontSize": "13px", "color": "#555"}),
                    ], style={"marginBottom": "20px"}),
                ], style={"padding": "16px 0"}),

                # Current snapshot stats
                html.Div(id="monitor-current-stats",
                         style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                                "marginBottom": "24px"}),

                # Time series charts
                html.Div([
                    dcc.Graph(id="monitor-f-chart"),
                    dcc.Graph(id="monitor-context-chart"),
                ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}),

                html.P(
                    "How to deploy: run 'python monitor.py --daemon' to collect hourly snapshots. "
                    "Add to cron: '0 * * * * cd /path/to/project && python monitor.py' for fully automated logging.",
                    style={"fontFamily": "Arial", "fontSize": "12px", "color": "#999",
                           "fontStyle": "italic", "marginTop": "16px"}
                ),
            ], style={"padding": "20px"}),
        ]),  # end Tab 2

        # ── TAB 3: FTX Backtest ───────────────────────────────────────────
        dcc.Tab(label="📅 FTX Backtest", value="backtest", children=[
            html.Div([
                html.Div([
                    html.H3("Empirical Validation: FTX Collapse (November 2022)",
                            style={"fontFamily": "Arial"}),
                    html.P(
                        "Reconstructs Aave V3 Ethereum pool conditions on November 8, 2022 "
                        "(the day Binance announced it would not rescue FTX) and runs the cascade "
                        "simulator from those starting conditions. Tests whether F would have "
                        "signalled ELEVATED RISK before the bad debt materialised on-chain.",
                        style={"fontFamily": "Arial", "color": "#555", "fontSize": "13px"}
                    ),
                    html.Button("▶ Run Backtest", id="backtest-run-btn",
                                style={"padding": "8px 18px", "backgroundColor": "#e74c3c",
                                       "color": "white", "border": "none", "borderRadius": "4px",
                                       "cursor": "pointer", "fontSize": "14px",
                                       "marginBottom": "20px"}),
                ], style={"padding": "16px 0"}),

                # Placeholder shown before backtest is run
                html.Div(
                    id="backtest-placeholder",
                    children=[
                        html.Div([
                            html.Span("📊", style={"fontSize": "40px"}),
                            html.P(
                                "Click ▶ Run Backtest to simulate the FTX cascade and generate all charts.",
                                style={"fontFamily": "Arial", "fontSize": "14px",
                                       "color": "#888", "margin": "10px 0 0 0"}
                            ),
                        ], style={"textAlign": "center", "padding": "60px 20px",
                                  "backgroundColor": "#f9f9f9", "borderRadius": "8px",
                                  "border": "1px dashed #ccc"}),
                    ],
                    style={"display": "block"},
                ),

                # All backtest content — hidden until button is clicked
                html.Div(
                    id="backtest-content",
                    style={"display": "none"},
                    children=[
                        # Key result callout
                        html.Div(id="backtest-result-callout",
                                 style={"marginBottom": "20px"}),

                        # Timeline + cascade charts
                        html.Div([
                            dcc.Graph(id="backtest-timeline-chart"),
                            dcc.Graph(id="backtest-cascade-chart"),
                        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}),

                        html.Div([
                            dcc.Graph(id="backtest-comparison-chart"),
                            dcc.Graph(id="backtest-specpremium-chart"),
                        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                                  "gap": "20px", "marginTop": "20px"}),

                        # Data sources footnote
                        html.P(
                            "Data sources: DeFiLlama TVL series (Nov 2022), Dune Analytics #1329110 "
                            "(HF distribution), Etherscan gas oracle export, CoinGecko ETH/USD OHLCV. "
                            "Actual liquidation figures: Messari Research / Rekt.news post-mortem.",
                            style={"fontFamily": "Arial", "fontSize": "11px", "color": "#aaa",
                                   "fontStyle": "italic", "marginTop": "16px"}
                        ),
                    ],
                ),
            ], style={"padding": "20px"}),
        ]),  # end Tab 3

    ]),  # end Tabs

], style={"maxWidth": "1400px", "margin": "auto", "padding": "20px"})


@app.callback(
    Output("price-drop", "value"),
    Output("liquidity-pct", "value"),
    Output("gas-cost", "value"),
    Output("preset-description", "children"),
    Output("summary-stats", "children"),
    Output("cascade-chart", "figure"),
    Output("theory-chart", "figure"),
    Output("distributions-chart", "figure"),
    Output("stress-test-chart", "figure"),
    Output("data-source-status", "children"),
    Input("scenario-preset", "value"),
    Input("price-drop", "value"),
    Input("liquidity-pct", "value"),
    Input("gas-cost", "value"),
    Input("data-source", "value"),
    Input("feedback-mode", "value"),
)
def update_dashboard(scenario_preset, price_drop, liquidity_pct, gas_cost, data_source, feedback_mode):

    preset_map = {
        "normal":    (30, 40, 80),
        "liquidity": (30, 3, 80),
        "gas":       (30, 40, 450),
        "combined":  (45, 5, 400),
    }

    preset_descriptions = {
        "normal":    "Baseline conditions. 30% price drop, 40% liquidity depth, $80 gas. Moderate φᵐ and low κ.",
        "liquidity": "φᵐ shock. Liquidity depth reduced to 3% of total debt, simulating capital flight before the crash.",
        "gas":       "κ shock. Gas cost raised to $450, raising flash crash risk as posting becomes more costly. Large positions remain profitable to liquidate — the scenario demonstrates how κ elevates F even without generating bad debt.",
        "combined":  "Simultaneous φᵐ collapse and κ shock with a 45% price drop. Analogue of March 2020 or November 2022.",
    }

    if ctx.triggered_id == "scenario-preset":
        price_drop, liquidity_pct, gas_cost = preset_map[scenario_preset]

    description = preset_descriptions.get(scenario_preset, "")
    use_feedback = (feedback_mode == "on")

    print(f"Running: preset={scenario_preset} drop={price_drop}% liquidity={liquidity_pct}% gas=${gas_cost} feedback={use_feedback}")

    # --- Generate pool ---
    data_status = ""
    if data_source == "live" and LIVE_AVAILABLE:
        try:
            positions = fetch_live_positions(n_sample=1000)
            data_status = f"Live Aave V3 — {len(positions)} positions fetched"
        except Exception as e:
            positions = generate_aave_positions(n=1000)
            data_status = f"Live fetch failed ({e}) — using synthetic data"
    else:
        positions = generate_aave_positions(n=1000)
        data_status = "Synthetic data (Aave V3 calibrated)" if data_source == "synthetic" else "Live data unavailable — using synthetic"

    total_debt = positions['debt_usd'].sum()
    stablecoin_depth = total_debt * (liquidity_pct / 100)

    # --- Pre-cascade initial F ---
    try:
        initial_model = calibrate_from_positions(positions, float(gas_cost), stablecoin_depth, 0.05)
        initial_F = initial_model.flash_crash_prob
        initial_theta = initial_model.theta
        initial_status = initial_model.summary()["market status"]
    except ValueError:
        initial_F = 1.0
        initial_theta = 0.0
        initial_status = "COLLAPSE"

    # --- Run simulation ---
    results, agents = run_cascade(
        price_drop_pct=price_drop / 100,
        gas_usd=float(gas_cost),
        initial_liquidity_pct=liquidity_pct / 100,
        use_feedback=use_feedback,
        rng_seed=42,
    )

    total_liquidated = sum(1 for a in agents if a.liquidated)
    total_bad_debt = results['bad_debt_usd'].sum()
    final_F = results['F (crash prob)'].iloc[-1]
    final_status = results['market_status'].iloc[-1]
    final_participation = results['participation_rate'].iloc[-1] if 'participation_rate' in results.columns else 1.0

    def status_color(s):
        return {"STABLE": "green", "ELEVATED RISK": "orange",
                "CRITICAL": "red"}.get(s, "grey")

    def stat_box(label, value, color="black"):
        return html.Div([
            html.H3(value, style={"color": color, "margin": "0", "fontSize": "24px"}),
            html.P(label, style={"margin": "0", "color": "grey", "fontSize": "12px"})
        ], style={"textAlign": "center", "padding": "15px",
                  "backgroundColor": "white", "borderRadius": "8px",
                  "boxShadow": "0 2px 4px rgba(0,0,0,0.1)", "minWidth": "150px"})

    summary = [
        stat_box("Positions Liquidated", f"{total_liquidated} / 1000"),
        stat_box("Bad Debt", f"${total_bad_debt/1e6:.1f}M",
                 color="red" if total_bad_debt > 0 else "green"),
        stat_box("Cascade Rounds", str(len(results))),
        stat_box("Flash Crash Risk (pre-cascade, per round)", f"{1-(1-initial_F)**1000:.1%}"),
        stat_box("Flash Crash Risk (post-cascade, per round)", f"{1-(1-final_F)**1000:.1%}"),
        stat_box("Final Participation Rate", f"{final_participation:.1%}",
                 color="red" if final_participation < 0.5 else ("orange" if final_participation < 0.9 else "green")),
        stat_box("Initial Status", initial_status, color=status_color(initial_status)),
        stat_box("Final Status", final_status, color=status_color(final_status)),
    ]

    # --- Chart 1: Cascade ---
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=results['round'], y=results['liquidations'],
        name='Liquidations', marker_color='crimson', yaxis='y'
    ))
    fig1.add_trace(go.Scatter(
        x=results['round'], y=results['price'],
        name='Price', line=dict(color='steelblue', width=2), yaxis='y2'
    ))
    fig1.update_layout(
        title="Liquidation Cascade by Round",
        xaxis_title="Round",
        yaxis=dict(title="Liquidations", side="left"),
        yaxis2=dict(title="Price (relative)", side="right", overlaying="y"),
        legend=dict(x=0.7, y=0.99), template="plotly_white"
    )

    # --- Chart 2: Theory ---
    CYCLES_PER_ROUND = 1000  # matches simulate.py — each cascade round = 1000 blocks (~3.3h)
    rounds = [0] + list(results['round'])
    thetas = [initial_theta] + list(results['theta'])
    # Show per-round crash probability: P = 1 - (1-F)^1000, consistent with simulation
    raw_fs = [initial_F] + list(results['F (crash prob)'])
    fs_per_round = [1 - (1 - f) ** CYCLES_PER_ROUND for f in raw_fs]
    participation = [1.0] + list(results['participation_rate']) if 'participation_rate' in results.columns else [1.0] * len(rounds)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=rounds, y=thetas,
        name='θ (quote intensity)', line=dict(color='green', width=2), yaxis='y'
    ))
    fig2.add_trace(go.Scatter(
        x=rounds, y=fs_per_round,
        name='P(flash crash | round)',
        line=dict(color='red', width=2, dash='dash'), yaxis='y2'
    ))
    if use_feedback:
        fig2.add_trace(go.Scatter(
            x=rounds, y=participation,
            name='Bot participation rate',
            line=dict(color='orange', width=2, dash='dot'), yaxis='y2'
        ))

    fig2.update_layout(
        title="Mishricky (2025): θ and F During Cascade", xaxis_title="Round",
        xaxis=dict(tickvals=rounds, ticktext=["Pre"] + [str(r) for r in results['round']]),
        yaxis=dict(title="θ (quote intensity)", side="left"),
        yaxis2=dict(title="P(flash crash | round)  [= 1 − (1−F)^1000]", side="right", overlaying="y",
                    tickformat=".0%", range=[0, 1]),
        legend=dict(x=0.5, y=0.99), template="plotly_white"
    )

    # --- Chart 3: Distributions — pre and post cascade ---
    # X-axis is in basis points from par (price=1.0), so sub-bps spread
    # widening is legible. Negative = bid side, positive = ask side.
    def dist_pdfs_bps(model, n=300):
        """Return ask/bid positions in bps from par, plus PDFs and inner edges."""
        p_hat = 1.0
        ask_lo_bps = (model.kappa / model.phi_m) * 10000
        ask_hi_bps = model.Gamma * 10000
        bid_hi_bps = -(model.kappa / model.phi_m) * 10000
        bid_lo_bps = -model.Gamma * 10000

        prices_ask_bps = np.linspace(ask_lo_bps, ask_hi_bps, n)
        prices_bid_bps = np.linspace(bid_lo_bps, bid_hi_bps, n)

        # CDF evaluated at actual prices, then gradient in bps space
        prices_ask = p_hat + prices_ask_bps / 10000
        prices_bid = p_hat + prices_bid_bps / 10000
        ask_pdf = np.gradient([model.ask_distribution(p) for p in prices_ask], prices_ask_bps)
        bid_pdf = np.gradient([model.bid_distribution(p) for p in prices_bid], prices_bid_bps)
        return prices_ask_bps, prices_bid_bps, ask_pdf, bid_pdf, ask_lo_bps, bid_hi_bps

    try:
        fig3 = go.Figure()

        # Pre-cascade
        pa, pb, apdf, bpdf, pre_ask_lo, pre_bid_hi = dist_pdfs_bps(initial_model)
        fig3.add_trace(go.Scatter(x=pa, y=apdf, fill='tozeroy', name='Ask (pre-cascade)',
                                  line=dict(color='darkred', width=2),
                                  fillcolor='rgba(180,0,0,0.30)'))
        fig3.add_trace(go.Scatter(x=pb, y=bpdf, fill='tozeroy', name='Bid (pre-cascade)',
                                  line=dict(color='darkblue', width=2),
                                  fillcolor='rgba(0,0,180,0.30)'))
        y_max = max(max(apdf), max(bpdf)) * 0.05

        # Post-cascade model — calibrate from surviving positions
        surviving_agents = [a for a in agents if not a.liquidated]
        post_model = None
        if len(surviving_agents) > 0:
            surviving_df = pd.DataFrame({
                'collateral_usd': [a.collateral for a in surviving_agents],
                'debt_usd':       [a.debt for a in surviving_agents],
                'health_factor':  [a.health_factor for a in surviving_agents]
            })
            try:
                post_model = calibrate_from_positions(
                    surviving_df,
                    gas_usd=float(gas_cost),
                    stablecoin_depth_usd=float(results['available_liquidity_usd'].iloc[-1]),
                    daily_volatility=0.05
                )
            except ValueError:
                post_model = None

        if post_model is not None:
            pa2, pb2, apdf2, bpdf2, post_ask_lo, post_bid_hi = dist_pdfs_bps(post_model)
            fig3.add_trace(go.Scatter(x=pa2, y=apdf2, fill='tozeroy', name='Ask (post-cascade)',
                                      line=dict(color='red', width=1.5, dash='dash'),
                                      fillcolor='rgba(255,80,80,0.12)'))
            fig3.add_trace(go.Scatter(x=pb2, y=bpdf2, fill='tozeroy', name='Bid (post-cascade)',
                                      line=dict(color='blue', width=1.5, dash='dash'),
                                      fillcolor='rgba(80,80,255,0.12)'))

            # Vertical lines at inner spread edges — one label per side only
            fig3.add_vline(x=pre_ask_lo,  line_dash='dot',  line_color='darkred',  opacity=0.5)
            fig3.add_vline(x=pre_bid_hi,  line_dash='dot',  line_color='darkblue', opacity=0.5)
            fig3.add_vline(x=post_ask_lo, line_dash='dash', line_color='red',      opacity=0.8)
            fig3.add_vline(x=post_bid_hi, line_dash='dash', line_color='blue',     opacity=0.8)

            y_max = max(y_max, max(max(apdf2), max(bpdf2)) * 0.05)

            pre_spread  = round(pre_ask_lo  - pre_bid_hi,  3)
            post_spread = round(post_ask_lo - post_bid_hi, 3)
            theta_pre  = round(initial_model.theta, 1)
            theta_post = round(post_model.theta, 1)
            title = (f"Spread: {pre_spread} → {post_spread} bps  |  θ: {theta_pre} → {theta_post}")
        else:
            title = f"Bid & Ask Distributions  |  θ={initial_model.theta:.1f}"

        fig3.add_vline(x=0, line_color='grey', line_width=1, opacity=0.4)
        fig3.update_layout(
            title=title,
            xaxis_title="Basis points from par (price = 1.0)",
            yaxis_title="Density",
            xaxis=dict(range=[-initial_model.Gamma * 10000 - 10, initial_model.Gamma * 10000 + 10]),
            legend=dict(x=0.01, y=0.99), template="plotly_white"
        )
        fig3.update_yaxes(range=[0, y_max])

    except (ValueError, AttributeError):
        fig3 = go.Figure()
        fig3.add_annotation(
            text="Market collapse — no quoting activity",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="red")
        )
        fig3.update_layout(title="Proposition 1: Bid & Ask Price Distributions",
                           template="plotly_white")

    # --- Chart 4: Stress test ---
    drops = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    scenario_results = []
    for d in drops:
        try:
            r, a = run_cascade(
                price_drop_pct=d,
                gas_usd=float(gas_cost),
                initial_liquidity_pct=liquidity_pct / 100,
                use_feedback=use_feedback,
                rng_seed=42,
            )
            scenario_results.append({
                "drop": f"{int(d*100)}%",
                "liquidated": sum(1 for x in a if x.liquidated),
                "bad_debt": r['bad_debt_usd'].sum() / 1e6,
            })
        except Exception:
            scenario_results.append({
                "drop": f"{int(d*100)}%",
                "liquidated": 1000,
                "bad_debt": 0,
            })

    scenario_df = pd.DataFrame(scenario_results)

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        x=scenario_df['drop'], y=scenario_df['bad_debt'],
        name='Bad Debt ($M)', marker_color='crimson', yaxis='y'
    ))
    fig4.add_trace(go.Scatter(
        x=scenario_df['drop'], y=scenario_df['liquidated'],
        name='Positions Liquidated',
        line=dict(color='steelblue', width=2), yaxis='y2'
    ))
    fig4.update_layout(
        title="Stress Test: All Scenarios", xaxis_title="Price Drop",
        yaxis=dict(title="Bad Debt ($M)", side="left"),
        yaxis2=dict(title="Positions Liquidated", side="right", overlaying="y"),
        legend=dict(x=0.01, y=0.99), template="plotly_white",
    )

    return price_drop, liquidity_pct, gas_cost, description, summary, fig1, fig2, fig3, fig4, data_status


# ── Monitor Tab Callbacks ─────────────────────────────────────────────────────

@app.callback(
    Output("monitor-current-stats", "children"),
    Output("monitor-f-chart", "figure"),
    Output("monitor-context-chart", "figure"),
    Output("monitor-status", "children"),
    Input("monitor-refresh-btn", "n_clicks"),
    Input("main-tabs", "value"),
    prevent_initial_call=False,
)
def update_monitor(n_clicks, active_tab):
    if active_tab != "monitor":
        return [], go.Figure(), go.Figure(), ""

    status_msg = ""
    if n_clicks and MONITOR_AVAILABLE:
        try:
            entry = monitor_run_once(verbose=False)
            status_msg = f"✓ Snapshot collected at {entry['timestamp_utc']}"
        except Exception as e:
            status_msg = f"⚠ Fetch failed: {str(e)[:80]}"

    # Load log
    if MONITOR_AVAILABLE:
        df = load_log()
    else:
        df = pd.DataFrame()

    def _stat_card(label, value, color="#2c7be5", bg="#f0f6ff"):
        return html.Div([
            html.Div(label, style={"fontSize": "11px", "color": "#888", "textTransform": "uppercase",
                                    "letterSpacing": "0.5px", "marginBottom": "4px"}),
            html.Div(value, style={"fontSize": "20px", "fontWeight": "bold", "color": color}),
        ], style={"backgroundColor": bg, "border": f"1px solid {color}40",
                  "borderRadius": "6px", "padding": "12px 18px",
                  "minWidth": "130px", "fontFamily": "Arial"})

    if df.empty:
        # Show demo snapshot computed right now from live params
        if MONITOR_AVAILABLE:
            try:
                from monitor import fetch_eth_price, fetch_gas_gwei, gas_gwei_to_usd, fetch_aave_liquidity, compute_F
                eth = fetch_eth_price()
                gwei = fetch_gas_gwei()
                gas = gas_gwei_to_usd(gwei, eth)
                depth, debt = fetch_aave_liquidity()
                m = compute_F(gas, depth, debt)
                status_color = {"STABLE": "#2ecc71", "ELEVATED RISK": "#f39c12", "CRITICAL": "#e74c3c"}.get(m["market_status"], "grey")
                cards = [
                    _stat_card("ETH Price", f"${eth:,.0f}"),
                    _stat_card("Gas", f"{gwei:.0f} gwei"),
                    _stat_card("Gas / Liquidation", f"${gas:.0f}"),
                    _stat_card("Stablecoin Depth", f"${depth/1e9:.1f}B"),
                    _stat_card("phi_m", f"{m['phi_m']:.4f}"),
                    _stat_card("P(fc | 24h)", f"{1-(1-m['F'])**24000:.2%}", color=status_color),
                    _stat_card("Market Status", m["market_status"], color=status_color,
                               bg={"STABLE": "#eafaf1", "ELEVATED RISK": "#fef9e7",
                                   "CRITICAL": "#fdedec"}.get(m["market_status"], "#f9f9f9")),
                ]
                return cards, go.Figure().update_layout(
                    title="No log data yet — click 'Fetch Snapshot Now' to start collecting",
                    template="plotly_white"), go.Figure(), "Live snapshot (not yet logged)"
            except Exception:
                pass
        cards = [_stat_card("Status", "No data — click 'Fetch Snapshot Now'", "#999")]
        return cards, go.Figure(), go.Figure(), status_msg

    latest = df.iloc[-1]
    status_color = {"STABLE": "#2ecc71", "ELEVATED RISK": "#f39c12", "CRITICAL": "#e74c3c"}.get(
        str(latest.get("market_status", "")), "grey")

    cards = [
        _stat_card("ETH Price", f"${float(latest.get('eth_price_usd', 0)):,.0f}"),
        _stat_card("Gas / Liq.", f"${float(latest.get('gas_usd', 0)):.0f}"),
        _stat_card("Stablecoin Depth", f"${float(latest.get('stablecoin_depth_usd', 0))/1e9:.2f}B"),
        _stat_card("phi_m", f"{float(latest.get('phi_m', 0)):.4f}"),
        _stat_card("theta", f"{float(latest.get('theta', 0)):.3f}"),
        _stat_card("P(flash crash | 24h)", f"{1-(1-float(latest.get('F', 0)))**24000:.2%}", color=status_color),
        _stat_card("Status", str(latest.get("market_status", "—")), color=status_color,
                   bg={"STABLE": "#eafaf1", "ELEVATED RISK": "#fef9e7",
                       "CRITICAL": "#fdedec"}.get(str(latest.get("market_status", "")), "#f9f9f9")),
    ]

    # Chart 1: P(flash crash | 24h) over time
    BLOCKS_PER_DAY = 24000
    df_daily_p = df["F"].apply(lambda f: 1 - (1 - f) ** BLOCKS_PER_DAY)
    # Threshold lines match monitor.py status classification: p_daily < 0.15 = STABLE, < 0.80 = ELEVATED
    elev_daily = 0.15
    crit_daily = 0.80

    fig_f = go.Figure()
    colour_map = {"STABLE": "#2ecc71", "ELEVATED RISK": "#f39c12", "CRITICAL": "#e74c3c"}
    bar_colours = [colour_map.get(str(s), "grey") for s in df["market_status"]]

    fig_f.add_trace(go.Bar(
        x=df["timestamp_utc"], y=df_daily_p,
        marker_color=bar_colours, name="P(flash crash | 24h)",
        opacity=0.85,
    ))
    fig_f.add_trace(go.Scatter(
        x=df["timestamp_utc"], y=df_daily_p,
        mode="lines", line=dict(color="black", width=1.5), name="trend",
    ))
    fig_f.update_layout(
        title="P(flash crash | 24h) = 1 − (1−F)^24000  —  Live Monitor",
        xaxis_title="Time (UTC)", yaxis_title="P(flash crash | 24h)", yaxis_tickformat=".1%",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
        shapes=[
            dict(type="line", x0=df["timestamp_utc"].min(), x1=df["timestamp_utc"].max(),
                 y0=elev_daily, y1=elev_daily, line=dict(color="orange", dash="dash", width=1.5)),
            dict(type="line", x0=df["timestamp_utc"].min(), x1=df["timestamp_utc"].max(),
                 y0=crit_daily, y1=crit_daily, line=dict(color="red", dash="dash", width=1.5)),
        ],
        annotations=[
            dict(x=df["timestamp_utc"].max(), y=elev_daily, text="ELEVATED (15%)",
                 showarrow=False, font=dict(color="orange", size=10),
                 xanchor="right", yanchor="bottom"),
            dict(x=df["timestamp_utc"].max(), y=crit_daily, text="CRITICAL (80%)",
                 showarrow=False, font=dict(color="red", size=10),
                 xanchor="right", yanchor="bottom"),
        ]
    )

    # Chart 2: ETH price + gas context
    fig_ctx = go.Figure()
    fig_ctx.add_trace(go.Scatter(
        x=df["timestamp_utc"], y=df["eth_price_usd"],
        name="ETH Price (USD)", line=dict(color="steelblue", width=2), yaxis="y"
    ))
    fig_ctx.add_trace(go.Scatter(
        x=df["timestamp_utc"], y=df["gas_usd"],
        name="Gas / Liquidation ($)", line=dict(color="darkorange", width=2, dash="dot"), yaxis="y2"
    ))
    fig_ctx.update_layout(
        title="Market Context: ETH Price & Gas Cost",
        xaxis_title="Time (UTC)",
        yaxis=dict(title="ETH Price (USD)", side="left"),
        yaxis2=dict(title="Gas / Liquidation ($)", side="right", overlaying="y"),
        legend=dict(x=0.01, y=0.99), template="plotly_white",
    )

    return cards, fig_f, fig_ctx, status_msg


# ── Backtest Tab Callbacks ─────────────────────────────────────────────────────

@app.callback(
    Output("backtest-result-callout", "children"),
    Output("backtest-timeline-chart", "figure"),
    Output("backtest-cascade-chart", "figure"),
    Output("backtest-comparison-chart", "figure"),
    Output("backtest-specpremium-chart", "figure"),
    Output("backtest-content", "style"),
    Output("backtest-placeholder", "style"),
    Input("backtest-run-btn", "n_clicks"),
    Input("main-tabs", "value"),
    prevent_initial_call=False,
)
def update_backtest(n_clicks, active_tab):
    hidden = {"display": "none"}
    visible_content = {"display": "block"}
    visible_placeholder = {"display": "block"}

    if active_tab != "backtest":
        return [], go.Figure(), go.Figure(), go.Figure(), go.Figure(), hidden, visible_placeholder

    empty = [go.Figure()] * 4

    if not BACKTEST_AVAILABLE:
        msg = html.Div("⚠ backtest_ftx.py not found. Ensure it is in the project directory.",
                       style={"color": "red", "fontFamily": "Arial"})
        return msg, *empty, hidden, visible_placeholder

    # Gate everything behind button click
    if not n_clicks or n_clicks == 0:
        return [], *empty, hidden, visible_placeholder

    # Build F timeline (cheap — no simulation)
    timeline = build_f_timeline()

    # Run cascade simulation
    results, agents, pre_crash, summary = None, None, None, None
    try:
        results, agents, pre_crash, summary = run_ftx_backtest(verbose=False)
    except Exception as e:
        pre_crash = {"flash crash prob (F)": None, "market status": "ERROR"}
        summary = {}

    # Result callout — read Nov 8 and Nov 9 directly from timeline (same source as chart)
    _colour_map_tl = {"STABLE": "#2ecc71", "ELEVATED RISK": "#f39c12", "CRITICAL": "#e74c3c"}
    _nov8  = timeline[timeline["date"] == "2022-11-08"].iloc[0] if "2022-11-08" in timeline["date"].values else None
    _nov9  = timeline[timeline["date"] == "2022-11-09"].iloc[0] if "2022-11-09" in timeline["date"].values else None
    _hf_tail = float(pre_crash.get("hf_tail_pct", 0) or 0) if pre_crash else 0.077

    if _nov8 is not None:
        p_open_daily = 1 - (1 - float(_nov8["F"])) ** 24000
        status       = str(_nov8["market_status"])
        s_colour     = _colour_map_tl.get(status, "grey")
    else:
        p_open_daily, status, s_colour = 0.0, "—", "grey"

    if _nov9 is not None:
        p_peak_daily = 1 - (1 - float(_nov9["F"])) ** 24000
        peak_status  = str(_nov9["market_status"])
        peak_colour  = _colour_map_tl.get(peak_status, "grey")
    else:
        p_peak_daily, peak_status, peak_colour = 0.989, "CRITICAL", "#e74c3c"

    # Callout always shown (reads from timeline, not simulation)
    bg_colour  = {"STABLE": "#eafaf1", "ELEVATED RISK": "#fef9e7", "CRITICAL": "#fdedec"}.get(status, "#f9f9f9")
    callout = html.Div([
        html.Div([
            html.Span("P(flash crash | 24h) Trajectory: ", style={"fontWeight": "bold", "fontSize": "15px"}),
            html.Span("Nov 8  ", style={"color": "#888", "fontSize": "13px"}),
            html.Span(f"{p_open_daily:.1%}", style={"fontSize": "20px", "fontWeight": "bold", "color": s_colour}),
            html.Span(f" [{status}]", style={"color": s_colour, "fontWeight": "bold"}),
            html.Span("  →  Nov 9 peak  ", style={"color": "#888", "fontSize": "13px", "marginLeft": "14px"}),
            html.Span(f"{p_peak_daily:.1%}", style={"fontSize": "20px", "fontWeight": "bold", "color": peak_colour}),
            html.Span(f" [{peak_status}]", style={"color": peak_colour, "fontWeight": "bold"}),
        ], style={"fontSize": "15px", "marginBottom": "10px"}),
        html.P([
            f"By Nov 8 (Binance's no-rescue announcement) flash-crash probability had already reached {p_open_daily:.1%}/day — "
            f"{status}: gas had spiked to $85/liq and stablecoin depth had drained to $180M. "
            f"With {_hf_tail:.1%} of positions already at HF < 1.2, ignition risk was severe. "
            f"As the cascade intensified (gas peaked at $118, depth fell to $120M), P(fc|24h) surged to {p_peak_daily:.1%} on Nov 9. "
            "This is the paper's central result: F captures mechanism fragility; the HF tail captures ignition risk. "
            "Both signals are needed for complete early warning."
        ], style={"fontFamily": "Arial", "fontSize": "13px", "color": "#444", "margin": "0"}),
    ], style={"backgroundColor": bg_colour,
              "border": f"1px solid {s_colour}60", "borderRadius": "6px",
              "padding": "14px 18px", "fontFamily": "Arial"})


    # Chart 1: Two-panel — ETH price (top) + P(flash crash|24h) (bottom)
    from plotly.subplots import make_subplots

    BLOCKS_PER_DAY = 24000
    timeline_daily_p = timeline["F"].apply(lambda f: 1 - (1 - f) ** BLOCKS_PER_DAY)
    elev_daily = 0.15   # matches build_f_timeline status classification
    crit_daily = 0.80   # matches build_f_timeline status classification
    colour_map = {"STABLE": "#2ecc71", "ELEVATED RISK": "#f39c12", "CRITICAL": "#e74c3c"}
    dates = list(timeline["date"])  # full ISO strings so Plotly parses year correctly
    dot_colours = [colour_map.get(s, "grey") for s in timeline["market_status"]]

    events = [
        (1,  "CoinDesk"),
        (5,  "Binance sells FTT"),
        (7,  "No rescue"),
        (8,  "Peak cascade"),
        (10, "Bankruptcy"),
        (11, "ETH bottom"),
    ]

    fig_tl = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.45, 0.55],
        vertical_spacing=0.06,
        subplot_titles=["ETH / USD", "P(flash crash | 24h)  =  1 − (1−F)^24000"]
    )

    # Panel 1: ETH price — tight y-axis so drawdown is visible
    eth_min = int(timeline["eth_price_usd"].min() * 0.94)
    eth_max = int(timeline["eth_price_usd"].max() * 1.04)
    fig_tl.add_trace(go.Scatter(
        x=dates, y=timeline["eth_price_usd"],
        mode="lines", line=dict(color="#7c5cbf", width=2.5),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    # One trace per status — guarantees Plotly renders correct hex colours (no colorscale coercion)
    for status, colour in colour_map.items():
        xs, ys, cd = [], [], []
        for i, (d, eth, s) in enumerate(zip(dates, timeline["eth_price_usd"], timeline["market_status"])):
            if s == status:
                xs.append(d); ys.append(eth)
                cd.append((timeline["gas_usd"].iloc[i],
                            timeline["stablecoin_depth_usd"].iloc[i]/1e6,
                            s, timeline["note"].iloc[i]))
        if xs:
            fig_tl.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(color=colour, size=8, line=dict(color="white", width=1.5)),
                showlegend=False,
                customdata=cd,
                hovertemplate="<b>%{x}</b>  ETH $%{y:,}<br>Gas: $%{customdata[0]:.0f}/liq  |  Depth: $%{customdata[1]:.0f}M<br>Status: %{customdata[2]}<br>%{customdata[3]}<extra></extra>",
            ), row=1, col=1)

    # Panel 2: P(flash crash|24h) — coloured line segments + dots, one trace per status
    # Line segments: use None-gap technique — one trace per status, gaps between non-contiguous points
    for status, colour in colour_map.items():
        seg_x, seg_y = [], []
        for i in range(len(timeline) - 1):
            s_i = timeline["market_status"].iloc[i]
            s_next = timeline["market_status"].iloc[i + 1]
            # Draw segment if either endpoint has this status (transition segments take the higher-risk colour)
            # Use: colour the segment by whichever endpoint has the worse status
            worse = s_i if (["STABLE","ELEVATED RISK","CRITICAL"].index(s_i) >=
                            ["STABLE","ELEVATED RISK","CRITICAL"].index(s_next)) else s_next
            if worse == status:
                seg_x += [dates[i], dates[i+1], None]
                seg_y += [float(timeline_daily_p.iloc[i]), float(timeline_daily_p.iloc[i+1]), None]
        if seg_x:
            fig_tl.add_trace(go.Scatter(
                x=seg_x, y=seg_y, mode="lines",
                line=dict(color=colour, width=3),
                showlegend=False, hoverinfo="skip",
            ), row=2, col=1)
    for status, colour in colour_map.items():
        xs2, ys2, cd2 = [], [], []
        for i, (d, p, s) in enumerate(zip(dates, timeline_daily_p, timeline["market_status"])):
            if s == status:
                xs2.append(d); ys2.append(p)
                cd2.append((timeline["gas_usd"].iloc[i],
                             timeline["stablecoin_depth_usd"].iloc[i]/1e6,
                             s, timeline["note"].iloc[i]))
        if xs2:
            fig_tl.add_trace(go.Scatter(
                x=xs2, y=ys2, mode="markers",
                marker=dict(color=colour, size=8, line=dict(color="white", width=1.5)),
                showlegend=False,
                customdata=cd2,
                hovertemplate="<b>%{x}</b>  P(flash crash|24h): %{y:.1%}<br>Gas: $%{customdata[0]:.0f}/liq  |  Depth: $%{customdata[1]:.0f}M<br>Status: %{customdata[2]}<br>%{customdata[3]}<extra></extra>",
            ), row=2, col=1)
    fig_tl.add_hline(y=elev_daily, row=2, col=1,
                     line=dict(color="#f59e0b", dash="dash", width=1.2),
                     annotation_text="ELEVATED RISK (15%)", annotation_position="right",
                     annotation_font=dict(color="#f59e0b", size=9))
    fig_tl.add_hline(y=crit_daily, row=2, col=1,
                     line=dict(color="#ef4444", dash="dash", width=1.2),
                     annotation_text="CRITICAL (80%)", annotation_position="right",
                     annotation_font=dict(color="#ef4444", size=9))

    # Event vertical lines + staggered labels
    label_y_positions = [0.97, 0.91, 0.97, 0.91, 0.97, 0.91]
    for i, (idx, label) in enumerate(events):
        for row_num in [1, 2]:
            fig_tl.add_shape(type="line",
                xref="x", yref="paper",
                x0=dates[idx], x1=dates[idx], y0=0, y1=1,
                line=dict(color="rgba(80,80,80,0.2)", dash="dot", width=1),
                row=row_num, col=1)
        fig_tl.add_annotation(
            x=dates[idx], y=label_y_positions[i], yref="paper",
            text=f"<b>{label}</b>", showarrow=False,
            font=dict(size=8, color="#555"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(80,80,80,0.25)", borderwidth=1, borderpad=2,
            xanchor="center",
        )

    # Legend
    for status, colour in colour_map.items():
        fig_tl.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=colour, size=9), name=status,
        ))

    fig_tl.update_layout(
        title=dict(
            text="FTX Collapse — Fragility Signal (Mishricky 2025)<br>"
                 "<sup>Dot colour: green = STABLE (&lt;15%) | orange = ELEVATED RISK (15–80%) | red = CRITICAL (&gt;80%)</sup>",
            font=dict(size=13),
        ),
        template="plotly_white",
        height=580,
        legend=dict(orientation="h", x=0.5, y=-0.18, xanchor="center", yanchor="top",
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="#ccc", borderwidth=1),
        margin=dict(r=120, t=90, b=100),
    )
    fig_tl.update_yaxes(title_text="ETH (USD)", tickprefix="$", range=[eth_min, eth_max], row=1, col=1)
    fig_tl.update_yaxes(title_text="P(flash crash | 24h)", tickformat=".0%", row=2, col=1)
    fig_tl.update_xaxes(tickangle=-45, tickformat="%b %d", row=2, col=1)

    # Chart 2: 3-panel fragility drivers — the three inputs that push F toward 1
    # Panel 1: Stablecoin liquidity (phi_m = depth / total_debt) — falling = more fragile
    # Panel 2: Gas fee (USD per liquidation) — rising = bots withdraw, cascade amplifies
    # Panel 3: Book width (Gamma) — proxied by 5-day rolling realised ETH vol, which
    #           compresses HF buffers as price swings widen; theoretically Gamma = mean
    #           distance of health factors above 1.0 (BurdettJudd DeFi, theory.py L11)
    from plotly.subplots import make_subplots as _msp

    _dates   = list(timeline["date"])
    _depth_m = [v / 1e6 for v in timeline["stablecoin_depth_usd"]]   # $M
    _gas     = list(timeline["gas_usd"])
    _gamma   = [v * 100 for v in timeline["Gamma"]]                   # % for readability
    _status  = list(timeline["market_status"])
    _colour_map_drv = {"STABLE": "#2ecc71", "ELEVATED RISK": "#f39c12", "CRITICAL": "#e74c3c"}
    _dot_clr = [_colour_map_drv.get(s, "grey") for s in _status]

    # Nov 8 index for the vertical annotation
    _nov8_idx = next((i for i, d in enumerate(_dates) if d == "2022-11-08"), None)

    fig_cas = _msp(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.34, 0.33, 0.33],
        vertical_spacing=0.07,
        subplot_titles=[
            "① Stablecoin Liquidity  (φ_m = depth / total debt)",
            "② Gas Cost per Liquidation  (κ driver)",
            "③ Book Width — Γ  (5-day realised ETH vol, proxy for HF buffer distance)",
        ],
    )

    # ── Panel 1: Stablecoin depth ─────────────────────────────────────────────
    fig_cas.add_trace(go.Scatter(
        x=_dates, y=_depth_m,
        mode="lines", line=dict(color="#3498db", width=2.5),
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>Depth: $%{y:.0f}M<extra></extra>",
    ), row=1, col=1)
    fig_cas.add_trace(go.Scatter(
        x=_dates, y=_depth_m,
        mode="markers", marker=dict(color=_dot_clr, size=8, line=dict(color="white", width=1.5)),
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>Depth: $%{y:.0f}M<extra></extra>",
    ), row=1, col=1)

    # ── Panel 2: Gas cost (USD) ───────────────────────────────────────────────
    fig_cas.add_trace(go.Scatter(
        x=_dates, y=_gas,
        mode="lines", line=dict(color="#e67e22", width=2.5),
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>Gas: $%{y:.0f}/liq<extra></extra>",
    ), row=2, col=1)
    fig_cas.add_trace(go.Scatter(
        x=_dates, y=_gas,
        mode="markers", marker=dict(color=_dot_clr, size=8, line=dict(color="white", width=1.5)),
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>Gas: $%{y:.0f}/liq<extra></extra>",
    ), row=2, col=1)

    # ── Panel 3: Book width (Gamma %) ─────────────────────────────────────────
    fig_cas.add_trace(go.Scatter(
        x=_dates, y=_gamma,
        mode="lines", line=dict(color="#8e44ad", width=2.5),
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>Γ (realised vol): %{y:.2f}%<extra></extra>",
    ), row=3, col=1)
    fig_cas.add_trace(go.Scatter(
        x=_dates, y=_gamma,
        mode="markers", marker=dict(color=_dot_clr, size=8, line=dict(color="white", width=1.5)),
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>Γ (realised vol): %{y:.2f}%<extra></extra>",
    ), row=3, col=1)

    # ── Vertical line at Nov 8 across all panels ──────────────────────────────
    if _nov8_idx is not None:
        for _row in [1, 2, 3]:
            fig_cas.add_shape(
                type="line", xref="x", yref="paper",
                x0=_dates[_nov8_idx], x1=_dates[_nov8_idx], y0=0, y1=1,
                line=dict(color="rgba(231,76,60,0.45)", dash="dot", width=1.5),
                row=_row, col=1,
            )
        fig_cas.add_annotation(
            x=_dates[_nov8_idx], y=1.01, yref="paper",
            text="<b>Nov 8 — no rescue</b>", showarrow=False,
            font=dict(size=8, color="#c0392b"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(231,76,60,0.4)", borderwidth=1, borderpad=2,
            xanchor="center",
        )

    # ── Legend dummy traces (status colours) ─────────────────────────────────
    for _lbl, _clr in _colour_map_drv.items():
        fig_cas.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=_clr, size=9), name=_lbl,
        ))

    fig_cas.update_layout(
        title=dict(
            text="Fragility Drivers: What Pushed F Toward Critical<br>"
                 "<sup>Dot colour: green = STABLE | orange = ELEVATED RISK | red = CRITICAL. "
                 "F rises when liquidity falls, gas spikes, or book width narrows.</sup>",
            font=dict(size=11),
        ),
        template="plotly_white",
        height=580,
        legend=dict(orientation="h", x=0.5, y=-0.18, xanchor="center", yanchor="top",
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="#ccc", borderwidth=1),
        margin=dict(t=90, b=110, r=30),
    )
    fig_cas.update_yaxes(title_text="Depth ($M)", tickprefix="$", row=1, col=1)
    fig_cas.update_yaxes(title_text="Gas (USD/liq)", tickprefix="$", row=2, col=1)
    fig_cas.update_yaxes(title_text="Γ (vol %)", ticksuffix="%", row=3, col=1)
    fig_cas.update_xaxes(tickangle=-45, tickformat="%b %d", row=3, col=1)

    # Chart 3: 2-panel — ETH price (top) + Health Factor trajectory (bottom)
    # HF median per day is derived by scaling from the Nov 8 Dune anchor:
    #   HF_median(t) ≈ HF_anchor * (ETH_price(t) / ETH_price_anchor)
    # since HF = collateral_value / debt and collateral is ETH-denominated.
    # % below HF 1.2 is similarly scaled inversely (more positions at risk as ETH falls).
    from plotly.subplots import make_subplots as _msp2

    _ETH_ANCHOR   = 1245.0   # Nov 8 close (our backtest entry point)
    _HF_MED_ANC   = 1.48     # Dune snapshot median HF on Nov 8
    _HF_TAIL_ANC  = 0.082    # 8.2% below HF 1.2 on Nov 8

    _eth_prices  = list(timeline["eth_price_usd"])
    _hf_median   = [_HF_MED_ANC  * (p / _ETH_ANCHOR) for p in _eth_prices]
    # tail risk scales inversely: lower ETH → more positions cross the 1.2 threshold
    _hf_tail_pct = [min(_HF_TAIL_ANC * (_ETH_ANCHOR / p), 0.60) for p in _eth_prices]

    _colour_map_p = {"STABLE": "#2ecc71", "ELEVATED RISK": "#f39c12", "CRITICAL": "#e74c3c"}
    _dot_clr_p = [_colour_map_p.get(s, "grey") for s in list(timeline["market_status"])]

    fig_cmp = _msp2(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.50],
        vertical_spacing=0.08,
        subplot_titles=[
            "ETH / USD",
            "Health Factor  (median + % positions below HF 1.2)",
        ],
    )

    # ── Panel 1: ETH price ────────────────────────────────────────────────────
    _eth_min = int(min(_eth_prices) * 0.94)
    _eth_max = int(max(_eth_prices) * 1.06)
    fig_cmp.add_trace(go.Scatter(
        x=_dates, y=_eth_prices,
        mode="lines", line=dict(color="#7c5cbf", width=2.5),
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>ETH: $%{y:,}<extra></extra>",
    ), row=1, col=1)
    fig_cmp.add_trace(go.Scatter(
        x=_dates, y=_eth_prices,
        mode="markers", marker=dict(color=_dot_clr_p, size=8, line=dict(color="white", width=1.5)),
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>ETH: $%{y:,}<extra></extra>",
    ), row=1, col=1)

    # ── Panel 2: HF median line ───────────────────────────────────────────────
    fig_cmp.add_trace(go.Scatter(
        x=_dates, y=_hf_median,
        mode="lines", line=dict(color="#2980b9", width=2.5),
        name="Median HF",
        hovertemplate="<b>%{x}</b><br>Median HF: %{y:.2f}<extra></extra>",
    ), row=2, col=1)
    fig_cmp.add_trace(go.Scatter(
        x=_dates, y=_hf_median,
        mode="markers", marker=dict(color=_dot_clr_p, size=8, line=dict(color="white", width=1.5)),
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>Median HF: %{y:.2f}<extra></extra>",
    ), row=2, col=1)

    # ── % below HF 1.2 as shaded area on secondary y ─────────────────────────
    fig_cmp.add_trace(go.Scatter(
        x=_dates, y=[v * 100 for v in _hf_tail_pct],
        mode="lines", line=dict(color="#e74c3c", width=1.5, dash="dot"),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.10)",
        name="% positions HF < 1.2",
        yaxis="y4",
        hovertemplate="<b>%{x}</b><br>HF < 1.2: %{y:.1f}%<extra></extra>",
    ), row=2, col=1)

    # ── HF = 1.2 danger threshold line ───────────────────────────────────────
    fig_cmp.add_hline(y=1.2, row=2, col=1,
                      line=dict(color="#e74c3c", dash="dash", width=1.2),
                      annotation_text="HF = 1.2 (liquidation risk)",
                      annotation_position="right",
                      annotation_font=dict(color="#e74c3c", size=9))

    # ── Nov 8 vertical line ───────────────────────────────────────────────────
    if _nov8_idx is not None:
        for _row in [1, 2]:
            fig_cmp.add_shape(
                type="line", xref="x", yref="paper",
                x0=_dates[_nov8_idx], x1=_dates[_nov8_idx], y0=0, y1=1,
                line=dict(color="rgba(231,76,60,0.45)", dash="dot", width=1.5),
                row=_row, col=1,
            )
        fig_cmp.add_annotation(
            x=_dates[_nov8_idx], y=1.01, yref="paper",
            text="<b>Nov 8 — no rescue</b>", showarrow=False,
            font=dict(size=8, color="#c0392b"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(231,76,60,0.4)", borderwidth=1, borderpad=2,
            xanchor="center",
        )

    # ── Legend dummy traces ───────────────────────────────────────────────────
    for _lbl, _clr in _colour_map_p.items():
        fig_cmp.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=_clr, size=9), name=_lbl,
        ))

    fig_cmp.update_layout(
        title=dict(
            text="ETH Price & Health Factor Trajectory<br>"
                 "<sup>HF median scaled daily from Nov 8 Dune anchor (1.48). "
                 "Dot colour: green = STABLE | orange = ELEVATED RISK | red = CRITICAL.</sup>",
            font=dict(size=11),
        ),
        template="plotly_white",
        height=580,
        legend=dict(orientation="h", x=0.5, y=-0.18, xanchor="center", yanchor="top",
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="#ccc", borderwidth=1),
        margin=dict(t=90, b=110, r=80),
        yaxis4=dict(
            title="HF < 1.2 (%)", overlaying="y3", side="right",
            ticksuffix="%", showgrid=False,
            range=[0, max(v * 100 for v in _hf_tail_pct) * 1.4],
        ),
    )
    fig_cmp.update_yaxes(title_text="ETH (USD)", tickprefix="$",
                         range=[_eth_min, _eth_max], row=1, col=1)
    fig_cmp.update_yaxes(title_text="Median HF", range=[1.0, max(_hf_median) * 1.1],
                         row=2, col=1)
    fig_cmp.update_xaxes(tickangle=-45, tickformat="%b %d", row=2, col=1)

    # Chart 4: Speculative discount  P = phi_s - phi_s_hat  (Proposition 12)
    # In this calibration P is always negative — a speculative discount, not a premium.
    # phi_s_hat is constant (fixed preference params), so all movement in P comes from
    # phi_m (liquidity) and kappa (gas cost) compressing the equilibrium price phi_s.
    # The chart shows: how much the discount deepened during the FTX collapse, and
    # which market-status regime the pool was in at each point.
    from theory import BurdettJuddDeFi as _BJD

    _spec_premium = []
    for _, _row in timeline.iterrows():
        try:
            _m = _BJD(kappa=float(_row["kappa"]), phi_m=float(_row["phi_m"]),
                      Gamma=float(_row["Gamma"]))
            _spec_premium.append(_m.speculative_premium)
        except Exception:
            _spec_premium.append(np.nan)

    # Baseline = Nov 1 value (pre-crisis)
    _baseline_P = _spec_premium[0]
    _deepening  = [p - _baseline_P for p in _spec_premium]   # negative = deeper discount vs pre-crisis

    _colour_map_sp = {"STABLE": "#2ecc71", "ELEVATED RISK": "#f39c12", "CRITICAL": "#e74c3c"}
    _dot_clr_sp = [_colour_map_sp.get(s, "grey") for s in list(timeline["market_status"])]

    fig_sp = go.Figure()

    # Shaded fill from baseline to P (shows widening discount)
    fig_sp.add_trace(go.Scatter(
        x=_dates + _dates[::-1],
        y=_spec_premium + [_baseline_P] * len(_dates),
        fill="toself",
        fillcolor="rgba(231,76,60,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))

    # Pre-crisis baseline reference line
    fig_sp.add_shape(
        type="line", x0=_dates[0], x1=_dates[-1],
        y0=_baseline_P, y1=_baseline_P,
        line=dict(color="#7f8c8d", dash="dot", width=1.5),
    )
    fig_sp.add_annotation(
        x=_dates[-1], y=_baseline_P,
        text="Pre-crisis baseline (Nov 1)",
        showarrow=False, xanchor="right", yanchor="bottom",
        font=dict(size=9, color="#7f8c8d"),
    )

    # P line
    fig_sp.add_trace(go.Scatter(
        x=_dates, y=_spec_premium,
        mode="lines", line=dict(color="#c0392b", width=2.5),
        name="Speculative discount P",
        hovertemplate="<b>%{x}</b><br>P = %{y:.4f}<extra></extra>",
    ))

    # Status-coloured dots on the line
    fig_sp.add_trace(go.Scatter(
        x=_dates, y=_spec_premium,
        mode="markers",
        marker=dict(color=_dot_clr_sp, size=9, line=dict(color="white", width=1.5)),
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>P = %{y:.4f}<extra></extra>",
    ))

    # Deepening annotation at trough (Nov 9)
    _trough_idx = int(np.argmin(_spec_premium))
    _deepening_at_trough = _spec_premium[_trough_idx] - _baseline_P
    fig_sp.add_annotation(
        x=_dates[_trough_idx],
        y=_spec_premium[_trough_idx],
        text=f"Peak deepening<br>ΔP = {_deepening_at_trough:.4f}",
        showarrow=True, arrowhead=2, arrowcolor="#c0392b",
        ax=40, ay=-40,
        font=dict(size=9, color="#c0392b"),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#c0392b", borderwidth=1, borderpad=3,
    )

    # Nov 8 vertical line
    if _nov8_idx is not None:
        fig_sp.add_shape(
            type="line", xref="x", yref="paper",
            x0=_dates[_nov8_idx], x1=_dates[_nov8_idx], y0=0, y1=1,
            line=dict(color="rgba(231,76,60,0.45)", dash="dot", width=1.5),
        )
        fig_sp.add_annotation(
            x=_dates[_nov8_idx], y=1.01, yref="paper",
            text="<b>Nov 8 — no rescue</b>", showarrow=False,
            font=dict(size=8, color="#c0392b"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(231,76,60,0.4)", borderwidth=1, borderpad=2,
            xanchor="center",
        )

    # Legend dummy traces for status colours
    for _lbl, _clr in _colour_map_sp.items():
        fig_sp.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=_clr, size=9), name=_lbl,
        ))

    fig_sp.update_layout(
        title=dict(
            text="Speculative Discount  P = φₛ − φ̂ₛ  (Proposition 12)<br>"
                 "<sup>P &lt; 0 throughout: fragility forces investors to price the asset below fundamental. "
                 "Discount deepens as φ_m (liquidity) falls and κ (gas) spikes.</sup>",
            font=dict(size=11),
        ),
        template="plotly_white",
        height=580,
        xaxis=dict(tickangle=-45, tickformat="%b %d"),
        yaxis=dict(title="Speculative discount P  (normalised units)"),
        legend=dict(orientation="h", x=0.5, y=-0.18, xanchor="center", yanchor="top",
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="#ccc", borderwidth=1),
        margin=dict(t=90, b=110, r=30),
    )

    return callout, fig_tl, fig_cas, fig_cmp, fig_sp, visible_content, hidden


if __name__ == "__main__":
    app.run(debug=True)
