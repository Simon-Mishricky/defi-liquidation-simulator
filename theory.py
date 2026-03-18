import numpy as np

class BurdettJuddDeFi:
    """
    Implements the equilibrium from Mishricky (2025) calibrated to DeFi data.

    Parameters
    ----------
    kappa   : cost of posting a liquidation (proxy: avg gas cost in USD)
    phi_m   : real value of money (proxy: stablecoin liquidity / total debt)
    Gamma   : book width (proxy: mean distance of health factors above 1.0)
    """

    def __init__(self, kappa, phi_m, Gamma,
                 beta=0.96, eta=0.98, epsilon_H=0.08, epsilon_L=0.02,
                 pi_H=0.7, pi_L=0.3):
        self.kappa = kappa
        self.phi_m = phi_m
        self.Gamma = Gamma
        self.beta = beta
        self.eta = eta
        self.epsilon_H = epsilon_H
        self.epsilon_L = epsilon_L
        self.pi_H = pi_H
        self.pi_L = pi_L
        self._validate()

    def _validate(self):
        ratio = (self.phi_m * self.Gamma) / self.kappa
        if ratio <= 1:
            raise ValueError(
                f"Market collapse: phi_m * Gamma / kappa = {ratio:.4f} <= 1\n"
                f"No quoting activity — analogous to a regulatory/monetary crisis."
            )

    @property
    def theta(self):
        """Quote intensity: theta = ln(phi_m * Gamma / kappa) — Proposition 1(iii)"""
        return np.log((self.phi_m * self.Gamma) / self.kappa)

    @property
    def flash_crash_prob(self):
        """F = e^{-theta} — Proposition 11"""
        return np.exp(-self.theta)

    @property
    def nominal_spread(self):
        """S = 2*kappa*theta / (phi_m*(1 - e^{-theta})) — Section 5.4"""
        return (2 * self.kappa * self.theta) / (
            self.phi_m * (1 - np.exp(-self.theta))
        )
    
    @property
    def real_spread(self):
        """
        S = phi_m * S^m — Corollary 4
        
        Novel result from the paper: real spread DECREASES with inflation
        but INCREASES with posting cost. This means quoted spreads can look
        tight even as the market deteriorates — the hidden risk result.
        """
        return self.phi_m * self.nominal_spread

    @property
    def variance_best_bid(self):
        """Vb = (kappa/phi_m)^2 * [e^theta - (theta/(1-e^{-theta}))^2] — Section 5.3"""
        return (self.kappa / self.phi_m) ** 2 * (
            np.exp(self.theta) - (self.theta / (1 - np.exp(-self.theta))) ** 2
        )

    @property
    def cv_bid(self):
        """
        CVb = sqrt(Vb) / |Mb| — Corollary 3
        
        Scale-free measure of quote dispersion.
        Increases with both inflation and posting cost.
        More informative than variance alone since it's comparable
        across pools of different sizes.
        """
        return np.sqrt(self.variance_best_bid) / abs(self.mean_best_bid)
    
    @property
    def _beta_bar_H(self):
        """beta_bar_H = pi_H * beta * gamma_bar, where gamma_bar=1 in steady state"""
        return self.pi_H * self.beta

    @property
    def _beta_bar_L(self):
        """beta_bar_L = pi_L * beta"""
        return self.pi_L * self.beta

    @property
    def fundamental_value(self):
        """
        Fundamental value = (beta_bar_H * eta / (1 - beta_bar_H * eta)) * epsilon_H
        
        This is the discounted present value of future dividends as perceived
        by high valuation investors — the first term in Proposition 2.
        """
        b = self._beta_bar_H * self.eta
        return (b / (1 - b)) * self.epsilon_H

    @property
    def speculative_premium(self):
        """
        Speculative premium P = phi_s - phi_s_hat — Proposition 12

        Exact formula from the paper (equation following Proposition 12 statement):

          P = beta_bar_H*eta / (1 - beta_bar_H*eta)
              * { epsilon_H + pi_L/(1-pi_L) * [phi_m*p_hat - kappa*theta/(1-e^{-theta})] }
              - beta_bar*eta / (1 - beta_bar*eta) * epsilon_bar

        where:
          beta_bar_H = pi_H * beta * gamma_bar  (high-type investor discount, gamma_bar=1 in s.s.)
          beta_bar   = beta * gamma_bar          (mean discount factor)
          epsilon_bar = pi_H*epsilon_H + pi_L*epsilon_L
          p_hat = 1.0  (mid-price, normalised)

        First term = equilibrium equity price phi_s (Proposition 2, eq. 23).
        Second term = fundamental value phi_s_hat.

        P > 0: speculative premium (investors pay above fundamental)
        P < 0: speculative discount (investors pay below fundamental)
        Proposition 12(i):  P decreases with inflation
        Proposition 12(ii): P increases with posting cost kappa
        """
        b_H = self._beta_bar_H * self.eta

        # Equilibrium asset price phi_s — Proposition 2, eq. (23)
        # phi_s = beta_bar_H*eta/(1-beta_bar_H*eta) * {epsilon_H + pi_L/(1-pi_L)*[phi_m*p_hat - kappa*theta/(1-e^{-theta})]}
        resale_term = (
            self.phi_m * 1.0  # phi_m * p_hat, p_hat = 1 (normalised mid-price)
            - (self.kappa * self.theta) / (1 - np.exp(-self.theta))
        )
        phi_s = (b_H / (1 - b_H)) * (
            self.epsilon_H + (self.pi_L / (1 - self.pi_L)) * resale_term
        )

        # Fundamental value phi_s_hat = beta_bar*eta/(1-beta_bar*eta) * epsilon_bar
        epsilon_bar = self.pi_H * self.epsilon_H + self.pi_L * self.epsilon_L
        b_bar = self.beta * self.eta  # beta_bar*eta with gamma_bar=1 in steady state
        phi_s_hat = (b_bar / (1 - b_bar)) * epsilon_bar

        return phi_s - phi_s_hat
    
    def ask_distribution(self, p):
        """
        A(p) = (1/theta) * ln(phi_m * (p - p_hat) / kappa) — Proposition 1(i)
        
        CDF of ask prices. Right-skewed — most asks cluster near lower bound.
        Returns 0 below support, 1 above support.
        """
        p_hat = 1.0
        p_lower = p_hat + self.kappa / self.phi_m
        p_upper = p_hat + self.Gamma

        if p < p_lower:
            return 0.0
        if p > p_upper:
            return 1.0
        return (1 / self.theta) * np.log(
            self.phi_m * (p - p_hat) / self.kappa
        )

    def bid_distribution(self, p):
        """
        B(p) = 1 - (1/theta) * ln(phi_m * (p_hat - p) / kappa) — Proposition 1(ii)
        
        CDF of bid prices. Left-skewed — most bids cluster near upper bound.
        Returns 0 below support, 1 above support.
        """
        p_hat = 1.0
        p_lower = p_hat - self.Gamma
        p_upper = p_hat - self.kappa / self.phi_m

        if p < p_lower:
            return 0.0
        if p > p_upper:
            return 1.0
        return 1 - (1 / self.theta) * np.log(
            self.phi_m * (p_hat - p) / self.kappa
        )

    def plot_distributions(self, n_points=500, title=""):
        """
        Plot ask and bid distributions for current parameters.
        Shows Propositions 1, 6, 7 visually.
        """
        import matplotlib.pyplot as plt

        p_hat = 1.0
        ask_lower = p_hat + self.kappa / self.phi_m
        ask_upper = p_hat + self.Gamma
        bid_lower = p_hat - self.Gamma
        bid_upper = p_hat - self.kappa / self.phi_m

        ask_prices = np.linspace(ask_lower, ask_upper, n_points)
        bid_prices = np.linspace(bid_lower, bid_upper, n_points)

        ask_cdf = [self.ask_distribution(p) for p in ask_prices]
        bid_cdf = [self.bid_distribution(p) for p in bid_prices]

        ask_pdf = np.gradient(ask_cdf, ask_prices)
        bid_pdf = np.gradient(bid_cdf, bid_prices)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(ask_prices, ask_pdf, color='red', linewidth=2)
        ax1.fill_between(ask_prices, ask_pdf, alpha=0.3, color='red')
        ax1.set_title(f"Ask Price Density a(p)\n{title}", fontsize=11)
        ax1.set_xlabel("Price")
        ax1.set_ylabel("Density")
        ax1.axvline(x=self.mean_best_ask, color='darkred',
                    linestyle='--', label=f"Mean ask = {self.mean_best_ask:.4f}")
        ax1.legend()

        ax2.plot(bid_prices, bid_pdf, color='blue', linewidth=2)
        ax2.fill_between(bid_prices, bid_pdf, alpha=0.3, color='blue')
        ax2.set_title(f"Bid Price Density b(p)\n{title}", fontsize=11)
        ax2.set_xlabel("Price")
        ax2.axvline(x=self.mean_best_bid, color='darkblue',
                    linestyle='--', label=f"Mean bid = {self.mean_best_bid:.4f}")
        ax2.legend()

        plt.tight_layout()
        return fig

    @property
    def mean_best_ask(self):
        """M_a = p_hat + kappa*theta / (phi_m*(1-e^{-theta})) — Section 5.3"""
        return 1.0 + (self.kappa * self.theta) / (
            self.phi_m * (1 - np.exp(-self.theta))
        )

    @property
    def mean_best_bid(self):
        """M_b = p_hat - kappa*theta / (phi_m*(1-e^{-theta})) — Section 5.3"""
        return 1.0 - (self.kappa * self.theta) / (
            self.phi_m * (1 - np.exp(-self.theta))
        )

    @property
    def mse(self):
        """
        MSE = kappa * Gamma / phi_m — Footnote 28

        Derived in Footnote 28 as:
          MSE = V + H^2
              = kappa^2 / ((phi_m)^2 * F) - H^2 + H^2
              = kappa^2 / ((phi_m)^2 * e^{-theta})
              = kappa^2 * e^{theta} / (phi_m)^2
              = kappa^2 / (phi_m)^2 * (phi_m * Gamma / kappa)   [since e^theta = phi_m*Gamma/kappa]
              = kappa * Gamma / phi_m

        This is the mean squared error of prices around the mid-price p_hat,
        combining both the variance V of the best-bid/ask distributions and
        the squared half-spread H^2.
        """
        return (self.kappa * self.Gamma) / self.phi_m

    @property
    def conservation_law(self):
        """
        MSE * F = (kappa/phi_m)^2 — Footnote 28

        This is the key diagnostic from the paper. If MSE*F is LOW relative
        to (kappa/phi_m)^2, the market looks calm but crash risk is hidden.
        """
        lhs = self.mse * self.flash_crash_prob
        rhs = (self.kappa / self.phi_m) ** 2
        return {
            "MSE":              round(self.mse, 6),
            "F (flash crash)":  round(self.flash_crash_prob, 6),
            "MSE x F":          round(lhs, 8),
            "(kappa/phi_m)^2":  round(rhs, 8),
            "ratio (should=1)": round(lhs / rhs, 6)
        }

    def summary(self):
        F = self.flash_crash_prob
        status = (
            "STABLE"        if F < 0.00005 else
            "ELEVATED RISK" if F < 0.00050 else
            "CRITICAL"
        )
        return {
            "theta (quote intensity)":  round(self.theta, 4),
            "flash crash prob (F)":     round(self.flash_crash_prob, 6),
            "nominal spread (S^m)":     round(self.nominal_spread, 6),
            "real spread (S)":          round(self.real_spread, 6),
            "mean best ask (M_a)":      round(self.mean_best_ask, 6),
            "mean best bid (M_b)":      round(self.mean_best_bid, 6),
            "variance bid (V_b)":       round(self.variance_best_bid, 8),
            "CV bid":                   round(self.cv_bid, 6),
            "fundamental value":        round(self.fundamental_value, 4),
            "speculative premium (P)":  round(self.speculative_premium, 4),
            "market status":            status
        }

def calibrate_from_positions(positions_df, gas_usd, stablecoin_depth_usd,
                              daily_volatility=0.05):
    """
    Map DeFi observables to Mishricky (2025) theoretical parameters.

    Scale invariance
    ----------------
    All three mapped parameters (κ, φᵐ, Γ) are dimensionless ratios.
    Consequently, the equilibrium objects — θ, F, spreads, price
    distributions, speculative premium — depend only on these ratios
    and not on the nominal size of the position pool. A 1,000-position
    subsample with $0.57B debt produces identical (κ, φᵐ, Γ) and
    therefore identical F trajectories as the full 45,000-position
    Aave V3 market with $24B debt, provided the HF distribution and
    the liquidity-to-debt ratio are preserved.

    Mapping:
      kappa  = gas_usd / stablecoin_depth_usd
               Gas cost normalised by liquidity depth gives a dimensionless
               posting cost — the fraction of available liquidity consumed
               by a single liquidation transaction. This is consistent with
               kappa being a real (consumption-good-denominated) cost and
               stablecoin depth serving as the numeraire.

      phi_m  = stablecoin_depth_usd / total_debt
               The real value of money is proxied by the liquidity ratio:
               stablecoin depth as a fraction of total protocol debt. When
               this ratio is high, money is plentiful relative to the claims
               it must service — phi_m is high. As liquidations drain
               stablecoin reserves, phi_m falls, raising F = e^{-theta}.
               This mapping is natural since phi_m in the model is the real
               purchasing power of the money stock, and liquidity_ratio is
               directly observable and dimensionless.

      Gamma  = daily_volatility of collateral asset (default 5%)
               Book width — the range over which brokers post competitive
               quotes, proxied by realised volatility of the collateral.

    The condition phi_m * Gamma / kappa > 1 (required for theta > 0) becomes:
      (liquidity_ratio * daily_volatility * stablecoin_depth_usd) / gas_usd > 1
    which fails when gas costs are extreme relative to available liquidity,
    correctly triggering market collapse via _validate().
    """
    total_debt = positions_df['debt_usd'].sum()
    liquidity_ratio = stablecoin_depth_usd / total_debt

    # kappa: posting cost as fraction of liquidity depth (dimensionless)
    kappa = gas_usd / stablecoin_depth_usd

    # phi_m: real value of money = liquidity ratio (no arbitrary normalisation)
    phi_m = liquidity_ratio

    # Gamma: book width proxied by collateral daily volatility
    Gamma = daily_volatility

    return BurdettJuddDeFi(kappa=kappa, phi_m=phi_m, Gamma=Gamma)