import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Buy vs Rent Analyzer", page_icon="🏠", layout="wide")

# ── ZIP code appreciation data (HUD / Zillow regional proxies) ──────────────
ZIP_APPRECIATION = {
    "100": 0.065, "101": 0.065, "102": 0.065,  # NYC
    "900": 0.072, "901": 0.072, "902": 0.072,  # LA
    "941": 0.075, "940": 0.075,                 # SF Bay
    "980": 0.068, "981": 0.068,                 # Seattle
    "850": 0.078, "852": 0.078,                 # Phoenix
    "733": 0.071, "770": 0.071,                 # Houston
    "606": 0.055, "607": 0.055,                 # Chicago
    "750": 0.073, "752": 0.073,                 # Dallas
    "321": 0.076, "322": 0.076,                 # Orlando
    "331": 0.074, "330": 0.074,                 # Miami
    "972": 0.066, "971": 0.066,                 # Portland
    "800": 0.069, "801": 0.069,                 # Denver
    "891": 0.082, "889": 0.082,                 # Las Vegas
    "787": 0.074, "786": 0.074,                 # Austin
    "919": 0.071, "275": 0.071,                 # Raleigh-Durham
    "300": 0.068,                               # Atlanta
    "191": 0.058, "190": 0.058,                 # Philadelphia
    "021": 0.061, "022": 0.061,                 # Boston
    "481": 0.052, "482": 0.052,                 # Detroit
    "441": 0.053, "440": 0.053,                 # Cleveland
    "553": 0.057, "554": 0.057,                 # Minneapolis
}
NATIONAL_APPRECIATION = 0.046


def get_appreciation_rate(zip_code: str) -> tuple[float, str]:
    prefix3 = zip_code[:3] if len(zip_code) >= 3 else ""
    prefix2 = zip_code[:2] if len(zip_code) >= 2 else ""
    rate = ZIP_APPRECIATION.get(prefix3) or ZIP_APPRECIATION.get(prefix2)
    if rate:
        return rate, f"Regional estimate for ZIP {zip_code}"
    return NATIONAL_APPRECIATION, "National historical average"


# ── Monte Carlo engine ───────────────────────────────────────────────────────

def run_monte_carlo(
    home_price, down_pct, loan_rate, years,
    appr_rate, appr_vol,
    rent_monthly, rent_inflation,
    invest_return, invest_vol,
    prop_tax_rate, insurance_rate, maintenance_rate,
    income, tax_bracket,
    n_sims=1000,
):
    np.random.seed(42)
    months = years * 12
    down = home_price * down_pct
    loan = home_price - down
    monthly_rate = loan_rate / 12
    pmt = (loan * (monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)
           if loan > 0 else 0.0)

    buy_wealth = np.zeros(n_sims)
    rent_wealth = np.zeros(n_sims)
    buy_paths = np.zeros((n_sims, years + 1))
    rent_paths = np.zeros((n_sims, years + 1))

    for s in range(n_sims):
        appr_shocks = np.random.normal(appr_rate, appr_vol, years)
        inv_shocks  = np.random.normal(invest_return, invest_vol, years)

        # BUY path
        balance = loan
        home_value = home_price
        buy_net = -down  # starts negative (down payment outlay); grows as costs accumulate

        for yr in range(years):
            home_value *= (1 + appr_shocks[yr])
            yr_interest = 0.0
            for m in range(12):
                interest = balance * monthly_rate
                principal = pmt - interest
                balance = max(0, balance - principal)
                yr_interest += interest

            prop_tax_annual = home_value * prop_tax_rate
            deductible = yr_interest + min(prop_tax_annual, 10_000)
            tax_saving = deductible * tax_bracket

            ins = home_value * insurance_rate
            maint = home_value * maintenance_rate
            # annual_cost is net of tax savings — buy_net already reflects them
            annual_cost = pmt * 12 + prop_tax_annual + ins + maint - tax_saving
            buy_net -= annual_cost
            # wealth = equity (appreciation + principal paydown) + net cash flows
            buy_paths[s, yr + 1] = (home_value - balance) + buy_net

        sale_costs = home_value * 0.06
        buy_wealth[s] = home_value - balance - sale_costs + buy_net

        # RENT path
        portfolio = down
        rent = rent_monthly
        rent_net = 0.0
        buy_monthly_cost = pmt + (home_price * (prop_tax_rate + insurance_rate + maintenance_rate)) / 12

        for yr in range(years):
            portfolio *= (1 + inv_shocks[yr])
            diff = max(0, buy_monthly_cost - rent)
            portfolio += diff * 12
            rent_net -= rent * 12
            rent *= (1 + rent_inflation)
            rent_paths[s, yr + 1] = portfolio + rent_net

        rent_wealth[s] = portfolio + rent_net

    return buy_wealth, rent_wealth, buy_paths, rent_paths


SELL_HORIZONS = [1, 3, 5, 10]

def sell_after_analysis(
    home_price, down_pct, loan_rate, loan_term,
    appr_rate, prop_tax_rate, insurance_rate, maintenance_rate,
    tax_bracket, rent_monthly, rent_inflation, invest_return,
):
    """
    Deterministic (median-rate) wealth for buyer and renter if the buyer sells
    after 1, 3, 5, and 10 years.  Returns a list of dicts, one per horizon.
    """
    down = home_price * down_pct
    loan = home_price - down
    monthly_rate = loan_rate / 12
    n_total = loan_term * 12
    pmt = (loan * (monthly_rate * (1 + monthly_rate) ** n_total) / ((1 + monthly_rate) ** n_total - 1)
           if loan > 0 else 0.0)
    selling_cost_pct = 0.06

    results = []
    balance = loan
    home_value = home_price
    buy_net = -down          # cumulative net cash position for buyer
    portfolio = down         # renter invests the down payment
    rent = rent_monthly
    rent_net = 0.0           # cumulative rent paid (negative)
    buy_monthly_cost = pmt + home_price * (prop_tax_rate + insurance_rate + maintenance_rate) / 12

    for yr in range(1, max(SELL_HORIZONS) + 1):
        # ── advance buy path one year ──
        home_value *= (1 + appr_rate)
        yr_interest = 0.0
        for _ in range(12):
            interest  = balance * monthly_rate
            principal = pmt - interest
            balance   = max(0, balance - principal)
            yr_interest += interest

        prop_tax_annual = home_value * prop_tax_rate
        deductible  = yr_interest + min(prop_tax_annual, 10_000)
        tax_saving  = deductible * tax_bracket
        ins   = home_value * insurance_rate
        maint = home_value * maintenance_rate
        annual_cost = pmt * 12 + prop_tax_annual + ins + maint - tax_saving
        buy_net -= annual_cost

        # ── advance rent path one year ──
        portfolio *= (1 + invest_return)
        diff = max(0, buy_monthly_cost - rent)
        portfolio += diff * 12
        rent_net  -= rent * 12
        rent      *= (1 + rent_inflation)

        if yr in SELL_HORIZONS:
            sale_proceeds = home_value - balance - home_value * selling_cost_pct
            buy_wealth  = sale_proceeds + buy_net          # net proceeds + cash flows
            rent_wealth = portfolio + rent_net

            results.append({
                "horizon":          yr,
                "buy_wealth":       buy_wealth,
                "rent_wealth":      rent_wealth,
                "advantage":        buy_wealth - rent_wealth,
                "equity":           home_value - balance,
                "principal_paid":   loan - balance,
                "appreciation":     home_value - home_price,
                "selling_costs":    home_value * selling_cost_pct,
                "home_value":       home_value,
            })

    return results


# ── Helpers ──────────────────────────────────────────────────────────────────

def compute_breakeven(buy_paths: np.ndarray, rent_paths: np.ndarray) -> dict:
    """
    Median-path breakeven: first year the median buy wealth >= median rent wealth.
    Also tracks per-simulation crossovers for the distribution histogram.
    """
    n_sims, n_years = buy_paths.shape

    # ── Median-path crossover (what the chart shows) ──────────────────────────
    med_buy  = np.median(buy_paths, axis=0)
    med_rent = np.median(rent_paths, axis=0)
    median_crossover = None
    for yr in range(1, n_years):
        if med_buy[yr] >= med_rent[yr]:
            median_crossover = yr
            break

    # ── Per-simulation crossover (distribution) ───────────────────────────────
    sim_crossovers = []
    for s in range(n_sims):
        crossed = np.where(buy_paths[s] >= rent_paths[s])[0]
        later = crossed[crossed > 0]
        sim_crossovers.append(int(later[0]) if len(later) > 0 else None)

    valid = [y for y in sim_crossovers if y is not None]
    never = n_sims - len(valid)
    return {
        "median_crossover": median_crossover,   # where the two median lines cross
        "sim_years": sim_crossovers,            # per-sim crossover (for histogram)
        "sim_median": float(np.median(valid)) if valid else None,
        "p25":        float(np.percentile(valid, 25)) if valid else None,
        "p75":        float(np.percentile(valid, 75)) if valid else None,
        "never_pct":  never / n_sims * 100,
        "n_valid":    len(valid),
        "med_buy":    med_buy,
        "med_rent":   med_rent,
    }


def input_fingerprint(**kwargs) -> tuple:
    """Hashable snapshot of all simulation inputs."""
    return tuple(kwargs.values())


def run_cash_vs_mortgage_mc(
    home_price, down_pct, loan_rate, loan_term,
    appr_rate, appr_vol,
    invest_return, invest_vol,
    prop_tax_rate, insurance_rate, maintenance_rate,
    tax_bracket,
    years,
    n_sims=1000,
):
    """
    All-cash vs leveraged mortgage — both start with `home_price` in available cash.
    Cash buyer:     pays full price upfront; invests the freed-up mortgage payment (pmt×12)
                    each year; only SALT prop-tax deduction (no mortgage interest).
    Mortgage buyer: pays down_pct * home_price; invests the loan-amount remainder as a
                    lump sum on day 1; pays P&I monthly; deducts mortgage interest + SALT.
    """
    np.random.seed(42)
    months_total = loan_term * 12
    down = home_price * down_pct
    loan = home_price - down
    monthly_rate = loan_rate / 12
    pmt = (loan * (monthly_rate * (1 + monthly_rate) ** months_total) /
           ((1 + monthly_rate) ** months_total - 1) if loan > 0 else 0.0)

    cash_wealth = np.zeros(n_sims)
    mtg_wealth  = np.zeros(n_sims)
    cash_paths  = np.zeros((n_sims, years + 1))
    mtg_paths   = np.zeros((n_sims, years + 1))

    for s in range(n_sims):
        appr_shocks = np.random.normal(appr_rate, appr_vol, years)
        inv_shocks  = np.random.normal(invest_return, invest_vol, years)

        # ── ALL-CASH BUYER ──────────────────────────────────────────────────────
        hv_c           = home_price
        cash_net       = -home_price   # full purchase price paid upfront
        cash_portfolio = 0.0           # starts with nothing invested; adds pmt each year

        for yr in range(years):
            hv_c           *= (1 + appr_shocks[yr])
            cash_portfolio *= (1 + inv_shocks[yr])
            cash_portfolio += pmt * 12   # invest the mortgage payments they're not making

            prop_tax_yr = hv_c * prop_tax_rate
            tax_saving  = min(prop_tax_yr, 10_000) * tax_bracket  # SALT only, no mtg interest
            cash_net   -= (prop_tax_yr + hv_c * insurance_rate + hv_c * maintenance_rate - tax_saving)
            cash_paths[s, yr + 1] = hv_c + cash_portfolio + cash_net

        cash_wealth[s] = hv_c - hv_c * 0.06 + cash_portfolio + cash_net

        # ── MORTGAGE BUYER ─────────────────────────────────────────────────────
        hv_m      = home_price
        balance   = loan
        portfolio = loan   # lump-sum: invests the portion not used for down payment
        mtg_net   = -down

        for yr in range(years):
            hv_m      *= (1 + appr_shocks[yr])
            portfolio *= (1 + inv_shocks[yr])

            yr_interest = 0.0
            for m in range(12):
                interest  = balance * monthly_rate
                principal = pmt - interest
                balance   = max(0, balance - principal)
                yr_interest += interest

            prop_tax_yr = hv_m * prop_tax_rate
            deductible  = yr_interest + min(prop_tax_yr, 10_000)
            tax_saving  = deductible * tax_bracket
            mtg_net -= (pmt * 12 + prop_tax_yr + hv_m * insurance_rate +
                        hv_m * maintenance_rate - tax_saving)
            mtg_paths[s, yr + 1] = (hv_m - balance) + portfolio + mtg_net

        mtg_wealth[s] = hv_m - balance - hv_m * 0.06 + portfolio + mtg_net

    return cash_wealth, mtg_wealth, cash_paths, mtg_paths


def cash_vs_mortgage_deterministic(
    home_price, down_pct, loan_rate, loan_term,
    appr_rate, invest_return,
    prop_tax_rate, insurance_rate, maintenance_rate,
    tax_bracket, years,
):
    """Year-by-year deterministic breakdown at median rates."""
    down = home_price * down_pct
    loan = home_price - down
    monthly_rate = loan_rate / 12
    months_total = loan_term * 12
    pmt = (loan * (monthly_rate * (1 + monthly_rate) ** months_total) /
           ((1 + monthly_rate) ** months_total - 1) if loan > 0 else 0.0)

    rows = []
    hv             = home_price
    balance        = loan
    mtg_portfolio  = loan    # mortgage buyer: lump-sum invested
    cash_portfolio = 0.0     # cash buyer: annual contributions of pmt*12
    cash_cum       = -home_price
    mtg_cum        = -down
    total_interest = 0.0

    for yr in range(1, years + 1):
        hv             *= (1 + appr_rate)
        mtg_portfolio  *= (1 + invest_return)
        cash_portfolio *= (1 + invest_return)
        cash_portfolio += pmt * 12   # annual contribution from saved mortgage payments

        yr_interest = 0.0
        for _ in range(12):
            interest  = balance * monthly_rate
            principal = pmt - interest
            balance   = max(0, balance - principal)
            yr_interest += interest
        total_interest += yr_interest

        prop_tax_yr = hv * prop_tax_rate
        ins   = hv * insurance_rate
        maint = hv * maintenance_rate

        cash_tax = min(prop_tax_yr, 10_000) * tax_bracket
        cash_cum -= (prop_tax_yr + ins + maint - cash_tax)
        cash_wealth_pre_sale = hv + cash_portfolio + cash_cum

        mtg_tax = (yr_interest + min(prop_tax_yr, 10_000)) * tax_bracket
        mtg_cum -= (pmt * 12 + prop_tax_yr + ins + maint - mtg_tax)
        mtg_wealth_pre_sale = (hv - balance) + mtg_portfolio + mtg_cum

        rows.append({
            "Year":                  yr,
            "Home Value ($k)":       round(hv / 1e3, 1),
            "Cash Wealth ($k)":      round(cash_wealth_pre_sale / 1e3, 1),
            "Mtg Wealth ($k)":       round(mtg_wealth_pre_sale / 1e3, 1),
            "Cash Portfolio ($k)":   round(cash_portfolio / 1e3, 1),
            "Mtg Portfolio ($k)":    round(mtg_portfolio / 1e3, 1),
            "Yr Interest ($k)":      round(yr_interest / 1e3, 1),
            "Mtg Tax Savings ($k)":  round(mtg_tax / 1e3, 1),
            "Cash Tax Savings ($k)": round(cash_tax / 1e3, 1),
            "Mtg Advantage ($k)":    round((mtg_wealth_pre_sale - cash_wealth_pre_sale) / 1e3, 1),
        })

    return rows, total_interest


def calc_affordability(home_price, down_pct, loan_rate, loan_term, prop_tax_rate,
                       insurance_rate, hoa_monthly, savings, income):
    down = home_price * down_pct
    loan = home_price - down
    monthly_rate = loan_rate / 12
    n = loan_term * 12
    pmt = (loan * (monthly_rate * (1 + monthly_rate) ** n) / ((1 + monthly_rate) ** n - 1)
           if loan > 0 else 0.0)
    total_monthly = pmt + home_price * prop_tax_rate / 12 + home_price * insurance_rate / 12 + hoa_monthly
    dti = total_monthly / (income / 12)
    return down, pmt, total_monthly, dti


# ── Sidebar Inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Your Profile")

    zip_code = st.text_input("ZIP Code", value="94103", max_chars=5)
    appr_est, appr_source = get_appreciation_rate(zip_code)
    st.caption(f"📍 {appr_source}: **{appr_est*100:.1f}%/yr**")

    st.divider()
    st.subheader("Income & Savings")
    income        = st.number_input("Annual Income ($)", value=120_000, step=5_000, format="%d")
    savings       = st.number_input("Total Savings ($)", value=100_000, step=5_000, format="%d")
    monthly_liquidity = st.number_input("Monthly Disposable ($)", value=2_000, step=100, format="%d")
    tax_bracket   = st.selectbox("Federal Tax Bracket",
                                 [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37],
                                 index=3, format_func=lambda x: f"{int(x*100)}%")

    st.divider()
    st.subheader("Home Purchase")
    home_price    = st.number_input("Home Price ($)", value=500_000, step=10_000, format="%d")
    down_pct      = st.slider("Down Payment (%)", 0, 100, 20) / 100
    loan_rate     = st.slider("Mortgage Rate (%)", 3.0, 10.0, 6.75, step=0.05) / 100
    loan_term     = st.selectbox("Loan Term (years)", [15, 20, 30], index=2)
    prop_tax_rate = st.slider("Property Tax Rate (%)", 0.3, 3.0, 1.1, step=0.05) / 100
    hoa_monthly   = st.number_input("HOA / Month ($)", value=0, step=50, format="%d")

    st.divider()
    st.subheader("Renting")
    rent_monthly  = st.number_input("Current Rent ($/mo)", value=2_500, step=50, format="%d")
    rent_inflation = st.slider("Rent Inflation (%/yr)", 1.0, 8.0, 3.5, step=0.1) / 100

    st.divider()
    st.subheader("Market Assumptions")
    appr_rate     = st.slider("Home Appreciation (%/yr)", 1.0, 12.0, float(f"{appr_est*100:.1f}"), step=0.1) / 100
    appr_vol      = st.slider("Appreciation Volatility (%)", 1.0, 10.0, 4.0, step=0.5) / 100
    invest_return = st.slider("Stock Return (%/yr)", 3.0, 14.0, 8.0, step=0.25) / 100
    invest_vol    = st.slider("Stock Volatility (%)", 5.0, 25.0, 14.0, step=0.5) / 100
    years         = st.slider("Time Horizon (years)", 3, 30, 10)
    n_sims        = st.selectbox("Monte Carlo Runs", [500, 1000, 2000], index=1)

# ── Constants ────────────────────────────────────────────────────────────────
INSURANCE_RATE    = 0.005
MAINTENANCE_RATE  = 0.01

# ── Page header ──────────────────────────────────────────────────────────────
st.title("🏠 Buy vs Rent Analyzer")
st.caption("Monte Carlo simulation with life event risk scenarios")

# ── Live Affordability Snapshot (always visible, always current) ──────────────
down, monthly_pmt_est, total_monthly_housing, dti = calc_affordability(
    home_price, down_pct, loan_rate, loan_term,
    prop_tax_rate, INSURANCE_RATE, hoa_monthly, savings, income
)

st.subheader("Affordability Snapshot")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Down Payment",      f"${down:,.0f}",               f"{down/savings*100:.0f}% of savings")
col_b.metric("Est. Monthly P&I",  f"${monthly_pmt_est:,.0f}")
col_c.metric("Total Housing/mo",  f"${total_monthly_housing:,.0f}")
col_d.metric("DTI Ratio",         f"{dti*100:.1f}%",
             "⚠️ High" if dti > 0.36 else "✅ OK", delta_color="inverse")

liquid_after_down = savings - down
liq1, liq2, liq3 = st.columns(3)
liq1.metric("Liquid After Down",   f"${liquid_after_down:,.0f}")
liq2.metric("Monthly Disposable",  f"${monthly_liquidity:,.0f}")
months_runway = liquid_after_down / total_monthly_housing if total_monthly_housing > 0 else 0
liq3.metric("Emergency Runway",    f"{months_runway:.1f} mo",
             "⚠️ Low" if months_runway < 6 else "✅ Adequate", delta_color="inverse")

if down > savings:
    st.error("⛔ Down payment exceeds available savings. Adjust inputs before running.")

# ── Input fingerprint for stale-result detection ──────────────────────────────
current_fp = input_fingerprint(
    home_price=home_price, down_pct=down_pct, loan_rate=loan_rate, loan_term=loan_term,
    prop_tax_rate=prop_tax_rate, hoa_monthly=hoa_monthly,
    rent_monthly=rent_monthly, rent_inflation=rent_inflation,
    appr_rate=appr_rate, appr_vol=appr_vol,
    invest_return=invest_return, invest_vol=invest_vol,
    income=income, savings=savings,
    tax_bracket=tax_bracket, years=years, n_sims=n_sims,
)

results_exist  = "sim_results" in st.session_state
inputs_changed = results_exist and st.session_state.get("sim_fingerprint") != current_fp

# ── Stale-results warning ─────────────────────────────────────────────────────
if inputs_changed:
    st.warning("⚠️ Inputs have changed — results below are from the previous run. Click **Run Simulation** to update.")

# ── Run Simulation button ─────────────────────────────────────────────────────
st.divider()
btn_col, _ = st.columns([1, 3])
run_clicked = btn_col.button(
    "▶ Run Simulation",
    type="primary",
    disabled=(down > savings),
    use_container_width=True,
)

if run_clicked:
    with st.spinner(f"Running {n_sims:,} simulations…"):
        buy_w, rent_w, buy_paths, rent_paths = run_monte_carlo(
            home_price, down_pct, loan_rate, years,
            appr_rate, appr_vol,
            rent_monthly, rent_inflation,
            invest_return, invest_vol,
            prop_tax_rate, INSURANCE_RATE, MAINTENANCE_RATE,
            income, tax_bracket,
            n_sims=n_sims,
        )
        cash_w, mtg_w, cash_paths_cm, mtg_paths_cm = run_cash_vs_mortgage_mc(
            home_price, down_pct, loan_rate, loan_term,
            appr_rate, appr_vol,
            invest_return, invest_vol,
            prop_tax_rate, INSURANCE_RATE, MAINTENANCE_RATE,
            tax_bracket, years, n_sims=n_sims,
        )
        cm_det_rows, cm_total_interest = cash_vs_mortgage_deterministic(
            home_price, down_pct, loan_rate, loan_term,
            appr_rate, invest_return,
            prop_tax_rate, INSURANCE_RATE, MAINTENANCE_RATE,
            tax_bracket, years,
        )

    breakeven = compute_breakeven(buy_paths, rent_paths)

    st.session_state["sim_results"] = {
        "buy_w": buy_w, "rent_w": rent_w,
        "buy_paths": buy_paths, "rent_paths": rent_paths,
        "breakeven": breakeven,
        "years": years, "n_sims": n_sims,
        "home_price": home_price, "loan_rate": loan_rate,
        "loan_term": loan_term, "down_pct": down_pct,
        "prop_tax_rate": prop_tax_rate, "tax_bracket": tax_bracket,
        "appr_rate": appr_rate, "savings": savings,
        "rent_monthly": rent_monthly, "rent_inflation": rent_inflation,
        "invest_return": invest_return,
        "monthly_liquidity": monthly_liquidity,
        "total_monthly_housing": total_monthly_housing,
        # cash vs mortgage
        "cash_w": cash_w, "mtg_w": mtg_w,
        "cash_paths_cm": cash_paths_cm, "mtg_paths_cm": mtg_paths_cm,
        "cm_det_rows": cm_det_rows, "cm_total_interest": cm_total_interest,
    }
    st.session_state["sim_fingerprint"] = current_fp
    st.rerun()

# ── Results (only rendered when session state has data) ───────────────────────
if "sim_results" not in st.session_state:
    st.info("Configure your inputs above, then click **Run Simulation** to see results.")
    st.stop()

R = st.session_state["sim_results"]
buy_w        = R["buy_w"];        rent_w     = R["rent_w"]
buy_paths    = R["buy_paths"];    rent_paths = R["rent_paths"]
breakeven    = R["breakeven"]
_years       = R["years"];        _n_sims    = R["n_sims"]
_home_price  = R["home_price"];   _loan_rate = R["loan_rate"]
_loan_term   = R["loan_term"];    _down_pct  = R["down_pct"]
_prop_tax    = R["prop_tax_rate"]; _tax_b    = R["tax_bracket"]
_appr_rate   = R["appr_rate"];    _savings   = R["savings"]
_rent        = R["rent_monthly"]
_liq_monthly = R["monthly_liquidity"]
_total_mth   = R["total_monthly_housing"]
# (rent_inflation and invest_return accessed directly via R["..."] where needed)

# ── Summary Stats ─────────────────────────────────────────────────────────────
buy_med  = np.median(buy_w)
rent_med = np.median(rent_w)
diff_med = buy_med - rent_med
prob_buy_wins = (buy_w > rent_w).mean()

be_med = breakeven["median_crossover"]   # year median buy line crosses median rent line
be_label = f"Yr {be_med}" if be_med is not None else f"Not in {_years}-yr window"
be_delta = "Median buy wealth overtakes rent" if be_med else "Rent stays ahead in median scenario"

st.divider()
st.subheader("Simulation Results")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Buy – Median Wealth",    f"${buy_med:,.0f}")
c2.metric("Rent – Median Wealth",   f"${rent_med:,.0f}")
c3.metric("Buy Advantage",          f"${diff_med:,.0f}",
          f"{'Buy' if diff_med > 0 else 'Rent'} wins median")
c4.metric("Prob. Buy Outperforms",  f"{prob_buy_wins*100:.0f}%")
c5.metric("Breakeven (median)",     be_label, be_delta, delta_color="off")

# ── Chart 1: Wealth Distribution ─────────────────────────────────────────────
fig1 = go.Figure()
fig1.add_trace(go.Histogram(x=buy_w/1e3,  name="Buy",         opacity=0.65,
                             marker_color="#2196F3", nbinsx=60,
                             hovertemplate="Wealth: $%{x:.0f}k<br>Count: %{y}"))
fig1.add_trace(go.Histogram(x=rent_w/1e3, name="Rent+Invest", opacity=0.65,
                             marker_color="#4CAF50", nbinsx=60,
                             hovertemplate="Wealth: $%{x:.0f}k<br>Count: %{y}"))
fig1.add_vline(x=buy_med/1e3,  line_dash="dash", line_color="#1565C0", annotation_text="Buy median")
fig1.add_vline(x=rent_med/1e3, line_dash="dash", line_color="#2E7D32", annotation_text="Rent median")
fig1.update_layout(barmode="overlay",
                   title=f"Wealth Distribution after {_years} Years ({_n_sims:,} simulations)",
                   xaxis_title="Net Wealth ($k)", yaxis_title="Frequency", height=380)
st.plotly_chart(fig1, use_container_width=True)

# ── Chart 2: Wealth Paths ─────────────────────────────────────────────────────
yr_axis       = list(range(_years + 1))
# Use stored medians from breakeven (already computed there)
buy_path_med  = breakeven["med_buy"]
rent_path_med = breakeven["med_rent"]
buy_path_p10  = np.percentile(buy_paths,  10, axis=0)
buy_path_p90  = np.percentile(buy_paths,  90, axis=0)
rent_path_p10 = np.percentile(rent_paths, 10, axis=0)
rent_path_p90 = np.percentile(rent_paths, 90, axis=0)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=yr_axis + yr_axis[::-1],
                           y=list(buy_path_p90/1e3) + list(buy_path_p10[::-1]/1e3),
                           fill="toself", fillcolor="rgba(33,150,243,0.15)",
                           line=dict(color="rgba(0,0,0,0)"), name="Buy P10–P90"))
fig2.add_trace(go.Scatter(x=yr_axis + yr_axis[::-1],
                           y=list(rent_path_p90/1e3) + list(rent_path_p10[::-1]/1e3),
                           fill="toself", fillcolor="rgba(76,175,80,0.15)",
                           line=dict(color="rgba(0,0,0,0)"), name="Rent P10–P90"))
fig2.add_trace(go.Scatter(x=yr_axis, y=buy_path_med/1e3,  name="Buy median",
                           line=dict(color="#2196F3", width=3)))
fig2.add_trace(go.Scatter(x=yr_axis, y=rent_path_med/1e3, name="Rent median",
                           line=dict(color="#4CAF50", width=3)))
# Only mark breakeven if the two MEDIAN lines actually cross
if be_med is not None:
    be_y = float(buy_path_med[be_med] / 1e3)
    fig2.add_vline(x=be_med, line_dash="dot", line_color="orange",
                   annotation_text=f"Breakeven yr {be_med}", annotation_position="top left")
    fig2.add_trace(go.Scatter(x=[be_med], y=[be_y], mode="markers",
                               marker=dict(color="orange", size=12, symbol="star"),
                               name=f"Breakeven yr {be_med}", showlegend=True))
fig2.update_layout(title="Wealth Accumulation Over Time (median ± P10/P90)",
                   xaxis_title="Year", yaxis_title="Net Wealth ($k)", height=400)
st.plotly_chart(fig2, use_container_width=True)

# ── Breakeven Analysis ────────────────────────────────────────────────────────
st.subheader("Breakeven Analysis")

be_years_valid = [y for y in breakeven["sim_years"] if y is not None]

# Two distinct concepts — explain them clearly
becol1, becol2, becol3, becol4 = st.columns(4)
becol1.metric(
    "Median-path Breakeven",
    f"Yr {be_med}" if be_med else f">{_years} yrs",
    "When the two median lines cross on the chart above",
    delta_color="off",
)
becol2.metric(
    "Sim-based Breakeven",
    f"Yr {breakeven['sim_median']:.0f}" if breakeven["sim_median"] else "Never",
    f"Median of {breakeven['n_valid']:,} sims where buy wins",
    delta_color="off",
)
becol3.metric("Sims Where Buy Wins", f"{breakeven['n_valid']:,} / {_n_sims:,}")
becol4.metric("Sims Where Rent Always Wins", f"{breakeven['never_pct']:.0f}%",
              "⚠️ High" if breakeven["never_pct"] > 30 else "✅ Low", delta_color="inverse")

# Histogram — only the per-sim distribution
if be_years_valid:
    fig_be = go.Figure()
    fig_be.add_trace(go.Histogram(
        x=be_years_valid, nbinsx=_years,
        marker_color="#FF9800", opacity=0.85,
        hovertemplate="Buy crosses rent in year %{x}<br>Count: %{y}<extra></extra>",
        name="Simulations",
    ))
    if breakeven["sim_median"] is not None:
        fig_be.add_vline(x=breakeven["sim_median"], line_dash="dash", line_color="#E65100",
                         annotation_text=f"Sim median yr {breakeven['sim_median']:.0f}")
    if breakeven["p25"] and breakeven["p75"]:
        fig_be.add_vrect(x0=breakeven["p25"], x1=breakeven["p75"],
                         fillcolor="rgba(255,152,0,0.15)", line_width=0,
                         annotation_text="P25–P75", annotation_position="top left")
    fig_be.update_layout(
        title=f"In which year does buying beat renting? ({breakeven['never_pct']:.0f}% of sims: never)",
        xaxis_title="First year buy wealth exceeds rent wealth (per simulation)",
        yaxis_title="Number of Simulations",
        xaxis=dict(range=[0, _years]),
        height=340,
    )
    st.plotly_chart(fig_be, use_container_width=True)
else:
    st.warning(f"In none of the {_n_sims:,} simulations does buying outperform renting within {_years} years.")

# Equity decomposition: show how principal + appreciation drive breakeven
st.markdown("**What drives the buy advantage?** Equity builds from two sources:")
_down_be  = _home_price * _down_pct
_loan_be  = _home_price - _down_be
_mr_be    = _loan_rate / 12
_n_be     = _loan_term * 12
_pmt_be   = (_loan_be * (_mr_be * (1 + _mr_be) ** _n_be) / ((1 + _mr_be) ** _n_be - 1)
             if _loan_be > 0 else 0.0)

eq_principal, eq_appreciation, eq_years = [], [], []
bal = _loan_be
hv  = _home_price
for yr in range(1, _years + 1):
    for _ in range(12):
        interest  = bal * _mr_be
        bal = max(0, bal - (_pmt_be - interest))
    hv *= (1 + _appr_rate)
    principal_equity    = _loan_be - bal
    appreciation_equity = hv - _home_price
    eq_years.append(yr)
    eq_principal.append(principal_equity / 1e3)
    eq_appreciation.append(appreciation_equity / 1e3)

fig_eq = go.Figure()
fig_eq.add_trace(go.Bar(x=eq_years, y=eq_principal,     name="Principal Paydown",
                         marker_color="#2196F3"))
fig_eq.add_trace(go.Bar(x=eq_years, y=eq_appreciation,  name="Appreciation Gain",
                         marker_color="#9C27B0"))
if be_med is not None and be_med <= _years:
    fig_eq.add_vline(x=be_med, line_dash="dot", line_color="orange",
                     annotation_text=f"Breakeven yr {be_med:.0f}")
fig_eq.update_layout(barmode="stack",
                      title="Equity Buildup: Principal Paydown vs Appreciation (deterministic, median rate)",
                      xaxis_title="Year", yaxis_title="Equity Gained ($k)", height=340)
st.plotly_chart(fig_eq, use_container_width=True)

# ── Chart 3: Amortization ─────────────────────────────────────────────────────
st.subheader("Mortgage Breakdown")
_down         = _home_price * _down_pct
_loan         = _home_price - _down
_monthly_rate = _loan_rate / 12
_months_total = _loan_term * 12
_pmt          = (_loan * (_monthly_rate * (1 + _monthly_rate) ** _months_total) /
                 ((1 + _monthly_rate) ** _months_total - 1) if _loan > 0 else 0.0)

balance = _loan
amort = []
for m in range(1, _months_total + 1):
    interest  = balance * _monthly_rate
    principal = _pmt - interest
    balance   = max(0, balance - principal)
    amort.append({"Month": m, "Year": m / 12, "Principal": principal,
                  "Interest": interest, "Balance": balance})
amort_df = pd.DataFrame(amort)
amort_annual = (amort_df.groupby(amort_df["Year"].astype(int) + 1)
                .agg(Principal=("Principal", "sum"),
                     Interest=("Interest", "sum"),
                     Balance=("Balance", "last"))
                .reset_index()
                .rename(columns={"Year": "Year"}))

fig3 = make_subplots(specs=[[{"secondary_y": True}]])
fig3.add_trace(go.Bar(x=amort_annual["Year"], y=amort_annual["Principal"]/1e3,
                       name="Principal", marker_color="#2196F3"), secondary_y=False)
fig3.add_trace(go.Bar(x=amort_annual["Year"], y=amort_annual["Interest"]/1e3,
                       name="Interest",   marker_color="#FF7043"), secondary_y=False)
fig3.add_trace(go.Scatter(x=amort_annual["Year"], y=amort_annual["Balance"]/1e3,
                           name="Remaining Balance", line=dict(color="#9C27B0", width=2)),
               secondary_y=True)
fig3.update_layout(barmode="stack", title="Annual Amortization Schedule",
                   xaxis_title="Year", height=350)
fig3.update_yaxes(title_text="Annual P&I ($k)", secondary_y=False)
fig3.update_yaxes(title_text="Loan Balance ($k)", secondary_y=True)
st.plotly_chart(fig3, use_container_width=True)

total_interest       = amort_df["Interest"].sum()
approx_tax_savings   = total_interest * _tax_b
m1, m2, m3 = st.columns(3)
m1.metric("Total Interest Paid",  f"${total_interest:,.0f}")
m2.metric("Est. Tax Savings",     f"${approx_tax_savings:,.0f}", f"{int(_tax_b*100)}% bracket")
m3.metric("Effective Loan Cost",  f"${total_interest - approx_tax_savings:,.0f}")

# ── Chart 4: Sell After X Years ───────────────────────────────────────────────
st.divider()
st.subheader("What If You Sell After...")

sell_results = sell_after_analysis(
    _home_price, _down_pct, _loan_rate, _loan_term,
    _appr_rate, _prop_tax, INSURANCE_RATE, MAINTENANCE_RATE,
    _tax_b, _rent, R["rent_inflation"], R["invest_return"],
)

labels      = [f"Sell Yr {r['horizon']}" for r in sell_results]
buy_vals4   = [r["buy_wealth"]  / 1e3 for r in sell_results]
rent_vals4  = [r["rent_wealth"] / 1e3 for r in sell_results]
adv_vals    = [r["advantage"]   / 1e3 for r in sell_results]

# Summary row metrics
s_cols = st.columns(len(sell_results))
for col, r in zip(s_cols, sell_results):
    adv = r["advantage"]
    winner = "Buy" if adv > 0 else "Rent"
    col.metric(
        f"Sell Year {r['horizon']}",
        f"${r['buy_wealth']/1e3:.0f}k buy  /  ${r['rent_wealth']/1e3:.0f}k rent",
        f"{winner} +${abs(adv)/1e3:.0f}k",
        delta_color="normal" if adv > 0 else "inverse",
    )

# Grouped bar: buy vs rent wealth at each horizon
fig4 = go.Figure()
fig4.add_trace(go.Bar(name="Buy (after sale)", x=labels, y=buy_vals4,
                       marker_color="#2196F3",
                       text=[f"${v:.0f}k" for v in buy_vals4], textposition="outside"))
fig4.add_trace(go.Bar(name="Rent + Invest",    x=labels, y=rent_vals4,
                       marker_color="#4CAF50",
                       text=[f"${v:.0f}k" for v in rent_vals4], textposition="outside"))
fig4.add_hline(y=0, line_color="gray", line_width=1)
fig4.update_layout(barmode="group",
                   title="Net Wealth at Sale (buy net proceeds vs rent+invest portfolio)",
                   yaxis_title="Net Wealth ($k)", height=400)
st.plotly_chart(fig4, use_container_width=True)

# Stacked bar: what makes up the buyer's equity at each horizon
fig4b = go.Figure()
fig4b.add_trace(go.Bar(name="Principal Paid Down",
                        x=labels, y=[r["principal_paid"]/1e3  for r in sell_results],
                        marker_color="#2196F3"))
fig4b.add_trace(go.Bar(name="Appreciation Gain",
                        x=labels, y=[r["appreciation"]/1e3    for r in sell_results],
                        marker_color="#9C27B0"))
fig4b.add_trace(go.Bar(name="Selling Costs (−6%)",
                        x=labels, y=[-r["selling_costs"]/1e3  for r in sell_results],
                        marker_color="#EF5350"))
fig4b.update_layout(barmode="relative",
                    title="Buyer's Equity Breakdown at Sale",
                    yaxis_title="$k", height=340)
st.plotly_chart(fig4b, use_container_width=True)

with st.expander("How this is calculated"):
    st.markdown("""
- **Buy net wealth** = Home value − Loan balance − Selling costs (6%) − Down payment − Cumulative housing costs + Tax savings
- **Rent net wealth** = Invested down payment (compounded) + Invested monthly savings vs buying − Cumulative rent paid
- All figures use the **median appreciation and return rates** (deterministic, no Monte Carlo noise)
- Selling costs include agent commissions, transfer taxes, and closing costs (~6%)
""")


# ── Chart 6: Liquidity Gauges ─────────────────────────────────────────────────
st.divider()
st.subheader("Liquidity & Emergency Buffer")
_liquid_after_down = _savings - _down
_months_runway     = _liquid_after_down / _total_mth if _total_mth > 0 else 0
_rent_runway       = _savings / _rent if _rent > 0 else 0

fig6 = go.Figure()
for domain_x, value, title, color in [
    ([0, 0.45], _months_runway, "Emergency Runway — Buy (months)", "#2196F3"),
    ([0.55, 1], _rent_runway,   "Emergency Runway — Rent (months)", "#4CAF50"),
]:
    fig6.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={"text": title},
        delta={"reference": 6, "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 24]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 3],  "color": "#FFCDD2"},
                {"range": [3, 6],  "color": "#FFF9C4"},
                {"range": [6, 24], "color": "#C8E6C9"},
            ],
            "threshold": {"line": {"color": "red", "width": 4},
                          "thickness": 0.75, "value": 6},
        },
        domain={"x": domain_x, "y": [0, 1]},
    ))
fig6.update_layout(height=300)
st.plotly_chart(fig6, use_container_width=True)

# ── Cash vs Mortgage Analysis ─────────────────────────────────────────────────
st.divider()
st.subheader("Cash vs Mortgage Analysis")
_cm_loan = _home_price * (1 - _down_pct)
_cm_mr   = _loan_rate / 12
_cm_n    = _loan_term * 12
_cm_pmt  = (_cm_loan * (_cm_mr * (1 + _cm_mr) ** _cm_n) / ((1 + _cm_mr) ** _cm_n - 1)
            if _cm_loan > 0 else 0.0)
st.caption(
    f"All-cash purchase vs {_down_pct*100:.0f}% down mortgage — both scenarios start with ${_home_price:,} in cash. "
    f"Cash buyer invests freed-up mortgage payments (${_cm_pmt*12:,.0f}/yr) annually. "
    f"Mortgage buyer invests ${_cm_loan:,.0f} as a lump sum on day 1."
)

_cash_w        = R["cash_w"]
_mtg_w         = R["mtg_w"]
_cash_paths_cm = R["cash_paths_cm"]
_mtg_paths_cm  = R["mtg_paths_cm"]
_cm_det_rows   = R["cm_det_rows"]
_cm_tot_int    = R["cm_total_interest"]

cash_med_cm   = float(np.median(_cash_w))
mtg_med_cm    = float(np.median(_mtg_w))
mtg_adv_cm    = mtg_med_cm - cash_med_cm
prob_mtg_wins = float((_mtg_w > _cash_w).mean())

# Deterministic: portfolio end values and after-tax total interest
_det_last      = _cm_det_rows[-1]
_mtg_port_end  = _det_last["Mtg Portfolio ($k)"] * 1e3
_invest_gain   = _mtg_port_end - _home_price * (1 - _down_pct)    # gain above principal (lump sum)
_total_int_at  = _cm_tot_int * (1 - _tax_b)                       # after-tax interest cost
_net_leverage  = _invest_gain - _total_int_at

cm1, cm2, cm3, cm4 = st.columns(4)
cm1.metric("All-Cash Median Wealth",   f"${cash_med_cm:,.0f}")
cm2.metric("Mortgage Median Wealth",   f"${mtg_med_cm:,.0f}")
cm3.metric(
    "Mortgage Advantage",
    f"${abs(mtg_adv_cm):,.0f}",
    f"{'Mortgage' if mtg_adv_cm >= 0 else 'Cash'} wins median",
    delta_color="normal" if mtg_adv_cm >= 0 else "inverse",
)
cm4.metric("Prob. Mortgage Outperforms", f"{prob_mtg_wins*100:.0f}%")

ck1, ck2, ck3 = st.columns(3)
ck1.metric("Total Mortgage Interest",    f"${_cm_tot_int:,.0f}",
           f"${_total_int_at:,.0f} after {int(_tax_b*100)}% deduction")
ck2.metric("Investment Gain on Float",   f"${_invest_gain:,.0f}",
           f"Portfolio − original loan principal")
ck3.metric(
    "Net Leverage Benefit",
    f"${_net_leverage:,.0f}",
    "Invest gain minus after-tax interest",
    delta_color="normal" if _net_leverage >= 0 else "inverse",
)

# ── Distribution chart ────────────────────────────────────────────────────────
fig_cm1 = go.Figure()
fig_cm1.add_trace(go.Histogram(
    x=_cash_w / 1e3, name="All-Cash", opacity=0.65,
    marker_color="#FF7043", nbinsx=60,
    hovertemplate="Wealth: $%{x:.0f}k<br>Count: %{y}",
))
fig_cm1.add_trace(go.Histogram(
    x=_mtg_w / 1e3, name=f"Mortgage ({int(_down_pct*100)}% down)", opacity=0.65,
    marker_color="#7C4DFF", nbinsx=60,
    hovertemplate="Wealth: $%{x:.0f}k<br>Count: %{y}",
))
fig_cm1.add_vline(x=cash_med_cm / 1e3, line_dash="dash", line_color="#BF360C",
                  annotation_text="Cash median")
fig_cm1.add_vline(x=mtg_med_cm  / 1e3, line_dash="dash", line_color="#4527A0",
                  annotation_text="Mortgage median")
fig_cm1.update_layout(
    barmode="overlay",
    title=f"Wealth Distribution after {_years} Years — Cash vs Mortgage ({_n_sims:,} simulations)",
    xaxis_title="Net Wealth ($k)", yaxis_title="Frequency", height=380,
)
st.plotly_chart(fig_cm1, use_container_width=True)

# ── Wealth paths chart ────────────────────────────────────────────────────────
yr_ax = list(range(_years + 1))
med_cash = np.median(_cash_paths_cm, axis=0)
med_mtg  = np.median(_mtg_paths_cm,  axis=0)
p10_cash = np.percentile(_cash_paths_cm, 10, axis=0)
p90_cash = np.percentile(_cash_paths_cm, 90, axis=0)
p10_mtg  = np.percentile(_mtg_paths_cm,  10, axis=0)
p90_mtg  = np.percentile(_mtg_paths_cm,  90, axis=0)

fig_cm2 = go.Figure()
fig_cm2.add_trace(go.Scatter(
    x=yr_ax + yr_ax[::-1],
    y=list(p90_cash / 1e3) + list(p10_cash[::-1] / 1e3),
    fill="toself", fillcolor="rgba(255,112,67,0.15)",
    line=dict(color="rgba(0,0,0,0)"), name="Cash P10–P90",
))
fig_cm2.add_trace(go.Scatter(
    x=yr_ax + yr_ax[::-1],
    y=list(p90_mtg / 1e3) + list(p10_mtg[::-1] / 1e3),
    fill="toself", fillcolor="rgba(124,77,255,0.15)",
    line=dict(color="rgba(0,0,0,0)"), name="Mortgage P10–P90",
))
fig_cm2.add_trace(go.Scatter(
    x=yr_ax, y=med_cash / 1e3, name="All-Cash median",
    line=dict(color="#FF7043", width=3),
))
fig_cm2.add_trace(go.Scatter(
    x=yr_ax, y=med_mtg / 1e3, name="Mortgage median",
    line=dict(color="#7C4DFF", width=3),
))
# Mark crossover on median paths
_cm_cross = None
for yr in range(1, _years + 1):
    if med_mtg[yr] >= med_cash[yr]:
        _cm_cross = yr
        break
if _cm_cross is not None:
    fig_cm2.add_vline(x=_cm_cross, line_dash="dot", line_color="orange",
                      annotation_text=f"Mtg ahead yr {_cm_cross}",
                      annotation_position="top left")
fig_cm2.update_layout(
    title="Wealth Accumulation Over Time — Cash vs Mortgage (median ± P10/P90)",
    xaxis_title="Year", yaxis_title="Net Wealth ($k)", height=400,
)
st.plotly_chart(fig_cm2, use_container_width=True)

# ── Year-by-year deterministic breakdown ─────────────────────────────────────
with st.expander("Year-by-Year Deterministic Breakdown"):
    det_df = pd.DataFrame(_cm_det_rows)
    # Stacked bar: cash buyer annual cost sources vs mortgage annual cost sources
    fig_det = go.Figure()
    fig_det.add_trace(go.Scatter(
        x=det_df["Year"], y=det_df["Cash Wealth ($k)"],
        name="All-Cash Wealth", line=dict(color="#FF7043", width=2),
    ))
    fig_det.add_trace(go.Scatter(
        x=det_df["Year"], y=det_df["Mtg Wealth ($k)"],
        name="Mortgage Wealth", line=dict(color="#7C4DFF", width=2),
    ))
    fig_det.add_trace(go.Scatter(
        x=det_df["Year"], y=det_df["Cash Portfolio ($k)"],
        name="Cash Portfolio (annual contributions)", line=dict(color="#FF7043", width=2, dash="dot"),
    ))
    fig_det.add_trace(go.Scatter(
        x=det_df["Year"], y=det_df["Mtg Portfolio ($k)"],
        name="Mtg Portfolio (lump sum)", line=dict(color="#7C4DFF", width=2, dash="dot"),
    ))
    fig_det.update_layout(
        title="Deterministic Wealth Paths & Portfolios (median rates, before selling costs)",
        xaxis_title="Year", yaxis_title="$k", height=340,
    )
    st.plotly_chart(fig_det, use_container_width=True)

    st.dataframe(
        det_df.style.format({c: "{:.1f}" for c in det_df.columns if c != "Year"}),
        use_container_width=True,
        hide_index=True,
    )

with st.expander("How this is calculated"):
    st.markdown(f"""
**Both scenarios assume you have ${_home_price:,} in cash available.**

| | All-Cash | Mortgage ({int(_down_pct*100)}% down) |
|---|---|---|
| Upfront outlay | ${_home_price:,} (full price) | ${_home_price*_down_pct:,.0f} (down payment) |
| Stock portfolio (day 1) | $0 | ${_home_price*(1-_down_pct):,.0f} lump sum |
| Annual stock contributions | ${_cm_pmt*12:,.0f}/yr (saved mortgage payments) | $0 (already invested) |
| Monthly mortgage | None | ${_cm_pmt:,.0f}/mo P&I |
| Tax deductions | Prop tax only (SALT ≤ $10k) | Mortgage interest + prop tax (SALT ≤ $10k) |

**Mortgage wins when:** lump-sum compounding at stock return > dollar-cost-averaging + interest savings.
**Cash wins when:** stock return < after-tax mortgage rate ({_loan_rate*(1-_tax_b)*100:.1f}%), or in market downturns.
Selling costs of 6% are applied to the home value at year {_years}.
""")

# ── Recommendation ────────────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Summary Recommendation")

score_buy = score_rent = 0
notes = []

if prob_buy_wins > 0.55:
    score_buy += 2
    notes.append(f"✅ Buy outperforms in **{prob_buy_wins*100:.0f}%** of simulations")
else:
    score_rent += 2
    notes.append(f"📈 Rent+Invest outperforms in **{(1-prob_buy_wins)*100:.0f}%** of simulations")

if dti > 0.36:
    score_rent += 2
    notes.append(f"⚠️ DTI of {dti*100:.1f}% is above the 36% guideline — affordability risk")
else:
    score_buy += 1

if _months_runway < 6:
    score_rent += 2
    notes.append(f"⚠️ Only **{_months_runway:.1f} months** emergency runway after down payment")
else:
    score_buy += 1

if diff_med > 0:
    score_buy += 1
    notes.append(f"💰 Median buy advantage: **${diff_med:,.0f}** over {_years} years")
else:
    score_rent += 1
    notes.append(f"💰 Median rent advantage: **${-diff_med:,.0f}** over {_years} years")

if _years < 5:
    score_rent += 1
    notes.append("⏳ Short horizon (<5 yrs) favors renting — transaction costs not recovered")

verdict = (
    "🏠 **Buy** appears stronger given your inputs."        if score_buy > score_rent else
    "📦 **Renting + Investing** appears stronger given your inputs." if score_rent > score_buy else
    "⚖️ **Too close to call** — both strategies are competitive."
)
st.info(verdict)
for n in notes:
    st.markdown(f"- {n}")

st.caption("Not financial advice. Projections use historical volatility and are not guaranteed.")

# ── Deploy instructions ───────────────────────────────────────────────────────
with st.expander("🚀 Deploy to Streamlit Cloud"):
    st.markdown("""
1. Push this project to a **GitHub repo**:
   ```bash
   git init && git add . && git commit -m 'Buy vs Rent app'
   git remote add origin https://github.com/YOUR_USERNAME/buy-vs-rent.git
   git push -u origin main
   ```
2. Go to **[share.streamlit.io](https://share.streamlit.io)** → *New app*
3. Select your repo, branch `main`, file `app.py`
4. Click **Deploy** — no secrets needed, all public data
""")
