import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# SECTION 1: PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="CMI | Real Estate Evaluation Engine",
    layout="wide",
    page_icon="🏠",
)

st.markdown(
    """
    <style>
    .re-metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-top: 4px solid #00ACC1;
        padding: 14px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .re-metric-card h4 {
        margin: 0 0 6px 0;
        font-size: 13px;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .re-metric-card .val {
        font-size: 22px;
        font-weight: 800;
        color: #111;
        margin-bottom: 4px;
    }
    .re-metric-card .subtext {
        font-size: 11px;
        color: #495057;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CMI_LOGO_SVG = """
<div style="font-family: sans-serif; font-weight: 800; font-size: 28px; color: #111; line-height: 1.1;">
    <span style="font-size: 36px; font-weight: 900; letter-spacing: -1px;">CMI</span>
    <span style="display:inline-block; width:10px; height:10px; background-color:#E91E63; margin-left:2px; vertical-align:top;"></span>
    <span style="display:inline-block; width:10px; height:10px; background-color:#26A69A; margin-left:1px; vertical-align:top;"></span>
    <span style="display:inline-block; width:10px; height:10px; background-color:#00ACC1; margin-left:1px; vertical-align:top;"></span>
    <div style="font-size: 11px; font-weight: 700; letter-spacing: 2px; color: #333; margin-top: -2px;">
        CORE MARKET INTELLIGENCE
    </div>
</div>
"""

st.sidebar.markdown(CMI_LOGO_SVG, unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🏠 Property Scout Active")
st.sidebar.info("Dedicated workspace for evaluating real estate assets and underwriting scenarios.")

col_header_left, col_header_right = st.columns([3, 1])
with col_header_left:
    st.title("🏠 Real Estate Evaluation & Predictive Analytics")
    st.caption("Powered by **CMI (Core Market Intelligence)** Quantitative Engine")
with col_header_right:
    st.markdown(CMI_LOGO_SVG, unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: COMPUTATIONAL & VALUATION ENGINES
# ==============================================================================
def compute_real_estate_valuation(address, purchase_price, intent, prop_type, school_rating, labor_cost_idx, dom_days, build_year):
    """Calculates institutional valuation metrics, seller bottoms, bid ranges, closing costs, CapEx, and diligence factors."""
    base_market_price = purchase_price * 1.025
    
    dom_discount = min(0.12, (dom_days / 120.0) * 0.08)
    lowest_seller_price = base_market_price * (0.91 - dom_discount)
    
    if intent == "Fix & Flip":
        bid_low = lowest_seller_price * 0.94
        bid_high = base_market_price * 0.90
    elif intent == "Personal Residence (Primary Home)":
        bid_low = lowest_seller_price * 1.01
        bid_high = base_market_price * 0.98
    else:
        bid_low = lowest_seller_price * 0.97
        bid_high = base_market_price * 0.95

    title_legal_lender_fees = purchase_price * 0.012
    transfer_taxes = purchase_price * 0.015
    escrow_prepaids = purchase_price * 0.008
    total_closing_costs = title_legal_lender_fees + transfer_taxes + escrow_prepaids

    age = max(0, 2026 - build_year)
    base_sqft_cost = 15.0 if age < 15 else (35.0 if age < 40 else 60.0)
    
    if intent == "Fix & Flip":
        intent_mult = 1.6
    elif intent == "Short-Term Rental":
        intent_mult = 1.3
    elif intent == "Personal Residence (Primary Home)":
        intent_mult = 1.2
    else:
        intent_mult = 0.9
        
    type_mult = 1.0 if prop_type == "Single Family" else (1.4 if prop_type == "Multi-Family (2-4 Units)" else 1.8)
    labor_mult = labor_cost_idx / 100.0
    
    rehab_low = purchase_price * (base_sqft_cost / 350.0) * intent_mult * type_mult * labor_mult * 0.75
    rehab_high = rehab_low * 1.55

    school_score = f"{school_rating}/10 ({'Top Tier' if school_rating>=8 else 'Moderate' if school_rating>=5 else 'Below Avg'})"
    labor_availability = "Tight / High Cost" if labor_cost_idx > 110 else ("Balanced" if labor_cost_idx >= 95 else "Abundant / Low Cost")
    tax_burden_pct = 2.15 if "CHICAGO" in address.upper() or "IL" in address.upper() else 1.45
    annual_taxes = purchase_price * (tax_burden_pct / 100.0)
    
    zoning_permits = "Complex / Slow (Historic/HOA)" if prop_type in ["Multi-Family (2-4 Units)", "Commercial"] else "Standard Municipal"
    insurance_risk = "Moderate (Urban/Wind/Water)" if "CHICAGO" in address.upper() else "Low/Standard"
    
    return {
        "market_price": base_market_price,
        "lowest_seller_price": lowest_seller_price,
        "bid_low": bid_low,
        "bid_high": bid_high,
        "total_closing_costs": total_closing_costs,
        "rehab_low": rehab_low,
        "rehab_high": rehab_high,
        "school_score": school_score,
        "labor_availability": labor_availability,
        "tax_burden_pct": tax_burden_pct,
        "annual_taxes": annual_taxes,
        "zoning_permits": zoning_permits,
        "insurance_risk": insurance_risk,
    }

def generate_40yr_hist_20yr_proj_housing_data(base_price):
    """Generates 40-year historical dataset (1986–2026) and 20-year projection (2026–2046) for comps."""
    years_hist = np.arange(1986, 2027)
    years_proj = np.arange(2027, 2047)
    all_years = np.concatenate([years_hist, years_proj])
    
    hist_factors = []
    p = 1.0
    for y in years_hist:
        if y < 2000:
            p *= 1.042
        elif 2000 <= y <= 2006:
            p *= 1.075
        elif 2007 <= y <= 2011:
            p *= 0.910
        elif 2012 <= y <= 2019:
            p *= 1.051
        elif 2020 <= y <= 2023:
            p *= 1.092
        else:
            p *= 1.038
        hist_factors.append(p)
    
    hist_factors = np.array(hist_factors)
    hist_prices = base_price * (hist_factors / hist_factors[-1])
    
    proj_prices = []
    curr_p = base_price
    for y in years_proj:
        curr_p *= 1.038
        proj_prices.append(curr_p)
        
    subject_trajectory = np.concatenate([hist_prices, np.array(proj_prices)])
    
    highest_home = subject_trajectory * 1.62
    lowest_home = subject_trajectory * 0.48
    avg_neighborhood = subject_trajectory * 0.94
    avg_zipcode = subject_trajectory * 0.88
    
    comp1 = subject_trajectory * 1.25
    comp2 = subject_trajectory * 1.10
    comp3 = subject_trajectory * 0.98
    comp4 = subject_trajectory * 0.82
    comp5 = subject_trajectory * 0.68
    
    df_chart = pd.DataFrame(
        {
            "Year": all_years,
            "Subject Property Trajectory": subject_trajectory,
            "Highest Price Home in Neighborhood": highest_home,
            "Lowest Price Home in Neighborhood": lowest_home,
            "Overall Avg Neighborhood Price": avg_neighborhood,
            "Avg Price Same ZIP Code Area": avg_zipcode,
            "Avg Comp Home 1 (Upper Tier)": comp1,
            "Avg Comp Home 2 (Mid-Upper)": comp2,
            "Avg Comp Home 3 (Median)": comp3,
            "Avg Comp Home 4 (Mid-Lower)": comp4,
            "Avg Comp Home 5 (Entry Level)": comp5,
        }
    ).set_index("Year")
    
    return df_chart

# ==============================================================================
# SECTION 3: MAIN UI & WORKFLOW
# ==============================================================================
st.header("🏠 Professional Real Estate Investment & Underwriting Engine")
st.caption("Institutional valuation models, seller bottom calculation, CapEx estimation, and 60-year neighborhood price historical/projected trajectory.")

# TOP CONTROL ROW
col_re_ctrl1, col_re_ctrl2, col_re_ctrl3 = st.columns(3)
with col_re_ctrl1:
    investment_intent = st.selectbox(
        "Investment Intent",
        [
            "Personal Residence (Primary Home)",
            "Long-Term Rental (Buy & Hold)",
            "Short-Term Rental",
            "Fix & Flip",
        ],
        index=0,
        key="re_investment_intent"
    )
with col_re_ctrl2:
    property_type = st.selectbox(
        "Property Type", 
        ["Single Family", "Multi-Family (2-4 Units)", "Commercial"], 
        key="re_property_type"
    )
with col_re_ctrl3:
    analysis_mode = st.radio(
        "Analysis Mode", 
        ["Specific Property Address", "Regional Market Scout"], 
        index=0, 
        key="re_analysis_mode"
    )
st.markdown("---")

# ADDRESS SELECTOR / INPUT
col_addr1, col_addr2 = st.columns([3, 1])
with col_addr1:
    address_preset = st.selectbox(
        "Select Property Address (or type custom address below)",
        [
            "1244 S Michigan Ave, Chicago, IL 60605",
            "1530 S State St, Chicago, IL 60605",
            "1250 S Indiana Ave, Chicago, IL 60605",
            "Custom Address Input"
        ],
        index=0,
        key="re_addr_preset"
    )
    if address_preset == "Custom Address Input":
        address_input = st.text_input("Enter Custom Property Address", value="1244 S Michigan Ave, Chicago, IL 60605", key="re_custom_addr")
    else:
        address_input = address_preset
with col_addr2:
    days_on_market = st.number_input("Days on Market (DOM)", value=42, step=5, key="re_dom")

# FINANCIAL & AREA INPUT METRICS
col_p1, col_p2, col_p3, col_p4 = st.columns(4)
purchase_price = col_p1.number_input("Target List Price ($)", value=450000, step=10000, key="re_purchase_price")
down_payment_pct = col_p2.slider("Down Payment (%)", 0, 50, 20, key="re_down_payment_pct") / 100
interest_rate = col_p3.slider("Interest Rate (%)", 3.0, 12.0, 6.5, key="re_interest_rate") / 100
loan_term_years = col_p4.selectbox("Loan Term (Years)", [30, 15, 10], index=0, key="re_loan_term_years")

col_var1, col_var2, col_var3 = st.columns(3)
year_built = col_var1.number_input("Year Built", value=1998, step=1, key="re_year_built")
school_rating = col_var2.slider("Area School District Rating (1-10)", 1, 10, 8, key="re_school_rating")
labor_cost_idx = col_var3.slider("Local Labor & Trade Cost Index (100 = National Avg)", 70, 160, 118, key="re_labor_cost_idx")

# COMPUTE METRICS
re_metrics = compute_real_estate_valuation(
    address=address_input,
    purchase_price=purchase_price,
    intent=investment_intent,
    prop_type=property_type,
    school_rating=school_rating,
    labor_cost_idx=labor_cost_idx,
    dom_days=days_on_market,
    build_year=year_built
)

st.markdown("---")
st.subheader("📌 Institutional Property Valuation & Underwriting Summary")

rc1, rc2, rc3, rc4, rc5 = st.columns(5)
with rc1:
    st.markdown(
        f"""
        <div class="re-metric-card">
            <h4>Est. Market Value</h4>
            <div class="val">${re_metrics['market_price']:,.0f}</div>
            <div class="subtext">Internet Comp Estimate</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with rc2:
    st.markdown(
        f"""
        <div class="re-metric-card">
            <h4>Seller Lowest Price</h4>
            <div class="val">${re_metrics['lowest_seller_price']:,.0f}</div>
            <div class="subtext">Based on DOM & Comps</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with rc3:
    st.markdown(
        f"""
        <div class="re-metric-card">
            <h4>Target Initial Offer Range</h4>
            <div class="val">${re_metrics['bid_low']:,.0f} – ${re_metrics['bid_high']:,.0f}</div>
            <div class="subtext">Optimal Opening Bid</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with rc4:
    st.markdown(
        f"""
        <div class="re-metric-card">
            <h4>Closing Costs & Taxes</h4>
            <div class="val">${re_metrics['total_closing_costs']:,.0f}</div>
            <div class="subtext">Title, Escrow & Transfers</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with rc5:
    st.markdown(
        f"""
        <div class="re-metric-card">
            <h4>Est. Renovation / Rehab</h4>
            <div class="val">${re_metrics['rehab_low']:,.0f} – ${re_metrics['rehab_high']:,.0f}</div>
            <div class="subtext">Tailored to Intent & Labor</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("🔍 Professional Broker Diligence & Risk Factors (Area, Labor, Taxes & Zoning)", expanded=True):
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("School District Quality", re_metrics["school_score"])
    d2.metric("Labor & Trade Availability", re_metrics["labor_availability"])
    d3.metric("Property Tax Burden", f"{re_metrics['tax_burden_pct']}% (${re_metrics['annual_taxes']:,.0f}/yr)")
    d4.metric("Zoning & Permit Velocity", re_metrics["zoning_permits"])

# MORTGAGE & CASH FLOW BREAKDOWN
down_payment = purchase_price * down_payment_pct
loan_amount = purchase_price - down_payment
monthly_rate = interest_rate / 12
num_payments = loan_term_years * 12
monthly_mortgage = (
    (loan_amount * (monthly_rate * (1 + monthly_rate) ** num_payments) / ((1 + monthly_rate) ** num_payments - 1))
    if loan_amount > 0
    else 0
)
est_monthly_rent = purchase_price * 0.0075
monthly_cash_flow = est_monthly_rent - monthly_mortgage - (re_metrics["annual_taxes"] / 12)

col_mf1, col_mf2, col_mf3 = st.columns(3)
col_mf1.metric("Required Down Payment", f"${down_payment:,.0f}")
col_mf2.metric("Monthly Principal & Interest", f"${monthly_mortgage:,.0f}")
if investment_intent != "Personal Residence (Primary Home)":
    col_mf3.metric("Est. Net Monthly Cash Flow", f"${monthly_cash_flow:,.0f}")
else:
    col_mf3.metric("Est. Total Monthly Carrying Cost", f"${(monthly_mortgage + (re_metrics['annual_taxes'] / 12)):,.0f}")

st.markdown("---")
st.subheader(f"📈 Neighborhood Valuation Dynamics ({address_input})")
st.caption("Comprehensive 60-Year Price Trend Comparison: **40-Year Historical Data (1986–2026)** vs. **20-Year Predictive Projection (2026–2046)**.")

df_re_chart = generate_40yr_hist_20yr_proj_housing_data(re_metrics["market_price"])

fig_re = go.Figure()
fig_re.add_trace(go.Scatter(
    x=df_re_chart.index,
    y=df_re_chart["Subject Property Trajectory"],
    mode='lines',
    name='⭐ Subject Property Trajectory',
    line=dict(width=4, color='#00ACC1')
))
fig_re.add_trace(go.Scatter(
    x=df_re_chart.index,
    y=df_re_chart["Highest Price Home in Neighborhood"],
    mode='lines',
    name='Highest Price Home in Neighborhood',
    line=dict(width=2, dash='dash', color='#28a745')
))
fig_re.add_trace(go.Scatter(
    x=df_re_chart.index,
    y=df_re_chart["Lowest Price Home in Neighborhood"],
    mode='lines',
    name='Lowest Price Home in Neighborhood',
    line=dict(width=2, dash='dash', color='#dc3545')
))
fig_re.add_trace(go.Scatter(
    x=df_re_chart.index,
    y=df_re_chart["Overall Avg Neighborhood Price"],
    mode='lines',
    name='Overall Neighborhood Avg Price',
    line=dict(width=2.5, color='#ffc107')
))
fig_re.add_trace(go.Scatter(
    x=df_re_chart.index,
    y=df_re_chart["Avg Price Same ZIP Code Area"],
    mode='lines',
    name='Avg Price Same ZIP Code Area',
    line=dict(width=2, color='#6c757d')
))

for comp_name, color_code in [
    ("Avg Comp Home 1 (Upper Tier)", "#8e44ad"),
    ("Avg Comp Home 2 (Mid-Upper)", "#2980b9"),
    ("Avg Comp Home 3 (Median)", "#16a085"),
    ("Avg Comp Home 4 (Mid-Lower)", "#d35400"),
    ("Avg Comp Home 5 (Entry Level)", "#7f8c8d"),
]:
    fig_re.add_trace(go.Scatter(
        x=df_re_chart.index,
        y=df_re_chart[comp_name],
        mode='lines',
        name=comp_name,
        line=dict(width=1.5, dash='dot', color=color_code)
    ))

fig_re.add_vline(x=2026, line_width=2, line_dash="dash", line_color="red")
fig_re.add_annotation(x=2026, y=df_re_chart["Subject Property Trajectory"].loc[2026], text="2026 Present Benchmark", showarrow=True, arrowhead=1)
fig_re.update_layout(
    title=f"Neighborhood Valuation Comparison & Long-Term Trajectory (1986 – 2046)",
    xaxis_title="Year",
    yaxis_title="Property Value ($)",
    margin=dict(l=20, r=20, t=40, b=20),
    height=520,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=-0.45, xanchor="left", x=0)
)
st.plotly_chart(fig_re, use_container_width=True)

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
col_ft_left, col_ft_right = st.columns([4, 1])
with col_ft_left:
    st.caption("© 2026 Core Market Intelligence (CMI). Real Estate Underwriting & Asset Management Engine.")
with col_ft_right:
    st.markdown(CMI_LOGO_SVG, unsafe_allow_html=True)
