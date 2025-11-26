# ---- START AIUpdatedData.py ----

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Desudo Final Dashboard", layout="wide")
st.title("Desudo Portfolio – Final Interactive Dashboard")

# Load and Prepare Data
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data():
    fr = pd.read_csv("data_desudo_france.csv", dtype=str)
    it = pd.read_csv("data_desudo_italy.csv", dtype=str)
    fr.columns = [c.lower() for c in fr.columns]
    it.columns = [c.lower() for c in it.columns]
    return fr, it

@st.cache_data(show_spinner=False)
def prepare(df, country):
    def parse_date(s):
        return pd.to_datetime(s, errors='coerce')
    def to_float(s):
        return pd.to_numeric(s.astype(str).str.replace(',','').str.strip(), errors='coerce')
    df['country'] = country
    for col in ['totalarrearamount','customertotalarrearamount','dpd']:
        df[col+'_num'] = to_float(df.get(col,''))
    df['bucket_flag'] = df.get('bucket','').str.contains('Bucket', case=False, na=False)
    sub = df.get('contract_substatus','').str.lower()
    df['is_arrears_substatus'] = sub.str.contains('arrears|bankruptcy|repossession|debt|fraud', regex=True)
    cst = df.get('contract_status','').fillna('')
    df['terminated_due_to_arrears'] = cst.str.contains('830') | cst.str.contains('Terminated due to Arrears', case=False)
    df['dpd_flag'] = (df['dpd_num'].fillna(0) > 0)
    df['arrears_amount_flag'] = (df['totalarrearamount_num'].fillna(0) > 0) | (df['customertotalarrearamount_num'].fillna(0) > 0)
    df['is_delinquent'] = df[['bucket_flag','is_arrears_substatus','terminated_due_to_arrears','dpd_flag','arrears_amount_flag']].any(axis=1)
    candidates = [parse_date(df.get('oldestduedate')), parse_date(df.get('contractenddate')), parse_date(df.get('workqueue_entrydate'))]
    event_date = candidates[0]
    for c in candidates[1:]:
        event_date = event_date.fillna(c)
    df['event_date'] = pd.to_datetime(event_date, errors='coerce').dt.tz_localize(None)
    df['month'] = df['event_date'].dt.to_period('M').astype(str)
    df['quarter'] = df['event_date'].dt.to_period('Q').astype(str)
    df['year'] = df['event_date'].dt.year
    df['q_num'] = df['event_date'].dt.quarter
    return df

fr, it = load_data()
fr = prepare(fr,'FRANCE')
it = prepare(it,'ITALY')
all_df = pd.concat([fr,it], ignore_index=True)
mask = (all_df['event_date'] >= pd.Timestamp('2018-01-01')) & (all_df['event_date'] <= pd.Timestamp('2025-12-31'))
all_df = all_df[mask]

# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")
selected_countries = st.sidebar.multiselect("Select Countries", options=sorted(all_df['country'].unique()), default=sorted(all_df['country'].unique()))
dealer_filter = st.sidebar.multiselect("Dealer(s)", options=sorted(all_df['dealerbpid'].dropna().unique()), default=[])
cust_filter = st.sidebar.multiselect("Customer Type(s)", options=sorted(all_df['legalentitycode'].dropna().unique()), default=[])
date_range = st.sidebar.date_input("Date Range", value=(all_df['event_date'].min().date(), all_df['event_date'].max().date()))

# Advanced filters
model_filter = st.sidebar.multiselect("Model Description(s)", options=sorted(all_df['modeldescription'].dropna().unique()), default=[], key="model_filter")
fuel_filter = st.sidebar.multiselect("Fuel Type(s)", options=sorted(all_df['fueltypecode'].dropna().unique()), default=[], key="fuel_filter")
model_search = st.sidebar.text_input("Search Model (partial match)", value="", key="model_search")

# Global revenue basis
st.sidebar.subheader("Revenue Basis")
rev_basis = st.sidebar.radio("Revenue Basis", ["Capital Only","Capital+Interest+Fees+Other","NetbookValue"], index=1, key="rev_basis")

# Currency symbol toggle
currency_symbol = st.sidebar.selectbox('Currency Symbol', options=['€','₹','$','None'], index=0, key='currency_symbol')

# Currency conversion
st.sidebar.subheader('Currency Conversion')
conversion_option = st.sidebar.selectbox('Convert To', options=['EUR','INR','USD'], index=0, key='conversion_option')
conversion_rates = {'EUR':1.0,'INR':90.0,'USD':1.1}
conversion_rate = conversion_rates.get(conversion_option,1.0)
st.sidebar.caption(f"Applied rate: 1 EUR = {conversion_rate:.2f} {conversion_option}")

# Reset buttons
reset_filters = st.sidebar.button("Reset Filters")
reset_conversion = st.sidebar.button("Reset Conversion")

if reset_filters:
    for k in ['selected_countries','dealer_filter','cust_filter','model_filter','fuel_filter','model_search','date_range','rev_basis','conversion_option']:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

if reset_conversion:
    for k in ['conversion_option']:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

# Apply filters
filtered = all_df[(all_df['country'].isin(selected_countries)) & (all_df['event_date'].dt.date >= date_range[0]) & (all_df['event_date'].dt.date <= date_range[1])]
if dealer_filter:
    filtered = filtered[filtered['dealerbpid'].isin(dealer_filter)]
if cust_filter:
    filtered = filtered[filtered['legalentitycode'].isin(cust_filter)]
if model_filter:
    filtered = filtered[filtered['modeldescription'].isin(model_filter)]
if fuel_filter:
    filtered = filtered[filtered['fueltypecode'].isin(fuel_filter)]
if model_search:
    filtered = filtered[filtered['modeldescription'].str.contains(model_search.strip(), case=False, na=False)]

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs(["Multi-Country Trends","Variance","Fiscal Analysis","Dealer Analysis","Seasonal","Export","Financial Revenue Analysis","Car Model Analysis"])

# Financial Revenue Analysis Tab
# -----------------------------
with tabs[6]:
    st.subheader("Financial Revenue Analysis")
    basis_option = st.session_state.get("rev_basis", "Capital+Interest+Fees+Other")
    if basis_option == "Capital Only":
        filtered['revenue_amount'] = pd.to_numeric(filtered['totalcapitalamount'].astype(str).str.replace(',','').str.strip(), errors='coerce')
    elif basis_option == "Capital+Interest+Fees+Other":
        cols = ['totalcapitalamount','totalinterestamount','totalfeeamount','totalotheramount']
        for c in cols:
            if c not in filtered.columns:
                filtered[c] = 0
        filtered['revenue_amount'] = filtered[cols].apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',','').str.strip(), errors='coerce')).sum(axis=1)
    else:
        filtered['revenue_amount'] = pd.to_numeric(filtered['netbookvalue'].astype(str).str.replace(',','').str.strip(), errors='coerce')

    # Apply conversion
    filtered['revenue_amount'] *= conversion_rate

    # KPIs
    total_revenue = filtered['revenue_amount'].sum()
    avg_rev_contract = filtered['revenue_amount'].mean()
    total_contracts = filtered['contractnumber'].count()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue", f"{currency_symbol} {total_revenue:,.2f}")
    c2.metric("Average Revenue per Contract", f"{currency_symbol} {avg_rev_contract:,.2f}")
    c3.metric("Contracts", f"{total_contracts:,}")
    st.caption(f"Applied conversion: 1 EUR = {conversion_rate:.2f} {conversion_option}")

    # Charts
    rev_country = filtered.groupby('country').agg(total_revenue=('revenue_amount','sum')).reset_index()
    fig_rev_country = px.bar(rev_country, x='country', y='total_revenue', title='Revenue by Country')
    fig_rev_country.update_traces(hovertemplate=currency_symbol + ' %{y:,.2f}')
    fig_rev_country.update_yaxes(tickprefix=currency_symbol, tickformat=',.2f')
    st.plotly_chart(fig_rev_country, use_container_width=True)

    rev_time = filtered.groupby('month').agg(total_revenue=('revenue_amount','sum')).reset_index()
    rev_time['month_dt'] = pd.to_datetime(rev_time['month']+'-01')
    fig_rev_time = px.line(rev_time, x='month_dt', y='total_revenue', markers=True, title='Monthly Revenue Trend')
    fig_rev_time.update_traces(hovertemplate=currency_symbol + ' %{y:,.2f}')
    fig_rev_time.update_yaxes(tickprefix=currency_symbol, tickformat=',.2f')
    st.plotly_chart(fig_rev_time, use_container_width=True)

    # Revenue trend by fuel type
    st.markdown("### Revenue Trend by Fuel Type (Monthly)")
    if 'month_dt' not in filtered.columns:
        filtered['month_dt'] = pd.to_datetime(filtered['month']+'-01')
    fuel_time = filtered.groupby(['month_dt','fueltypecode']).agg(total_revenue=('revenue_amount','sum')).reset_index()
    fuel_time['total_revenue'] *= 1
    view_mode = st.radio("Fuel Trend View", ["Multi-line","Stacked area"], index=0, horizontal=True, key="fuel_trend_view")
    if view_mode == "Multi-line":
        fig_fuel_time = px.line(fuel_time, x='month_dt', y='total_revenue', color='fueltypecode', markers=True, title='Monthly Revenue by Fuel Type')
    else:
        fig_fuel_time = px.area(fuel_time, x='month_dt', y='total_revenue', color='fueltypecode', title='Monthly Revenue by Fuel Type (Stacked)')
    fig_fuel_time.update_traces(hovertemplate=currency_symbol + ' %{y:,.2f}')
    fig_fuel_time.update_yaxes(tickprefix=currency_symbol, tickformat=',.2f')
    st.plotly_chart(fig_fuel_time, use_container_width=True)

    # Revenue trend by country
    st.markdown("### Revenue Trend by Country (Monthly)")
    rev_country_time = filtered.groupby(['month_dt','country']).agg(total_revenue=('revenue_amount','sum')).reset_index()
    fig_country_time = px.line(rev_country_time, x='month_dt', y='total_revenue', color='country', markers=True, title='Monthly Revenue by Country')
    fig_country_time.update_traces(hovertemplate=currency_symbol + ' %{y:,.2f}')
    fig_country_time.update_yaxes(tickprefix=currency_symbol, tickformat=',.2f')
    st.plotly_chart(fig_country_time, use_container_width=True)

    # Fuel type comparison
    st.markdown("### Fuel Type Comparison")
    fuel_comp = filtered.groupby('fueltypecode').agg(total_revenue=('revenue_amount','sum'), avg_rev_contract=('revenue_amount','mean'), contracts=('contractnumber','count')).reset_index()
    fuel_comp['total_revenue'] *= 1
    fuel_comp['avg_rev_contract'] *= 1
    fig_fuel_comp = px.bar(fuel_comp, x='fueltypecode', y='total_revenue', color='fueltypecode', title='Total Revenue by Fuel Type')
    fig_fuel_comp.update_traces(hovertemplate=currency_symbol + ' %{y:,.2f}')
    fig_fuel_comp.update_yaxes(tickprefix=currency_symbol, tickformat=',.2f')
    st.plotly_chart(fig_fuel_comp, use_container_width=True)

    fig_fuel_avg = px.bar(fuel_comp, x='fueltypecode', y='avg_rev_contract', color='fueltypecode', title='Average Revenue per Contract by Fuel Type')
    fig_fuel_avg.update_traces(hovertemplate=currency_symbol + ' %{y:,.2f}')
    fig_fuel_avg.update_yaxes(tickprefix=currency_symbol, tickformat=',.2f')
    st.plotly_chart(fig_fuel_avg, use_container_width=True)

    st.dataframe(fuel_comp)

# -----------------------------
# Car Model Analysis Tab
# -----------------------------
with tabs[7]:
    st.subheader("Car Model Analysis by Fuel Type")
    basis_option = st.session_state.get("rev_basis", "Capital+Interest+Fees+Other")
    if basis_option == "Capital Only":
        filtered['revenue_amount'] = pd.to_numeric(filtered['totalcapitalamount'].astype(str).str.replace(',','').str.strip(), errors='coerce')
    elif basis_option == "Capital+Interest+Fees+Other":
        cols = ['totalcapitalamount','totalinterestamount','totalfeeamount','totalotheramount']
        for c in cols:
            if c not in filtered.columns:
                filtered[c] = 0
        filtered['revenue_amount'] = filtered[cols].apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',','').str.strip(), errors='coerce')).sum(axis=1)
    else:
        filtered['revenue_amount'] = pd.to_numeric(filtered['netbookvalue'].astype(str).str.replace(',','').str.strip(), errors='coerce')

    filtered['revenue_amount'] *= conversion_rate
    top_n = st.slider("Select Top N Models", min_value=5, max_value=20, value=10)
    st.caption(f"Applied conversion: 1 EUR = {conversion_rate:.2f} {conversion_option}")

    model_fuel_rev = filtered.groupby(['fueltypecode','modeldescription']).agg(revenue=('revenue_amount','sum'), contracts=('contractnumber','count'), avg_rev_contract=('revenue_amount','mean')).reset_index()
    top_by_fuel = model_fuel_rev.sort_values(['fueltypecode','revenue'], ascending=[True, False]).groupby('fueltypecode').head(top_n)
    fig_top_by_fuel = px.bar(top_by_fuel, x='modeldescription', y='revenue', color='fueltypecode', title=f'Top {top_n} Models by Revenue within Each Fuel Type')
    fig_top_by_fuel.update_xaxes(tickangle=45)
    fig_top_by_fuel.update_traces(hovertemplate=currency_symbol + ' %{y:,.2f}')
    fig_top_by_fuel.update_yaxes(tickprefix=currency_symbol, tickformat=',.2f')
    st.plotly_chart(fig_top_by_fuel, use_container_width=True)

    st.download_button(label="Download Model-Fuel Revenue CSV", data=model_fuel_rev.to_csv(index=False), file_name="model_fuel_revenue.csv", mime="text/csv")
    st.dataframe(model_fuel_rev.sort_values(['fueltypecode','revenue'], ascending=[True, False]))


# ---- START AIData.py ----


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Desudo Final Dashboard", layout="wide")
st.title("Desudo Portfolio – Final Interactive Dashboard")

# Load and Prepare Data
# -----------------------------
@st.cache_data(show_spinner=False)
    fr = pd.read_csv("data_desudo_france.csv", dtype=str)
    it = pd.read_csv("data_desudo_italy.csv", dtype=str)
    fr.columns = [c.lower() for c in fr.columns]
    it.columns = [c.lower() for c in it.columns]
    return fr, it

@st.cache_data(show_spinner=False)
        return pd.to_datetime(s, errors='coerce')
        return pd.to_numeric(s.astype(str).str.replace(',','').str.strip(), errors='coerce')
    df['country'] = country
    for col in ['totalarrearamount','customertotalarrearamount','dpd']:
        df[col+'_num'] = to_float(df.get(col,''))
    df['bucket_flag'] = df.get('bucket','').str.contains('Bucket', case=False, na=False)
    sub = df.get('contract_substatus','').str.lower()
    df['is_arrears_substatus'] = sub.str.contains('arrears|bankruptcy|repossession|debt|fraud', regex=True)
    cst = df.get('contract_status','').fillna('')
    df['terminated_due_to_arrears'] = cst.str.contains('830') | cst.str.contains('Terminated due to Arrears', case=False)
    df['dpd_flag'] = (df['dpd_num'].fillna(0) > 0)
    df['arrears_amount_flag'] = (df['totalarrearamount_num'].fillna(0) > 0) | (df['customertotalarrearamount_num'].fillna(0) > 0)
    df['is_delinquent'] = df[['bucket_flag','is_arrears_substatus','terminated_due_to_arrears','dpd_flag','arrears_amount_flag']].any(axis=1)
    candidates = [parse_date(df.get('oldestduedate')), parse_date(df.get('contractenddate')), parse_date(df.get('workqueue_entrydate'))]
    event_date = candidates[0]
    for c in candidates[1:]:
        event_date = event_date.fillna(c)
    df['event_date'] = pd.to_datetime(event_date, errors='coerce').dt.tz_localize(None)
    df['month'] = df['event_date'].dt.to_period('M').astype(str)
    df['quarter'] = df['event_date'].dt.to_period('Q').astype(str)
    df['year'] = df['event_date'].dt.year
    df['q_num'] = df['event_date'].dt.quarter
    return df

fr, it = load_data()
fr = prepare(fr,'FRANCE')
it = prepare(it,'ITALY')
all_df = pd.concat([fr,it], ignore_index=True)
mask = (all_df['event_date'] >= pd.Timestamp('2018-01-01')) & (all_df['event_date'] <= pd.Timestamp('2025-12-31'))
all_df = all_df[mask]

# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")
selected_countries = st.sidebar.multiselect("Select Countries", options=sorted(all_df['country'].unique()), default=sorted(all_df['country'].unique()))
dealer_filter = st.sidebar.multiselect("Dealer(s)", options=sorted(all_df['dealerbpid'].dropna().unique()), default=[])
cust_filter = st.sidebar.multiselect("Customer Type(s)", options=sorted(all_df['legalentitycode'].dropna().unique()), default=[])
date_range = st.sidebar.date_input("Date Range", value=(all_df['event_date'].min().date(), all_df['event_date'].max().date()))

filtered = all_df[(all_df['country'].isin(selected_countries)) & (all_df['event_date'].dt.date >= date_range[0]) & (all_df['event_date'].dt.date <= date_range[1])]
if dealer_filter:
    filtered = filtered[filtered['dealerbpid'].isin(dealer_filter)]
if cust_filter:
    filtered = filtered[filtered['legalentitycode'].isin(cust_filter)]

# Tabs
# -----------------------------
tabs = st.tabs(["Multi-Country Trends","Variance","Fiscal Analysis","Dealer Analysis","Seasonal","Export"])

# Multi-Country Trends
with tabs[0]:
    st.subheader("Monthly Delinquency Rate by Country")
    monthly = filtered.groupby(['country','month']).agg(total=('contractnumber','count'),delinq=('is_delinquent','sum')).reset_index()
    monthly['rate_pct'] = (monthly['delinq']/monthly['total']*100).round(2)
    monthly['month_dt'] = pd.to_datetime(monthly['month']+'-01')
    fig_month = px.line(monthly, x='month_dt', y='rate_pct', color='country', markers=True, title='Monthly Delinquency Rate')
    st.plotly_chart(fig_month, use_container_width=True)

    st.subheader("Quarterly Delinquency Rate by Country")
    quarterly = filtered.groupby(['country','quarter']).agg(total=('contractnumber','count'),delinq=('is_delinquent','sum')).reset_index()
    quarterly['rate_pct'] = (quarterly['delinq']/quarterly['total']*100).round(2)
    quarterly['q_dt'] = pd.PeriodIndex(quarterly['quarter'], freq='Q').to_timestamp()
    fig_quarter = px.line(quarterly, x='q_dt', y='rate_pct', color='country', markers=True, title='Quarterly Delinquency Rate')
    st.plotly_chart(fig_quarter, use_container_width=True)

# Variance Tab
with tabs[1]:
    st.subheader("MoM Variance")
    if not monthly.empty:
        monthly_sorted = monthly.sort_values(['country','month_dt'])
        monthly_sorted['mom_var'] = monthly_sorted.groupby('country')['rate_pct'].pct_change()*100
        fig_mom = px.bar(monthly_sorted, x='month_dt', y='mom_var', color='country', title='MoM Variance (%)')
        st.plotly_chart(fig_mom, use_container_width=True)

    st.subheader("QoQ Variance")
    if not quarterly.empty:
        quarterly_sorted = quarterly.sort_values(['country','q_dt'])
        quarterly_sorted['qoq_var'] = quarterly_sorted.groupby('country')['rate_pct'].pct_change()*100
        fig_qoq = px.bar(quarterly_sorted, x='q_dt', y='qoq_var', color='country', title='QoQ Variance (%)')
        st.plotly_chart(fig_qoq, use_container_width=True)

# Fiscal Analysis Tab
with tabs[2]:
    st.subheader("Fiscal Q4 vs Q1 Comparison")
    q4q1 = filtered[filtered['q_num'].isin([1,4])].groupby(['country','year','q_num']).agg(rate=('is_delinquent','mean')).reset_index()
    q4q1['rate'] = q4q1['rate']*100
    pivot_q4q1 = q4q1.pivot_table(index=['country','year'], columns='q_num', values='rate').reset_index()
    fig_q4q1 = go.Figure()
    for country in selected_countries:
        d = pivot_q4q1[pivot_q4q1['country']==country]
        fig_q4q1.add_trace(go.Scatter(x=d['year'], y=d.get(4,[]), mode='lines+markers', name=f'{country} Q4'))
        fig_q4q1.add_trace(go.Scatter(x=d['year'], y=d.get(1,[]), mode='lines+markers', name=f'{country} Q1'))
    fig_q4q1.update_layout(title='Fiscal Q4 vs Q1 Comparison')
    st.plotly_chart(fig_q4q1, use_container_width=True)

# Dealer Analysis Tab
with tabs[3]:
    st.subheader("Dealer-Level Analysis")
    dealer_data = filtered.groupby(['dealerbpid','country']).agg(rate=('is_delinquent','mean')).reset_index()
    dealer_data['rate'] = dealer_data['rate']*100
    fig_dealer = px.bar(dealer_data, x='dealerbpid', y='rate', color='country', title='Dealer-Level Delinquency Rate')
    st.plotly_chart(fig_dealer, use_container_width=True)

# Seasonal Tab
with tabs[4]:
    st.subheader("Seasonal Trend by Month")
    seasonal = filtered.groupby([filtered['event_date'].dt.month_name(),'country']).agg(rate=('is_delinquent','mean')).reset_index()
    seasonal['rate'] = seasonal['rate']*100
    fig_seasonal = px.bar(seasonal, x='event_date', y='rate', color='country', title='Seasonal Trend')
    st.plotly_chart(fig_seasonal, use_container_width=True)

# Export Tab
with tabs[5]:
    st.write("Report export functionality can be added here (Word/PDF generation).")

