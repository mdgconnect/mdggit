
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="RiskIQ Dashboard", layout="wide")
st.title("RiskIQ Interactive Dashboard")

# -----------------------------
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

# -----------------------------
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

# -----------------------------
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

