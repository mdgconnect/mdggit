
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AutoCred Global", layout="wide")
st.title("AutoCred Global Dashboard")

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
def prepare(df, Country):
    def parse_date(s):
        return pd.to_datetime(s, errors='coerce')
    def to_float(s):
        return pd.to_numeric(s.astype(str).str.replace(',','').str.strip(), errors='coerce')
    df['Country'] = Country
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
    Event = candidates[0]
    for c in candidates[1:]:
        Event = Event.fillna(c)
    df['Event'] = pd.to_datetime(Event, errors='coerce').dt.tz_localize(None)
    df['month'] = df['Event'].dt.to_period('M').astype(str)
    df['quarter'] = df['Event'].dt.to_period('Q').astype(str)
    df['year'] = df['Event'].dt.year
    df['q_num'] = df['Event'].dt.quarter
    return df

fr, it = load_data()
fr = prepare(fr,'FRANCE')
it = prepare(it,'ITALY')
all_df = pd.concat([fr,it], ignore_index=True)
mask = (all_df['Event'] >= pd.Timestamp('2018-01-01')) & (all_df['Event'] <= pd.Timestamp('2025-12-31'))
all_df = all_df[mask]

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")
selected_countries = st.sidebar.multiselect("Select Countries", options=sorted(all_df['Country'].unique()), default=sorted(all_df['Country'].unique()))
cust_filter = st.sidebar.multiselect("Customer Type", options=sorted(all_df['legalentitycode'].dropna().unique()), default=[])
date_range = st.sidebar.date_input("Date Range", value=(all_df['Event'].min().date(), all_df['Event'].max().date()))

# Advanced filters
model_filter = st.sidebar.multiselect("Vehicle Model", options=sorted(all_df['modeldescription'].dropna().unique()), default=[], key="model_filter")
fuel_filter = st.sidebar.multiselect(
    "fueltypecode",
    options=sorted(all_df['fueltypecode'].dropna().unique()),
    default=[],
    key="fuel_filter")
model_search = st.sidebar.text_input("Search Model", value="", key="model_search")

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
    for k in ['selected_countries','dealer_filter','cust_filter','model_filter','fuel_filter','model_search','date_range','rev_basis','conversion_option','currency_symbol','fuel_trend_view']:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

if reset_conversion:
    for k in ['conversion_option']:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

# Apply filters
filtered = all_df[(all_df['Country'].isin(selected_countries)) & (all_df['Event'].dt.date >= date_range[0]) & (all_df['Event'].dt.date <= date_range[1])]
if cust_filter:
    filtered = filtered[filtered['legalentitycode'].isin(cust_filter)]
if model_filter:
    filtered = filtered[filtered['modeldescription'].isin(model_filter)]
if fuel_filter:
    filtered = filtered[filtered['FuelType'].isin(fuel_filter)]
if model_search:
    filtered = filtered[filtered['modeldescription'].str.contains(model_search.strip(), case=False, na=False)]

# Tabs
# -----------------------------
tabs = st.tabs(["Multi-Country Trends","Variance","Fiscal Analysis","Seasonal","Financial Revenue Analysis","Car Model Analysis"])
# Multi-Country Trends
with tabs[0]:
    st.subheader("Monthly Delinquency Rate by Country")
    monthly = filtered.groupby(['Country','month']).agg(total=('contractnumber','count'),delinq=('is_delinquent','sum')).reset_index()
    monthly['Rate Percentage'] = (monthly['delinq']/monthly['total']*100).round(2)
    monthly['Month'] = pd.to_datetime(monthly['month']+'-01')
    fig_month = px.line(monthly, x='Month', y='Rate Percentage', color='Country', markers=True, title='Monthly Delinquency Rate')
    st.plotly_chart(fig_month, use_container_width=True)

    st.subheader("Quarterly Delinquency Rate by Country")
    quarterly = filtered.groupby(['Country','quarter']).agg(total=('contractnumber','count'),delinq=('is_delinquent','sum')).reset_index()
    quarterly['Rate Percentage'] = (quarterly['delinq']/quarterly['total']*100).round(2)
    quarterly['Quarter'] = pd.PeriodIndex(quarterly['quarter'], freq='Q').to_timestamp()
    fig_quarter = px.line(quarterly, x='Quarter', y='Rate Percentage', color='Country', markers=True, title='Quarterly Delinquency Rate')
    st.plotly_chart(fig_quarter, use_container_width=True)

# Variance Tab
with tabs[1]:
    st.subheader("MoM Variance")
    if not monthly.empty:
        monthly_sorted = monthly.sort_values(['Country','Month'])
        monthly_sorted['MoM Variance'] = monthly_sorted.groupby('Country')['Rate Percentage'].pct_change()*100
        fig_mom = px.bar(monthly_sorted, x='Month', y='MoM Variance', color='Country', title='MoM Variance (%)')
        st.plotly_chart(fig_mom, use_container_width=True)

    st.subheader("QoQ Variance")
    if not quarterly.empty:
        quarterly_sorted = quarterly.sort_values(['Country','Quarter'])
        quarterly_sorted['QoQ Variance'] = quarterly_sorted.groupby('Country')['Rate Percentage'].pct_change()*100
        fig_qoq = px.bar(quarterly_sorted, x='Quarter', y='QoQ Variance', color='Country', title='QoQ Variance (%)')
        st.plotly_chart(fig_qoq, use_container_width=True)

# Fiscal Analysis Tab
with tabs[2]:
    st.subheader("Fiscal Q4 vs Q1 Comparison")
    q4q1 = filtered[filtered['q_num'].isin([1,4])].groupby(['Country','year','q_num']).agg(rate=('is_delinquent','mean')).reset_index()
    q4q1['rate'] = q4q1['rate']*100
    pivot_q4q1 = q4q1.pivot_table(index=['Country','year'], columns='q_num', values='rate').reset_index()
    fig_q4q1 = go.Figure()
    for Country in selected_countries:
        d = pivot_q4q1[pivot_q4q1['Country']==Country]
        fig_q4q1.add_trace(go.Scatter(x=d['year'], y=d.get(4,[]), mode='lines+markers', name=f'{Country} Q4'))
        fig_q4q1.add_trace(go.Scatter(x=d['year'], y=d.get(1,[]), mode='lines+markers', name=f'{Country} Q1'))
    fig_q4q1.update_layout(title='Fiscal Q4 vs Q1 Comparison')
    st.plotly_chart(fig_q4q1, use_container_width=True)


# Seasonal Tab
with tabs[3]:
    st.subheader("Seasonal Trend by Month")

    seasonal = (
        filtered
        .groupby([filtered['Event'].dt.month_name().rename("Event"), 'Country'])
        .agg({'is_delinquent': 'mean'})
        .reset_index()
        .rename(columns={'is_delinquent': 'Rate Percentage'})
    )

    seasonal['Rate Percentage'] *= 100

    fig_seasonal = px.bar(
        seasonal,
        x='Event',
        y='Rate Percentage',
        color='Country',
        title='Seasonal Trend'
    )

    st.plotly_chart(fig_seasonal, use_container_width=True)


# -----------------------------
# Financial Revenue Analysis Tab
# -----------------------------
with tabs[4]:
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
    Turnover = filtered['revenue_amount'].sum()
    avg_rev_contract = filtered['revenue_amount'].mean()
    total_contracts = filtered['contractnumber'].count()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue", f"{currency_symbol} {Turnover:,.2f}")
    c2.metric("Average Revenue per Contract", f"{currency_symbol} {avg_rev_contract:,.2f}")
    c3.metric("Contracts", f"{total_contracts:,}")
    st.caption(f"Applied conversion: 1 EUR = {conversion_rate:.2f} {conversion_option}")

    # Charts
    rev_Country = filtered.groupby('Country').agg(Turnover=('revenue_amount','sum')).reset_index()
    fig_rev_Country = px.bar(rev_Country, x='Country', y='Turnover', title='Revenue by Country')
    fig_rev_Country.update_traces(hovertemplate=(('' if currency_symbol == 'None' else currency_symbol) + ' %{y:,.2f}'))
    fig_rev_Country.update_yaxes(tickprefix=currency_symbol, tickformat=',.2f')
    st.plotly_chart(fig_rev_Country, use_container_width=True)

    rev_time = filtered.groupby('month').agg(Turnover=('revenue_amount','sum')).reset_index()
    rev_time['Month'] = pd.to_datetime(rev_time['month']+'-01')
    fig_rev_time = px.line(rev_time, x='Month', y='Turnover', markers=True, title='Monthly Revenue Trend')
    fig_rev_time.update_traces(hovertemplate=(('' if currency_symbol == 'None' else currency_symbol) + ' %{y:,.2f}'))
    fig_rev_time.update_yaxes(tickprefix=currency_symbol, tickformat=',.2f')
    st.plotly_chart(fig_rev_time, use_container_width=True)

    # Revenue trend by fuel type
    st.markdown("### Revenue Trend by Fuel Type (Monthly)")
    if 'Month' not in filtered.columns:
        filtered['Month'] = pd.to_datetime(filtered['month']+'-01')
    fuel_time = filtered.groupby(['Month','FuelType']).agg(Turnover=('revenue_amount','sum')).reset_index()
    fuel_time['Turnover'] *= 1
    view_mode = st.radio("Fuel Trend View", ["Multi-line","Stacked area"], index=0, horizontal=True, key="fuel_trend_view")
    if view_mode == "Multi-line":
        fig_fuel_time = px.line(fuel_time, x='Month', y='Turnover', color='FuelType', markers=True, title='Monthly Revenue by Fuel Type')
    else:
        fig_fuel_time = px.area(fuel_time, x='Month', y='Turnover', color='FuelType', title='Monthly Revenue by Fuel Type (Stacked)')
    fig_fuel_time.update_traces(hovertemplate=(('' if currency_symbol == 'None' else currency_symbol) + ' %{y:,.2f}'))
    fig_fuel_time.update_yaxes(tickprefix=currency_symbol, tickformat=',.2f')
    st.plotly_chart(fig_fuel_time, use_container_width=True)

    # Revenue trend by Country
    st.markdown("### Revenue Trend by Country (Monthly)")
    rev_Country_time = filtered.groupby(['Month','Country']).agg(Turnover=('revenue_amount','sum')).reset_index()
    fig_Country_time = px.line(rev_Country_time, x='Month', y='Turnover', color='Country', markers=True, title='Monthly Revenue by Country')
    fig_Country_time.update_traces(hovertemplate=(('' if currency_symbol == 'None' else currency_symbol) + ' %{y:,.2f}'))
    fig_Country_time.update_yaxes(tickprefix=currency_symbol, tickformat=',.2f')
    st.plotly_chart(fig_Country_time, use_container_width=True)

    # Fuel type comparison
    st.markdown("### Fuel Type Comparison")
    fuel_comp = filtered.groupby('FuelType').agg(Turnover=('revenue_amount','sum'), avg_rev_contract=('revenue_amount','mean'), contracts=('contractnumber','count')).reset_index()
    fuel_comp['Turnover'] *= 1
    fuel_comp['avg_rev_contract'] *= 1
    fig_fuel_comp = px.bar(fuel_comp, x='FuelType', y='Turnover', color='FuelType', title='Total Revenue by Fuel Type')
    fig_fuel_comp.update_traces(hovertemplate=(('' if currency_symbol == 'None' else currency_symbol) + ' %{y:,.2f}'))
    fig_fuel_comp.update_yaxes(tickprefix=currency_symbol, tickformat=',.2f')
    st.plotly_chart(fig_fuel_comp, use_container_width=True)

    fig_fuel_avg = px.bar(fuel_comp, x='FuelType', y='avg_rev_contract', color='FuelType', title='Average Revenue per Contract by Fuel Type')
    fig_fuel_avg.update_traces(hovertemplate=(('' if currency_symbol == 'None' else currency_symbol) + ' %{y:,.2f}'))
    fig_fuel_avg.update_yaxes(tickprefix=currency_symbol, tickformat=',.2f')
    st.plotly_chart(fig_fuel_avg, use_container_width=True)

    st.dataframe(fuel_comp)

# -----------------------------
# Car Model Analysis Tab
# -----------------------------
with tabs[5]:
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

    model_fuel_rev = filtered.groupby(['FuelType','modeldescription']).agg(revenue=('revenue_amount','sum'), contracts=('contractnumber','count'), avg_rev_contract=('revenue_amount','mean')).reset_index()
    top_by_fuel = model_fuel_rev.sort_values(['FuelType','revenue'], ascending=[True, False]).groupby('FuelType').head(top_n)
    fig_top_by_fuel = px.bar(top_by_fuel, x='modeldescription', y='revenue', color='FuelType', title=f'Top {top_n} Models by Revenue within Each Fuel Type')
    fig_top_by_fuel.update_xaxes(tickangle=45)
    fig_top_by_fuel.update_traces(hovertemplate=(('' if currency_symbol == 'None' else currency_symbol) + ' %{y:,.2f}'))
    fig_top_by_fuel.update_yaxes(tickprefix=currency_symbol, tickformat=',.2f')
    st.plotly_chart(fig_top_by_fuel, use_container_width=True)

    st.download_button(label="Download Model-Fuel Revenue CSV", data=model_fuel_rev.to_csv(index=False), file_name="model_fuel_revenue.csv", mime="text/csv")
    st.dataframe(model_fuel_rev.sort_values(['FuelType','revenue'], ascending=[True, False]))
