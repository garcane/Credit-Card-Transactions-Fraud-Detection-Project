import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from risk_model import FraudRiskModel
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Fraud Intelligence Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# DARK THEME CSS INJECTION
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    .stMetric {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    
    .stMetric > div:first-child {
        color: #94a3b8 !important;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stMetric .css-1xarl3l {
        color: #f8fafc !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e293b;
        padding: 8px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #94a3b8;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }
    
    .stSidebar {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    
    div[data-testid="stMarkdownContainer"] > p {
        color: #cbd5e1;
        line-height: 1.6;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }
    
    .risk-low { color: #4ade80; }
    .risk-medium { color: #fbbf24; }
    .risk-high { color: #fb923c; }
    .risk-critical { color: #f87171; }
    
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.5);
    }
    
    div[data-testid="stExpander"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CACHED DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv('fraudTrain.csv', low_memory=False)
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['year_month'] = df['trans_date_trans_time'].dt.to_period('M').astype(str)
    return df

@st.cache_resource
def get_model():
    try:
        return FraudRiskModel.load('fraud_model.pkl')
    except Exception:
        with st.spinner('Training initial model on sample...'):
            model = FraudRiskModel()
            sample = load_data().sample(min(50000, len(load_data())), random_state=42)
            model.fit(sample)
            model.save('fraud_model.pkl')
        return model

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div style="text-align: center; padding: 30px 0 20px;">
    <h1 style="font-size: 3.2rem; margin-bottom: 5px; background: linear-gradient(90deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        🛡️ Fraud Intelligence
    </h1>
    <p style="color: #64748b; font-size: 1.1rem; margin-top: 0;">
        From exploratory analysis to operational decision support
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================
try:
    df = load_data()
    model = get_model()
    data_ready = True
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.info("Ensure `fraudTrain.csv` is in the working directory.")
    data_ready = False

if not data_ready:
    st.stop()

# ============================================================
# SIDEBAR FILTERS
# ============================================================
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    
    date_range = st.date_input(
        "Date Range",
        value=[df['trans_date_trans_time'].min().date(),
               df['trans_date_trans_time'].max().date()]
    )
    
    selected_states = st.multiselect(
        "States",
        options=sorted(df['state'].unique()),
        default=sorted(df['state'].unique())[:10]
    )
    
    amt_min, amt_max = st.slider(
        "Amount Range ($)",
        float(df['amt'].min()), 
        float(df['amt'].quantile(0.995)),
        (0.0, 1000.0)
    )
    
    cats = st.multiselect(
        "Categories",
        options=sorted(df['category'].unique()),
        default=sorted(df['category'].unique())
    )
    
    st.markdown("---")
    st.markdown("### 📊 Dataset Snapshot")
    st.metric("Total Records", f"{len(df):,}")
    st.metric("Baseline Fraud Rate", f"{df['is_fraud'].mean()*100:.2f}%")
    st.metric("Unique Merchants", f"{df['merchant'].nunique():,}")

# ============================================================
# FILTERING LOGIC
# ============================================================
mask = (
    (df['trans_date_trans_time'].dt.date >= date_range[0]) &
    (df['trans_date_trans_time'].dt.date <= date_range[1]) &
    (df['state'].isin(selected_states)) &
    (df['amt'] >= amt_min) &
    (df['amt'] <= amt_max) &
    (df['category'].isin(cats))
)
d = df[mask]

# ============================================================
# KPI CARDS
# ============================================================
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    fraud_rate = d['is_fraud'].mean() * 100
    delta = fraud_rate - (df['is_fraud'].mean() * 100)
    st.metric("Fraud Rate", f"{fraud_rate:.2f}%", f"{delta:+.2f}% vs baseline")

with kpi2:
    avg_fraud = d[d['is_fraud']==1]['amt'].mean() if d['is_fraud'].any() else 0
    st.metric("Avg Fraud Loss", f"${avg_fraud:,.0f}")

with kpi3:
    total_loss = d[d['is_fraud']==1]['amt'].sum()
    st.metric("Total Exposure", f"${total_loss:,.0f}")

with kpi4:
    block_potential = len(d[(d['is_fraud']==1) & (d['amt'] > 500)])
    st.metric("High-Value Flags", f"{block_potential:,}")

st.markdown("---")

# ============================================================
# TABS
# ============================================================
tab_geo, tab_temporal, tab_demo, tab_scorer = st.tabs([
    "🗺️ Geographic Risk", 
    "📈 Category & Temporal", 
    "👤 Demographics", 
    "🎯 Live Risk Scorer"
])

# ------------------------------------------------------------
# TAB 1: GEOGRAPHIC
# ------------------------------------------------------------
with tab_geo:
    geo_col, state_col = st.columns([3, 2])
    
    with geo_col:
        st.markdown("### Fraud Rate by State")
        state_stats = d.groupby('state').agg({
            'is_fraud': ['count', 'sum', 'mean']
        }).reset_index()
        state_stats.columns = ['state', 'volume', 'frauds', 'rate']
        state_stats['rate'] *= 100
        
        fig = px.choropleth(
            state_stats,
            locations='state',
            locationmode='USA-states',
            color='rate',
            scope="usa",
            color_continuous_scale='RdYlBu_r',
            range_color=[0, max(state_stats['rate'].quantile(0.90), 1.5)],
            template='plotly_dark',
            height=650,
            title=''
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            geo=dict(bgcolor='rgba(0,0,0,0)', lakecolor='#0b0f19'),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with state_col:
        st.markdown("### 🚨 Highest Risk States")
        top_states = state_stats.nlargest(12, 'rate')
        fig = px.bar(
            top_states,
            x='rate',
            y='state',
            orientation='h',
            color='rate',
            color_continuous_scale='Reds',
            template='plotly_dark',
            text=top_states['rate'].round(2),
            labels={'rate': 'Fraud %', 'state': ''}
        )
        fig.update_traces(textposition='outside', textfont_color='white')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(autorange="reversed", tickfont_size=11),
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📍 Risk by Volume")
        fig2 = px.scatter(
            state_stats,
            x='volume',
            y='rate',
            size='frauds',
            color='rate',
            hover_name='state',
            template='plotly_dark',
            labels={'volume': 'Transaction Volume', 'rate': 'Fraud Rate (%)'},
            color_continuous_scale='Plasma'
        )
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------
# TAB 2: CATEGORY & TEMPORAL
# ------------------------------------------------------------
with tab_temporal:
    cat_col, time_col = st.columns(2)
    
    with cat_col:
        st.markdown("### Merchant Category Risk Ranking")
        cat_stats = d.groupby('category').agg(
            volume=('is_fraud', 'count'),
            fraud_rate=('is_fraud', 'mean')
        ).reset_index()
        cat_stats = cat_stats[cat_stats['volume'] >= 50]
        cat_stats['fraud_rate'] *= 100
        cat_stats = cat_stats.sort_values('fraud_rate', ascending=True)
        
        fig = px.bar(
            cat_stats,
            x='fraud_rate',
            y='category',
            orientation='h',
            color='fraud_rate',
            color_continuous_scale='Tealrose',
            template='plotly_dark',
            text=cat_stats['fraud_rate'].round(2),
            labels={'fraud_rate': 'Fraud Rate (%)', 'category': ''}
        )
        fig.update_traces(textposition='outside', textfont_color='white')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(tickfont_size=11),
            margin=dict(l=10, r=30, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with time_col:
        st.markdown("### ⏰ Fraud Rate by Hour")
        hourly = d.groupby('hour')['is_fraud'].mean() * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hourly.index,
            y=hourly.values,
            mode='lines+markers',
            line=dict(color='#60a5fa', width=3),
            marker=dict(size=8, color=hourly.values, colorscale='Reds', showscale=False),
            fill='tozeroy',
            fillcolor='rgba(96, 165, 250, 0.1)'
        ))
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Hour of Day', tickmode='linear', dtick=2, gridcolor='#1e293b'),
            yaxis=dict(title='Fraud Rate (%)', gridcolor='#1e293b'),
            margin=dict(l=10, r=10, t=10, b=10),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📅 Monthly Trend")
        monthly = d.groupby('year_month')['is_fraud'].mean() * 100
        fig2 = px.area(
            x=monthly.index,
            y=monthly.values,
            template='plotly_dark',
            labels={'x': 'Month', 'y': 'Fraud Rate (%)'},
            color_discrete_sequence=['#f87171']
        )
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='#1e293b'),
            yaxis=dict(gridcolor='#1e293b'),
            margin=dict(l=10, r=10, t=10, b=10),
            height=350
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Full-width amount analysis
    st.markdown("### 💰 Amount Risk Profile")
    amt_col1, amt_col2 = st.columns(2)
    
    with amt_col1:
        # Violin plot
        sample = d[d['amt'] <= d['amt'].quantile(0.98)]
        fig = px.violin(
            sample,
            y='amt',
            color='is_fraud',
            box=True,
            template='plotly_dark',
            labels={'amt': 'Amount ($)', 'is_fraud': 'Status'},
            color_discrete_map={0: '#3b82f6', 1: '#ef4444'},
            category_orders={'is_fraud': [0, 1]}
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend_title_text='Fraud'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with amt_col2:
        # Binned amount fraud rate
        d['amt_bin'] = pd.qcut(d['amt'], q=10, duplicates='drop')
        amt_bin_stats = d.groupby('amt_bin')['is_fraud'].mean() * 100
        
        fig = px.bar(
            x=[str(b) for b in amt_bin_stats.index],
            y=amt_bin_stats.values,
            template='plotly_dark',
            labels={'x': 'Amount Decile', 'y': 'Fraud Rate (%)'},
            color=amt_bin_stats.values,
            color_continuous_scale='Reds'
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(tickangle=45, tickfont_size=10),
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# TAB 3: DEMOGRAPHICS
# ------------------------------------------------------------
with tab_demo:
    demo_col1, demo_col2 = st.columns(2)
    
    with demo_col1:
        st.markdown("### 🎂 Age Group Vulnerability")
        bins = [0, 25, 35, 45, 55, 65, 100]
        labels = ['<25', '25-34', '35-44', '45-54', '55-64', '65+']
        d['age_group'] = pd.cut(d['age'], bins=bins, labels=labels)
        age_stats = d.groupby('age_group')['is_fraud'].agg(['count', 'mean']).reset_index()
        age_stats['mean'] *= 100
        
        fig = px.bar(
            age_stats,
            x='age_group',
            y='mean',
            color='mean',
            color_continuous_scale='Blues',
            template='plotly_dark',
            text=age_stats['mean'].round(2),
            labels={'age_group': 'Age Group', 'mean': 'Fraud Rate (%)'}
        )
        fig.update_traces(textposition='outside', textfont_color='white')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with demo_col2:
        st.markdown("### 🏙️ City Population vs Fraud")
        d['pop_tier'] = pd.qcut(d['city_pop'], q=5, labels=['Rural', 'Small', 'Mid', 'Large', 'Metro'], duplicates='drop')
        pop_stats = d.groupby('pop_tier')['is_fraud'].mean() * 100
        
        fig = px.bar(
            x=pop_stats.index.astype(str),
            y=pop_stats.values,
            template='plotly_dark',
            labels={'x': 'Population Tier', 'y': 'Fraud Rate (%)'},
            color=pop_stats.values,
            color_continuous_scale='Purples',
            text=pop_stats.values.round(2)
        )
        fig.update_traces(textposition='outside', textfont_color='white')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance from model
    st.markdown("### 🧠 Model Feature Importance")
    imp = model.importance_df.head(12)
    fig = px.bar(
        imp,
        x='importance',
        y='feature',
        orientation='h',
        template='plotly_dark',
        color='importance',
        color_continuous_scale='Cividis',
        labels={'importance': 'Importance', 'feature': ''}
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(autorange="reversed", tickfont_size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# TAB 4: LIVE RISK SCORER
# ------------------------------------------------------------
with tab_scorer:
    st.markdown("""
    <div style="background: linear-gradient(145deg, #1e293b, #0f172a); padding: 25px; border-radius: 16px; border: 1px solid #334155; margin-bottom: 20px;">
        <h3 style="margin-top: 0;">🎯 Real-Time Transaction Risk Scorer</h3>
        <p style="color: #94a3b8; margin-bottom: 0;">
            Simulate a production scoring engine. Enter transaction parameters to generate an instant risk assessment with model-driven recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    in_col1, in_col2 = st.columns(2)
    
    with in_col1:
        st.markdown("#### Transaction Details")
        in_amt = st.number_input("Amount ($)", 0.0, 50000.0, 250.0, step=10.0)
        in_cat = st.selectbox("Category", options=sorted(df['category'].unique()), index=0)
        in_hour = st.slider("Hour", 0, 23, 14)
        in_state = st.selectbox("State", options=sorted(df['state'].unique()), index=0)
        in_gender = st.radio("Gender", options=['M', 'F'], horizontal=True)
    
    with in_col2:
        st.markdown("#### Customer Profile")
        in_age = st.number_input("Age", 18, 100, 35)
        in_city_pop = st.number_input("City Population", 100, 5000000, 300000, step=1000)
        in_lat = st.number_input("Customer Lat", value=36.0, format="%.4f")
        in_long = st.number_input("Customer Long", value=-115.0, format="%.4f")
        in_merch_lat = st.number_input("Merchant Lat", value=36.0, format="%.4f")
        in_merch_long = st.number_input("Merchant Long", value=-115.0, format="%.4f")
        
        # Distance preview
        dist = FraudRiskModel.haversine(in_lat, in_long, in_merch_lat, in_merch_long)
        st.caption(f"📍 Distance: **{dist:.1f} km** from home")
    
    if st.button("🔮 Assess Transaction Risk", type="primary", use_container_width=True):
        # Build input dataframe
        now = datetime.now()
        dob = now - timedelta(days=int(in_age * 365.25))
        
        input_df = pd.DataFrame([{
            'trans_date_trans_time': now,
            'category': in_cat,
            'amt': in_amt,
            'state': in_state,
            'lat': in_lat,
            'long': in_long,
            'city_pop': in_city_pop,
            'dob': dob,
            'merch_lat': in_merch_lat,
            'merch_long': in_merch_long,
            'job': 'unknown',
            'gender': in_gender,
            'is_fraud': 0  # placeholder
        }])
        
        try:
            result = model.predict_risk(input_df)
            prob = result['fraud_probability'].iloc[0]
            tier = result['risk_tier'].iloc[0]
            rec = result['recommendation'].iloc[0]
            
            # Color maps
            tier_colors = {
                'Low': '#4ade80',
                'Medium': '#fbbf24', 
                'High': '#fb923c',
                'Critical': '#f87171'
            }
            tier_color = tier_colors.get(str(tier), '#94a3b8')
            
            action_colors = {
                '⛔ Block Transaction': '#f87171',
                '🔍 Manual Review Required': '#fb923c',
                '✅ Approve with Monitoring': '#fbbf24',
                '✅ Auto-Approve': '#4ade80'
            }
            action_color = action_colors.get(rec, '#94a3b8')
            
            st.markdown("---")
            
            # Result cards
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f"""
                <div style="text-align: center; padding: 30px 20px; background: linear-gradient(145deg, #1e293b, #0f172a); border-radius: 16px; border: 1px solid #334155;">
                    <p style="color: #94a3b8; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;">Probability</p>
                    <h2 style="margin: 0; font-size: 3rem; color: {tier_color};">{prob:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with r2:
                st.markdown(f"""
                <div style="text-align: center; padding: 30px 20px; background: linear-gradient(145deg, #1e293b, #0f172a); border-radius: 16px; border: 1px solid #334155;">
                    <p style="color: #94a3b8; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;">Risk Tier</p>
                    <h2 style="margin: 0; font-size: 3rem; color: {tier_color};">{tier}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with r3:
                st.markdown(f"""
                <div style="text-align: center; padding: 30px 20px; background: linear-gradient(145deg, #1e293b, #0f172a); border-radius: 16px; border: 1px solid #334155;">
                    <p style="color: #94a3b8; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;">Action</p>
                    <h2 style="margin: 0; font-size: 1.8rem; color: {action_color};">{rec}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Gauge
            st.markdown("<br>", unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': "%", 'font': {'size': 48, 'color': 'white'}},
                title={'text': "Risk Score", 'font': {'size': 20, 'color': '#94a3b8'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#334155"},
                    'bar': {'color': tier_color, 'thickness': 0.75},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [0, 30], 'color': '#0f281e'},
                        {'range': [30, 60], 'color': '#2a1f0f'},
                        {'range': [60, 85], 'color': '#2a1515'},
                        {'range': [85, 100], 'color': '#3f1010'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 3},
                        'thickness': 0.8,
                        'value': prob * 100
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': "white", 'family': "Inter"},
                height=350,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Local explanation
            st.markdown("### 🔍 Top Risk Drivers")
            expl = model.explain_local(input_df)
            expl['contribution_abs'] = expl['contribution'].abs()
            expl = expl.sort_values('contribution_abs', ascending=False)
            
            for _, row in expl.iterrows():
                direction = "increases" if row['contribution'] > 0 else "decreases"
                st.markdown(f"""
                <div style="padding: 12px 16px; background-color: #1e293b; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid {tier_color};">
                    <span style="color: #e2e8f0; font-weight: 600;">{row['feature']}</span> = <code>{row['value']:.3f}</code>
                    <br><span style="color: #94a3b8; font-size: 0.9rem;">This {direction} risk (contribution: {row['contribution']:.4f})</span>
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Scoring failed: {e}")
            st.info("Tip: If this is the first run, the model trains on a sample. Ensure your input matches the training schema.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #475569; padding: 20px;">
    <p style="margin: 0; font-size: 0.85rem;">
        Fraud Intelligence Dashboard • Built with Streamlit & Plotly • Dark Mode
    </p>
</div>
""", unsafe_allow_html=True)
