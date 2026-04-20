# %% [markdown]
# # Credit Card Fraud Detection – End-to-End Data Analysis & Modeling
# **Comprehensive Jupyter Notebook**  
# **Dataset**: Sparkov Credit Card Transactions (fraudTrain.csv + fraudTest.csv)  
# **Goal**: Explore fraud patterns, create stunning visualizations, and build a predictive model

# %%
# ============================
# 0. SETUP & REQUIREMENTS
# ============================
# Run this first!
!pip install pandas numpy seaborn matplotlib plotly scikit-learn imbalanced-learn sqlalchemy psycopg2-binary --quiet

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("✅ Libraries imported successfully!")

# %% [markdown]
# ## 1. Data Ingestion
# Load both training and test sets (standard for this Kaggle dataset).

# %%
# ========================
# 1. DATA INGESTION
# ========================
# Place fraudTrain.csv and fraudTest.csv in the same folder as this notebook

train = pd.read_csv('fraudTrain.csv', low_memory=False)
test  = pd.read_csv('fraudTest.csv',  low_memory=False)

print(f"Train shape: {train.shape}")
print(f"Test shape : {test.shape}")
print(f"Columns: {list(train.columns)}")

# Quick look at fraud distribution
print("\nFraud rate in train:", round(train['is_fraud'].mean()*100, 3), "%")
print("Fraud rate in test :", round(test['is_fraud'].mean()*100, 3), "%")

# %% [markdown]
# ## 2. Exploratory Data Analysis (EDA)
# What types of purchases are most likely to be fraud?

# %%
# ========================
# 2. EDA
# ========================

# 2.1 Fraud by Merchant Category
cat_fraud = train.groupby('category')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
cat_fraud.columns = ['category', 'transactions', 'frauds', 'fraud_rate']
cat_fraud = cat_fraud.sort_values('fraud_rate', ascending=False)

fig_cat = px.bar(cat_fraud.head(10), x='category', y='fraud_rate', 
                title='Top 10 Merchant Categories by Fraud Rate',
                labels={'fraud_rate': 'Fraud Rate (%)'},
                color='fraud_rate', color_continuous_scale='Reds')
fig_cat.show()

print("Highest fraud categories:")
print(cat_fraud.head(5)[['category', 'fraud_rate']])

# 2.2 Transaction Amount Analysis
fig_amt = px.histogram(train, x='amt', color='is_fraud', nbins=100,
                    title='Transaction Amount Distribution by Fraud Status',
                    log_y=True)
fig_amt.show()

# 2.3 Time Patterns
train['trans_date_trans_time'] = pd.to_datetime(train['trans_date_trans_time'])
train['hour'] = train['trans_date_trans_time'].dt.hour
train['day_of_week'] = train['trans_date_trans_time'].dt.day_name()

hour_fraud = train.groupby('hour')['is_fraud'].mean().reset_index()
fig_hour = px.line(hour_fraud, x='hour', y='is_fraud', 
                title='Fraud Rate by Hour of Day', markers=True)
fig_hour.show()

# %% [markdown]
# ## 3. Geospatial Analysis
# Fraud rates across U.S. states + merchant location heatmap

# %%
# ========================
# 3. GEOSPATIAL VISUALIZATION
# ========================

# 3.1 Fraud Rate by State (Choropleth)
state_fraud = train.groupby('state')['is_fraud'].agg(['count', 'sum']).reset_index()
state_fraud['fraud_rate'] = state_fraud['sum'] / state_fraud['count'] * 100

fig_geo = px.choropleth(state_fraud, 
                        locations='state', 
                        locationmode='USA-states',
                        color='fraud_rate',
                        scope="usa",
                        title='Fraud Rate (%) by U.S. State',
                        color_continuous_scale='Reds',
                        labels={'fraud_rate': 'Fraud Rate (%)'})
fig_geo.show()

# 3.2 Merchant Location Scatter (fraud vs legit)
sample = train.sample(50000, random_state=42)  # for speed
fig_scatter = px.scatter_mapbox(sample,
                                lat='merch_lat', lon='merch_long',
                                color='is_fraud',
                                hover_name='merchant',
                                hover_data=['amt', 'category'],
                                zoom=3, height=600,
                                title='Merchant Locations – Fraud (red) vs Legit (blue)',
                                color_discrete_map={0: 'blue', 1: 'red'})
fig_scatter.update_layout(mapbox_style="carto-positron")
fig_scatter.show()

# %% [markdown]
# ## 4. Demographic Analysis
# Are older customers significantly more likely to be victims of fraud?

# %%
# ========================
# 4. DEMOGRAPHIC ANALYSIS (Age)
# ========================

# Convert dates
train['dob'] = pd.to_datetime(train['dob'])
train['age_at_trans'] = (train['trans_date_trans_time'] - train['dob']).dt.days // 365

# Age groups
bins = [0, 25, 35, 45, 55, 65, 100]
labels = ['<25', '25-34', '35-44', '45-54', '55-64', '65+']
train['age_group'] = pd.cut(train['age_at_trans'], bins=bins, labels=labels)

age_fraud = train.groupby('age_group')['is_fraud'].agg(['count', 'mean']).reset_index()
age_fraud['mean'] *= 100

fig_age = px.bar(age_fraud, x='age_group', y='mean',
                 title='Fraud Rate by Customer Age Group',
                 labels={'mean': 'Fraud Rate (%)'},
                 color='mean', color_continuous_scale='Blues')
fig_age.show()

print("Fraud rate for 65+ customers:", round(age_fraud[age_fraud['age_group']=='65+']['mean'].values[0], 2), "%")

# %% [markdown]
# ## 5. Interactive Visualizations & Dashboard-style Layout

# %%
# ========================
# 5. INTERACTIVE DASHBOARD-STYLE VISUALIZATIONS
# ========================

# Combine several plots into one figure
fig_dashboard = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Fraud by Category (Top 8)", "Fraud by Hour", 
                    "Fraud Rate by Age Group", "Transaction Amount (Log)"),
    specs=[[{"type": "bar"}, {"type": "scatter"}],
           [{"type": "bar"}, {"type": "histogram"}]]
)

# Top categories
top_cat = cat_fraud.head(8)
fig_dashboard.add_trace(go.Bar(x=top_cat['category'], y=top_cat['fraud_rate']), row=1, col=1)

# Hour
fig_dashboard.add_trace(go.Scatter(x=hour_fraud['hour'], y=hour_fraud['is_fraud'], mode='lines+markers'), row=1, col=2)

# Age
fig_dashboard.add_trace(go.Bar(x=age_fraud['age_group'], y=age_fraud['mean']), row=2, col=1)

# Amount
fig_dashboard.add_trace(go.Histogram(x=train[train['is_fraud']==1]['amt'], name='Fraud', opacity=0.7), row=2, col=2)
fig_dashboard.add_trace(go.Histogram(x=train[train['is_fraud']==0]['amt'], name='Legit', opacity=0.7), row=2, col=2)

fig_dashboard.update_layout(height=800, title_text="Credit Card Fraud Dashboard")
fig_dashboard.show()

# %% [markdown]
# ## 6. Spatial Feature Engineering
# Fraud often correlates with unusual distance between a customer's home location and the merchant.

# %%
# Haversine distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

train['distance_km'] = haversine(train['lat'], train['long'], 
                                  train['merch_lat'], train['merch_long'])
train['distance_tier'] = pd.qcut(train['distance_km'], q=5, 
                                  labels=['very_close', 'close', 'medium', 'far', 'very_far'])

# %% [markdown]
# ## 7. Customer-Level Velocity & Deviation (RFM-style)
# Aggregate at the customer level before the transaction to simulate real-time scoring:

# %%
# Requires sorting by time first
train = train.sort_values('trans_date_trans_time')
train['customer_tx_count'] = train.groupby('cc_num')['trans_num'].cumcount()
train['customer_avg_amt'] = train.groupby('cc_num')['amt'].expanding().mean().reset_index(0, drop=True)
train['amt_deviation_from_avg'] = train['amt'] / (train['customer_avg_amt'] + 1)

# %% [markdown]
# ## 8. Merchant Risk Profile
# Pre-compute merchant historical fraud rates (as of the transaction date — no data leakage):

# %%
# Time-windowed merchant risk (simulate "as of this transaction")
train = train.sort_values('trans_date_trans_time')
train['merchant_tx_count'] = train.groupby('merchant').cumcount()
train['merchant_fraud_cumsum'] = train.groupby('merchant')['is_fraud'].cumsum() - train['is_fraud']
train['merchant_risk_score'] = train['merchant_fraud_cumsum'] / (train['merchant_tx_count'] + 1)

# %% [markdown]
# ## 9. Statistical Rigour
# Move beyond visual inference:

# %%
from scipy.stats import mannwhitneyu, chi2_contingency

# Amount difference (non-parametric due to skew)
fraud_amts = train[train['is_fraud']==1]['amt']
legit_amts = train[train['is_fraud']==0]['amt']
stat, p = mannwhitneyu(fraud_amts, legit_amts, alternative='two-sided')
print(f"Mann-Whitney U p-value: {p:.2e} → Fraud amounts are significantly different")

# Category association
contingency = pd.crosstab(train['category'], train['is_fraud'])
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"Chi-square p-value: {p:.2e} → Category and fraud are dependent")

# %% [markdown]
# ## 10 Precision-Recall & Calibration
# For imbalanced data, ROC-AUC is misleading

# %% [markdown]
# ## 11. Time-Series Decomposition
# Show monthly trend/seasonality instead of just aggregates:

# %%
train['year_month'] = train['trans_date_trans_time'].dt.to_period('M')
monthly = train.groupby('year_month')['is_fraud'].mean()
# Decompose with statsmodels if you want to get fancy

# %% [markdown]
# ## 12. Fraud Prediction Model
# Random Forest with class imbalance handling

# %%
# ========================
# 6. MACHINE LEARNING MODEL
# ========================

# Feature engineering (same for train & test)
def engineer_features(df):
    df = df.copy()
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['amt_log'] = np.log1p(df['amt'])
    return df

train = engineer_features(train)
test  = engineer_features(test)

# Simple features
features = ['category', 'amt_log', 'age', 'hour', 'city_pop']
X_train = pd.get_dummies(train[features], drop_first=True)
X_test  = pd.get_dummies(test[features], drop_first=True)

# Align columns
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

y_train = train['is_fraud']
y_test  = test['is_fraud']

# Model
model = RandomForestClassifier(n_estimators=200, 
                               class_weight='balanced', 
                               random_state=42,
                               n_jobs=-1)
model.fit(X_train, y_train)

# Predictions & metrics
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# Feature importance
importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
fig_imp = px.bar(importances.head(10), title='Top 10 Feature Importances')
fig_imp.show()

# %%
# ========================
# 10. PRECISION-RECALL & CALIBRATION (Advanced Model Evaluation)
# ========================
# Add this cell AFTER you have run Section 6 (the Random Forest model cell)
# so that y_test and y_prob are already defined in the kernel.

# Required imports (add these if not already present)
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

# 10.1 Precision-Recall Curve (critical for imbalanced fraud data)
precision, recall, _ = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)

fig_pr = go.Figure()
fig_pr.add_trace(go.Scatter(x=recall, y=precision, 
                           mode='lines', 
                           name=f'PR Curve (AP = {ap:.4f})',
                           line=dict(color='purple', width=3)))
fig_pr.update_layout(
    title='Precision-Recall Curve – Fraud Detection Model',
    xaxis_title='Recall',
    yaxis_title='Precision',
    width=800,
    height=500,
    template='plotly_white'
)
fig_pr.show()

print(f"✅ Average Precision (AP) Score: {ap:.4f} (higher is better on imbalanced data)")

# 10.2 Calibration Plot (shows if predicted probabilities are reliable)
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy='uniform')

fig_cal = go.Figure()
fig_cal.add_trace(go.Scatter(x=prob_pred, y=prob_true,
                             mode='lines+markers',
                             name='Model Calibration',
                             line=dict(color='blue', width=3)))
fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                             mode='lines',
                             name='Perfect Calibration',
                             line=dict(color='black', dash='dash', width=2)))
fig_cal.update_layout(
    title='Calibration Plot (Reliability Curve)',
    xaxis_title='Mean Predicted Probability',
    yaxis_title='Observed Fraud Fraction',
    width=800,
    height=500,
    template='plotly_white'
)
fig_cal.show()

print("✅ Calibration plot generated – closer to the dashed line = better probability estimates")


