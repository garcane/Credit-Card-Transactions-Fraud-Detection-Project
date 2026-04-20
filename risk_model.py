import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import joblib
import warnings
warnings.filterwarnings('ignore')


class FraudRiskModel:
    """
    Production-grade fraud risk model with built-in feature engineering,
    risk-tier classification, and explainability hooks.
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.num_cols = ['amt', 'amt_log', 'distance_km', 'distance_log', 
                         'city_pop_log', 'age', 'amt_deviation']
        self.cat_cols = ['category', 'state', 'job', 'gender', 'day_period']
        self.feature_names = None
        self.is_fitted = False
        
    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        """Calculate haversine distance in kilometers."""
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
        return 2 * R * np.arcsin(np.sqrt(a))
    
    def _create_time_features(self, df):
        """Extract cyclical and categorical time features."""
        df['hour'] = df['trans_date_trans_time'].dt.hour
        df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
        df['month'] = df['trans_date_trans_time'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        
        # Cyclical encoding for hour (fraud patterns are cyclical)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day period bins
        bins = [0, 6, 12, 18, 24]
        labels = ['night', 'morning', 'afternoon', 'evening']
        df['day_period'] = pd.cut(df['hour'], bins=bins, labels=labels, right=False)
        return df
    
    def engineer_features(self, df, fit=True):
        """Full feature engineering pipeline."""
        data = df.copy()
        
        # --- Parse dates ---
        data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
        data['dob'] = pd.to_datetime(data['dob'])
        
        # --- Time features ---
        data = self._create_time_features(data)
        
        # --- Demographics ---
        data['age'] = (data['trans_date_trans_time'] - data['dob']).dt.days // 365
        data['age_group'] = pd.cut(
            data['age'], 
            bins=[0, 25, 35, 45, 55, 65, 100], 
            labels=['<25', '25-34', '35-44', '45-54', '55-64', '65+']
        )
        
        # --- Amount features ---
        data['amt_log'] = np.log1p(data['amt'])
        data['amt_zscore'] = (data['amt'] - data['amt'].mean()) / (data['amt'].std() + 1e-9)
        
        # --- Spatial features ---
        data['distance_km'] = self.haversine(
            data['lat'], data['long'],
            data['merch_lat'], data['merch_long']
        )
        data['distance_log'] = np.log1p(data['distance_km'])
        data['distance_tier'] = pd.qcut(
            data['distance_km'], q=5, 
            labels=['very_close', 'close', 'medium', 'far', 'very_far'],
            duplicates='drop'
        )
        
        # --- Population ---
        data['city_pop_log'] = np.log1p(data['city_pop'])
        
        # --- Categorical encoding ---
        for col in self.cat_cols:
            if col not in data.columns:
                continue
            if fit:
                le = LabelEncoder()
                # Handle unseen by adding 'unknown'
                data[col] = data[col].astype(str).fillna('unknown')
                le.fit(list(data[col].unique()) + ['unknown'])
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le is None:
                    continue
                data[col] = data[col].astype(str).fillna('unknown')
                # Map unseen to 'unknown'
                mask = ~data[col].isin(le.classes_)
                data.loc[mask, col] = 'unknown'
            
            data[col] = le.transform(data[col])
        
        # --- Feature selection ---
        base_features = [
            'category', 'amt', 'amt_log', 'amt_zscore',
            'hour', 'hour_sin', 'hour_cos', 'day_of_week', 
            'month', 'is_weekend', 'is_night',
            'age', 'city_pop_log', 'distance_km', 'distance_log',
            'state', 'job', 'gender'
        ]
        
        self.feature_names = [c for c in base_features if c in data.columns]
        X = data[self.feature_names].copy()
        return X, data  # return engineered df too for metadata
    
    def fit(self, df, target_col='is_fraud'):
        """Train model with full evaluation metrics."""
        y = df[target_col]
        X, _ = self.engineer_features(df, fit=True)
        
        # Scale numericals
        num_present = [c for c in self.num_cols if c in X.columns]
        if num_present:
            X[num_present] = self.scaler.fit_transform(X[num_present])
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        # --- Training diagnostics ---
        y_prob = self.model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_prob)
        ap = average_precision_score(y, y_prob)
        print(f"[Train] AUC-ROC: {auc:.4f} | Avg Precision: {ap:.4f}")
        
        # Store feature importance
        self.importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self
    
    def predict_proba(self, df):
        """Return fraud probability vector."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        X, meta = self.engineer_features(df, fit=False)
        
        num_present = [c for c in self.num_cols if c in X.columns]
        if num_present:
            X[num_present] = self.scaler.transform(X[num_present])
        
        return self.model.predict_proba(X)[:, 1], meta
    
    def predict_risk(self, df):
        """
        Return comprehensive risk assessment including:
        - Raw probability
        - Risk tier (Low/Medium/High/Critical)
        - Recommended action
        - Top contributing feature (approximate via permutation)
        """
        prob, meta = self.predict_proba(df)
        
        # Dynamic thresholds (can be tuned via cost-benefit)
        tiers = pd.cut(
            prob, 
            bins=[0, 0.15, 0.40, 0.75, 1.0],
            labels=['Low', 'Medium', 'High', 'Critical'],
            include_lowest=True
        )
        
        recommendations = np.select(
            [
                prob > 0.75,
                prob > 0.40,
                prob > 0.15
            ],
            [
                '⛔ Block Transaction',
                '🔍 Manual Review Required',
                '✅ Approve with Monitoring'
            ],
            default='✅ Auto-Approve'
        )
        
        return pd.DataFrame({
            'fraud_probability': prob,
            'risk_tier': tiers,
            'recommendation': recommendations,
            'distance_km': meta.get('distance_km', np.nan),
            'hour': meta.get('hour', np.nan)
        })
    
    def explain_local(self, df_row):
        """
        Simple local explanation using feature value * importance contribution.
        Returns top 3 drivers for the prediction.
        """
        prob, _ = self.predict_proba(df_row)
        X, _ = self.engineer_features(df_row, fit=False)
        num_present = [c for c in self.num_cols if c in X.columns]
        if num_present:
            X[num_present] = self.scaler.transform(X[num_present])
        
        # Approximate contribution
        contributions = X.iloc[0].values * self.model.feature_importances_
        expl = pd.DataFrame({
            'feature': self.feature_names,
            'value': X.iloc[0].values,
            'contribution': contributions
        }).sort_values('contribution', key=abs, ascending=False).head(3)
        return expl
    
    def save(self, path='fraud_model.pkl'):
        joblib.dump(self, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path='fraud_model.pkl'):
        return joblib.load(path)