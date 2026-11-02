# Homework 6 - Decision Trees and Ensemble Learning (ML Zoomcamp 2026)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb

DATA_PATH = "/home/abdoulaye/Documents/Orange Digital center/Machine Learning Zoomcamp 2026/car_fuel_efficiency_2026.csv"

# ---------- Data preparation ----------
df = pd.read_csv(DATA_PATH)
df = df.fillna(0)                                   # fill missing values with zeros

# exact split calls required by the homework
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

y_train = df_train['fuel_efficiency_mpg'].values
y_val = df_val['fuel_efficiency_mpg'].values

features = [c for c in df.columns if c != 'fuel_efficiency_mpg']
dv = DictVectorizer(sparse=True)
X_train = dv.fit_transform(df_train[features].to_dict(orient='records'))
X_val = dv.transform(df_val[features].to_dict(orient='records'))
feature_names = list(dv.get_feature_names_out())

def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

# ---------- Q1: feature used by a tree with max_depth=1 ----------
dt = DecisionTreeRegressor(max_depth=1)
dt.fit(X_train, y_train)
print("Q1 - splitting feature:", feature_names[dt.tree_.feature[0]])

# ---------- Q2: random forest with 10 trees ----------
rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)
print("Q2 - validation RMSE:", round(rmse(y_val, rf.predict(X_val)), 4))

# ---------- Q3: number of estimators ----------
for n in [10, 50, 100, 150]:
    rf = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
    rf.fit(X_train, y_train)
    print("Q3 - n_estimators =", n, "| RMSE:", round(rmse(y_val, rf.predict(X_val)), 3))

# ---------- Q4: best max_depth (mean RMSE over n_estimators) ----------
for depth in [10, 15, 20, 25]:
    scores = []
    for n in [10, 50, 100, 150]:
        rf = RandomForestRegressor(n_estimators=n, max_depth=depth, random_state=1, n_jobs=-1)
        rf.fit(X_train, y_train)
        scores.append(rmse(y_val, rf.predict(X_val)))
    print("Q4 - max_depth =", depth, "| mean RMSE:", round(float(np.mean(scores)), 4))

# ---------- Q5: feature importance ----------
rf = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
print("Q5 - feature importances:")
print(importances.head(6))

# ---------- Q6: XGBoost, eta 0.3 vs 0.1 ----------
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

for eta in [0.3, 0.1]:
    xgb_params = {
        'eta': eta,
        'max_depth': 6,
        'min_child_weight': 1,
        'objective': 'reg:squarederror',
        'nthread': 8,
        'seed': 1,
        'verbosity': 0,
    }
    model = xgb.train(xgb_params, dtrain, num_boost_round=100)
    print("Q6 - eta =", eta, "| RMSE:", round(rmse(y_val, model.predict(dval)), 4))