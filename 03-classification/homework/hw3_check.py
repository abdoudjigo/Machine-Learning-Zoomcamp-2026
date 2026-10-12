# Quick check for Q5 and Q6 of Homework 3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("course_lead_scoring_2026.csv")

categorical = ['lead_source', 'industry', 'employment_status', 'location']
numerical = ['annual_income', 'number_of_courses_viewed', 'interaction_count', 'lead_score']
df[categorical] = df[categorical].fillna('NA')
df[numerical] = df[numerical].fillna(0.0)

df_full, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full, test_size=0.25, random_state=42)
y_train, y_val = df_train['converted'].values, df_val['converted'].values

def fit_acc(features, C=1.0):
    dv = DictVectorizer(sparse=False)
    X_tr = dv.fit_transform(df_train[features].to_dict(orient='records'))
    X_va = dv.transform(df_val[features].to_dict(orient='records'))
    m = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42)
    m.fit(X_tr, y_train)
    return (m.predict(X_va) == y_val).mean()

all_features = categorical + numerical
base = fit_acc(all_features)
print("Original accuracy:", base)

print("--- Q5 ---")
for f in ['lead_source', 'number_of_courses_viewed', 'interaction_count']:
    print(f, "| difference:", base - fit_acc([x for x in all_features if x != f]))

print("--- Q6 ---")
for C in [0.000001, 0.00001, 0.0001, 0.001]:
    print(C, round(fit_acc(all_features, C), 3))
