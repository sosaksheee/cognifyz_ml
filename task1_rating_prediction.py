import os
os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), 'matplotlib_cache')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib

# === Step 1: Load Dataset ===
df = pd.read_csv(r'C:\Users\Dhatrak\Desktop\cognifyz_ml\Dataset .csv')
print("Dataset Loaded | Shape:", df.shape)
print(df.head(2), "\n")

# === Step 2: Clean Data ===
cols_to_drop = ['Restaurant ID', 'Restaurant Name', 'URL', 'Address']
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True, errors='ignore')

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# === Step 3: Define Features & Target ===
target_col = 'Aggregate rating' if 'Aggregate rating' in df.columns else df.columns[-1]
y = df[target_col]
X = df.drop(columns=[target_col])

# === Step 4: Split Dataset ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f" Training set: {X_train.shape}, Test set: {X_test.shape}")

# === Step 5: Preprocessing Setup ===
numeric_features = X.select_dtypes(include=[np.number]).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# === Step 6: Define Models ===
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.01),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# === Step 7: Train & Evaluate Models ===
results = []
for name, model in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({'Model': name, 'MSE': mse, 'MAE': mae, 'R2': r2})
    print(f"\n {name} Results:")
    print(f" MSE={mse:.4f} | MAE={mae:.4f} | R²={r2:.4f}")
    
# === Step 8: Compare Models ===
results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)
print("\n=== Model Comparison ===")
print(results_df)

# === Step 9: Plot Best Model Predictions ===
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

final_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                             ('model', best_model)])
final_pipe.fit(X_train, y_train)
y_pred_best = final_pipe.predict(X_test)

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_best, alpha=0.7)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title(f'{best_model_name}: Actual vs Predicted')
plt.grid(True)
plt.show()

# === Step 10: Feature Importance (Tree Models Only) ===
if hasattr(best_model, 'feature_importances_'):
    encoded_cols = list(preprocessor.named_transformers_['cat']
                        .named_steps['encoder'].get_feature_names_out(categorical_features))
    all_features = list(numeric_features) + encoded_cols
    fi = pd.Series(best_model.feature_importances_, index=all_features).sort_values(ascending=False).head(15)
    plt.figure(figsize=(10,5))
    plt.barh(fi.index, fi.values)
    plt.title(f'{best_model_name} - Top Feature Importances')
    plt.xlabel('Importance Score')
    plt.gca().invert_yaxis()
    plt.show()

# === Step 11: Save Best Model ===
joblib.dump(final_pipe, f'{best_model_name.replace(" ", "_").lower()}_model.pkl')
print(f"\n Model saved as '{best_model_name.replace(' ', '_').lower()}_model.pkl'")

print("\n Pipeline Complete — Ready for Analysis & Deployment.")
