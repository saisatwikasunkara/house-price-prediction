import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
import warnings

# =========================
# LOAD DATA
# =========================
data = pd.read_csv('data/House Price India.csv')

# =========================
# FEATURES
# =========================
FEATURES = [
    'living area',
    'number of bedrooms',
    'number of bathrooms',
    'number of floors',
    'condition of the house',
    'Distance from the airport'
]

TARGET = 'Price'

X = data[FEATURES]
y = data[TARGET]

# =========================
# HARD LIMITS (YOUR DATA BOUNDS)
# =========================
LIMITS = {
    'living area': (370.0, 13540.0),
    'number of bedrooms': (1.0, 33.0),
    'number of bathrooms': (0.5, 8.0),
    'number of floors': (1.0, 3.5),
    'condition of the house': (1.0, 5.0),
    'Distance from the airport': (50.0, 80.0)
}

# =========================
# CLEANING FUNCTION
# =========================
def clip_data(df):
    df = df.copy()

    for col in FEATURES:
        min_v, max_v = LIMITS[col]

        out = (df[col] < min_v) | (df[col] > max_v)
        if out.any():
            warnings.warn(f"{col} out of bounds → clipping applied")

        df[col] = np.clip(df[col], min_v, max_v)

    return df

X = clip_data(X)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODEL
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
pred = model.predict(X_test)
r2 = r2_score(y_test, pred)

print("R2 Score:", r2)


joblib.dump(model, "house_price_model.pkl")
print("Model saved successfully!")