from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load('house_price_model.pkl')

# Limits
LIMITS = {
    'living_area': (370.0, 13540.0),
    'bedrooms': (1.0, 33.0),
    'bathrooms': (0.5, 8.0),
    'floors': (1.0, 3.5),
    'condition': (1.0, 5.0),
    'airport': (50.0, 80.0)
}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    try:
        # Inputs
        living_area = float(request.form['living_area'])
        bedrooms = float(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        floors = float(request.form['floors'])
        condition = float(request.form['condition'])
        airport = float(request.form['airport'])

        # Validation
        errors = []

        if not (370 <= living_area <= 13540):
            errors.append("Living area must be 370 - 13540")

        if not (1 <= bedrooms <= 33):
            errors.append("Bedrooms must be 1 - 33")

        if not (1 <= bathrooms <= 8):
            errors.append("Bathrooms must be 1 - 8")

        if not (1 <= floors <= 3):
            errors.append("Floors must be 1 - 3")

        if not (1 <= condition <= 5):
            errors.append("Condition must be 1 - 5")

        if not (50 <= airport <= 80):
            errors.append("Airport distance must be 50 - 80")

        # If errors → show message
        if errors:
            return render_template(
                "index.html",
                prediction_text="❌ Please enter correct values:\n" + "\n".join(errors)
            )

        # Dataframe
        features = pd.DataFrame([[
            living_area,
            bedrooms,
            bathrooms,
            floors,
            condition,
            airport
        ]], columns=[
            'living area',
            'number of bedrooms',
            'number of bathrooms',
            'number of floors',
            'condition of the house',
            'Distance from the airport'
        ])

        # Predict
        prediction = model.predict(features)[0]
        prediction = max(0, prediction)

        return render_template(
            "index.html",
            prediction_text=f"🏠 Predicted Price: ₹ {prediction:,.0f}"
        )

    except:
        return render_template(
            "index.html",
            prediction_text="❌ Invalid input. Please enter numbers only."
        )


if __name__ == "__main__":
    app.run(debug=True)