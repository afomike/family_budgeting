from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model and encoders
model = joblib.load('model/best_budget_recommender_model.pkl')
age_encoder = joblib.load('model/age_encoder.pkl')
employment_encoder = joblib.load('model/employment_encoder.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendation = None
    budget_breakdown = None
    
    if request.method == 'POST':
        age_range = age_encoder.transform([request.form['age_range']])[0]
        household = int(request.form['household'])
        employment_status = employment_encoder.transform([request.form['employment_status']])[0]
        total_income = int(request.form['total_income'])
        
        # Create input vector for prediction
        user_input = [age_range, household, employment_status, total_income]
        
        # Predict budget allocations
        predicted_allocations = model.predict([user_input])[0]
        
        # Generate detailed budget breakdown
        budget_breakdown = {
            'Food': predicted_allocations[0],
            'Housing': predicted_allocations[1],
            'Transportation': predicted_allocations[2],
            'Utilities': predicted_allocations[3],
            'Insurance': predicted_allocations[4],
            'Savings': predicted_allocations[5],
            'Other Expenses': predicted_allocations[6]
        }
    
    return render_template('index.html', budget_breakdown=budget_breakdown)

if __name__ == '__main__':
    app.run(debug=True)
