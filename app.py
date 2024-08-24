from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model and encoders
model = joblib.load('model/best_budget_recommender_model.pkl')
age_encoder = joblib.load('model/age_encoder.pkl')
employment_encoder = joblib.load('model/employment_encoder.pkl')
def preprocess_input(data):
    df = pd.DataFrame(data, index=[0])
    # Label Encoding for categorical features
    label_encode_features = ['age_range','employment_status' ]
    for column in label_encode_features:
        if column in df.columns:
            le = label_encoders[column]
            known_classes = set(le.classes_)
            df[column] = df[column].apply(lambda x: le.transform([x])[0] if x in known_classes else -1)
    
    
    df['household'] = df['household'] = pd.to_numeric(df['household'], errors='coerce').fillna(0)
    df['total_income'] = pd.to_numeric(df['total_income'], errors='coerce').fillna(0)
    
    return df 
        

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendation = None
    budget_breakdown = None
    
    if request.method == 'POST':
        data = request.form.to_dict()
        
        # Create input vector for prediction
        user_input = preprocess_input(data)
        
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
