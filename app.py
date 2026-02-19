from flask import Flask, request, render_template, jsonify
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import os

# This tells Flask to look in the current folder for templates, which is more reliable on Render
app = Flask(__name__, template_folder='.', static_folder='.')

# --- Train a simple model with synthetic data ---
np.random.seed(42)
n = 1000

age                = np.random.randint(18, 45, n)
academic_score     = np.random.randint(40, 100, n)
family_income      = np.random.randint(10000, 200000, n)
existing_loans     = np.random.randint(0, 5, n)
college_rank       = np.random.randint(1, 500, n)
coapplicant_income = np.random.randint(0, 100000, n)
location_tier      = np.random.randint(1, 4, n)

eligible = (
    (academic_score > 65) &
    (family_income + coapplicant_income > 50000) &
    (existing_loans < 3) &
    (college_rank < 300)
).astype(int)

X = np.column_stack([age, academic_score, family_income,
                     existing_loans, college_rank,
                     coapplicant_income, location_tier])
y = eligible

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

FEATURE_NAMES = [
    "Age", "Academic Score", "Family Income",
    "Existing Loans", "College Rank",
    "Co-applicant Income", "Location Tier"
]

def compute_factor_scores(features):
    """Return a 0-100 score for each factor (how good it is for eligibility)."""
    age_v, score_v, finc_v, loans_v, rank_v, coinc_v, tier_v = features
    return {
        "Academic Score":      min(100, int(score_v)),
        "Family Income":       min(100, int(finc_v / 2000)),
        "Co-applicant Income": min(100, int(coinc_v / 1000)),
        "Existing Loans":      max(0, 100 - loans_v * 20),
        "College Rank":        max(0, 100 - int(rank_v / 5)),
        "Age":                 min(100, int((age_v - 16) / 30 * 100)),
        "Location Tier":       {1: 100, 2: 65, 3: 35}.get(int(tier_v), 50),
    }

@app.route('/')
def home():
    # Try multiple paths just to be safe for deployment
    for path in ['templates/index.html', 'index.html']:
        if os.path.exists(path):
            return render_template(path.split('/')[-1])
    return "Error: index.html not found in root or templates/ folder."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            int(request.form['age']),
            int(request.form['academic_score']),
            int(request.form['family_income']),
            int(request.form['existing_loans']),
            int(request.form['college_rank']),
            int(request.form['coapplicant_income']),
            int(request.form['location_tier']),
        ]

        prediction   = model.predict([features])[0]
        proba_arr    = model.predict_proba([features])[0]
        eligible_pct = round(float(proba_arr[1]) * 100, 1)
        conf_pct     = round(float(proba_arr[prediction]) * 100, 1)

        importances  = model.feature_importances_
        factor_scores = compute_factor_scores(features)

        # Build importance list sorted descending
        importance_data = sorted(
            [{"name": n, "importance": round(float(v)*100, 1)}
             for n, v in zip(FEATURE_NAMES, importances)],
            key=lambda x: x["importance"], reverse=True
        )

        if prediction == 1:
            result_text  = f"✅ Eligible for Loan!"
            result_class = "eligible"
        else:
            result_text  = f"❌ Not Eligible for Loan"
            result_class = "not-eligible"

        return render_template(
            'index.html',
            prediction_text  = result_text,
            result_class     = result_class,
            eligible_pct     = eligible_pct,
            conf_pct         = conf_pct,
            importance_data  = importance_data,
            factor_scores    = factor_scores,
            form_data        = request.form,
            prediction       = int(prediction)
        )
    except Exception as e:
        return render_template('index.html',
                               prediction_text=f"Error: {str(e)}",
                               result_class="error")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
