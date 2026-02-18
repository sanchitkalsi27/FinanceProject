# Kuhoo Finance Demo

AI-powered Student Loan Eligibility Predictor + EMI Calculator  
Built by **Sanchit Kalsi**

## Features
- 🤖 AI Loan Eligibility Predictor (Random Forest ML model)
- 💰 EMI Calculator with amortization schedule
- 📊 Visual charts: Gauge, Donut, Pie, Bar charts
- 🌙 Premium dark glassmorphism UI

## Local Setup
```bash
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

## Deploy to Render
1. Push this folder to a GitHub repository
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
5. Click **Deploy** — done! 🚀
