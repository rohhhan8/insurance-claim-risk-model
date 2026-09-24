# 🚗 Insurance Claim Risk Model

A machine learning system that predicts **how likely** a car insurance policy is to file a claim, and **how expensive** that claim will be — then combines both into an expected payout, just like a real insurance pricing team would.

## 🧠 The idea, in one analogy

Think of a food delivery app estimating your order's cost before you check out:

- **"Will this order get cancelled?"** → a *probability* (frequency model)
- **"If it doesn't get cancelled, how much will it cost?"** → an *amount* (severity model)
- **Final estimate shown to you** → `P(not cancelled) × expected cost`

This project does the same thing for car insurance:

- 📊 **Frequency model** → probability a policy files a claim
- 💰 **Severity model** → expected claim amount, *if* one happens
- ➗ **Combined** → `expected_loss = P(claim) × P(real payout | claim) × E(severity)`

Both are trained on the real-world [French Motor Third-Party Liability](https://www.openml.org/d/41214) dataset (~678K policies) using **XGBoost**, calibrated, and served through a **FastAPI** endpoint.

## 📁 What's inside

```
src/            → data cleaning, feature engineering, model training
api/            → FastAPI service that serves predictions
notebooks/      → exploratory data analysis
reports/        → charts, metrics, SHAP explanations
tests/          → unit tests
Dockerfile      → container build for the API
```

## 🚀 Running it

### Option A — Run locally

**1. Clone the repo**
```bash
git clone https://github.com/rohhhan8/insurance-claim-risk-model.git
cd insurance-claim-risk-model
```

**2. Create a virtual environment & install dependencies**
```bash
python3.11 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**3. Train the models** (downloads data automatically, trains + saves everything)
```bash
python -m src.models.train_frequency
python -m src.models.train_severity
python -m src.models.train_combine
python -m src.interpret.run_shap
```

**4. Start the API**
```bash
uvicorn api.main:app --reload
```

**5. Try it out** 🎉
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Exposure": 0.8, "VehPower": 7, "VehAge": 5, "DrivAge": 35,
    "BonusMalus": 60, "VehBrand": "B12", "VehGas": "Diesel",
    "Density": 1200, "Region": "R82", "Area": "D"
  }'
```

Or open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the interactive Swagger UI.

### Option B — Run with Docker 🐳

**1. Train the models first** (Docker only serves the API, so model files need to exist locally — repeat step 3 above if you haven't).

**2. Build the image**
```bash
docker build -t claim-risk-api .
```

**3. Run the container**
```bash
docker run -p 8000:8000 claim-risk-api
```

**4. Test it** — same `curl` command as above, or visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Running the tests
```bash
pytest tests/
```

## 📈 Results at a glance

| Model | Metric | Value |
|---|---|---|
| Frequency (claim probability) | PR-AUC | **0.139** vs 0.050 baseline (~2.75x lift) |
| Severity (claim amount) | MAE | **€2,302** |
| Combined | Predicted vs. actual total loss | **1.10x** (well calibrated) |

The model correctly identifies its riskiest decile of policies (predicted **€312**/policy vs actual **€205**), and SHAP analysis shows `Exposure` and `BonusMalus` (no-claims bonus score) as the top drivers of risk for both models.

## 🛠️ Built with

`Python` · `XGBoost` · `scikit-learn` · `SHAP` · `FastAPI` · `Docker` · `pandas`
