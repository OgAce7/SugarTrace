# 🩺 SugarTrace — Diabetes Prediction System

A student machine learning project that predicts whether a patient is likely to have diabetes, using the Pima Indians Diabetes Dataset.

---

## What This Project Does

- Takes 8 health features as input (like Glucose, BMI, Age, etc.)
- Uses two ML models to make a binary prediction: **Diabetic** or **Not Diabetic**
- Shows a risk score (probability) and risk level (Low / Moderate / High)
- Displays feature importances from Random Forest
- Has a simple web interface to interact with the model

---

## Models Used

| Model | Description |
|-------|-------------|
| Logistic Regression | Simple linear classification model |
| Random Forest | Ensemble of decision trees |

Both models are evaluated using:
- **Accuracy** — overall correctness
- **Recall** — how well it catches actual diabetic cases (important for medical)
- **ROC-AUC** — overall discrimination ability

---

## Project Structure

```
SugarTrace/
│
├── train_model.py          # Downloads dataset, trains and saves both models
├── preprocess.py           # Data cleaning and feature scaling functions
├── app.py                  # Flask backend with /predict and /model-info endpoints
│
├── frontend/
│   ├── index.html          # Main web page
│   ├── style.css           # Basic styling
│   └── script.js           # JS for form handling and API calls
│
├── models/                 # Created after training
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   └── results.pkl
│
├── requirements.txt
└── README.md
```

---

## How to Run

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Set up Kaggle API key (for dataset download)

1. Go to https://www.kaggle.com/settings/account
2. Scroll to "API" section and click **Create New API Token**
3. This downloads `kaggle.json`
4. Place it at `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<user>\.kaggle\kaggle.json` (Windows)

### Step 3: Train the models

```bash
python train_model.py
```

This will:
- Download the Pima Indians Diabetes dataset from Kaggle
- Clean and preprocess the data
- Train Logistic Regression and Random Forest models
- Save models to the `models/` folder
- Print evaluation metrics

### Step 4: Start the Flask server

```bash
python app.py
```

### Step 5: Open in browser

Visit: [http://localhost:5000](http://localhost:5000)

---

## Dataset

- **Name:** Pima Indians Diabetes Database
- **Source:** UCI Machine Learning Repository (via Kaggle)
- **Records:** 768 patients
- **Features:** 8 numerical health measurements
- **Target:** 0 = Not Diabetic, 1 = Diabetic

### Features Explained

| Feature | Description |
|---------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (2hr oral test) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (μU/mL) |
| BMI | Body mass index (weight / height²) |
| DiabetesPedigreeFunction | Family history risk score |
| Age | Age in years |

---

## API Endpoints

### POST `/predict`

Send patient data and get a prediction.

**Request body (JSON):**
```json
{
  "model": "random_forest",
  "Pregnancies": 3,
  "Glucose": 148,
  "BloodPressure": 72,
  "SkinThickness": 35,
  "Insulin": 125,
  "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.627,
  "Age": 50
}
```

**Response:**
```json
{
  "prediction": 1,
  "result": "Diabetic",
  "probability": 78.5,
  "risk_level": "High",
  "model_used": "Random Forest",
  "feature_importance": {
    "Glucose": 0.2341,
    "BMI": 0.1832,
    ...
  }
}
```

### GET `/model-info`

Returns accuracy, recall, and ROC-AUC scores for both models.

---

## Notes

- Zero values in Glucose, BloodPressure, etc. are treated as missing and replaced with column medians
- All features are scaled using StandardScaler before model input
- Recall is prioritized as a metric because missing a diabetic case is more costly than a false positive

---

## Disclaimer

⚠️ This is a **student learning project**. It is NOT meant for actual medical diagnosis. Always consult a qualified doctor for health-related decisions.

---

*Built with Python, Flask, Scikit-learn, and basic HTML/CSS/JS*
