from flask import Flask, render_template, request
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline
from src.utils import probability_to_risk_score, get_risk_category_and_limit

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "LIMIT_BAL": float(request.form["LIMIT_BAL"]),
    "SEX": int(request.form["SEX"]),
    "EDUCATION": int(request.form["EDUCATION"]),
    "MARRIAGE": int(request.form["MARRIAGE"]),
    "AGE": int(request.form["AGE"]),

    "PAY_0": int(request.form["PAY_0"]),
    "PAY_2": int(request.form["PAY_2"]),
    "PAY_3": int(request.form["PAY_3"]),
    "PAY_4": int(request.form["PAY_4"]),
    "PAY_5": int(request.form["PAY_5"]),
    "PAY_6": int(request.form["PAY_6"]),

    "BILL_AMT1": float(request.form["BILL_AMT1"]),
    "BILL_AMT2": float(request.form["BILL_AMT2"]),
    "BILL_AMT3": float(request.form["BILL_AMT3"]),
    "BILL_AMT4": float(request.form["BILL_AMT4"]),
    "BILL_AMT5": float(request.form["BILL_AMT5"]),
    "BILL_AMT6": float(request.form["BILL_AMT6"]),

    "PAY_AMT1": float(request.form["PAY_AMT1"]),
    "PAY_AMT2": float(request.form["PAY_AMT2"]),
    "PAY_AMT3": float(request.form["PAY_AMT3"]),
    "PAY_AMT4": float(request.form["PAY_AMT4"]),
    "PAY_AMT5": float(request.form["PAY_AMT5"]),
    "PAY_AMT6": float(request.form["PAY_AMT6"]),
        }

        df = pd.DataFrame([data])

        pipeline = PredictPipeline()
        prob_default = pipeline.predict(df)[0]

        risk_score = probability_to_risk_score(prob_default)
        risk_category, credit_limit = get_risk_category_and_limit(risk_score)

        return render_template(
            "result.html",
            risk_score=risk_score,
            risk_category=risk_category,
            credit_limit=credit_limit,
        )

    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(host="0.0.0.0")