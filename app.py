import os
from pathlib import Path

from flask import Flask, render_template, request
import joblib


app = Flask(__name__, template_folder=".")

# Load trained model and scaler
# Keep these files in the same folder as app.py for deployment.
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "placement_model.pkl")
scaler = joblib.load(BASE_DIR / "placement_scaler.pkl")

# Company requirements are kept separate from the ML model.
# These are project/demo criteria and can be edited as required.
COMPANIES = {
    "tcs": {
        "name": "TCS",
        "role": "Software Developer",
        "cgpa": 7.0,
        "ssc": 60,
        "hsc": 60,
        "aptitude": 60,
        "internships": 1,
        "projects": 2
    },
    "infosys": {
        "name": "Infosys",
        "role": "System Engineer",
        "cgpa": 6.5,
        "ssc": 60,
        "hsc": 60,
        "aptitude": 55,
        "internships": 0,
        "projects": 1
    },
    "accenture": {
        "name": "Accenture",
        "role": "Associate Software Engineer",
        "cgpa": 6.5,
        "ssc": 60,
        "hsc": 60,
        "aptitude": 55,
        "internships": 0,
        "projects": 2
    },
    "wipro": {
        "name": "Wipro",
        "role": "Project Engineer",
        "cgpa": 6.0,
        "ssc": 55,
        "hsc": 55,
        "aptitude": 50,
        "internships": 0,
        "projects": 1
    }
}

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    selected_company = None
    company = None
    eligibility = None
    requirements = None
    missing = []

    if request.method == "POST":
        try:
            student = {
                "cgpa": float(request.form["cgpa"]),
                "internships": float(request.form["internships"]),
                "projects": float(request.form["projects"]),
                "workshops": float(request.form["workshops"]),
                "aptitude": float(request.form["aptitude"]),
                "soft_skills": float(request.form["soft_skills"]),
                "extracurricular": float(request.form["extracurricular"]),
                "placement_training": float(request.form["placement_training"]),
                "ssc": float(request.form["ssc"]),
                "hsc": float(request.form["hsc"])
            }

            # Same feature order used during model training.
            model_input = [[
                student["cgpa"],
                student["internships"],
                student["projects"],
                student["workshops"],
                student["aptitude"],
                student["soft_skills"],
                student["extracurricular"],
                student["placement_training"],
                student["ssc"],
                student["hsc"]
            ]]

            student_scaled = scaler.transform(model_input)
            prediction = int(model.predict(student_scaled)[0])
            probability = float(model.predict_proba(student_scaled)[0][1] * 100)

            selected_company = request.form.get("company", "tcs")
            company = COMPANIES.get(selected_company, COMPANIES["tcs"])

            requirements = [
                ("CGPA", student["cgpa"], company["cgpa"], student["cgpa"] >= company["cgpa"]),
                ("SSC Marks", student["ssc"], company["ssc"], student["ssc"] >= company["ssc"]),
                ("HSC Marks", student["hsc"], company["hsc"], student["hsc"] >= company["hsc"]),
                ("Aptitude Score", student["aptitude"], company["aptitude"], student["aptitude"] >= company["aptitude"]),
                ("Internships", student["internships"], company["internships"], student["internships"] >= company["internships"]),
                ("Projects", student["projects"], company["projects"], student["projects"] >= company["projects"])
            ]

            missing = [item[0] for item in requirements if not item[3]]
            eligibility = len(missing) == 0

        except (ValueError, KeyError):
            prediction = None

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        companies=COMPANIES,
        selected_company=selected_company,
        company=company,
        eligibility=eligibility,
        requirements=requirements,
        missing=missing
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
