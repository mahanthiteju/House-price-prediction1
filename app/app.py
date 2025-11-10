from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("../models/model.pkl", "rb"))


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        size = float(request.form["size"])
        bhk = int(request.form["bhk"])
        under_construction = int(request.form["under_construction"])
        ready_to_move = int(request.form["ready_to_move"])
        resale = int(request.form["resale"])
        rera = int(request.form["rera"])
        
        # Create input DataFrame with required features
        input_data = {
            "UNDER_CONSTRUCTION": [under_construction],
            "RERA": [rera],
            "BHK_NO.": [bhk],
            "size": [size],
            "READY_TO_MOVE": [ready_to_move],
            "RESALE": [resale],
            "LONGITUDE": [0],  # Using default values for location
            "LATITUDE": [0]    # Using default values for location
        }

        df = pd.DataFrame(input_data)
        prediction = model.predict(df)[0]

        return render_template("result.html", price=round(prediction, 2))
    return render_template("form.html")


if __name__ == "__main__":
    app.run(debug=True)