
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

with open("traffic_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        
        hour = int(request.form["hour"])
        day = int(request.form["day"])
        weather = int(request.form["weather"])
        vehicle_count = int(request.form["vehicle_count"])

    
        new_data = pd.DataFrame({
            'Hour': [hour],
            'DayOfWeek': [day],
            'Weather': [weather],
            'VehicleCount': [vehicle_count]
        })

        
        prediction = model.predict(new_data)[0]

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
