from flask import Flask, request, render_template
from flask_cors import cross_origin
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("flight_rf.pkl", "rb"))

@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        # Date_of_Journey
        date_dep = request.form["Dep_Time"]
        journey_date = pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M")
        Journey_day = journey_date.day
        Journey_month = journey_date.month

        # Departure
        Dep_hour = journey_date.hour
        Dep_min = journey_date.minute

        # Arrival
        date_arr = request.form["Arrival_Time"]
        arrival_date = pd.to_datetime(date_arr, format="%Y-%m-%dT%H:%M")
        Arrival_hour = arrival_date.hour
        Arrival_min = arrival_date.minute

        # Duration
        dur_hour = abs(Arrival_hour - Dep_hour)
        dur_min = abs(Arrival_min - Dep_min)

        # Total Stops
        Total_stops = int(request.form["stops"])

        # Airline
        airline = request.form['airline']
        airlines_dict = {
            'Jet Airways': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'IndiGo': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'Air India': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            'Multiple carriers': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            'SpiceJet': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            'Vistara': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            'GoAir': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            'Multiple carriers Premium economy': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            'Jet Airways Business': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            'Vistara Premium economy': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            'Trujet': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        }
        airline_features = airlines_dict.get(airline, [0]*11)
        Jet_Airways, IndiGo, Air_India, Multiple_carriers, SpiceJet, Vistara, GoAir, Multiple_carriers_Premium_economy, Jet_Airways_Business, Vistara_Premium_economy, Trujet = airline_features

        # Source
        Source = request.form["Source"]
        sources_dict = {
            'Delhi': [1, 0, 0, 0],
            'Kolkata': [0, 1, 0, 0],
            'Mumbai': [0, 0, 1, 0],
            'Chennai': [0, 0, 0, 1]
        }
        source_features = sources_dict.get(Source, [0]*4)
        s_Delhi, s_Kolkata, s_Mumbai, s_Chennai = source_features

        # Destination
        Destination = request.form["Destination"]
        destinations_dict = {
            'Cochin': [1, 0, 0, 0, 0],
            'Delhi': [0, 1, 0, 0, 0],
            'Hyderabad': [0, 0, 1, 0, 0],
            'Kolkata': [0, 0, 0, 1, 0]
        }
        destination_features = destinations_dict.get(Destination, [0]*5)
        d_Cochin, d_Delhi, d_Hyderabad, d_Kolkata, d_New_Delhi = destination_features

        prediction = model.predict([[
            Total_stops, Journey_day, Journey_month, Dep_hour, Dep_min,
            Arrival_hour, Arrival_min, dur_hour, dur_min, Air_India, GoAir,
            IndiGo, Jet_Airways, Jet_Airways_Business, Multiple_carriers,
            Multiple_carriers_Premium_economy, SpiceJet, Trujet, Vistara,
            Vistara_Premium_economy, s_Chennai, s_Delhi, s_Kolkata, s_Mumbai,
            d_Cochin, d_Delhi, d_Hyderabad, d_Kolkata
        ]])

        output = round(prediction[0], 2)
        return render_template('home.html', prediction_text=f"Your Flight price is Rs. {output}")

    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
