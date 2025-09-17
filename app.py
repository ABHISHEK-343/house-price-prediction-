from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("bangalore_house_price_model.pkl", "rb"))

data = pd.read_csv("cleaned_data.csv")
locations = sorted(data['location'].unique())

@app.route("/")
def index():
    return render_template("index.html", locations=locations)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.content_type != "application/json":
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        print("Received JSON:", data) 

        location = data.get("location")
        bhk = data.get("bhk")
        bath = data.get("bath")
        sqft = data.get("total_sqft")  

    
        if not location or bhk is None or bath is None or sqft is None:
            return jsonify({"error": "Missing required parameters"}), 400

       
        try:
            bhk = int(bhk)
            bath = int(bath)
            sqft = float(sqft)
        except ValueError:
            return jsonify({"error": "Invalid data format"}), 400

        print(f"Processing: Location={location}, BHK={bhk}, Bath={bath}, sqft={sqft}")

      
        if location not in locations:
            return jsonify({"error": f"Invalid location: {location}"}), 400


        input_data = pd.DataFrame([[location, sqft, bath, bhk]], 
                                  columns=["location", "total_sqft", "bath", "BHK"]) 

        
        prediction = model.predict(input_data)[0]
        price = round(prediction, 2)

        return jsonify({"price":f" {price} lakhs"})  

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
