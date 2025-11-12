import numpy as np
import pickle
from flask import Flask, request, render_template

# --- Load saved model and scalers ---
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("standscaler.pkl", "rb") as file:
    sc = pickle.load(file)

with open("minmaxscaler.pkl", "rb") as file:
    ms = pickle.load(file)

# --- Create Flask app ---
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Get user inputs from form
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['Ph'])
    rainfall = float(request.form['Rainfall'])

    # Create numpy array
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Apply scaling
    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)

    # Predict crop
    prediction = model.predict(final_features)

    # Crop mapping
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya",
        7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes",
        12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
        17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans",
        21: "Chickpea", 22: "Coffee"
    }

    crop = crop_dict.get(prediction[0], "Unknown Crop")
    result = f"{crop} is the best crop to be cultivated right there"

    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
