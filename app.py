import streamlit as st
import numpy as np
import pickle

# ----------------------------
# Load model and scalers safely
# ----------------------------
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("standscaler.pkl", "rb") as file:
    sc = pickle.load(file)

with open("minmaxscaler.pkl", "rb") as file:
    ms = pickle.load(file)

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="🌱 Crop Recommendation System", layout="centered")
st.title("🌾 Crop Recommendation System")
st.write("Enter the soil and weather conditions to get the best crop recommendation.")

# ----------------------------
# Input fields
# ----------------------------
N = st.number_input("Nitrogen", min_value=0.0, step=0.1)
P = st.number_input("Phosphorus", min_value=0.0, step=0.1)
K = st.number_input("Potassium", min_value=0.0, step=0.1)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0.0, step=0.1)
ph = st.number_input("pH value", min_value=0.0, step=0.01)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=0.1)

# ----------------------------
# Predict button
# ----------------------------
if st.button("🌿 Get Recommendation"):
    # Prepare input
    feature_list = [N, P, K, temperature, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Scale features
    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)

    # Make prediction
    prediction = model.predict(final_features)

    # Crop dictionary
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya",
        7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes",
        12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
        17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans",
        21: "Chickpea", 22: "Coffee"
    }

    # Get crop name
    crop = crop_dict.get(prediction[0], "Unknown Crop")
    st.success(f"✅ Recommended Crop: **{crop}**")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and Machine Learning.")