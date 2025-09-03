import streamlit as st
import joblib
import json
import numpy as np

# =======================================================
# 1. Page Configuration
# =======================================================
st.set_page_config(page_title="Disease Prediction", page_icon="🩺", layout="wide")

# =======================================================
# 2. Load Model and Symptom Columns
# =======================================================
@st.cache_resource
def load_model_and_columns():
    model = joblib.load("model.joblib")
    with open("symptom_columns.json", "r") as f:
        symptom_columns = json.load(f)
    return model, symptom_columns

model, symptom_columns = load_model_and_columns()

# =======================================================
# 3. UI Header
# =======================================================
st.title("🩺 Disease Prediction System")
st.write("Enter your symptoms to predict the most probable diseases.")

# =======================================================
# 4. Symptom Selection
# =======================================================
selected_symptoms = st.multiselect(
    "Select your symptoms:",
    options=symptom_columns,
)

# =======================================================
# 5. Prediction
# =======================================================
if st.button("🔍 Predict Disease"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        # Create input vector
        input_vector = np.zeros(len(symptom_columns))
        for s in selected_symptoms:
            idx = symptom_columns.index(s)
            input_vector[idx] = 1

        input_vector = input_vector.reshape(1, -1)

        # Get prediction probabilities
        probs = model.predict_proba(input_vector)[0]

        # Sort top 5 diseases by probability
        top_indices = np.argsort(probs)[::-1][:5]

        st.subheader("🧾 Prediction Results:")
        for i in top_indices:
            st.write(f"**{model.classes_[i]}** → {probs[i]*100:.2f}%")

        # Highlight most probable disease
        best_idx = top_indices[0]
        st.success(f"✅ Most probable disease: **{model.classes_[best_idx]}**")

# =======================================================
# 6. Footer
# =======================================================
st.markdown("---")
