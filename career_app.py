import streamlit as st
import pandas as pd
import joblib
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
from difflib import get_close_matches

# Load model and encoders
model = joblib.load("career_model.pkl")
mlb_interests = joblib.load("encoder_interests.pkl")
mlb_subjects = joblib.load("encoder_subjects.pkl")
mlb_traits = joblib.load("encoder_traits.pkl")
workstyle_cols = joblib.load("encoder_workstyle.pkl")
env_cols = joblib.load("encoder_env.pkl")
risk_cols = joblib.load("encoder_risk.pkl")
df = pd.read_csv("career_dataset.csv")

# Load dataset (must include Career, Exams, Resources)
df = pd.read_csv("career_dataset.csv")

def fuzzy_match_list(user_inputs, valid_choices):
    matched = []
    for item in user_inputs:
        match = get_close_matches(item, valid_choices, n=1, cutoff=0.6)
        if match:
            matched.append(match[0])
    return matched

# Title
st.title("🎓 CareerPathAI – Advanced Career Recommendations")
st.subheader("📥 Fill Your Preferences:")

interests = st.multiselect("Your Interests:", mlb_interests.classes_)
subjects = st.multiselect("Your Favorite Subjects:", mlb_subjects.classes_)
traits = st.multiselect("Your Personality Traits:", mlb_traits.classes_)
work_style = st.selectbox("Preferred Work Style:", [col.split("_")[-1] for col in workstyle_cols])
environment = st.selectbox("Preferred Environment:", [col.split("_")[-1] for col in env_cols])
risk = st.selectbox("Risk Level:", [col.split("_")[-1] for col in risk_cols])

if st.button("🎯 Get Career Matches"):
    if not interests or not subjects or not traits:
        st.error("Please fill all the required fields.")
    else:
        # Apply fuzzy match
        interests_mapped = fuzzy_match_list(interests, mlb_interests.classes_)
        subjects_mapped = fuzzy_match_list(subjects, mlb_subjects.classes_)
        traits_mapped = fuzzy_match_list(traits, mlb_traits.classes_)

        # Debug
        st.markdown(f"🔍 **Matched Interests**: {interests_mapped}")
        st.markdown(f"🔍 **Matched Subjects**: {subjects_mapped}")
        st.markdown(f"🔍 **Matched Traits**: {traits_mapped}")

        # Encode
        interests_vec = mlb_interests.transform([interests_mapped])
        subjects_vec = mlb_subjects.transform([subjects_mapped])
        traits_vec = mlb_traits.transform([traits_mapped])
        workstyle_vec = np.array([1 if f"Workstyle_{work_style}" == col else 0 for col in workstyle_cols]).reshape(1, -1)
        env_vec = np.array([1 if f"Env_{environment}" == col else 0 for col in env_cols]).reshape(1, -1)
        risk_vec = np.array([1 if f"Risk_{risk}" == col else 0 for col in risk_cols]).reshape(1, -1)

        # Combine all
        final_input = np.hstack((
            interests_vec,
            subjects_vec,
            traits_vec,
            workstyle_vec,
            env_vec,
            risk_vec
        ))

        # Predict
        predictions = model.predict_proba(final_input)[0]
        top_indices = predictions.argsort()[::-1][:3]
        careers = model.classes_

        st.subheader("💡 Your Top 3 Career Matches:")
        for idx in top_indices:
            career_name = careers[idx]
            career_row = df[df["Career"] == career_name].iloc[0]

            st.markdown(f"### 🔹 {career_name}")
            st.write(f"📚 **Entrance Exams**: {career_row['Exams']}")
            st.write(f"🔗 **Free Resources**: {career_row['Resources']}")
            st.markdown("---")


# OPEN AI PART
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_chatbot(question):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=[
            {"role": "system", "content": "You are a helpful career counselor."},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content

st.subheader("💬 Ask CareerPathAI Anything!")

user_question = st.text_input("Ask your career-related question here:")

if st.button("Ask"):
    if user_question:
        reply = ask_chatbot(user_question)
        st.success(reply)
    else:
        st.error("Please enter a question.")