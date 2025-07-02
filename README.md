# CareerPathAI

# 🎓 CareerPathAI

**CareerPathAI** is an intelligent, AI-powered career guidance system that recommends personalized career paths based on your interests, subjects, personality traits, work style, risk appetite, and preferred work environment. It combines machine learning predictions with an interactive chatbot counselor to help students and young professionals find their ideal career path.

---

## 🧠 Features

- 🎯 Personalized career prediction using a trained ML model.
- 📊 Multiselect inputs for interests, subjects, and traits.
- 🧬 One-hot encoding for categorical features like work style, environment, and risk level.
- 🤖 Real-time chatbot powered by OpenAI for career guidance.
- 📚 Displays relevant exams and free learning resources.
- 🔍 Fuzzy matching to improve input accuracy and usability.

---

## 💻 Tech Stack

| **Component**      | **Technology**              |
|--------------------|-----------------------------|
| Frontend           | [Streamlit](https://streamlit.io) |
| Backend / ML       | Python, scikit-learn, joblib |
| AI Chatbot         | [OpenAI GPT-3.5 Turbo](https://platform.openai.com) |
| Data Handling      | pandas, NumPy               |
| Environment Config | python-dotenv               |
| Model Algorithm    | RandomForestClassifier (scikit-learn) |


---

## 📁 Project Structure

```bash
CareerPathAI/
│
├── .env                        # API key for OpenAI
├── .gitignore
├── career_dataset.csv         # Training dataset with career info
├── career_model.pkl           # Trained ML model
├── encoder_*.pkl              # Saved encoders for inputs
├── career_app.py              # Streamlit web app
├── main.py                    # Model training script
├── requirements.txt           # Python dependencies
└── README.md
