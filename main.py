##########################################################################################################################################
'''
Create Dataset
Load the dataset
convert into encodings (First multiple values then single values)
combile all the features in one dict
Target Y 
Test train and spilt them so we get better result
Train the model using random forest
saved it
'''
##########################################################################################################################################

import pandas as pd                                      #For Load CSV files
from sklearn.preprocessing import MultiLabelBinarizer    #For convert words into 0/1
from sklearn.model_selection import train_test_split     #For Train and test and split the data 
from sklearn.ensemble import RandomForestClassifier      #An algorithm which finds the best match
import joblib                                            #To save this model
import numpy as np                                       #for data manipulation

# LOAD THE CSV FILE 
df = pd.read_csv("career_dataset.csv")

# CONVERT STRING INTO LIST
for col in ['Interests', 'Subjects', 'Traits']:
    df[col] = df[col].apply(lambda x: [i.strip() for i in str(x).split(",")])

#ENCODING (CONVERT THIS LIST IN 0 AND 1)
mlb_interests = MultiLabelBinarizer()
mlb_subjects = MultiLabelBinarizer()
mlb_traits = MultiLabelBinarizer()

interests_encoded = mlb_interests.fit_transform(df["Interests"])
subjects_encoded = mlb_subjects.fit_transform(df["Subjects"])
traits_encoded = mlb_traits.fit_transform(df["Traits"])

#ENCODING SINGLE VALUE FEATURES
work_style_encoded = pd.get_dummies(df["Work Style"], prefix="Workstyle")
environment_encoded = pd.get_dummies(df["Environment"], prefix="Env")
risk_encoded = pd.get_dummies(df["Risk Level"], prefix="Risk")

#COMBINE ALL THE FEATURES
X = np.hstack(
    (
        interests_encoded,
        subjects_encoded,
        traits_encoded,
        work_style_encoded.values,
        environment_encoded.values,risk_encoded.values
    )
)

#TARGET LABELS (Y = TARGETED VALUE)
y = df["Career"]

# TRAIN , TEST AND SPLIT THE DATA
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state=42)

# TRAIN THE MODEL USING RANDOM FOREST
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# SAVE THESE MODEL + ENCODERS
joblib.dump(model, "career_model.pkl")
joblib.dump(mlb_interests, "encoder_interests.pkl")
joblib.dump(mlb_subjects, "encoder_subjects.pkl")
joblib.dump(mlb_traits, "encoder_traits.pkl")
joblib.dump(work_style_encoded.columns.tolist(), "encoder_workstyle.pkl")
joblib.dump(environment_encoded.columns.tolist(), "encoder_env.pkl")
joblib.dump(risk_encoded.columns.tolist(), "encoder_risk.pkl")

# TO CHECK AND CONFIRM THE MESSAGE
print("Model trained and saved!")








