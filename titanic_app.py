import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow import keras

# ------------------------------
# Page config
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# ------------------------------
st.title("🚢 Titanic Survival Prediction")
st.markdown(
    """
    Predict whether a passenger would survive the Titanic disaster using a trained ML model.
    Adjust the passenger details below and see the prediction instantly!
    """
)

# ------------------------------
# Load & preprocess dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Titanic-Dataset.csv")
    df.drop(["PassengerId","Name","Ticket","Cabin"], axis=1, inplace=True)
    df["Age"] = df["Age"].fillna(df["Age"].mode()[0])
    
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])
    
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return df, X_scaled, y, scaler

df, X_scaled, y, scaler = load_data()

# ------------------------------
# Sidebar for user input
st.sidebar.header("Passenger Information")

def user_input_features():
    Pclass = st.sidebar.selectbox("Pclass (1=1st, 2=2nd, 3=3rd)", [1,2,3])
    Sex = st.sidebar.selectbox("Sex (0=Female, 1=Male)", [0,1])
    Age = st.sidebar.slider("Age", 0, 80, 25)
    SibSp = st.sidebar.slider("Siblings/Spouses Aboard", 0, 8, 0)
    Parch = st.sidebar.slider("Parents/Children Aboard", 0, 6, 0)
    Fare = st.sidebar.number_input("Fare ($)", 0.0, 600.0, 32.0)
    Embarked = st.sidebar.selectbox("Embarked (0=C, 1=Q, 2=S)", [0,1,2])
    
    data = pd.DataFrame([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]],
                        columns=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"])
    return data

input_df = user_input_features()

# ------------------------------
# Train model (could also save & load a trained model for speed)
@st.cache_resource
def train_model(X_scaled, y):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = keras.Sequential([
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(22, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=50, verbose=0)
    
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    
    return model, acc

model, accuracy = train_model(X_scaled, y)

st.subheader("Model Accuracy")
st.success(f"{accuracy*100:.2f}%")

# ------------------------------
# Make prediction
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)
prediction_label = "🚀 Survived" if prediction[0][0] > 0.5 else "💀 Not Survived"

st.subheader("Prediction")
st.write(prediction_label)

# ------------------------------
# Data visualization
st.subheader("Passenger Distribution Charts")

fig1 = px.histogram(df, x="Age", color="Survived", barmode="overlay",
                    labels={"Survived":"Survival"}, title="Age Distribution by Survival")
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.histogram(df, x="Fare", color="Survived", nbins=30,
                    labels={"Survived":"Survival"}, title="Fare Distribution by Survival")
st.plotly_chart(fig2, use_container_width=True)
