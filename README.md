🚢 Titanic Survival Prediction

A machine learning web application that predicts whether a passenger would survive the Titanic disaster based on key features such as age, gender, ticket class, and fare.
Built using Python, TensorFlow/Keras, and Streamlit, this project demonstrates an end-to-end ML workflow — from data preprocessing and model training to deployment in an interactive UI.

📌 Features
🔍 Predict passenger survival in real-time
📊 Interactive data visualizations (Age & Fare distributions)
🧠 Neural network model built with TensorFlow/Keras
⚡ Fast and responsive UI using Streamlit
🧹 Data preprocessing (handling missing values, encoding, scaling)

🗂️ Project Structure
├── Titanic Passenger Survival Prediction.ipynb   # Data analysis & model experimentation
├── titanic_app.py                               # Streamlit web application
├── Titanic-Dataset.csv                          # Dataset used for training
├── README.md                                    # Project documentation

⚙️ Tech Stack
Frontend/UI: Streamlit
Backend: Python
Machine Learning: TensorFlow / Keras
Data Processing: Pandas, Scikit-learn
Visualization: Plotly
📊 Dataset

The dataset contains passenger details such as:

Pclass (Ticket class)
Sex
Age
SibSp (Siblings/Spouses aboard)
Parch (Parents/Children aboard)
Fare
Embarked
Survival status

Basic preprocessing steps include:

Dropping irrelevant columns (Name, Ticket, Cabin, PassengerId)
Filling missing Age values
Encoding categorical variables
Feature scaling using StandardScaler

🧠 Model Details
Neural Network Architecture:
Dense Layer (32 neurons, ReLU)
Dense Layer (22 neurons, ReLU)
Output Layer (1 neuron, Sigmoid)
Loss Function: Binary Crossentropy
Optimizer: Adam
Evaluation Metric: Accuracy

🚀 How to Run the Project
1. Install Dependencies
pip install streamlit pandas plotly scikit-learn tensorflow
2. Run the Streamlit App
streamlit run titanic_app.py
3. Open in Browser
Streamlit will automatically open in your browser (usually at http://localhost:8501)

🎯 How It Works
User inputs passenger details via sidebar
Data is preprocessed and scaled
Model predicts survival probability
Result is displayed instantly
Visual charts provide insights into dataset patterns

📈 Future Improvements
Save and load trained model instead of retraining every run
Add more advanced models (Random Forest, XGBoost)
Improve UI/UX design (better styling, themes)
Add probability score instead of just binary output
Deploy on cloud (Streamlit Cloud / AWS / Render)

💡 Learning Outcomes
This project demonstrates:
End-to-end ML pipeline
Model building with neural networks
Data preprocessing techniques
Deploying ML models using Streamlit
Creating interactive dashboards
