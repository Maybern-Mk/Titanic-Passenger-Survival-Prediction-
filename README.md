# Titanic Survival Prediction

## Overview  
This project is a machine learning web application that predicts whether a passenger would have survived the Titanic disaster based on key attributes such as age, gender, ticket class, and fare.  

Developed using Python, TensorFlow/Keras, and Streamlit, the application demonstrates a complete end-to-end machine learning workflow, including data preprocessing, model development, and deployment through an interactive user interface.

---

## Features  
- Real-time prediction of passenger survival  
- Interactive data visualizations, including **age** and **fare** distributions  
- Neural network model implemented using **TensorFlow/Keras**  
- Fast and responsive user interface built with **Streamlit**  
- Comprehensive data preprocessing, including handling missing values, encoding, and scaling  

---

## Project Structure  
├── Titanic Passenger Survival Prediction.ipynb # Data analysis and model experimentation
├── titanic_app.py # Streamlit web application
├── Titanic-Dataset.csv # Dataset used for training
├── README.md # Project documentation

---

## Technology Stack  
- Frontend/UI: **Streamlit**  
- Backend: **Python**  
- Machine Learning: **TensorFlow / Keras**  
- Data Processing: **Pandas**, **Scikit-learn**  
- Visualization: **Plotly**  

---

## Dataset  
The dataset includes passenger details such as:  
- Pclass (Ticket class)  
- Sex  
- Age  
- SibSp (Siblings/Spouses aboard)  
- Parch (Parents/Children aboard)  
- Fare  
- Embarked  
- Survival status  

### Data Preprocessing  
- Removal of irrelevant columns (Name, Ticket, Cabin, PassengerId)  
- Handling missing values (e.g., filling missing Age values)  
- Encoding categorical variables  
- Feature scaling using **StandardScaler**  

---

## Model Details  

### Neural Network Architecture  
- Dense layer with 32 neurons and ReLU activation  
- Dense layer with 22 neurons and ReLU activation  
- Output layer with 1 neuron and Sigmoid activation  

### Training Configuration  
- Loss function: **Binary Crossentropy**  
- Optimizer: **Adam**  
- Evaluation metric: **Accuracy**

### 2. Run the Application  
streamlit run titanic_app.py


---

### 3. Access in Browser  
The application will automatically open in your default browser, typically at:  

---

## How It Works  
- The user inputs passenger details through the sidebar  
- Input data is preprocessed and scaled  
- The model predicts survival probability  
- Results are displayed instantly  
- Visualizations provide insights into dataset patterns  

---

## Future Improvements  
- Save and load the trained model instead of retraining on each run  
- Integrate additional models such as Random Forest or XGBoost  
- Enhance UI/UX with improved styling and themes  
- Display survival probability instead of only binary output  
- Deploy the application on cloud platforms such as Streamlit Cloud, AWS, or Render  

---

## Learning Outcomes  
This project demonstrates:  
- End-to-end machine learning pipeline development  
- Neural network model building  
- Data preprocessing techniques  
- Deployment of ML models using Streamlit  
- Creation of interactive data dashboards  


### 1. Install Dependencies  
