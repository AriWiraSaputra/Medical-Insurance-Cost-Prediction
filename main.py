import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# Load data
df = pd.read_csv('insurance.csv')

def app(df):
    # Set page configuration
    st.set_page_config(page_title='Insurance Cost Prediction', page_icon=':money_with_wings:', layout='wide')

    # Display dataset
    st.subheader('Insurance Dataset')
    st.dataframe(df)

    # Preprocess data
    df.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
    df.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
    df.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)
    X = df.drop(columns="charges", axis=1)
    y = df['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    # Prediction input form
    st.sidebar.title("Insurance Cost Prediction")
    st.sidebar.subheader("Input Data")
    age = st.sidebar.number_input('Age', min_value=1, max_value=100, value=30)
    sex = st.sidebar.selectbox('Sex', ['male', 'female'])
    bmi = st.sidebar.slider('BMI', min_value=10, max_value=50, value=25, step=1)
    children = st.sidebar.number_input('Number of Children', min_value=0, max_value=10, value=0)
    smoker = st.sidebar.selectbox('Smoker', ['yes', 'no'])
    region = st.sidebar.selectbox('Region', ['southeast', 'southwest', 'northeast', 'northwest'])

    # Convert user input into a DataFrame
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })

    # Preprocess user input
    input_data.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
    input_data.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
    input_data.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2,'northwest': 3}}, inplace=True)
    
    # Generate prediction based on user input
    prediction = reg.predict(input_data)

    # Add user data to the main DataFrame
    df = df.append(input_data, ignore_index=True)

    # Display user data
    st.subheader('Patient Data')
    st.dataframe(input_data)

    # Display updated dataset
    st.subheader('Updated Dataset')
    st.dataframe(df)

    # Display prediction
    st.subheader('Insurance Cost Prediction')
    st.write(f'The predicted insurance cost is ${round(prediction[0], 2)}.')

    # Calculate R-squared score on test data
    test_data_prediction = reg.predict(X_test)
    r2_score = metrics.r2_score(y_test, test_data_prediction)

    # Display R-squared score
    st.subheader('R-squared Score (Test Data)')
    st.write(r2_score)

    # Age Distribution (Scatter Plot)
    plt.figure(figsize=(6, 3))
    sns.scatterplot(x='age', y='bmi', data=df, color='blue', alpha=0.5)
    sns.scatterplot(x='age', y='bmi', data=input_data, color='red', marker='o')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('BMI')
    st.subheader('Age Distribution')
    st.pyplot(plt)

    # Sex Distribution
    plt.figure(figsize=(6, 3))
    sns.countplot(x="sex", data=df)
    plt.title("Sex Distribution")
    plt.xlabel("Sex")
    plt.ylabel("Count")
    st.subheader("Sex Distribution")
    st.pyplot(plt)

    # BMI Distribution
    plt.figure(figsize=(6, 3))
    sns.distplot(df['bmi'])
    plt.title("BMI Distribution")
    plt.xlabel("BMI")
    plt.ylabel("Density")
    st.subheader("BMI Distribution")
    st.pyplot(plt)



    # Run the app
if __name__=='__main__':
    app(df)
