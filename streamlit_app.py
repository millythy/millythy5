

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score


def main():
    st.title("Brain Stroke Classification Web App")
    st.sidebar.title("Brain Stroke Classification Web App")
    st.markdown("Predict if a person is likely to have a stroke 🧠")
    st.sidebar.markdown("Predict if a person is likely to have a stroke 🧠")


    @st.cache_data(persist=True)
    def load_data():
        # Load the brain stroke dataset
        data = pd.read_csv('brain_stroke.csv')


        # Handle categorical variables encoding
        label = LabelEncoder()
        data['gender'] = label.fit_transform(data['gender'])
        data['ever_married'] = label.fit_transform(data['ever_married'])
        data['work_type'] = label.fit_transform(data['work_type'])
        data['Residence_type'] = label.fit_transform(data['Residence_type'])
        data['smoking_status'] = label.fit_transform(data['smoking_status'])


        return data


    @st.cache_data(persist=True)
    def split(df):
        # Set the target variable (stroke) and features
        y = df['stroke']
        X = df.drop(columns=['stroke'])


        # Split data into training and test sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test


    def plot_metrics(metrics_list, model, x_test, y_test, y_pred):
        st.set_option('deprecation.showPyplotGlobalUse', False)


        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            cm_display = ConfusionMatrixDisplay.from_estimator(
                model, x_test, y_test, display_labels=['No Stroke', 'Stroke'], cmap="Blues"
            )
            cm_display.plot()
            st.pyplot()


    # Load dataset and prepare the data
    df = load_data()


    # Split the data into train and test sets
    x_train, x_test, y_train, y_test = split(df)


    # Sidebar for classifier selection
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))


    # SVM Classifier
    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
        metrics = st.sidebar.multiselect("What metrics to plot?", ['Confusion Matrix'])


        if st.sidebar.button("Classify", key='classify'):
            st.subheader("SVM Classifier Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)


            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)


            # Display metrics
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision.round(2))
            st.write("Recall: ", recall.round(2))


            # Plot metrics
            plot_metrics(metrics, model, x_test, y_test, y_pred)


    # Option to display raw data
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Brain Stroke Dataset")
        st.write(df)


if __name__ == '__main__':
    main()