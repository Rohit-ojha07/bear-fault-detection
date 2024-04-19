import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Load the saved model
with open('souvik_1800.pkl', 'rb') as file:
    clf_loaded = pickle.load(file)

# User input for RPM and AXIS
rpm = st.number_input("Enter RPM", value=500, step=100)
axis = st.selectbox("Enter AXIS", options=['Z', 'Y'])

# Dataframe to store RPM and AXIS values
input_df = pd.DataFrame({'RPM ': [rpm], 'AXIS': [axis]})

# Create a Streamlit file uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file and skip the first and third rows
    df = pd.read_csv(uploaded_file, skiprows=[0, 2,3])

    # Rename columns
    df.rename(columns={'X Unit(Hz)': 'FREQUENCY (Hz)', 'Y Unit(mm)': 'AMPLITUDE (mm)'}, inplace=True)

    # Add RPM and AXIS columns from user input
    df['RPM '] = rpm
    df['AXIS'] = axis

    # Create new columns AXIS _Z and AXIS _Y based on AXIS value
    df['AXIS _Z'] = 0
    df['AXIS _Y'] = 0
    if axis == 'Z':
        df['AXIS _Z'] = 1
    elif axis == 'Y':
        df['AXIS _Y'] = 1

    # Drop AXIS column
    df.drop(columns=['AXIS'], inplace=True)

    # Select columns for prediction
    prediction_data = df[['RPM ', 'FREQUENCY (Hz)', 'AMPLITUDE (mm)', 'AXIS _Y', 'AXIS _Z']]

    # Make predictions
    predictions = clf_loaded.predict(prediction_data)

    # Add predictions to the DataFrame
    df['Prediction'] = predictions

    # Display the DataFrame
    st.write(df)

    # Allow the user to download the DataFrame as a CSV file
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='predicted_results.csv',
        mime='text/csv'
    )

    # Display predicted fault for Label=0
    for index, row in df.iterrows():
        if row['Prediction'] == 0:
            st.write(f"Predicted fault at {row['FREQUENCY (Hz)']} Hz at {row['RPM ']} RPM")

    # plt.figure(figsize=(8, 6))
    # y_proba_train = clf.predict_proba(X_train)[:, 1]
    # fpr_train, tpr_train, _ = roc_curve(y_train, y_proba_train)
    # auc_train = roc_auc_score(y_train, y_proba_train)
    # plt.plot(fpr_train, tpr_train, label=f'Training AUC = {auc_train:.2f}')

    # y_proba_test = clf.predict_proba(X_test)[:, 1]
    # fpr_test, tpr_test, _ = roc_curve(y_test, y_proba_test)
    # auc_test = roc_auc_score(y_test, y_proba_test)
    # plt.plot(fpr_test, tpr_test, label=f'Testing AUC = {auc_test:.2f}')

    # plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend()
    # plt.show()

    # st.write(f"PLot")
    # frequencies = df['FREQUENCY (Hz)'].unique()
    # OR_accuracy = clf_loaded.score(prediction_data, predictions)
    # HB_accuracy = clf_loaded.score(prediction_data, predictions)
    # plt.plot(frequencies, [OR_accuracy]*len(frequencies), label='Outer Race (OR) Accuracy')
    # plt.plot(frequencies, [HB_accuracy]*len(frequencies), label='Healthy Bearing (HB) Accuracy')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Classification Accuracy')
    # plt.title('Classification Accuracy for OR and HB')
    # plt.legend()
    # st.pyplot(plt)

    # frequencies = df['FREQUENCY (Hz)'].unique()
    # OR_accuracy = predictions[predictions == 0].shape[0] / predictions.shape[0]
    # HB_accuracy = predictions[predictions == 1].shape[0] / predictions.shape[0]
    # plt.bar(['Outer Race (OR)', 'Healthy Bearing (HB)'], [OR_accuracy, HB_accuracy], color=['blue', 'green'])
    # plt.ylim(0, 1)  # Set y-axis limit from 0 to 1 for accuracy
    # plt.ylabel('Classification Accuracy')
    # plt.title('Classification Accuracy for OR and HB')
    # st.pyplot(plt)
