import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Load the saved model
with open('souvik.pkl', 'rb') as file:
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

    st.write(f"PLot")
    frequencies = df['FREQUENCY (Hz)'].unique()
    OR_accuracy = clf_loaded.score(prediction_data, predictions)
    HB_accuracy = clf_loaded.score(prediction_data, predictions)
    plt.plot(frequencies, [OR_accuracy]*len(frequencies), label='Outer Race (OR) Accuracy')
    plt.plot(frequencies, [HB_accuracy]*len(frequencies), label='Healthy Bearing (HB) Accuracy')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Classification Accuracy')
    plt.title('Classification Accuracy for OR and HB')
    plt.legend()
    st.pyplot(plt)
