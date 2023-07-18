
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OrdinalEncoder
import re
import numpy as np

# Function to perform prediction on the DataFrame
def perform_prediction(data):
    model = joblib.load("model.pkl")
    # Perform your prediction logic here
    # Replace this dummy example with your actual prediction code
    predictions = model.predict(data)

    return predictions

def to_ordinal(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    ordinal_encoder = OrdinalEncoder()
    df[categorical_columns] = ordinal_encoder.fit_transform(df[categorical_columns])
    return df

def preprocess(df):
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in df.columns.values]
    return df

def mapping(pred):
    value_mapping = {1: 'No Failure', 2: 'Power Failure', 3: 'Tool Wear Failure', 4: 'Overstrain Failure', 5: 'Random Failures', 6: 'Heat Dissipation Failure'}
    pred_int = pred.astype(int)
    mapper = np.vectorize(lambda x: value_mapping.get(x, 'Unknown'))
    # Apply the mapping to the input array
    return mapper(pred_int)
def display_home_content():
    st.header("Introducing MechAlert")
    st.write("""
    MechAlert is a tool that performs predictive maintenance
      classification for machinery. Users can upload a CSV file
        or manually input data to analyze and classify maintenance
        requirements. By leveraging machine learning algorithms, 
        MechAssist helps users make informed decisions about preventive
        maintenance, reducing equipment failures and improving efficiency.
    """)
    
    st.markdown("---")
    
    st.header("Steps to Use the App")
    
    st.subheader("Step 1: Choose Input Type")
    st.write("Select the input type from the sidebar on the left:")
    st.write("- **Upload CSV File:** Upload a CSV file containing the features related to the scenario you want to predict the failure type for.")
    st.write("- **Simple Prediction:** Enter the required information manually in the provided input fields.")
    
    st.subheader("Step 2: Perform Prediction")
    st.write("Click the 'Predict' button to initiate the prediction process. The app will analyze the provided data and predict the possible failure types.")
    
    st.subheader("Step 3: View Results")
    st.write("Once the prediction is complete, you can view the predicted failure types on the screen.")
    
    st.markdown("---")

# Streamlit app
def main():
    st.set_page_config(layout="wide")
    # Logo and Title Side by Side
    logo_col, title_col = st.columns(2)
    with logo_col:
        st.image("logo.png", width=250)
    with title_col:
        st.markdown("<br>", unsafe_allow_html=True)
        st.title("MechAlert")
        text = "<span style='color: gray;font-size: 20px;'>A Simple Predictive Maintenance App</span>"
        st.markdown(text, unsafe_allow_html=True)
    st.markdown("---")
    #st.sidebar.button("Home")
    if st.sidebar.button("Home"):
        display_home_content()   
    # # Display a sidebar to choose input type
    input_type = st.sidebar.radio("Input Type", options=["Unselected","Upload CSV File", "Simple Prediction"])
    
    if input_type == "Upload CSV File":
        # Upload CSV file
        file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if file is not None:
            st.write("File uploaded successfully.")
            
            # Read CSV file into DataFrame
            data = pd.read_csv(file)
            
            # Display the DataFrame and predictions side by side
            columns_to_drop = ['UDI', 'Product ID', 'Target', 'Failure Type']
            df = data.drop(columns_to_drop, axis=1)
            df_init = data.drop(['Target', 'Failure Type'],axis=1)          
            st.subheader("Input DataFrame")
            st.write(df_init)
            if st.button("Predict"):
            # Display the predictions in the second column   
                st.subheader("Predictions")
                df1 = to_ordinal(df)
                df1 = preprocess(df1)
                predictions = perform_prediction(df1)
                final = mapping(predictions)
                st.write(pd.concat([df_init[['UDI','Product ID']], pd.DataFrame(final, columns=['Failure Type Prediction'])], axis=1))
    elif input_type == "Unselected":
        display_home_content()   
    elif input_type == "Simple Prediction":
        #st.header('Plant Features')
        col1, col2 = st.columns(2)
        with col1:
            st.text('Identifiers')
            udi = st.text_input('UDI')
            ID= st.text_input('Product ID')
            type = st.radio('Type', ('M','L','H'))
        with col2:
            st.text('Measurements')
            air_temp = st.number_input('Air temperature [K]')
            proc_temp = st.number_input('Process temperature _K_')
            rot_speed = st.number_input('Rotational speed _rpm_')
            torque = st.number_input('Torque _Nm_')
            tool_wear = st.slider('Tool wear _min_', 0, 100, 1)
   
        if st.button("Predict"):
            Data = [{  "Type": type,
                'Air temperature _K_':float(air_temp)	,'Process temperature _K_':float(proc_temp) ,'Rotational speed _rpm_':float(rot_speed),	
                'Torque _Nm_':float(torque),	'Tool wear _min_':int(tool_wear)
              }]
            df = pd.DataFrame(Data)
    
            # Display the filled table and perform predictions
            df1 = to_ordinal(df)
            df1 = preprocess(df1)
            predictions = perform_prediction(df1)
            final = mapping(predictions)
            # st.subheader("Input DataFrame")
            # st.write(df)
            st.subheader("Prediction")
            st.write(final[0])

if __name__ == "__main__":
    main()
