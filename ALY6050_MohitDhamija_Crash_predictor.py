import streamlit as st
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'crash_data_with_severity_category.csv'
df = pd.read_csv(file_path)

# Selecting only the specified variables
selected_features = ['Active School Zone Flag', 'Construction Zone Flag', 'Crash Time', 'Day of Week',
                     'Roadway Part', 'Speed Limit', 'Surface Condition', 'Person Helmet', 'Severity Category']
df_selected = df[selected_features]

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df_selected, columns=['Active School Zone Flag', 'Construction Zone Flag',
                                                  'Day of Week', 'Roadway Part', 'Surface Condition', 'Person Helmet'])

# Split the data into features (X) and target variable (y)
X = df_encoded.drop('Severity Category', axis=1)
y = df_encoded['Severity Category']

# Perform SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate model accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Set up the layout for the title and introduction text
st.title('Crash Severity Prediction')
st.sidebar.title('User Input')

# Introduction Text
st.sidebar.markdown("""
This app predicts the severity category of a crash based on various factors such as the presence of active school zones, construction zones, crash time, day of the week, roadway part, speed limit, surface condition, and whether the person was wearing a helmet.

Please select the relevant options from the sidebar and click the "Predict" button to see the predicted severity category.
""")
# Sidebar for user input
st.sidebar.title('User Input')

# Active School Zone Flag
with st.sidebar.expander('Active School Zone Flag', expanded=True):
    st.image('https://github.com/dhamijam/ALY6050/raw/main/school.png', width=50)
    st.markdown('Select whether you will be riding through active school zone.')
    active_school_zone_key = 'active_school_zone'
    active_school_zone = st.selectbox('', df['Active School Zone Flag'].unique(), key=active_school_zone_key)

# Construction Zone Flag
with st.sidebar.expander('Construction Zone Flag', expanded=True):
    st.image('https://github.com/dhamijam/ALY6050/raw/main/construction.png', width=50)
    st.markdown('Select whether you will be riding through construction zone.')
    construction_zone_key = 'construction_zone'
    construction_zone = st.selectbox('', df['Construction Zone Flag'].unique(), key=construction_zone_key)

# Crash Time
with st.sidebar.expander('Crash Time', expanded=True):
    st.image('https://github.com/dhamijam/ALY6050/raw/main/clock.png', width=50)
    st.markdown('Select the time of day when you will ride (in military time).')
    crash_time_key = 'crash_time'
    crash_time = st.slider('', 0, 2400, step=100, key=crash_time_key)

# Day of Week
with st.sidebar.expander('Day of Week', expanded=True):
    st.image('https://github.com/dhamijam/ALY6050/raw/main/calendar.png', width=50)
    st.markdown('Select the day of the week when you will ride.')
    day_of_week_key = 'day_of_week'
    day_of_week = st.selectbox('', df['Day of Week'].unique(), key=day_of_week_key)

# Roadway Part
with st.sidebar.expander('Roadway Part', expanded=True):
    st.image('https://github.com/dhamijam/ALY6050/raw/main/highway.png', width=50)
    st.markdown('Select the part of the roadway where you will ride.')
    roadway_part_key = 'roadway_part'
    roadway_part = st.selectbox('', df['Roadway Part'].unique(), key=roadway_part_key)

# Speed Limit
with st.sidebar.expander('Speed Limit', expanded=True):
    st.image('https://github.com/dhamijam/ALY6050/raw/main/speed.png', width=50)
    st.markdown('Select the speed limit on the roadway where you will ride.')
    speed_limit_key = 'speed_limit'
    speed_limit = st.slider('', min(df['Speed Limit']), max(df['Speed Limit']), step=5, key=speed_limit_key)

# Surface Condition
with st.sidebar.expander('Surface Condition', expanded=True):
    st.image('https://github.com/dhamijam/ALY6050/raw/main/snow.png', width=50)
    st.markdown('Select the surface condition of the roadway where you will ride.')
    surface_condition_key = 'surface_condition'
    surface_condition = st.selectbox('', df['Surface Condition'].unique(), key=surface_condition_key)

# Person Helmet
with st.sidebar.expander('Person Helmet', expanded=True):
    st.image('https://github.com/dhamijam/ALY6050/raw/main/helmet.png', width=50)
    st.markdown('Select whether you will be wearing a helmet.')
    person_helmet_key = 'person_helmet'
    person_helmet = st.selectbox('', df['Person Helmet'].unique(), key=person_helmet_key)

# Function to preprocess user input and make prediction
def preprocess_input(active_school_zone, construction_zone, crash_time, day_of_week, roadway_part,
                     speed_limit, surface_condition, person_helmet):
    input_data = {'Active School Zone Flag': [active_school_zone],
                  'Construction Zone Flag': [construction_zone],
                  'Crash Time': [crash_time],
                  'Day of Week': [day_of_week],
                  'Roadway Part': [roadway_part],
                  'Speed Limit': [speed_limit],
                  'Surface Condition': [surface_condition],
                  'Person Helmet': [person_helmet]}
    input_df = pd.DataFrame(input_data)

    # One-hot encode categorical variables
    input_encoded = pd.get_dummies(input_df, columns=['Active School Zone Flag', 'Construction Zone Flag',
                                                      'Day of Week', 'Roadway Part', 'Surface Condition', 'Person Helmet'])
    # Align columns with original dataset
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)
    
    return input_encoded

# Predict severity category
input_features = preprocess_input(active_school_zone, construction_zone, crash_time, day_of_week,
                                  roadway_part, speed_limit, surface_condition, person_helmet)

# Predict function
def predict(input_features):
    prediction = clf.predict(input_features)
    return prediction

def display_severity_icon(severity_category):
    if severity_category == 'Low':
        st.image('https://github.com/dhamijam/ALY6050/raw/main/green.png', caption='Low Severity', use_column_width=150)
    elif severity_category == 'Moderate':
        st.image('https://github.com/dhamijam/ALY6050/raw/main/caution.png', caption='Moderate Severity', use_column_width=150)
    elif severity_category == 'High':
        st.image('https://github.com/dhamijam/ALY6050/raw/main/warning.png', caption='High Severity', use_column_width=150)
    else:
        st.write("No icon available for the predicted severity category.")

# Display model accuracy when prediction is made
if st.sidebar.button('Predict', key='predict_button'):
    prediction = predict(input_features)
    
    # Determine font color based on severity category
    if prediction[0] == 'Low':
        font_color = 'green'
        prediction_text = 'There is a low chance of injury'
    elif prediction[0] == 'Moderate':
        font_color = 'yellow'
        prediction_text = 'There is a moderate chance of injury'
    elif prediction[0] == 'High':
        font_color = 'red'
        prediction_text = 'There is a high chance of injury'
    else:
        font_color = 'black'  # Default font color
    
    # Introduction Text with spacing
    st.markdown("""
    &nbsp;
    
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align: center;'>Predicted Severity Category</h3>", unsafe_allow_html=True)
    
    # Increase the font size and change color based on severity category for predicted severity category
    st.write("<span style='font-size:60px; color:{};'>{}</span>".format(font_color, prediction_text), unsafe_allow_html=True)
    
    # Display model accuracy with increased font size and spacing
    st.write("<span style='font-size:30px;'>Model Accuracy: {}</span>".format(round(accuracy, 2)), unsafe_allow_html=True)
    
    # Display prediction icon based on severity category
    display_severity_icon(prediction[0])
    
    # Insert the code to display the three most important features here
    # Get feature importances from the trained model
    feature_importances = clf.feature_importances_

    # Create a dictionary to map feature names to their importance scores
    feature_importance_dict = dict(zip(X.columns, feature_importances))

    # Sort the dictionary by importance scores in descending order
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    # Extract the names and importance scores of the top 3 most important features
    top_features = [feature[0] for feature in sorted_features[:3]]
    top_feature_importance = [feature[1] for feature in sorted_features[:3]]

    # Plot a bar graph to visualize the importance of the top 3 variables
    plt.figure(figsize=(6, 3))
    bars = plt.bar(top_features, top_feature_importance, color='red')
    plt.xlabel('Features', color='white')  # Setting color for x-axis label
    plt.ylabel('Importance', color='white')  # Setting color for y-axis label
    plt.title('Top 3 Most Important Features', color='white')  # Setting color for title
    plt.xticks(rotation=45, color='white')  # Setting color for x-axis ticks
    plt.yticks(color='white')  # Setting color for y-axis ticks
    plt.gca().set_facecolor('black')
    plt.gcf().set_facecolor('black')
    plt.gcf().subplots_adjust(bottom=0.25)
    
    # Set visibility of grid lines for x-axis and y-axis
    plt.grid(axis='x', color='white')
    plt.grid(axis='y', color='white')
    plt.grid(False)
    
    # Adding border color to the bars
    for bar in bars:
        bar.set_edgecolor('white')

    st.pyplot(plt)

    
    # Function to display the three most important features
    def display_top_features(top_features):
        st.markdown("<h3 style='text-align: center;'>Top 3 Most Important Features</h3>", unsafe_allow_html=True)
        for feature in top_features:
            st.write(feature)

    # Display the three most important features
    display_top_features(top_features)

    # Debugging statement to check prediction value
    print("Prediction:", prediction)
