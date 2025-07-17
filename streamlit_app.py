import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Cardiovascular Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #FF6B6B;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F0F2F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-positive {
        background-color: #FFE6E6;
        border: 2px solid #FF4B4B;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .prediction-negative {
        background-color: #E6F7E6;
        border: 2px solid #4CAF50;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_components():
    """Load the trained model and preprocessing components"""
    try:
        # Load model
        with open('models/cardio_model_gradient_boosting.pkl', 'rb') as f:
            model = cloudpickle.load(f)
        
        # Load scaler
        with open('models/scaler.pkl', 'rb') as f:
            scaler = cloudpickle.load(f)
        
        # Load feature columns
        with open('models/feature_columns.pkl', 'rb') as f:
            feature_columns = cloudpickle.load(f)
        
        # Load metadata
        with open('models/model_metadata.pkl', 'rb') as f:
            metadata = cloudpickle.load(f)
        
        return model, scaler, feature_columns, metadata
    except FileNotFoundError:
        st.error("Model files not found. Please run the training notebook first to generate the models.")
        return None, None, None, None

def calculate_bmi(weight, height):
    """Calculate BMI from weight and height"""
    height_m = height / 100  # Convert cm to meters
    return weight / (height_m ** 2)

def get_risk_category(probability):
    """Categorize risk based on probability"""
    if probability < 0.3:
        return "Low Risk", "#4CAF50"
    elif probability < 0.6:
        return "Moderate Risk", "#FF9800"
    else:
        return "High Risk", "#FF4B4B"

def create_gauge_chart(probability):
    """Create a gauge chart for risk visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Cardiovascular Disease Risk (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_feature_importance_chart(feature_values, feature_names):
    """Create a horizontal bar chart showing feature contributions"""
    # This is a simplified feature importance visualization
    # In a real scenario, you might want to use SHAP values or similar
    
    # Normalize values for visualization
    normalized_values = [(val - 0.5) * 100 if isinstance(val, (int, float)) else val for val in feature_values]
    
    fig = go.Figure(go.Bar(
        x=normalized_values[:6],  # Show top 6 features
        y=feature_names[:6],
        orientation='h',
        marker_color=['red' if x > 0 else 'green' for x in normalized_values[:6]]
    ))
    
    fig.update_layout(
        title="Key Risk Factors (Your Values)",
        xaxis_title="Relative Impact",
        height=300,
        margin=dict(l=100, r=50, t=50, b=50)
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">❤️ Cardiovascular Disease Risk Predictor</h1>', unsafe_allow_html=True)
    
    # Load model components
    model, scaler, feature_columns, metadata = load_model_components()
    
    if model is None:
        st.stop()
    
    # Sidebar for model information
    with st.sidebar:
        st.markdown('<h2 class="sub-header">Model Information</h2>', unsafe_allow_html=True)
        if metadata is not None:
            st.info(f"""
            **Model Type:** {metadata['model_name']}
            
            **Performance Metrics:**
            - Accuracy: {metadata['performance']['accuracy']:.3f}
            - Precision: {metadata['performance']['precision']:.3f}
            - Recall: {metadata['performance']['recall']:.3f}
            - F1-Score: {metadata['performance']['f1_score']:.3f}
            - AUC-ROC: {metadata['performance']['auc']:.3f}
            """)
        else:
            st.warning("Model metadata is not available. Please ensure the model files are present and loaded correctly.")
        
        st.markdown("---")
        st.markdown("### About This App")
        st.markdown("""
        This application predicts the risk of cardiovascular disease based on various health parameters. 
        
        **Important:** This is for educational purposes only and should not replace professional medical advice.
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">Enter Your Health Information</h2>', unsafe_allow_html=True)
        
        # Create input form
        with st.form("health_form"):
            # Basic Information
            st.subheader("Basic Information")
            col_basic1, col_basic2 = st.columns(2)
            
            with col_basic1:
                age_years = st.number_input(
                    "Age (years)", 
                    min_value=18, 
                    max_value=100, 
                    value=45,
                    help="Your current age in years"
                )
                
                height = st.number_input(
                    "Height (cm)", 
                    min_value=120, 
                    max_value=220, 
                    value=170,
                    help="Your height in centimeters"
                )
                
                weight = st.number_input(
                    "Weight (kg)", 
                    min_value=30.0, 
                    max_value=200.0, 
                    value=70.0,
                    help="Your weight in kilograms"
                )
            
            with col_basic2:
                gender = st.selectbox(
                    "Gender", 
                    options=[1, 2], 
                    format_func=lambda x: "Female" if x == 1 else "Male",
                    help="Select your gender"
                )
                
                # Display calculated BMI
                bmi = calculate_bmi(weight, height)
                st.metric("Calculated BMI", f"{bmi:.1f}")
                
                if bmi < 18.5:
                    st.warning("Underweight")
                elif bmi < 25:
                    st.success("Normal weight")
                elif bmi < 30:
                    st.warning("Overweight")
                else:
                    st.error("Obese")
            
            # Blood Pressure
            st.subheader("Blood Pressure")
            col_bp1, col_bp2 = st.columns(2)
            
            with col_bp1:
                ap_hi = st.number_input(
                    "Systolic Blood Pressure (mmHg)", 
                    min_value=70, 
                    max_value=250, 
                    value=120,
                    help="The pressure when your heart beats (top number)"
                )
            
            with col_bp2:
                ap_lo = st.number_input(
                    "Diastolic Blood Pressure (mmHg)", 
                    min_value=40, 
                    max_value=150, 
                    value=80,
                    help="The pressure when your heart rests (bottom number)"
                )
            
            # Validate blood pressure
            if ap_hi <= ap_lo:
                st.error("⚠️ Systolic pressure should be higher than diastolic pressure")
            
            # Medical Conditions
            st.subheader("Medical Conditions")
            col_med1, col_med2 = st.columns(2)
            
            with col_med1:
                cholesterol = st.selectbox(
                    "Cholesterol Level",
                    options=[1, 2, 3],
                    format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x],
                    help="Your cholesterol level category"
                )
                
                gluc = st.selectbox(
                    "Glucose Level",
                    options=[1, 2, 3],
                    format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x],
                    help="Your glucose level category"
                )
            
            with col_med2:
                smoke = st.selectbox(
                    "Smoking Status",
                    options=[0, 1],
                    format_func=lambda x: "Non-smoker" if x == 0 else "Smoker",
                    help="Do you smoke?"
                )
                
                alco = st.selectbox(
                    "Alcohol Consumption",
                    options=[0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes",
                    help="Do you consume alcohol regularly?"
                )
                
                active = st.selectbox(
                    "Physical Activity",
                    options=[0, 1],
                    format_func=lambda x: "Not Active" if x == 0 else "Active",
                    help="Do you engage in regular physical activity?"
                )
            
            # Submit button
            submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)
        
        # Prediction results
        if submitted:
            # Validate inputs
            if ap_hi <= ap_lo:
                st.error("Please correct the blood pressure values before proceeding.")
                return
            
            # Prepare input data
            input_data = pd.DataFrame({
                'age_years': [age_years],
                'gender': [gender],
                'height': [height],
                'weight': [weight],
                'ap_hi': [ap_hi],
                'ap_lo': [ap_lo],
                'cholesterol': [cholesterol],
                'gluc': [gluc],
                'smoke': [smoke],
                'alco': [alco],
                'active': [active],
                'bmi': [bmi]
            })
            
            # Ensure correct column order
            input_data = input_data[feature_columns]
            
            # Make prediction
            if metadata is not None and scaler is not None and metadata.get('requires_scaling', False):
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][1]
            else:
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0][1]
            
            # Display results
            st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)
            
            risk_category, risk_color = get_risk_category(probability)
            
            # Main prediction result
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-positive">
                    <h3>⚠️ High Risk Detected</h3>
                    <p>The model indicates a higher likelihood of cardiovascular disease.</p>
                    <p><strong>Risk Probability: {probability:.1%}</strong></p>
                    <p><strong>Risk Category: {risk_category}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-negative">
                    <h3>✅ Lower Risk Detected</h3>
                    <p>The model indicates a lower likelihood of cardiovascular disease.</p>
                    <p><strong>Risk Probability: {probability:.1%}</strong></p>
                    <p><strong>Risk Category: {risk_category}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualizations
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                st.plotly_chart(create_gauge_chart(probability), use_container_width=True)
            
            with col_viz2:
                # Feature importance chart
                feature_values = input_data.iloc[0].values
                if feature_columns is not None:
                    feature_names = [col.replace('_', ' ').title() for col in feature_columns]
                else:
                    feature_names = ["Feature " + str(i+1) for i in range(len(feature_values))]
                st.plotly_chart(create_feature_importance_chart(feature_values, feature_names), use_container_width=True)
            
            # Recommendations
            st.markdown('<h3 class="sub-header">Health Recommendations</h3>', unsafe_allow_html=True)
            
            recommendations = []
            
            # Blood pressure recommendations
            if ap_hi > 140 or ap_lo > 90:
                recommendations.append("🩺 **High Blood Pressure**: Consult with a healthcare provider about blood pressure management.")
            
            # BMI recommendations
            if bmi >= 30:
                recommendations.append("🏃‍♂️ **Weight Management**: Consider a balanced diet and regular exercise to achieve a healthy weight.")
            elif bmi >= 25:
                recommendations.append("🥗 **Healthy Weight**: Maintain current weight through balanced nutrition and regular activity.")
            
            # Lifestyle recommendations
            if smoke == 1:
                recommendations.append("🚭 **Smoking Cessation**: Quitting smoking is one of the best things you can do for your heart health.")
            
            if active == 0:
                recommendations.append("💪 **Physical Activity**: Aim for at least 150 minutes of moderate exercise per week.")
            
            if cholesterol >= 2:
                recommendations.append("🥑 **Cholesterol Management**: Consider a heart-healthy diet low in saturated fats.")
            
            if gluc >= 2:
                recommendations.append("🍎 **Blood Sugar**: Monitor your blood glucose levels and maintain a balanced diet.")
            
            # General recommendations
            recommendations.extend([
                "👨‍⚕️ **Regular Check-ups**: Schedule regular health screenings with your healthcare provider.",
                "😴 **Quality Sleep**: Aim for 7-9 hours of quality sleep per night.",
                "🧘‍♀️ **Stress Management**: Practice stress-reduction techniques like meditation or yoga.",
                "🥬 **Heart-Healthy Diet**: Focus on fruits, vegetables, whole grains, and lean proteins."
            ])
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
            
            # Disclaimer
            st.warning("""
            **Important Disclaimer:** This prediction is based on a machine learning model and is for educational purposes only. 
            It should not be used as a substitute for professional medical advice, diagnosis, or treatment. 
            Always consult with qualified healthcare providers for medical decisions.
            """)
    
    with col2:
        st.markdown('<h2 class="sub-header">Understanding the Features</h2>', unsafe_allow_html=True)
        
        # Feature explanations
        feature_explanations = {
            "Age": "Cardiovascular disease risk increases with age. Most cases occur in people over 45.",
            "Gender": "Men generally have higher risk of heart disease, especially at younger ages.",
            "Blood Pressure": "High blood pressure (hypertension) is a major risk factor for heart disease.",
            "BMI": "Body Mass Index indicates if you're at a healthy weight. Higher BMI increases heart disease risk.",
            "Cholesterol": "High cholesterol levels can lead to plaque buildup in arteries.",
            "Glucose": "High blood glucose levels may indicate diabetes, which increases heart disease risk.",
            "Smoking": "Smoking damages blood vessels and significantly increases cardiovascular risk.",
            "Alcohol": "Excessive alcohol consumption can contribute to high blood pressure and heart problems.",
            "Physical Activity": "Regular exercise strengthens the heart and reduces disease risk."
        }
        
        for feature, explanation in feature_explanations.items():
            with st.expander(f"📖 {feature}"):
                st.write(explanation)
        
        # Risk factors summary
        st.markdown("### Common Risk Factors")
        st.markdown("""
        **Modifiable Factors:**
        - High blood pressure
        - High cholesterol
        - Smoking
        - Diabetes
        - Obesity
        - Physical inactivity
        - Poor diet
        - Excessive alcohol use
        
        **Non-Modifiable Factors:**
        - Age
        - Gender
        - Family history
        - Race/ethnicity
        """)

if __name__ == "__main__":
    main()
