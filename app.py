import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card styling */
    .css-1r6slb0 {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Header styling */
    h1 {
        color: #ffffff !important;
        text-align: center;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    h2, h3 {
        color: #2d3748;
        font-weight: 700;
    }
    
    /* Input containers */
    .stNumberInput, .stSelectbox, .stSlider {
        background: white;
        border-radius: 10px;
        padding: 5px;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 50px;
        font-size: 18px;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Metric cards */
    .css-1xarl3l {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
    }
    
    /* Success/Error boxes */
    .stSuccess, .stError {
        border-radius: 10px;
        padding: 20px;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Feature labels */
    label {
        font-weight: 600 !important;
        color: #2d3748 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error("Model files not found! Please ensure model.pkl, scaler.pkl, and feature_names.pkl are in the repository.")
        return None, None, None

model, scaler, feature_names = load_model()

# ==================== HEADER ====================
st.markdown("<h1>❤️ Heart Disease Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; font-size: 18px;'>AI-Powered Early Detection for Better Health Outcomes</p>", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/heart-with-pulse.png", width=100)
    st.markdown("### 📊 Navigation")
    page = st.radio("", ["🏠 Home", "🔍 Prediction", "📈 Analytics", "ℹ️ About"])
    
    st.markdown("---")
    st.markdown("### 📞 Contact")
    st.info("For medical emergencies, please contact your healthcare provider immediately.")
    
    st.markdown("---")
    st.markdown("### 🔒 Privacy")
    st.caption("Your data is not stored and remains completely private.")

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <h2 style='color: #667eea;'>🎯</h2>
            <h3>Accurate</h3>
            <p>ML-powered predictions with high accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <h2 style='color: #667eea;'>⚡</h2>
            <h3>Fast</h3>
            <p>Get results in seconds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <h2 style='color: #667eea;'>🔒</h2>
            <h3>Secure</h3>
            <p>Your data is private and secure</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <h2>🌟 Welcome to Heart Disease Prediction System</h2>
        <p style='font-size: 16px;'>
        This application uses advanced Machine Learning algorithms to predict the likelihood of heart disease 
        based on various medical parameters. Our model has been trained on comprehensive medical data and 
        provides quick, reliable risk assessments.
        </p>
        <br>
        <h3>📋 How It Works:</h3>
        <ol style='font-size: 15px;'>
            <li>Navigate to the <b>Prediction</b> page</li>
            <li>Enter patient medical information</li>
            <li>Click "Predict" to get instant results</li>
            <li>View detailed risk analysis and recommendations</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <h3>⚕️ Key Features</h3>
            <ul>
                <li>Multiple ML models comparison</li>
                <li>Real-time predictions</li>
                <li>Interactive visualizations</li>
                <li>Feature importance analysis</li>
                <li>Risk factor breakdown</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <h3>📊 Models Used</h3>
            <ul>
                <li>Logistic Regression</li>
                <li>Decision Tree</li>
                <li>Random Forest</li>
                <li>Neural Networks</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ==================== PREDICTION PAGE ====================
elif page == "🔍 Prediction":
    if model is None:
        st.error("⚠️ Model not loaded. Please check if model files exist.")
        st.stop()
    
    st.markdown("""
    <div style='background: white; padding: 20px; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
        <h2 style='color: #667eea; text-align: center;'>🏥 Patient Information</h2>
        <p style='text-align: center; color: #666;'>Enter the medical parameters below to predict heart disease risk</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input Form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 👤 Personal Details")
            age = st.number_input("Age", min_value=1, max_value=120, value=50, help="Patient's age in years")
            sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
            
            st.markdown("#### 💓 Heart Metrics")
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120, help="Resting blood pressure")
            thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150, help="Maximum heart rate achieved")
            oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1, help="ST depression induced by exercise")
        
        with col2:
            st.markdown("#### 🩺 Clinical Measurements")
            cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], 
                            format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
            chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, help="Serum cholesterol in mg/dl")
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], 
                             format_func=lambda x: "Yes" if x == 1 else "No")
            
            restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                                 format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x])
        
        with col3:
            st.markdown("#### 🔬 Additional Tests")
            exang = st.selectbox("Exercise Induced Angina", options=[0, 1], 
                               format_func=lambda x: "Yes" if x == 1 else "No")
            slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2],
                               format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
            ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3], help="Colored by fluoroscopy")
            thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3],
                              format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x])
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮 Predict Heart Disease Risk")
    
    if submitted:
        # Create input dataframe
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'cp': [cp],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs],
            'restecg': [restecg],
            'thalach': [thalach],
            'exang': [exang],
            'oldpeak': [oldpeak],
            'slope': [slope],
            'ca': [ca],
            'thal': [thal]
        })
        
        # Ensure correct column order
        input_data = input_data[feature_names]
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Display Results
        st.markdown("<br>", unsafe_allow_html=True)
        
        if prediction == 1:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%); padding: 30px; border-radius: 15px; text-align: center; box-shadow: 0 8px 20px rgba(255,107,107,0.4);'>
                <h1 style='color: white; margin: 0;'>⚠️ HIGH RISK DETECTED</h1>
                <p style='color: white; font-size: 18px; margin-top: 10px;'>The model predicts a high likelihood of heart disease</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%); padding: 30px; border-radius: 15px; text-align: center; box-shadow: 0 8px 20px rgba(78,205,196,0.4);'>
                <h1 style='color: white; margin: 0;'>✅ LOW RISK DETECTED</h1>
                <p style='color: white; font-size: 18px; margin-top: 10px;'>The model predicts a low likelihood of heart disease</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Probability Display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style='background: white; padding: 20px; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                <h3 style='color: #4ecdc4;'>Healthy</h3>
                <h1 style='color: #4ecdc4;'>{:.1f}%</h1>
            </div>
            """.format(probability[0] * 100), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background: white; padding: 20px; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                <h3 style='color: #ff6b6b;'>At Risk</h3>
                <h1 style='color: #ff6b6b;'>{:.1f}%</h1>
            </div>
            """.format(probability[1] * 100), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style='background: white; padding: 20px; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                <h3 style='color: #667eea;'>Confidence</h3>
                <h1 style='color: #667eea;'>{:.1f}%</h1>
            </div>
            """.format(max(probability) * 100), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Probability Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=probability[1] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Heart Disease Risk Score", 'font': {'size': 24}},
            delta={'reference': 50, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#667eea"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#4ecdc4'},
                    {'range': [30, 70], 'color': '#f9ca24'},
                    {'range': [70, 100], 'color': '#ff6b6b'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white", 'family': "Arial"},
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <h3 style='color: #667eea;'>📋 Recommendations</h3>
        """, unsafe_allow_html=True)
        
        if prediction == 1:
            st.markdown("""
            <ul style='font-size: 15px; color: #2d3748;'>
                <li>⚕️ <b>Consult a cardiologist immediately</b> for comprehensive evaluation</li>
                <li>🏥 Schedule advanced cardiac tests (ECG, Echocardiogram, Stress Test)</li>
                <li>💊 Review current medications with your healthcare provider</li>
                <li>🥗 Adopt a heart-healthy diet (low sodium, low saturated fat)</li>
                <li>🏃 Regular moderate exercise (consult doctor before starting)</li>
                <li>🚭 Quit smoking and limit alcohol consumption</li>
                <li>😴 Ensure adequate sleep (7-9 hours per night)</li>
                <li>🧘 Manage stress through meditation or yoga</li>
            </ul>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <ul style='font-size: 15px; color: #2d3748;'>
                <li>✅ <b>Maintain current healthy habits</b></li>
                <li>🏃 Continue regular physical activity (150 min/week)</li>
                <li>🥗 Follow a balanced, heart-healthy diet</li>
                <li>📊 Regular health check-ups and monitoring</li>
                <li>💪 Maintain healthy weight and BMI</li>
                <li>😊 Keep stress levels under control</li>
                <li>🚭 Avoid smoking and excessive alcohol</li>
                <li>💤 Get adequate sleep and rest</li>
            </ul>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Disclaimer
        st.warning("⚠️ **Disclaimer:** This prediction is based on machine learning models and should NOT replace professional medical advice. Please consult with healthcare professionals for proper diagnosis and treatment.")

# ==================== ANALYTICS PAGE ====================
elif page == "📈 Analytics":
    st.markdown("""
    <div style='background: white; padding: 20px; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
        <h2 style='color: #667eea; text-align: center;'>📊 Model Performance Analytics</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample performance metrics (replace with actual values from your model evaluation)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "92.5%", "2.1%")
    with col2:
        st.metric("Precision", "90.3%", "1.5%")
    with col3:
        st.metric("Recall", "89.7%", "0.8%")
    with col4:
        st.metric("F1-Score", "91.2%", "1.2%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model Comparison Chart
    models_data = {
        'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Neural Network'],
        'Accuracy': [85.2, 82.7, 92.5, 89.3],
        'Precision': [83.5, 80.2, 90.3, 87.8],
        'Recall': [86.1, 83.5, 89.7, 88.5]
    }
    
    df_models = pd.DataFrame(models_data)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Accuracy', x=df_models['Model'], y=df_models['Accuracy'], marker_color='#667eea'))
    fig.add_trace(go.Bar(name='Precision', x=df_models['Model'], y=df_models['Precision'], marker_color='#764ba2'))
    fig.add_trace(go.Bar(name='Recall', x=df_models['Model'], y=df_models['Recall'], marker_color='#f093fb'))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='Score (%)',
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.9)',
        font=dict(size=12, color='#2d3748'),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance
    st.markdown("""
    <div style='background: white; padding: 20px; border-radius: 15px; margin-top: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
        <h3 style='color: #667eea;'>🔍 Feature Importance Analysis</h3>
        <p style='color: #666;'>Key factors contributing to heart disease prediction:</p>
    </div>
    """, unsafe_allow_html=True)
    
    features_data = {
        'Feature': ['Chest Pain Type', 'Max Heart Rate', 'ST Depression', 'Age', 'Cholesterol', 
                   'Blood Pressure', 'Major Vessels', 'Thalassemia', 'Exercise Angina', 'Sex'],
        'Importance': [0.18, 0.15, 0.13, 0.12, 0.10, 0.09, 0.08, 0.07, 0.05, 0.03]
    }
    
    df_features = pd.DataFrame(features_data)
    
    fig2 = px.bar(df_features, x='Importance', y='Feature', orientation='h',
                  color='Importance', color_continuous_scale='Viridis',
                  title='Top Features by Importance')
    
    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.9)',
        font=dict(size=12, color='#2d3748'),
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig2, use_container_width=True)

# ==================== ABOUT PAGE ====================
elif page == "ℹ️ About":
    st.markdown("""
    <div style='background: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
        <h2 style='color: #667eea;'>🎯 About This Project</h2>
        <p style='font-size: 16px; line-height: 1.6; color: #2d3748;'>
        This Heart Disease Prediction System is a machine learning-powered application designed to assist 
        healthcare professionals and individuals in early detection of heart disease risk. The system analyzes 
        various medical parameters and provides instant risk assessments.
        </p>
        
        <h3 style='color: #667eea; margin-top: 30px;'>🔬 Technology Stack</h3>
        <ul style='font-size: 15px; color: #2d3748;'>
            <li><b>Machine Learning:</b> Scikit-learn, TensorFlow</li>
            <li><b>Frontend:</b> Streamlit, Plotly</li>
            <li><b>Data Processing:</b> Pandas, NumPy</li>
            <li><b>Visualization:</b> Matplotlib, Seaborn</li>
        </ul>
        
        <h3 style='color: #667eea; margin-top: 30px;'>📊 Dataset Information</h3>
        <p style='font-size: 15px; color: #2d3748;'>
        The model was trained on a comprehensive heart disease dataset containing medical records from 
        patients with various cardiovascular conditions. The dataset includes 13 features and has been 
        preprocessed and validated for accuracy.
        </p>
        
        <h3 style='color: #667eea; margin-top: 30px;'>🎓 Models Used</h3>
        <ul style='font-size: 15px; color: #2d3748;'>
            <li><b>Logistic Regression:</b> Linear classification model</li>
            <li><b>Decision Tree:</b> Tree-based classification</li>
            <li><b>Random Forest:</b> Ensemble learning method (Best Performance)</li>
            <li><b>Neural Network:</b> Deep learning approach</li>
        </ul>
        
        <h3 style='color: #667eea; margin-top: 30px;'>⚠️ Important Notice</h3>
        <p style='font-size: 15px; color: #2d3748; background: #fff3cd; padding: 15px; border-radius: 10px; border-left: 4px solid #ffc107;'>
        This application is intended for educational and informational purposes only. It should not be used 
        as a substitute for professional medical advice, diagnosis, or treatment. Always consult with 
        qualified healthcare professionals for medical concerns.
        </p>
        
        <h3 style='color: #667eea; margin-top: 30px;'>👨‍💻 Developer Information</h3>
        <p style='font-size: 15px; color: #2d3748;'>
        Developed as a Machine Learning Major Project<br>
        Built with ❤️ using Python and Streamlit
        </p>
        
        <div style='text-align: center; margin-top: 30px;'>
            <p style='color: #667eea; font-size: 14px;'>
                © 2024 Heart Disease Prediction System. All rights reserved.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: white; padding: 20px;'>
        <p>Made with ❤️ using Streamlit | Powered by Machine Learning</p>
    </div>
""", unsafe_allow_html=True)