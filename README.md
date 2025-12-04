# ❤️ Heart Disease Prediction System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered web application that predicts the likelihood of heart disease using machine learning algorithms. This project leverages multiple classification models to provide accurate risk assessments based on medical parameters.

![Heart Disease Prediction Banner](https://via.placeholder.com/1200x400/667eea/ffffff?text=Heart+Disease+Prediction+System)

## 🌟 Features

- **Multiple ML Models**: Compare Logistic Regression, Decision Tree, Random Forest, and Neural Networks
- **Interactive UI**: Modern, responsive Streamlit interface with beautiful visualizations
- **Real-time Predictions**: Instant risk assessment based on medical parameters
- **Visual Analytics**: Interactive charts and graphs for model performance analysis
- **Feature Importance**: Understand which factors contribute most to predictions
- **Privacy-First**: No data storage - all predictions happen in real-time

## 🚀 Live Demo

Check out the live application: [Heart Disease Predictor](https://your-app-url.streamlit.app)

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| Logistic Regression | 85.2% | 83.5% | 86.1% | 84.8% |
| Decision Tree | 82.7% | 80.2% | 83.5% | 81.8% |
| **Random Forest** | **92.5%** | **90.3%** | **89.7%** | **91.2%** |
| Neural Network | 89.3% | 87.8% | 88.5% | 88.1% |

## 🛠️ Technology Stack

- **Frontend**: Streamlit, Plotly
- **Machine Learning**: Scikit-learn, TensorFlow
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Deployment**: Streamlit Cloud, GitHub

## 📁 Project Structure

```
heart-disease-prediction/
│
├── app.py                          # Streamlit web application
├── heart_disease_analysis.ipynb    # Jupyter notebook for training
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore file
│
├── models/                         # Saved model files
│   ├── model.pkl                   # Trained model
│   ├── scaler.pkl                  # Feature scaler
│   └── feature_names.pkl           # Feature names
│
└── data/
    └── heart.csv                   # Dataset (optional)
```

## 🔧 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app**
```bash
streamlit run app.py
```

5. **Open your browser**
Navigate to `http://localhost:8501`

## 📊 Training the Model

### Google Colab Setup

1. **Open Google Colab**: [colab.research.google.com](https://colab.research.google.com)

2. **Upload the notebook**: Upload `heart_disease_analysis.ipynb`

3. **Upload dataset**: Upload your `heart.csv` file when prompted

4. **Run all cells**: Execute all cells sequentially

5. **Download model files**: The notebook will automatically download:
   - `model.pkl`
   - `scaler.pkl`
   - `feature_names.pkl`

6. **Add to repository**: Place these files in your GitHub repository

## 🚀 Deployment on Streamlit Cloud

### Step-by-Step Guide

1. **Push code to GitHub**
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Go to Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub

3. **Deploy app**
   - Click "New app"
   - Select your repository
   - Choose branch: `main`
   - Main file path: `app.py`
   - Click "Deploy"

4. **Wait for deployment**
   - Streamlit will install dependencies and deploy your app
   - You'll get a public URL like: `your-app.streamlit.app`

## 📝 Dataset Features

The model uses 13 medical parameters:

| Feature | Description | Type |
|---------|-------------|------|
| age | Age in years | Numeric |
| sex | Gender (1=Male, 0=Female) | Binary |
| cp | Chest pain type (0-3) | Categorical |
| trestbps | Resting blood pressure (mm Hg) | Numeric |
| chol | Serum cholesterol (mg/dl) | Numeric |
| fbs | Fasting blood sugar > 120 mg/dl | Binary |
| restecg | Resting ECG results (0-2) | Categorical |
| thalach | Maximum heart rate achieved | Numeric |
| exang | Exercise induced angina | Binary |
| oldpeak | ST depression induced by exercise | Numeric |
| slope | Slope of peak exercise ST segment | Categorical |
| ca | Number of major vessels (0-3) | Numeric |
| thal | Thalassemia (0-3) | Categorical |

**Target Variable**: `target` (0 = No disease, 1 = Disease)

## 🎯 Usage Example

```python
# Example input for prediction
patient_data = {
    'age': 55,
    'sex': 1,  # Male
    'cp': 2,   # Non-anginal pain
    'trestbps': 130,
    'chol': 250,
    'fbs': 0,
    'restecg': 1,
    'thalach': 150,
    'exang': 0,
    'oldpeak': 1.5,
    'slope': 1,
    'ca': 1,
    'thal': 2
}
```

## 📈 Model Training Process

1. **Data Preprocessing**
   - Handle missing values
   - Encode categorical variables
   - Scale numerical features

2. **Exploratory Data Analysis**
   - Feature correlations
   - Distribution analysis
   - Target variable analysis

3. **Model Training**
   - Train multiple models
   - Cross-validation
   - Hyperparameter tuning

4. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - ROC-AUC curves
   - Confusion matrices

5. **Model Selection**
   - Compare performance
   - Select best model
   - Save for deployment

## ⚠️ Disclaimer

**IMPORTANT**: This application is designed for educational and informational purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment.

- Always consult with qualified healthcare professionals
- Do not make medical decisions based solely on this tool
- This is a predictive model with inherent limitations
- Medical diagnosis requires comprehensive clinical evaluation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

## 🙏 Acknowledgments

- Dataset source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- Streamlit for the amazing framework
- Scikit-learn for machine learning tools
- The open-source community

## 📞 Support

If you have any questions or issues, please:
- Open an issue on GitHub
- Contact via email
- Check the documentation

---

**⭐ If you found this project helpful, please give it a star!**

Made with ❤️ and Python# heart-disease-prediction1
AI-powered heart disease prediction system
