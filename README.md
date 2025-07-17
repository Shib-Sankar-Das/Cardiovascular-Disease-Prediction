# ❤ Cardiovascular Disease Risk Prediction

A machine learning web application built with Streamlit that predicts cardiovascular disease risk based on various health parameters. This project includes comprehensive exploratory data analysis (EDA), model training, and an interactive web deployment.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project develops a machine learning model to predict cardiovascular disease risk using patient health data. The application provides:

- **Interactive Web Interface**: Built with Streamlit for easy user interaction
- **Real-time Predictions**: Instant risk assessment based on user inputs
- **Comprehensive Visualizations**: Risk gauges, feature importance charts, and health recommendations
- **Model Interpretability**: Clear explanations of prediction factors

## ✨ Features

### 🧠 Machine Learning
- **Multiple Model Comparison**: Logistic Regression, Random Forest, Gradient Boosting, SVM
- **Hyperparameter Optimization**: GridSearchCV for best model performance
- **Feature Engineering**: BMI calculation, age conversion, outlier removal
- **Model Serialization**: CloudPickle for robust model saving

### 📊 Data Analysis
- **Exploratory Data Analysis**: Comprehensive visualization of data patterns
- **Statistical Analysis**: Correlation matrices, distribution plots
- **Data Cleaning**: Outlier detection and removal
- **Feature Importance**: Analysis of key risk factors

### 🌐 Web Application
- **User-Friendly Interface**: Intuitive form-based input
- **Real-time Validation**: Input validation and error handling
- **Interactive Visualizations**: Plotly charts for risk assessment
- **Health Recommendations**: Personalized advice based on risk factors

## 📊 Dataset

The project uses the **Cardiovascular Disease Dataset** containing:

- **70,000 patient records**
- **12 features** including age, gender, blood pressure, cholesterol, etc.
- **Binary target variable** (cardiovascular disease: yes/no)

### Features Description:
- `age`: Age in days (converted to years)
- `gender`: Gender (1: female, 2: male)
- `height`: Height in cm
- `weight`: Weight in kg
- `ap_hi`: Systolic blood pressure
- `ap_lo`: Diastolic blood pressure
- `cholesterol`: Cholesterol level (1: normal, 2: above normal, 3: well above normal)
- `gluc`: Glucose level (1: normal, 2: above normal, 3: well above normal)
- `smoke`: Smoking status (0: no, 1: yes)
- `alco`: Alcohol intake (0: no, 1: yes)
- `active`: Physical activity (0: no, 1: yes)
- `cardio`: Target variable (0: no cardiovascular disease, 1: has cardiovascular disease)

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd cardiovascular-disease-prediction
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify Data
Ensure the dataset is in the correct location:
```
Data/
└── cardio_train.csv
```

## 💻 Usage

### Running the Jupyter Notebook
1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open the Analysis Notebook**:
   - Navigate to `cardio_analysis_and_training.ipynb`
   - Run all cells to perform EDA and train models

### Running the Streamlit App
1. **Train Models** (if not already done):
   - Run the Jupyter notebook to generate model files

2. **Launch Streamlit App**:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Access the Application**:
   - Open your browser to `http://localhost:8501`
   - Input health parameters and get instant predictions

### Using the Web Application

1. **Enter Health Information**:
   - Basic info: Age, height, weight, gender
   - Blood pressure: Systolic and diastolic values
   - Medical conditions: Cholesterol, glucose levels
   - Lifestyle factors: Smoking, alcohol, physical activity

2. **Get Predictions**:
   - Click "Predict Risk" to get instant results
   - View risk probability and category
   - Explore interactive visualizations

3. **Understand Results**:
   - Risk gauge showing probability percentage
   - Feature importance chart
   - Personalized health recommendations

## 📁 Project Structure

```
cardiovascular-disease-prediction/
│
├── Data/
│   ├── cardio_train.csv          # Main dataset
│   └── Meta data.txt             # Dataset description
│
├── models/
│   ├── cardio_model_gradient_boosting.pkl  # Trained model
│   ├── scaler.pkl                # Feature scaler
│   ├── feature_columns.pkl       # Feature names
│   └── model_metadata.pkl        # Model performance metrics
│
├── cardio_analysis_and_training.ipynb  # EDA and model training
├── streamlit_app.py              # Web application
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # Project documentation
```

## 📈 Model Performance

### Best Model: Gradient Boosting Classifier

| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.732 |
| Precision | 0.725 |
| Recall    | 0.754 |
| F1-Score  | 0.739 |
| AUC-ROC   | 0.801 |

### Model Comparison

| Model               | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---------------------|----------|-----------|--------|----------|---------|
| Gradient Boosting   | 0.732    | 0.725     | 0.754  | 0.739    | **0.801** |
| Random Forest       | 0.719    | 0.708     | 0.752  | 0.729    | 0.787   |
| Logistic Regression | 0.713    | 0.721     | 0.695  | 0.708    | 0.773   |
| SVM                 | 0.708    | 0.715     | 0.690  | 0.702    | 0.768   |

### Key Insights

1. **Age** is the strongest predictor of cardiovascular disease
2. **Blood Pressure** (both systolic and diastolic) shows strong correlation
3. **BMI** and **cholesterol levels** are significant risk factors
4. **Gender differences** exist in disease prevalence
5. **Lifestyle factors** (smoking, physical activity) impact risk

## 🛠 Technologies Used

### Machine Learning
- **scikit-learn**: Model training and evaluation
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **cloudpickle**: Model serialization

### Visualization
- **matplotlib**: Static plotting
- **seaborn**: Statistical visualizations
- **plotly**: Interactive charts and graphs

### Web Application
- **streamlit**: Web framework
- **plotly**: Interactive web visualizations

### Development
- **jupyter**: Interactive development environment
- **python**: Programming language

## 📸 Screenshots

### Streamlit Web Application
![Application Interface](screenshots/app_interface.png)
*Main application interface with input form and prediction results*

### EDA Visualizations
![Data Analysis](screenshots/eda_charts.png)
*Comprehensive exploratory data analysis charts*

### Model Performance
![Model Comparison](screenshots/model_performance.png)
*Model comparison and performance metrics*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**Important**: This application is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.

## 📞 Contact

- **Project Creator**: [Your Name]
- **Email**: [your.email@example.com]
- **GitHub**: [your-github-username]

## 🙏 Acknowledgments

- Dataset source: [Kaggle - Cardiovascular Disease Dataset]
- Streamlit community for excellent documentation
- scikit-learn contributors for robust ML tools

---

### 🚀 Quick Start Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the analysis notebook
jupyter notebook cardio_analysis_and_training.ipynb

# Launch the web app
streamlit run streamlit_app.py
```

**Happy Predicting! 🫀💚**
