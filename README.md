# 🌧️ Rainfall Prediction of Kannauj District using LSTM 🌾

## 🌟 Project Overview
This project predicts rainfall for the Kannauj district using a **Long Short-Term Memory (LSTM)** neural network.  
With a custom-collected dataset and advanced techniques like **5-fold cross-validation** and **confusion matrix visualization**, this model aims to provide accurate rainfall forecasts to assist in agricultural and water resource management.

---

## 📊 Dataset
- **Type**: Custom-collected data.  
- **Format**: CSV file (`rain_data.csv`).  
- **Features**: Time-series data including rainfall and other influential factors.  
- **Preprocessing**: Includes normalization/scaling and handling missing values.  

---

## 🤖 Model Details

### 🏗️ Architecture
- **Input Layer**: Time-series data sequences (`sequence_length`).  
- **Hidden Layers**:  
  - LSTM layer with **64 units** and sigmoid activation.  
  - LSTM layer with **32 units** and sigmoid activation.  
  - LSTM layer with **16 units** and sigmoid activation.  
  - LSTM layer with **8 units** and tanh activation.  
- **Output Layer**: Dense layer with **1 unit** for rainfall prediction.  

### ⚙️ Compilation
- **Optimizer**: Adam  
- **Loss Function**: Mean Squared Error (MSE)  
- **Metrics**: Accuracy  

---

## 📈 Results and Evaluation
- **Cross-Validation**: Performed **5-fold cross-validation** for robust model evaluation.  
- **Metrics**:  
  - Mean Squared Error (MSE)  
  - Confusion Matrix Visualization  

---

## 🛠️ Prerequisites
Ensure you have the following installed:  
- Python 3.x  
- Required libraries:  
  - `numpy`  
  - `pandas`  
  - `tensorflow`  
  - `scikit-learn`  
  - `matplotlib`  

---

## 🚀 Installation and Usage

1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/yourusername/rainfall-prediction.git

2. **Install Dependencies**:  
   ```bash
   pip install -r requirements.txt

3. **Add the Dataset**:
Place the `rain_data.csv` file in the project directory.

4. **Run the Project**:
 Open and run the `main.ipynb` file to:
   - Preprocess the data.
   - Train and evaluate the LSTM model.
   - Visualize results, including the confusion matrix.

## ✨ Features
- **5-Fold Cross-Validation**: Enhances model robustness.
- **Confusion Matrix**: Provides a visual understanding of prediction accuracy.
- **Custom Dataset**: Reflects real-world data specific to the Kannauj district.

## 🔗 References
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Scikit-Learn Documentation](https://scikit-learn.org/)
