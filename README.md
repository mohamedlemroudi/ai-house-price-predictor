# 🏡 **AI House Price Predictor**

## 📋 **Description**

This project demonstrates how to predict house prices using a neural network built with TensorFlow and Keras. The model leverages a multi-layer neural network to estimate house prices based on the size in square meters.

## 📦 **Requirements**

- **TensorFlow:** `pip install tensorflow`
- **NumPy:** `pip install numpy`
- **Matplotlib:** `pip install matplotlib`

## 📂 **Structure**

- **📚 Libraries Import:** Imports necessary libraries for tensor operations, numerical computations, and visualizations.
- **📊 Data Preparation:** Creates a synthetic dataset with property sizes and corresponding prices.
- **🛠️ Model Creation:** Builds a sequential neural network with three dense layers.
- **⚙️ Model Compilation:** Configures the model with the Adam optimizer and Mean Squared Error loss function.
- **🕒 Model Training:** Trains the model for 1000 epochs.
- **🔮 Prediction:** Makes a prediction for a property with a specified size.
- **📈 Visualization & Evaluation:** Plots the loss over epochs and prints the internal weights of the model.

## 🚀 **How to Run**

```bash
python house_price_prediction.py
