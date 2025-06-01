# End-to-End Deep Learning Project Using ANN 🧠🚀

This repository contains a complete end-to-end deep learning project using an **Artificial Neural Network (ANN)** for binary classification. The model is trained on a real-world dataset and demonstrates the key stages of building and deploying a deep learning model using Python.

## 🔍 Project Overview

- **Objective:** Predict whether a customer will exit a bank based on various attributes such as credit score, geography, age, balance, and more.
- **Model:** Artificial Neural Network (ANN)
- **Tools Used:** Python, TensorFlow, Keras, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn

---

## 📁 Project Structure

```

📦 End-to-End-Deep-Learning-Project-Using-ANN
├── data/                        # Dataset used for training/testing
├── model/                       # Trained ANN model
├── ann\_classifier.ipynb        # Main Jupyter Notebook (model building + EDA)
├── README.md                   # Project description and setup

````

---

## 🛠️ Tech Stack & Libraries

- Python 3.x
- TensorFlow / Keras
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn
- Jupyter Notebook

---

## 🔬 Steps Involved

1. **Data Preprocessing**
   - Load dataset
   - Handle missing values
   - Encode categorical data
   - Normalize features
   - Train-test split

2. **Model Building**
   - Create ANN using Keras Sequential API
   - Add hidden layers with ReLU activation
   - Use Sigmoid activation in the output layer
   - Compile with Adam optimizer and binary cross-entropy loss

3. **Model Training**
   - Train on the processed dataset
   - Track loss and accuracy

4. **Evaluation**
   - Evaluate using confusion matrix, accuracy, precision, recall, and F1-score
   - Plot training/validation accuracy and loss

5. **Prediction**
   - Test the model on new/unseen data

---

## 📊 Results

- Achieved high accuracy and consistent performance on both training and testing data.
- Model generalizes well and can be deployed in a production-ready pipeline.

---

## 📌 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/udityamerit/End-to-End-Deep-Learning-Project-Using-ANN.git
   cd End-to-End-Deep-Learning-Project-Using-ANN


2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:

   ```bash
   jupyter notebook ann_classifier.ipynb
   ```

---

## 📄 Dataset

The dataset used in this project is a modified version of a customer churn dataset. You can find it in the `/data` folder or from sources like Kaggle.

---

## 📬 Contact

**Uditya Narayan Tiwari**
B.Tech CSE (AI & ML) | VIT Bhopal University
🔗 [Portfolio Website](https://udityanarayantiwari.netlify.app)
💼 [LinkedIn](https://www.linkedin.com/in/uditya-narayan-tiwari-562332289/)
📁 [GitHub](https://github.com/udityamerit)

---

## ⭐️ Star the repo

If you found this project helpful or interesting, please consider ⭐️ starring the repository to show your support!

---

## 📌 License

This project is licensed under the [MIT License](LICENSE).
