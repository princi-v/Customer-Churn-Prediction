# Customer Churn Prediction using Machine Learning

A beginner-friendly end-to-end machine learning project that estimates whether a telecom customer may leave the service. It includes dataset exploration, a preprocessing pipeline, a Random Forest classifier, evaluation metrics, and an interactive Streamlit dashboard.

> **Dataset note:** To make the project runnable immediately, `data/customer_churn.csv` contains **synthetic demo records** generated for this project. It is not real customer data and results should not be interpreted as real business performance. If you have internet access, you can replace it with the public IBM Telco Customer Churn sample dataset using `py download_real_dataset.py`.

## Features

- Data cleaning and preprocessing
- Numerical scaling and categorical one-hot encoding
- Random Forest classification
- Accuracy, balanced accuracy, ROC-AUC, classification report and confusion matrix
- Interactive churn prediction form
- Churn distribution and contract comparison charts
- Trained model and evaluation metrics saved locally

## Project Structure

```text
Customer-Churn-Prediction/
├── app.py                         # Interactive web dashboard
├── train_model.py                 # Train and evaluate the model
├── download_real_dataset.py       # Optional public dataset download
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── data/
│   └── customer_churn.csv          # Synthetic demo dataset
├── models/                         # Saved model generated after training
├── reports/                        # Evaluation metrics generated after training
└── src/
    └── model_utils.py              # Data, preprocessing and model functions
```

## Run in VS Code (Windows)

1. Install **Python 3.11 or 3.12** from [python.org](https://www.python.org/downloads/) and **Visual Studio Code** from [code.visualstudio.com](https://code.visualstudio.com/). During Python installation, check **Add Python to PATH**.
2. Download the project ZIP from GitHub or extract this project folder.
3. In VS Code, choose **File → Open Folder** and select `Customer-Churn-Prediction`.
4. Open **Terminal → New Terminal**.
5. Install dependencies:

   ```powershell
   py -m pip install -r requirements.txt
   ```

6. Train the model and print evaluation metrics:

   ```powershell
   py train_model.py
   ```

7. Launch the interactive dashboard:

   ```powershell
   py -m streamlit run app.py
   ```

8. A browser tab will open. Enter sample customer details and press **Predict churn risk**. If it does not open automatically, copy the local URL shown in the terminal (usually `http://localhost:8501`) into your browser.

### If `py` is not recognized
Try replacing `py` with `python` in the commands. If Python is not recognized either, reinstall Python and enable **Add Python to PATH**.

## Optional: use the public IBM sample data

If internet is available, run:

```powershell
py download_real_dataset.py
py train_model.py
py -m streamlit run app.py
```

This replaces the bundled synthetic demo CSV with the public IBM Telco Customer Churn sample. The source is the [IBM public GitHub dataset](https://github.com/IBM/telco-customer-churn-on-icp4d).

## Model Workflow

1. Load customer-level data and remove customer IDs from the features.
2. Convert `TotalCharges` to numeric values and handle missing values through preprocessing pipelines.
3. Split data into training (80%) and testing (20%) sets using stratification.
4. Encode categorical variables and scale numeric variables.
5. Train a Random Forest classifier.
6. Evaluate the test split using accuracy, balanced accuracy, ROC-AUC, precision, recall, F1-score, and confusion matrix.
7. Use the trained pipeline to estimate churn risk for one customer in the dashboard.

## Important limitations

- Churn is a binary classification exercise; output is a model estimate, not a guarantee.
- The default bundled dataset is synthetic for demonstration.
- Performance depends on the dataset and should be re-evaluated after replacing it with another source.
- Do not use this demo as the sole basis for real customer decisions.

## Technologies

Python · Pandas · NumPy · Scikit-learn · Streamlit · Matplotlib
