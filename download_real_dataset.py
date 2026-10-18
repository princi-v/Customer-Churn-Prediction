"""Optional: download the public IBM Telco Churn sample dataset.
This replaces the included synthetic demo CSV with the real public sample file.
Internet access is required.
"""
from pathlib import Path
import urllib.request

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
OUT = Path(__file__).resolve().parent / "data" / "customer_churn.csv"

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading public IBM Telco Customer Churn sample dataset...")
    urllib.request.urlretrieve(URL, OUT)
    print(f"Downloaded to: {OUT}")
    print("Next run: py train_model.py")
