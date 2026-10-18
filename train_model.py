"""Train and evaluate the customer churn classifier."""
import json
from src.model_utils import train_and_evaluate, METRICS_PATH


def main():
    model, metrics, _ = train_and_evaluate(save_model=True)
    print("\nCUSTOMER CHURN MODEL TRAINED SUCCESSFULLY")
    print("-" * 44)
    print(f"Rows used:           {metrics['rows']}")
    print(f"Input features:      {metrics['features']}")
    print(f"Accuracy:            {metrics['accuracy']:.2%}")
    print(f"Balanced accuracy:   {metrics['balanced_accuracy']:.2%}")
    print(f"ROC-AUC:             {metrics['roc_auc']:.3f}")
    print("\nClassification report:")
    for label in ("No Churn", "Churn"):
        report = metrics["classification_report"][label]
        print(f"{label:10s} | precision={report['precision']:.2f} | recall={report['recall']:.2f} | f1={report['f1-score']:.2f}")
    print(f"\nSaved model: models/churn_model.joblib")
    print(f"Saved metrics: {METRICS_PATH.relative_to(METRICS_PATH.parents[1])}")
    print("\nNote: the bundled CSV is synthetic demo data, not real customer records.")

if __name__ == "__main__":
    main()
