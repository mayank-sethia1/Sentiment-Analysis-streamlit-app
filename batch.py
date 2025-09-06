import sys
import time
import pandas as pd
from sentiment_llm import MovieSentimentClassifier

CLASSES = ["Positive", "Neutral", "Negative"]

# batch.py
def run_batch(csv_or_df, mode: str = "Strict", rpm: float = 15.0, return_details: bool = True) -> pd.DataFrame:
    if isinstance(csv_or_df, str):
        df = pd.read_csv(csv_or_df)
    else:
        df = csv_or_df.copy()

    gap = 0.0 if rpm <= 0 else (60.0 / rpm) + 0.25
    clf = MovieSentimentClassifier(temperature=0.1 if mode == "Strict" else 0.3)

    texts = df.iloc[:, 0].astype(str).str.strip()  # first col = review text
    labels, confs, expls, evids = [], [], [], []
    for t in texts:
        res = clf.classify(t, mode=mode)
        labels.append(res.label)
        if return_details:
            confs.append(float(res.confidence))
            expls.append(res.explanation)
            evids.append(res.evidence_phrases)
        if gap > 0:
            time.sleep(gap)

    df_out = df.copy()
    df_out["pred"] = labels
    if return_details:
        df_out["confidence"] = confs
        df_out["explanation"] = expls
        df_out["evidence_phrases"] = evids
    return df_out


def report(csv_or_df) -> dict:
    if isinstance(csv_or_df, str):
        df = pd.read_csv(csv_or_df)
    else:
        df = csv_or_df.copy()

    # gold labels (2nd column) and predictions
    actual = df.iloc[:, 1].astype(str).str.strip().str.title()
    pred   = df["pred"].astype(str).str.strip().str.title()

    total = len(df)
    correct = (actual == pred).sum()
    acc = (correct / total) if total else 0.0

    actual_counts = actual.value_counts().reindex(CLASSES, fill_value=0)
    pred_counts   = pred.value_counts().reindex(CLASSES, fill_value=0)

    return {
        "accuracy": acc,
        "correct": int(correct),
        "total": int(total),
        "actual_counts": actual_counts.to_dict(),
        "pred_counts": pred_counts.to_dict(),
    }

if __name__ == "__main__":
    # usage: python main.py <strict|lenient> <in_csv> [out_csv] [rpm]
    mode = sys.argv[1].title()
    in_csv = sys.argv[2]
    out_csv = sys.argv[3] if len(sys.argv) > 3 else "predictions.csv"
    rpm = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0

    # Run batch (no prints), then save + print here for CLI usage
    df_out = run_batch(in_csv, mode=mode, rpm=rpm)
    df_out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    metrics = report(df_out)  # or report(out_csv)
    acc = metrics["accuracy"] * 100
    print(f"Accuracy: {acc:.2f}% ({metrics['correct']}/{metrics['total']})")

    print("\nActual label counts:")
    for c in CLASSES:
        print(f"  {c}: {metrics['actual_counts'].get(c, 0)}")
    print("\nPredicted label counts:")
    for c in CLASSES:
        print(f"  {c}: {metrics['pred_counts'].get(c, 0)}")
