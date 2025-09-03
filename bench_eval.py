# bench_eval.py
import argparse, csv, os, sys
import numpy as np

# Assumes The_Code.py is in the same directory or on PYTHONPATH
try:
    import The_Code as lrc
except Exception as e:
    print("ERROR: Could not import The_Code.py. Place bench_eval.py next to The_Code.py or add to PYTHONPATH.")
    print(e)
    sys.exit(1)

def _parse_targets(s):
    if not s:
        return None
    return [t.strip().lower() for t in s.split(",") if t.strip()]

def score_file(path, args):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    res = lrc.analyze_text(
        txt,
        target_word=lrc.TARGET_WORD_DEFAULT,
        min_rs_points=args.min_rs,
        targets=_parse_targets(args.targets),
        presence_mode=args.presence_mode,
        window=args.window,
        dfa_only=args.dfa_only,
        use_surrogates=args.surrogates,
        max_sents=args.max_sents
    )
    return res.aggregate_score

def cm_counts(y_true, y_pred):
    # Positive class = human-like
    tp = sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==1)
    tn = sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==0)
    fp = sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==1)
    fn = sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==0)
    return tp, fp, fn, tn

def metrics(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2*prec*rec) / (prec + rec) if (prec + rec) else 0.0
    return acc, prec, rec, f1

def main():
    ap = argparse.ArgumentParser(description="Benchmark & confusion-style summary for The_Code")
    ap.add_argument("manifest", help="CSV with columns: path,label,domain")
    ap.add_argument("--threshold", type=float, default=0.56,
                    help="Aggregate threshold: >=T => predict human-like (positive)")
    ap.add_argument("--presence-mode", choices=["or","gap","density"], default="density")
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--min-rs", type=int, default=lrc.MIN_RS_POINTS)
    ap.add_argument("--dfa-only", action="store_true")
    ap.add_argument("--surrogates", action="store_true")
    ap.add_argument("--targets", type=str, default=None,
                    help="Optional targets list, e.g. 'myth,reason'")
    ap.add_argument("--max-sents", type=int, default=5000)
    args = ap.parse_args()

    rows=[]
    with open(args.manifest, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            if not r.get("path") or not r.get("label"):
                continue
            rows.append(r)

    y_true=[]; y_pred=[]; domains=[]; scores=[]
    for r in rows:
        path = r["path"].strip()
        lab  = r["label"].strip().lower()  # "human" or "ai"
        dom  = (r.get("domain") or r.get("genre") or "general").strip().lower()
        if not os.path.exists(path):
            print(f"WARNING: missing file {path}; skipping")
            continue

        score = score_file(path, args)
        scores.append(score)
        # Map true labels: positive=human-like
        true = 1 if lab in ("human","human_like","h") else 0
        pred = 1 if score >= args.threshold else 0

        y_true.append(true)
        y_pred.append(pred)
        domains.append(dom)

        print(f"{path} | {lab:5s} | {dom:14s} | agg={score:.3f} | pred={'human' if pred==1 else 'ai'}")

    # Overall confusion-style summary
    tp, fp, fn, tn = cm_counts(y_true, y_pred)
    acc, prec, rec, f1 = metrics(tp, fp, fn, tn)
    print("\n=== Overall confusion-style summary (positive = human-like) ===")
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")

    # Per-domain summaries
    print("\n=== Per-domain ===")
    for dom in sorted(set(domains)):
        idx = [i for i,d in enumerate(domains) if d==dom]
        yt=[y_true[i] for i in idx]; yp=[y_pred[i] for i in idx]
        tp, fp, fn, tn = cm_counts(yt, yp)
        acc, prec, rec, f1 = metrics(tp, fp, fn, tn)
        print(f"[{dom}] N={len(idx)} | TP={tp} FP={fp} FN={fn} TN={tn} | "
              f"Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")

    # Score distributions for threshold tuning
    print("\n=== Score distributions (for threshold tuning) ===")
    human_scores = [s for s,t in zip(scores,y_true) if t==1]
    ai_scores    = [s for s,t in zip(scores,y_true) if t==0]
    if human_scores:
        print(f"Human-like  mean={np.mean(human_scores):.3f}  sd={np.std(human_scores):.3f}  n={len(human_scores)}")
    if ai_scores:
        print(f"AI-like     mean={np.mean(ai_scores):.3f}  sd={np.std(ai_scores):.3f}  n={len(ai_scores)}")
    print(f"Current threshold={args.threshold:.2f}. "
          f"Try sweeping 0.50..0.60 to trade off FP/FN by domain.")

if __name__ == "__main__":
    main()
