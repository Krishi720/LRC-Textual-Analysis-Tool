# calibration.py
# Calibrate and score using features from The_Code.analyze_text
# Works with either a flattened calibration CSV (path,label,genre)
# or your manifest.csv (domain,text1,text2,label1,label2)

import os
import json
import argparse
import csv
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import joblib

# Import your analyzer and dataclasses
from The_Code import analyze_text, EncodingStat, LRCResult  # type: ignore


# ---------------- Feature extraction ----------------

def _result_to_features(
    res: LRCResult,
    include_presence: bool = False,
    prefer_delta: bool = True,
    include_punc_fano: bool = True,
) -> Tuple[np.ndarray, List[str], int]:
    """
    Turn an LRCResult into a numeric vector.
    prefer_delta=True -> use ΔH when available; fallback to raw exponent H otherwise.
    include_presence=False to avoid topical leakage by default.
    Returns: (Xvec, feature_names, n_sentences)
    """
    feats: Dict[str, float] = {}

    # Optionally include one presence-like channel if present
    pres_keys = [k for k in res.encoding_values.keys() if k.startswith("word_")]
    if include_presence and pres_keys:
        pk = pres_keys[0]
        stat = res.encoding_values[pk]
        feats[f"{pk}"] = (
            float(stat.delta) if (prefer_delta and stat.delta is not None) else float(stat.value)
        )

    # Topic-agnostic encoders
    for key in ["sentence_lengths", "punctuation_cadence", "semantic_drift", "function_words"]:
        if key in res.encoding_values:
            stat = res.encoding_values[key]
            feats[f"{key}"] = (
                float(stat.delta) if (prefer_delta and stat.delta is not None) else float(stat.value)
            )

    if include_punc_fano and res.punc_fano is not None:
        feats["punctuation_fano"] = float(res.punc_fano)

    names = list(feats.keys())
    Xvec = np.array([feats[n] for n in names], dtype=float)

    n_sents = res.encoding_values.get("sentence_lengths", EncodingStat(0.0, "", 0)).n
    return Xvec, names, n_sents


def _length_bin(n_sents: int) -> str:
    if n_sents < 60:
        return "short"
    if n_sents < 140:
        return "medium"
    return "long"


# ---------------- Data loading ----------------

def load_rows_from_manifest(manifest_path: str) -> List[Dict[str, str]]:
    """Read domain,text1,text2,label1,label2 → rows of {path,label,genre}."""
    rows: List[Dict[str, str]] = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        required = {"domain", "text1", "text2", "label1", "label2"}
        if not required.issubset(set(r.fieldnames or [])):
            raise ValueError(
                f"manifest must have columns: {sorted(required)}; found {r.fieldnames}"
            )
        for row in r:
            domain = (row.get("domain") or "generic").strip()
            t1, t2 = row["text1"].strip(), row["text2"].strip()
            l1, l2 = row["label1"].strip().lower(), row["label2"].strip().lower()
            rows.append({"path": t1, "label": l1, "genre": domain})
            rows.append({"path": t2, "label": l2, "genre": domain})
    return rows


def load_rows_from_calib_csv(csv_path: str) -> List[Dict[str, str]]:
    """Read path,label,genre rows."""
    df = pd.read_csv(csv_path)
    need = {"path", "label"}
    if not need.issubset(df.columns):
        raise ValueError(f"calibration csv must have at least columns {sorted(need)}")
    if "genre" not in df.columns:
        df["genre"] = "generic"
    out: List[Dict[str, str]] = []
    for _, r in df.iterrows():
        out.append({"path": str(r["path"]), "label": str(r["label"]).lower(), "genre": str(r["genre"])})
    return out


# ---------------- Calibration core ----------------

def run_calibration(
    rows: List[Dict[str, str]],
    dfa_only: bool = True,
    use_surrogates: bool = True,
    include_presence: bool = False,
    prefer_delta: bool = True,
    presence_mode: str = "gap",
    targets: Optional[List[str]] = None,
    window: int = 128,
    min_rs_points: int = 500,
    max_sents: int = 5000,
    cv_mode: str = "stratified",
    k_folds: int = 5,
    save_calib_json: Optional[str] = None,
    save_model_path: Optional[str] = None,
) -> None:
    X_list, y_list, genres, lenbins = [], [], [], []
    feat_names: Optional[List[str]] = None

    for item in rows:
        path = item["path"]
        label = item["label"]
        genre = item.get("genre", "generic")
        if not os.path.isfile(path):
            print(f"[WARN] Missing file: {path}; skipping.")
            continue
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()

        res = analyze_text(
            txt,
            min_rs_points=min_rs_points,
            targets=targets,
            presence_mode=presence_mode,
            window=window,
            dfa_only=dfa_only,
            use_surrogates=use_surrogates,
            max_sents=max_sents,
        )

        Xvec, names, n_sents = _result_to_features(
            res,
            include_presence=include_presence,
            prefer_delta=prefer_delta,
            include_punc_fano=True,
        )
        if feat_names is None:
            feat_names = names
        X_list.append(Xvec)
        y_list.append(1 if label == "human" else 0)
        genres.append(genre)
        lenbins.append(_length_bin(n_sents))

    if not X_list:
        print("[ERROR] No usable examples.")
        return

    X = np.vstack(X_list)
    y = np.asarray(y_list, dtype=int)
    genres = np.asarray(genres)
    lenbins = np.asarray(lenbins)

    # Scale
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # CV split
    if cv_mode == "group-genre":
        splitter = GroupKFold(n_splits=min(k_folds, len(np.unique(genres))))
        splits = splitter.split(Xs, y, groups=genres)
    else:
        splitter = StratifiedKFold(n_splits=min(k_folds, int(np.bincount(y).min())), shuffle=True, random_state=42)
        splits = splitter.split(Xs, y)

    # CV predict
    y_prob = np.zeros_like(y, dtype=float)
    for tr, te in splits:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
        clf.fit(Xs[tr], y[tr])
        y_prob[te] = clf.predict_proba(Xs[te])[:, 1]

    # Global metrics
    roc = roc_auc_score(y, y_prob)
    pr = average_precision_score(y, y_prob)
    fpr, tpr, thr = roc_curve(y, y_prob)
    J = tpr - fpr
    j_idx = int(np.argmax(J))
    best_thr = float(thr[j_idx])

    print("\n=== CALIBRATION SUMMARY ===")
    print(f"Examples: {len(y)}")
    print(f"Features: {feat_names}")
    print(f"Global ROC AUC: {roc:.3f} | PR AUC: {pr:.3f}")
    print(f"Best global threshold (Youden J): {best_thr:.3f}")

    # Per-bucket breakdowns
    def bucket_report(mask: np.ndarray):
        if mask.sum() < 6 or len(np.unique(y[mask])) < 2:
            return None
        _roc = roc_auc_score(y[mask], y_prob[mask])
        _pr = average_precision_score(y[mask], y_prob[mask])
        _fpr, _tpr, _thr = roc_curve(y[mask], y_prob[mask])
        _J = _tpr - _fpr
        _j = int(np.argmax(_J))
        return {"n": int(mask.sum()), "roc_auc": float(_roc), "pr_auc": float(_pr), "threshold": float(_thr[_j])}

    buckets: Dict[str, Optional[Dict[str, float]]] = {}
    for g in np.unique(genres):
        buckets[f"genre={g}"] = bucket_report(genres == g)
    for b in np.unique(lenbins):
        buckets[f"lenbin={b}"] = bucket_report(lenbins == b)

    print("\nPer-bucket AUCs & thresholds:")
    for k, v in buckets.items():
        if v is None:
            print(f"  {k}: n/a")
        else:
            print(f"  {k}: n={v['n']}  ROC AUC={v['roc_auc']:.3f}  PR AUC={v['pr_auc']:.3f}  thr={v['threshold']:.3f}")

    # Save artifacts
    if save_calib_json:
        blob = {
            "feat_names": feat_names,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "global_threshold": best_thr,
            "buckets": buckets,
            "use_surrogates": use_surrogates,
            "dfa_only": dfa_only,
            "presence_mode": presence_mode,
            "targets": targets,
            "window": window,
            "min_rs_points": min_rs_points,
            "max_sents": max_sents,
            "include_presence": include_presence,
            "prefer_delta": prefer_delta,
        }
        with open(save_calib_json, "w", encoding="utf-8") as f:
            json.dump(blob, f, ensure_ascii=False, indent=2)
        print(f"\nSaved calibration JSON → {save_calib_json}")

    if save_model_path:
        final_clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
        final_clf.fit(Xs, y)
        joblib.dump({"model": final_clf, "scaler": scaler, "feat_names": feat_names}, save_model_path)
        print(f"Saved model → {save_model_path}")


# ---------------- Scoring ----------------

def score_file(
    text_path: str,
    calib_json: str,
    model_path: Optional[str] = None,
    presence_mode: str = "gap",
    targets: Optional[List[str]] = None,
    window: int = 128,
    min_rs_points: int = 500,
    max_sents: int = 5000,
    dfa_only: bool = True,
    use_surrogates: bool = True,
    include_presence: bool = False,
    prefer_delta: bool = True,
) -> None:
    if not os.path.isfile(text_path):
        print(f"[ERROR] Missing: {text_path}")
        return
    with open(text_path, "r", encoding="utf-8") as f:
        txt = f.read()

    with open(calib_json, "r", encoding="utf-8") as f:
        calib = json.load(f)
    saved_feat_names = calib["feat_names"]

    # Use saved analyzer settings for consistency where possible
    dfa_only = calib.get("dfa_only", dfa_only)
    use_surrogates = calib.get("use_surrogates", use_surrogates)
    presence_mode = calib.get("presence_mode", presence_mode)
    targets = calib.get("targets", targets)
    window = calib.get("window", window)
    min_rs_points = calib.get("min_rs_points", min_rs_points)
    max_sents = calib.get("max_sents", max_sents)
    include_presence = calib.get("include_presence", include_presence)
    prefer_delta = calib.get("prefer_delta", prefer_delta)

    res = analyze_text(
        txt,
        min_rs_points=min_rs_points,
        targets=targets,
        presence_mode=presence_mode,
        window=window,
        dfa_only=dfa_only,
        use_surrogates=use_surrogates,
        max_sents=max_sents,
    )

    Xvec, names, _ = _result_to_features(
        res,
        include_presence=include_presence,
        prefer_delta=prefer_delta,
        include_punc_fano=True,
    )

    # Reorder to saved feature order
    name_to_val = {n: v for n, v in zip(names, Xvec)}
    Xordered = np.array([name_to_val[n] for n in saved_feat_names], dtype=float)

    if model_path and os.path.isfile(model_path):
        bundle = joblib.load(model_path)
        scaler = bundle["scaler"]
        model = bundle["model"]
        Xs = scaler.transform(Xordered.reshape(1, -1))
        prob = float(model.predict_proba(Xs)[0, 1])
        thr = float(calib.get("global_threshold", 0.5))
        decision = "human" if prob >= thr else "ai"
        print(f"Score: P(human)={prob:.3f} | threshold={thr:.3f} → decision: {decision}")
    else:
        # No model: just print the vector and note missing model
        print("[WARN] No model provided; printing feature vector only.")
        print({n: float(name_to_val[n]) for n in saved_feat_names})


# ---------------- CLI ----------------

def parse_targets_arg(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    return [t.strip().lower() for t in s.split(",") if t.strip()]


def main():
    ap = argparse.ArgumentParser(description="Calibrate/score LRC features using The_Code.analyze_text")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # calibrate
    cap = sub.add_parser("calibrate", help="Run calibration from a manifest or calibration CSV")
    gsrc = cap.add_mutually_exclusive_group(required=True)
    gsrc.add_argument("--manifest", type=str, help="manifest.csv with domain,text1,text2,label1,label2")
    gsrc.add_argument("--csv", type=str, help="calibration CSV with path,label[,genre]")
    cap.add_argument("--dfa-only", action="store_true", default=True)
    cap.add_argument("--no-surrogates", dest="use_surrogates", action="store_false", help="Disable surrogate ΔH")
    cap.add_argument("--presence-mode", choices=["or","gap","density"], default="gap")
    cap.add_argument("--targets", type=str, help="Comma-separated targets; omit to skip presence feature")
    cap.add_argument("--include-presence", action="store_true", help="Include word_* feature in classifier")
    cap.add_argument("--window", type=int, default=128)
    cap.add_argument("--min-rs", type=int, default=500)
    cap.add_argument("--max-sents", type=int, default=5000)
    cap.add_argument("--cv-mode", choices=["stratified","group-genre"], default="stratified")
    cap.add_argument("--folds", type=int, default=5)
    cap.add_argument("--save-calib-json", type=str, default="calib.json")
    cap.add_argument("--save-model", type=str, help="Optional path to save trained model (joblib)")

    # score
    sp = sub.add_parser("score", help="Score a new text using saved calibration (and optional model)")
    sp.add_argument("--text", required=True, type=str)
    sp.add_argument("--calib-json", required=True, type=str)
    sp.add_argument("--model", type=str, help="Optional path to joblib with {'model','scaler','feat_names'}")

    # shared analyzer options (for score only; calibrate uses calib JSON settings)
    sp.add_argument("--presence-mode", choices=["or","gap","density"], default="gap")
    sp.add_argument("--targets", type=str)
    sp.add_argument("--window", type=int, default=128)
    sp.add_argument("--min-rs", type=int, default=500)
    sp.add_argument("--max-sents", type=int, default=5000)
    sp.add_argument("--dfa-only", action="store_true", default=True)
    sp.add_argument("--no-surrogates", dest="use_surrogates", action="store_false")
    sp.add_argument("--include-presence", action="store_true")

    args = ap.parse_args()

    if args.cmd == "calibrate":
        if args.manifest:
            rows = load_rows_from_manifest(args.manifest)
        else:
            rows = load_rows_from_calib_csv(args.csv)
        run_calibration(
            rows=rows,
            dfa_only=args.dfa_only,
            use_surrogates=args.use_surrogates,
            include_presence=args.include_presence,
            prefer_delta=True,
            presence_mode=args.presence_mode,
            targets=parse_targets_arg(args.targets),
            window=args.window,
            min_rs_points=args.min_rs,
            max_sents=args.max_sents,
            cv_mode=args.cv_mode,
            k_folds=args.folds,
            save_calib_json=args.save_calib_json,
            save_model_path=args.save_model,
        )
        return

    if args.cmd == "score":
        score_file(
            text_path=args.text,
            calib_json=args.calib_json,
            model_path=args.model,
            presence_mode=args.presence_mode,
            targets=parse_targets_arg(args.targets),
            window=args.window,
            min_rs_points=args.min_rs,
            max_sents=args.max_sents,
            dfa_only=args.dfa_only,
            use_surrogates=args.use_surrogates,
            include_presence=args.include_presence,
            prefer_delta=True,
        )
        return


if __name__ == "__main__":
    main()
