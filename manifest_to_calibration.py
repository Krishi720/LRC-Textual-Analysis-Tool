# manifest_to_calibration.py
import csv, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="manifest.csv with columns: domain,text1,text2,label1,label2")
    ap.add_argument("--out", default="calibration.csv", help="output CSV with columns: path,label,genre")
    args = ap.parse_args()

    rows = []
    with open(args.manifest, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r, 1):
            domain = (row.get("domain") or "generic").strip()
            # text1
            rows.append({"path": row["text1"].strip(),
                         "label": row["label1"].strip().lower(),
                         "genre": domain})
            # text2
            rows.append({"path": row["text2"].strip(),
                         "label": row["label2"].strip().lower(),
                         "genre": domain})

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path","label","genre"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
