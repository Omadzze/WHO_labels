#!/usr/bin/env python
"""
Read `column_labels.csv` –> open each referenced table –> collect the unique
values for every label in ONE pass.

Example call
------------
python collect_unique_values.py \
       --mapping column_labels.csv \
       --tables_dir tables \
       --out values_per_label.json
"""
import argparse, json, csv
from pathlib import Path
from collections import defaultdict
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mapping",    required=None, type=Path,
                  help="CSV with columns: table_name,column_index,label", default="../gold/cta_gt.csv")
    p.add_argument("--tables_dir", required=None, type=Path,
                   help="Directory that holds the raw CSV tables", default=Path("../tables"))
    p.add_argument("--out",        required=None, type=Path,
                   help="Where to write the JSON with unique values per label", default=("./output.json"))
    p.add_argument("--sample_rows", type=int, default=None,
                   help="Read only the first N rows of each table "
                        "(speed-up for huge files)")
    args = p.parse_args()

    # ------------------------------------------------------------------ #
    # 1. Load the mapping file                                           #
    # ------------------------------------------------------------------ #
    mapping = pd.read_csv(args.mapping)
    # ensure column_index is int for grouping
    mapping["column_index"] = mapping["column_index"].astype(int)

    # ------------------------------------------------------------------ #
    # 2. Collect values label-wise                                       #
    # ------------------------------------------------------------------ #
    values_per_label = defaultdict(set)

    # group so we open each table only once
    for table_name, group in mapping.groupby("table_name"):
        table_path = args.tables_dir / table_name
        if not table_path.exists():
            print(f"⚠  Table file missing: {table_path}")
            continue

        # Pandas can read just the needed columns
        usecols = group["column_index"].tolist()
        try:
            df = pd.read_csv(
                table_path,
                header=None,          # assume raw CSV has no header rows
                usecols=usecols,
                nrows=args.sample_rows,
                dtype=str             # read everything as string to avoid NaN fuzz
            )
        except Exception as e:
            print(f"⚠  Could not read {table_path}: {e}")
            continue

        # df columns keep original order; map back index→Series
        for _, row in group.iterrows():
            col_idx, label = row["column_index"], row["label"]
            # pandas renames integer columns to the col_idx itself
            col_series = df[col_idx]
            values_per_label[label].update(
                x.strip() for x in col_series.dropna().tolist()
            )

    # ------------------------------------------------------------------ #
    # 3. Save to JSON so you can reload quickly later                     #
    # ------------------------------------------------------------------ #
    # convert sets ➜ sorted lists to make JSON serialisable & human-readable
    out_obj = {lbl: sorted(vals) for lbl, vals in values_per_label.items()}
    args.out.write_text(json.dumps(out_obj, indent=2))
    print(f"Done. Wrote {len(out_obj)} labels to {args.out}")

if __name__ == "__main__":
    main()