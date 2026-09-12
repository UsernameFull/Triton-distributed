#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
import argparse
import csv
from pathlib import Path


def gbps(bytes_value, us_value):
    if us_value <= 0:
        return 0.0
    return bytes_value / us_value * 1_000_000.0 / 1024.0 / 1024.0 / 1024.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--output", default="combine_perf_summary.csv")
    args = parser.parse_args()

    root = Path(args.dir)
    rows = []
    header = None
    for csv_path in sorted(root.glob("combine_perf_rank*.csv")):
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if header is None:
                header = reader.fieldnames
            for row in reader:
                row.pop(None, None)
                rows.append(row)

    if not rows or header is None:
        return

    grouped = {}
    key_fields = ["Metric", "DataSize/B", "GlobalDataSize/B", "Npus", "Blocks", "UBsize/KB",
                  "BS", "H", "TopK", "ExpertPerPe", "Dtype", "Warmup", "Loops", "CaseId"]
    for row in rows:
        key = tuple(row.get(field, "") for field in key_fields)
        current = grouped.get(key)
        row_time = float(row.get("CoreMaxTime/us", "0") or 0)
        current_time = float(current.get("CoreMaxTime/us", "0") or 0) if current is not None else -1
        if current is None or row_time > current_time:
            grouped[key] = dict(row)

    output_rows = []
    for row in grouped.values():
        global_bytes = float(row.get("GlobalDataSize/B", "0") or 0)
        per_pe_bytes = float(row.get("DataSize/B", "0") or 0)
        max_time = float(row.get("CoreMaxTime/us", "0") or 0)
        row["Bandwidth/GB/s"] = f"{gbps(global_bytes, max_time):.4f}"
        row["PerPeBandwidth/GB/s"] = f"{gbps(per_pe_bytes, max_time):.4f}"
        row["ProfPe"] = "max"
        output_rows.append(row)

    output_rows.sort(key=lambda r: (
        r.get("Metric", ""),
        int(float(r.get("Npus", "0") or 0)),
        int(float(r.get("DataSize/B", "0") or 0)),
        r.get("CaseId", ""),
    ))

    output_path = root / args.output
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"[Combine] wrote summary csv: {output_path}")


if __name__ == "__main__":
    main()
