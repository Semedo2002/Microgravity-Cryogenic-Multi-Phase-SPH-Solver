#!/usr/bin/env python3
"""Download the SPHERIC Test 2 (3D dam break, Kleefsman et al. 2005) data.

The experimental time series (MARIN) are distributed by the SPHERIC ERCOFTAC
Special Interest Group at https://www.spheric-sph.org/tests/test-02 . They are
not redistributed with this repository; this script fetches the official
archive and converts the spreadsheet to data/experiment.csv with columns
    time, P1..P8 [Pa], H1..H4 [m].

Requires `xlrd` (pip install xlrd).
"""
import csv
import io
import os
import sys
import urllib.request
import zipfile

ARCHIVE = ("https://9449af45-2363-44f6-8b03-03b6b7c2aee5.usrfiles.com/archives/"
           "9449af_6ce73873d6d04315853e481c337dbfca.zip")
HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    out_dir = os.path.join(HERE, "data")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "experiment.csv")
    if os.path.exists(out_csv):
        print(f"{out_csv} already exists")
        return 0
    try:
        import xlrd
    except ImportError:
        print("this script needs xlrd: pip install xlrd", file=sys.stderr)
        return 1
    print(f"downloading {ARCHIVE}")
    with urllib.request.urlopen(ARCHIVE, timeout=120) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = next(n for n in z.namelist() if n.endswith("exp_data.xls"))
        book = xlrd.open_workbook(file_contents=z.read(name))
    sheet = book.sheet_by_index(0)
    header = ["time"] + [f"P{i}" for i in range(1, 9)] + [f"H{i}" for i in range(1, 5)]
    cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]  # column 9 is empty in the sheet
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in range(1, sheet.nrows):
            row = sheet.row_values(r)
            if row[0] == "":
                continue
            w.writerow([row[c] for c in cols])
    print(f"wrote {out_csv} ({sheet.nrows - 1} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
