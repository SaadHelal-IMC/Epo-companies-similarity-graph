"""
Patent Company Graph - Exploratory Data Analysis (EDA)
======================================================
Cleans applicant names, generates visualizations, and prints
summary statistics for the collected EPO patent data.

Setup:
  pip install pandas matplotlib seaborn
"""

import re
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (saves to files)
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data")
PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)

INPUT_FILE = DATA_DIR / "epo_patents_all.csv"

# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("Patent Company Graph - EDA")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)
print(f"\nLoaded {len(df)} patents from {INPUT_FILE}")
print(f"Columns: {list(df.columns)}")

# ============================================================
# 2. BASIC CLEANING
# ============================================================

# Fix publication_date: convert float like 20231228.0 → "20231228"
df["publication_date"] = df["publication_date"].apply(
    lambda x: str(int(x)) if pd.notna(x) else ""
)

# Extract year from publication date
df["pub_year"] = df["publication_date"].str[:4]

# ============================================================
# 3. CLEAN APPLICANT NAMES → map to canonical company
# ============================================================

# The search_company column already has the canonical name, but
# the applicants column has the real EPO names like:
#   "SIEMENS AG [DE]", "SIEMENS CORP [US]", "SIEMENS LTD CHINA; 西门子（中国）有限公司"
#
# We'll create a "primary_applicant" column that maps to the
# search company when it's a match, and keeps the original otherwise.

# First, get the first applicant (before the semicolon)
df["primary_applicant_raw"] = df["applicants"].fillna("").str.split(";").str[0].str.strip()

# Map: if the search_company appears in the applicant string, use search_company
# This handles "SIEMENS AG [DE]" → "SIEMENS", "BOSCH GMBH ROBERT [DE]" → "BOSCH", etc.
def map_to_canonical(row):
    applicant = row["primary_applicant_raw"].upper()
    company = row["search_company"].upper()
    if company in applicant:
        return company
    # Handle special cases
    special = {
        "BAYERISCHE MOTOREN WERKE": "BMW",
        "KONINKLIJKE PHILIPS": "PHILIPS",
        "ROBERT BOSCH": "BOSCH",
        "TELEFON AB L M": "ERICSSON",
        "TELEFONAKTIEBOLAGET": "ERICSSON",
    }
    for pattern, canonical in special.items():
        if pattern in applicant:
            return canonical
    return row["search_company"]  # fallback to search company

df["company"] = df.apply(map_to_canonical, axis=1)

print(f"\n--- Applicant name mapping ---")
print(f"Raw unique applicants: {df['primary_applicant_raw'].nunique()}")
print(f"Canonical companies: {df['company'].nunique()}")

# ============================================================
# 4. SUMMARY STATISTICS
# ============================================================

print(f"\n{'='*60}")
print("SUMMARY STATISTICS")
print(f"{'='*60}")

print(f"\nTotal patents: {len(df)}")
print(f"Companies: {df['company'].nunique()}")
print(f"IPC sections: {df['search_ipc'].nunique()}")
print(f"Date range: {df['publication_date'].min()} - {df['publication_date'].max()}")

# Abstract stats
has_abstract = df["abstract"].notna() & (df["abstract"].astype(str) != "")
print(f"\nPatents with abstracts: {has_abstract.sum()} / {len(df)} ({100*has_abstract.sum()/len(df):.1f}%)")
abstract_lengths = df.loc[has_abstract, "abstract"].astype(str).str.len()
print(f"Abstract length: min={abstract_lengths.min()}, median={abstract_lengths.median():.0f}, max={abstract_lengths.max()}")

# IPC codes per patent
df["ipc_count"] = df["ipc_codes"].fillna("").str.split(";").apply(lambda x: len([i for i in x if i.strip()]))
print(f"\nIPC codes per patent: min={df['ipc_count'].min()}, median={df['ipc_count'].median():.0f}, max={df['ipc_count'].max()}")

print(f"\n--- Patents per company ---")
company_counts = df["company"].value_counts()
print(company_counts.to_string())

print(f"\n--- Patents per IPC section ---")
print(df["search_ipc"].value_counts().to_string())

print(f"\n--- Patents per year ---")
print(df["pub_year"].value_counts().sort_index().to_string())

# ============================================================
# 5. VISUALIZATIONS
# ============================================================

sns.set_theme(style="whitegrid", font_scale=1.1)

# --- Plot 1: Patents per company (bar chart) ---
fig, ax = plt.subplots(figsize=(14, 7))
company_counts.plot(kind="barh", ax=ax, color=sns.color_palette("viridis", len(company_counts)))
ax.set_xlabel("Number of Patents")
ax.set_ylabel("")
ax.set_title("Patents per Company (2023-2024)")
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(PLOT_DIR / "01_patents_per_company.png", dpi=150)
print(f"\nSaved: plots/01_patents_per_company.png")

# --- Plot 2: Patents per IPC section (bar chart) ---
fig, ax = plt.subplots(figsize=(10, 5))
ipc_labels = {
    "G06N": "G06N\n(AI/ML)",
    "H04L": "H04L\n(Telecom)",
    "H04W": "H04W\n(Wireless)",
    "H01M": "H01M\n(Batteries)",
    "B60L": "B60L\n(EV)",
}
ipc_counts = df["search_ipc"].value_counts()
ipc_counts.index = [ipc_labels.get(x, x) for x in ipc_counts.index]
ipc_counts.plot(kind="bar", ax=ax, color=sns.color_palette("Set2", 5))
ax.set_ylabel("Number of Patents")
ax.set_title("Patents per Technology Area (IPC Section)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
fig.savefig(PLOT_DIR / "02_patents_per_ipc.png", dpi=150)
print(f"Saved: plots/02_patents_per_ipc.png")

# --- Plot 3: Company × IPC heatmap ---
fig, ax = plt.subplots(figsize=(12, 10))
pivot = df.groupby(["company", "search_ipc"]).size().unstack(fill_value=0)
# Rename columns for readability
pivot.columns = [ipc_labels.get(c, c) for c in pivot.columns]
sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax, linewidths=0.5)
ax.set_title("Patent Count: Company × Technology Area")
ax.set_ylabel("")
ax.set_xlabel("")
plt.tight_layout()
fig.savefig(PLOT_DIR / "03_company_ipc_heatmap.png", dpi=150)
print(f"Saved: plots/03_company_ipc_heatmap.png")

# --- Plot 4: Abstract length distribution ---
fig, ax = plt.subplots(figsize=(10, 5))
abstract_lengths.plot(kind="hist", bins=50, ax=ax, color="steelblue", edgecolor="white")
ax.axvline(abstract_lengths.median(), color="red", linestyle="--", label=f"Median: {abstract_lengths.median():.0f}")
ax.set_xlabel("Abstract Length (characters)")
ax.set_ylabel("Count")
ax.set_title("Distribution of Patent Abstract Lengths")
ax.legend()
plt.tight_layout()
fig.savefig(PLOT_DIR / "04_abstract_length_dist.png", dpi=150)
print(f"Saved: plots/04_abstract_length_dist.png")

# --- Plot 5: Top 20 applicants (real names from EPO) ---
fig, ax = plt.subplots(figsize=(14, 7))
top_applicants = df["primary_applicant_raw"].value_counts().head(20)
top_applicants.plot(kind="barh", ax=ax, color=sns.color_palette("mako", 20))
ax.set_xlabel("Number of Patents")
ax.set_ylabel("")
ax.set_title("Top 20 Patent Applicants (EPO names)")
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(PLOT_DIR / "05_top_applicants.png", dpi=150)
print(f"Saved: plots/05_top_applicants.png")

# --- Plot 6: Company × Year breakdown ---
fig, ax = plt.subplots(figsize=(14, 7))
year_pivot = df.groupby(["company", "pub_year"]).size().unstack(fill_value=0)
year_pivot.plot(kind="barh", stacked=True, ax=ax, colormap="Paired")
ax.set_xlabel("Number of Patents")
ax.set_ylabel("")
ax.set_title("Patents per Company by Year")
ax.invert_yaxis()
ax.legend(title="Year")
plt.tight_layout()
fig.savefig(PLOT_DIR / "06_company_year.png", dpi=150)
print(f"Saved: plots/06_company_year.png")

# ============================================================
# 6. SAVE CLEANED DATA
# ============================================================

# Save the cleaned version with the canonical company column
output_path = DATA_DIR / "epo_patents_cleaned.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved cleaned data: {output_path} ({len(df)} rows)")

# Also save a summary table
summary = df.groupby("company").agg(
    patent_count=("doc_number", "count"),
    unique_ipc_sections=("search_ipc", "nunique"),
    avg_abstract_len=("abstract", lambda x: x.dropna().astype(str).str.len().mean()),
    has_abstract_pct=("abstract", lambda x: 100 * x.notna().sum() / len(x)),
).round(1)
summary = summary.sort_values("patent_count", ascending=False)
summary.to_csv(DATA_DIR / "company_summary.csv")
print(f"Saved company summary: data/company_summary.csv")

print(f"\n{'='*60}")
print("EDA COMPLETE")
print(f"{'='*60}")
print(f"Plots saved to: {PLOT_DIR}/")
print(f"Cleaned data saved to: {output_path}")
print(f"\nNext step: python 03_nlp_similarity.py")
