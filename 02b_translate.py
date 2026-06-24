"""
Patent Company Graph - Translate Non-English Abstracts
======================================================
Detects non-English patent abstracts (mostly German from
Bosch, Siemens, VW, Infineon) and translates them to English
using Google Translate (free, no API key needed).

Input:  data/epo_patents_cleaned.csv
Output: data/epo_patents_translated.csv

Setup:
  pip install pandas langdetect deep-translator
"""

import time
from pathlib import Path

import pandas as pd
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "epo_patents_cleaned.csv"
OUTPUT_FILE = DATA_DIR / "epo_patents_translated.csv"

# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("Patent Company Graph - Abstract Translation")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)
print(f"\nLoaded {len(df)} patents from {INPUT_FILE}")

# Keep only patents with abstracts (same filter as 03_nlp_similarity.py)
df = df[df["abstract"].notna() & (df["abstract"].astype(str).str.len() > 50)].copy()
print(f"After filtering (abstract > 50 chars): {len(df)} patents")

# ============================================================
# 2. DETECT LANGUAGES
# ============================================================

print(f"\nDetecting languages...")
langs = []
for text in df["abstract"].astype(str):
    try:
        lang = detect(text)
    except LangDetectException:
        lang = "unknown"
    langs.append(lang)

df["abstract_lang"] = langs
df["abstract_original"] = df["abstract"].copy()

lang_counts = df["abstract_lang"].value_counts()
print(f"\nLanguage distribution:")
for lang, count in lang_counts.items():
    pct = 100 * count / len(df)
    print(f"  {lang}: {count} patents ({pct:.1f}%)")

# Show which companies have non-English abstracts
non_en = df[df["abstract_lang"] != "en"]
if len(non_en) > 0:
    print(f"\nNon-English abstracts by company:")
    for company, group in non_en.groupby("company"):
        lang_breakdown = group["abstract_lang"].value_counts().to_dict()
        details = ", ".join(f"{l}: {c}" for l, c in lang_breakdown.items())
        print(f"  {company:<22s}: {len(group)} patents ({details})")
else:
    print("\nAll abstracts are already in English!")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")
    exit(0)

# ============================================================
# 3. TRANSLATE TO ENGLISH
# ============================================================

print(f"\nTranslating {len(non_en)} non-English abstracts to English...")

translator = GoogleTranslator(source="auto", target="en")
translated_count = 0
failed_count = 0

for idx in non_en.index:
    text = df.at[idx, "abstract_original"]
    try:
        # Google Translate has a 5000 char limit per request
        if len(text) > 4500:
            text = text[:4500]
        translated = translator.translate(text)
        if translated:
            df.at[idx, "abstract"] = translated
            translated_count += 1
        else:
            failed_count += 1
    except Exception as e:
        failed_count += 1
        if failed_count <= 5:
            print(f"  Failed for patent at index {idx}: {e}")

    # Progress update every 50 patents
    if (translated_count + failed_count) % 50 == 0:
        print(f"  Progress: {translated_count + failed_count}/{len(non_en)} "
              f"({translated_count} translated, {failed_count} failed)")
        time.sleep(1)  # small delay to avoid rate limiting

# ============================================================
# 4. SAVE RESULTS
# ============================================================

df.to_csv(OUTPUT_FILE, index=False)

print(f"\n{'='*60}")
print(f"TRANSLATION COMPLETE")
print(f"{'='*60}")
print(f"  Translated: {translated_count}")
print(f"  Failed:     {failed_count}")
print(f"  Saved to:   {OUTPUT_FILE}")
print(f"\nNext step: python 03_nlp_similarity.py")
