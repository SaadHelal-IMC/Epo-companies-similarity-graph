"""
Patent Company Graph - Data Collection Script (v3)
===================================================
Collects European patent data from the EPO OPS API.
Uses company+IPC+year queries to keep result sets small.

Setup:
  pip install requests pandas python-dotenv
"""

import os
import time
import base64
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[CONFIG] Loaded credentials from .env file")
except ImportError:
    pass

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

EPO_API_KEY = os.environ.get("EPO_API_KEY", "YOUR_KEY_HERE")
EPO_API_SECRET = os.environ.get("EPO_API_SECRET", "YOUR_SECRET_HERE")

NS = "http://www.epo.org/exchange"


# ============================================================
# AUTH
# ============================================================

def get_token(key, secret):
    """Get OAuth2 access token from EPO."""
    creds = base64.b64encode(f"{key}:{secret}".encode()).decode()
    r = requests.post(
        "https://ops.epo.org/3.2/auth/accesstoken",
        headers={
            "Authorization": f"Basic {creds}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data="grant_type=client_credentials",
    )
    r.raise_for_status()
    data = r.json()
    print(f"[AUTH] Got access token (expires in {data['expires_in']}s)")
    return data["access_token"]


# ============================================================
# SEARCH - returns list of (country, number, kind)
# ============================================================

def search_patents(token, query, start=1, end=25, max_retries=5):
    """
    Search EPO OPS. Returns list of doc IDs.
    Returns empty list on any error (413, 404, etc.)
    Retries up to max_retries times on 403 (rate limit) or connection errors.
    """
    url = "https://ops.epo.org/3.2/rest-services/published-data/search"

    for attempt in range(max_retries + 1):
        try:
            r = requests.get(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/xml",
                    "Range": f"{start}-{end}",
                },
                params={"q": query},
                timeout=30,
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if attempt < max_retries:
                print(f"    [NETWORK] Connection issue, retry {attempt+1}/{max_retries}, waiting 30s...")
                time.sleep(30)
                continue
            else:
                print(f"    [NETWORK] Still failing after {max_retries+1} attempts, skipping.")
                return []

        if r.status_code == 403:
            wait = min(int(r.headers.get("Retry-After", "30")), 60)
            if attempt < max_retries:
                print(f"    [THROTTLE] 403 on attempt {attempt+1}/{max_retries+1}, waiting {wait}s...")
                time.sleep(wait)
                continue
            else:
                print(f"    [THROTTLE] Still 403 after {max_retries+1} attempts, skipping.")
                return []

        if r.status_code == 200:
            break

        # 413 = too many results, 404 = no results
        if r.status_code in (404, 413):
            return []

        # Other errors
        print(f"    [ERROR] HTTP {r.status_code}: {r.text[:200]}")
        return []

    root = ET.fromstring(r.content)
    docs = []
    for doc_id in root.iter(f"{{{NS}}}document-id"):
        if doc_id.get("document-id-type") == "docdb":
            country = doc_id.findtext(f"{{{NS}}}country", "")
            number = doc_id.findtext(f"{{{NS}}}doc-number", "")
            kind = doc_id.findtext(f"{{{NS}}}kind", "")
            if country and number:
                docs.append((country, number, kind))
    return docs


# ============================================================
# BIBLIO - get details for one patent
# ============================================================

def get_biblio(token, country, number, kind, max_retries=3):
    """Get bibliographic details for a single patent."""
    url = (
        f"https://ops.epo.org/3.2/rest-services/published-data/publication/"
        f"docdb/{country}.{number}.{kind}/biblio"
    )

    for attempt in range(max_retries + 1):
        try:
            r = requests.get(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/xml",
                },
                timeout=30,
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if attempt < max_retries:
                print(f"      [NETWORK] Connection issue, retry {attempt+1}, waiting 15s...")
                time.sleep(15)
                continue
            else:
                return None

        if r.status_code == 403:
            wait = min(int(r.headers.get("Retry-After", "30")), 60)
            if attempt < max_retries:
                print(f"      [BIBLIO THROTTLE] attempt {attempt+1}, waiting {wait}s...")
                time.sleep(wait)
                continue
            else:
                return None

        if r.status_code == 200:
            break

        return None

    root = ET.fromstring(r.content)

    result = {
        "country": country,
        "doc_number": number,
        "kind": kind,
        "title": "",
        "abstract": "",
        "applicants": "",
        "inventors": "",
        "ipc_codes": "",
        "cpc_codes": "",
        "publication_date": "",
        "filing_date": "",
    }

    # Title (prefer English)
    for t in root.iter(f"{{{NS}}}invention-title"):
        if t.get("lang") == "en":
            result["title"] = (t.text or "").strip()
            break
        elif not result["title"]:
            result["title"] = (t.text or "").strip()

    # Abstract (prefer English)
    for ab in root.iter(f"{{{NS}}}abstract"):
        if ab.get("lang") == "en" or not result["abstract"]:
            paras = ab.findall(f".//{{{NS}}}p")
            if paras:
                result["abstract"] = " ".join((p.text or "") for p in paras).strip()
            elif ab.text:
                result["abstract"] = ab.text.strip()
            if ab.get("lang") == "en":
                break

    # Applicants
    apps = []
    for a in root.iter(f"{{{NS}}}applicant"):
        name = a.find(f".//{{{NS}}}name")
        if name is not None and name.text:
            apps.append(name.text.strip())
    result["applicants"] = "; ".join(dict.fromkeys(apps))  # deduplicate, keep order

    # Inventors
    invs = []
    for i in root.iter(f"{{{NS}}}inventor"):
        name = i.find(f".//{{{NS}}}name")
        if name is not None and name.text:
            invs.append(name.text.strip())
    result["inventors"] = "; ".join(dict.fromkeys(invs))

    # IPC codes
    ipcs = []
    for ipcr in root.iter(f"{{{NS}}}classification-ipcr"):
        text_el = ipcr.find(f"{{{NS}}}text")
        if text_el is not None and text_el.text:
            ipcs.append(text_el.text.strip())
    result["ipc_codes"] = "; ".join(ipcs)

    # CPC codes
    cpcs = []
    for cpc in root.iter(f"{{{NS}}}patent-classification"):
        section = cpc.findtext(f"{{{NS}}}section", "")
        cl = cpc.findtext(f"{{{NS}}}class", "")
        subclass = cpc.findtext(f"{{{NS}}}subclass", "")
        if section:
            cpcs.append(f"{section}{cl}{subclass}".strip())
    result["cpc_codes"] = "; ".join(cpcs)

    # Publication date
    for pr in root.iter(f"{{{NS}}}publication-reference"):
        d = pr.find(f".//{{{NS}}}date")
        if d is not None and d.text:
            result["publication_date"] = d.text
            break

    # Filing date
    for ar in root.iter(f"{{{NS}}}application-reference"):
        d = ar.find(f".//{{{NS}}}date")
        if d is not None and d.text:
            result["filing_date"] = d.text
            break

    return result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Patent Company Graph - Data Collection v3")
    print("=" * 60)

    if EPO_API_KEY == "YOUR_KEY_HERE":
        print("[ERROR] Set EPO_API_KEY and EPO_API_SECRET in .env file")
        exit(1)

    token = get_token(EPO_API_KEY, EPO_API_SECRET)
    token_time = time.time()

    # ---------------------------------------------------------------
    # QUERY STRATEGY:
    # Search by COMPANY + IPC section + single YEAR
    # This produces small, focused result sets (< 100 per query).
    #
    # pa = applicant name
    # ipc = IPC classification (first 4 chars = section+class+subclass)
    # pd = publication date (YYYYMMDD format)
    # ---------------------------------------------------------------

    COMPANIES = [
        "SIEMENS", "BOSCH", "SAMSUNG", "HUAWEI", "LG ELECTRONICS",
        "QUALCOMM", "SONY", "TOYOTA", "BMW", "BASF",
        "NOKIA", "ERICSSON", "PHILIPS", "APPLE", "MICROSOFT",
        "GOOGLE", "IBM", "VOLKSWAGEN", "BAYER",
        "ROCHE", "NOVARTIS", "PANASONIC", "HITACHI", "TOSHIBA",
        "INTEL", "ABB", "SCHNEIDER ELECTRIC",
        "AIRBUS", "SAP", "INFINEON", "HONDA",
        "GENERAL ELECTRIC", "HONEYWELL", "CANON",
    ]

    # IPC sections to combine with companies (keeps results small)
    IPC_SECTIONS = [
        "G06N",   # AI / ML
        "H04L",   # Telecom
        "H04W",   # Wireless
        "H01M",   # Batteries
        "B60L",   # Electric vehicles
    ]

    YEARS = [2023, 2024]

    # ---------------------------------------------------------------
    # RESUME: Load checkpoint if it exists
    # ---------------------------------------------------------------
    all_patents = []
    seen_doc_numbers = set()
    completed_queries = set()

    checkpoint_path = DATA_DIR / "epo_patents_all.csv"
    if checkpoint_path.exists() and checkpoint_path.stat().st_size > 100:
        df_prev = pd.read_csv(checkpoint_path)
        all_patents = df_prev.to_dict("records")
        seen_doc_numbers = set(df_prev["doc_number"].astype(str))
        # Mark which (company, ipc) combos are already done
        for _, row in df_prev.iterrows():
            completed_queries.add((row["search_company"], row["search_ipc"]))
        print(f"[RESUME] Loaded {len(all_patents)} patents from checkpoint")
        print(f"[RESUME] {len(completed_queries)} company+IPC combos already done, skipping those")

    total_queries = len(COMPANIES) * len(IPC_SECTIONS) * len(YEARS)
    query_num = 0

    for company in COMPANIES:
        for ipc in IPC_SECTIONS:
            # Skip if we already have data for this company+IPC combo
            if (company, ipc) in completed_queries:
                query_num += len(YEARS)  # count the skipped queries
                continue

            for year in YEARS:
                query_num += 1
                query = f'pa="{company}" and ipc="{ipc}" and pd within "{year}0101,{year}1231"'

                # Refresh token every 15 minutes
                if time.time() - token_time > 900:
                    token = get_token(EPO_API_KEY, EPO_API_SECRET)
                    token_time = time.time()

                # Search (fetch up to 15 per query — keeps us under retrieval limits)
                docs = search_patents(token, query, start=1, end=15)

                if docs:
                    print(f"[{query_num}/{total_queries}] {company} + {ipc} + {year}: {len(docs)} found", end="", flush=True)

                    # Fetch biblio for each
                    count = 0
                    for country, number, kind in docs:
                        if number in seen_doc_numbers:
                            continue  # skip duplicates

                        bib = get_biblio(token, country, number, kind)
                        if bib and bib["title"]:
                            bib["search_company"] = company
                            bib["search_ipc"] = ipc
                            all_patents.append(bib)
                            seen_doc_numbers.add(number)
                            count += 1

                        time.sleep(0.7)  # ~85 retrievals/min (limit is 100)

                    print(f" → saved {count}")
                else:
                    # Print progress every 10 queries
                    if query_num % 10 == 0:
                        print(f"[{query_num}/{total_queries}] Progress... ({len(all_patents)} patents so far)")

                # EPO allows 15 searches/min → 1 search every 4s minimum
                # Use 5s to be safe (many queries return 0 and are fast)
                time.sleep(5)

                # Save progress every 30 queries (so you don't lose data if interrupted)
                if query_num % 30 == 0 and all_patents:
                    df_temp = pd.DataFrame(all_patents)
                    df_temp.to_csv(DATA_DIR / "epo_patents_all.csv", index=False)
                    print(f"    [CHECKPOINT] Saved {len(all_patents)} patents so far")

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    if all_patents:
        df = pd.DataFrame(all_patents)
        output_path = DATA_DIR / "epo_patents_all.csv"
        df.to_csv(output_path, index=False)

        print(f"\n{'='*60}")
        print(f"COLLECTION COMPLETE")
        print(f"{'='*60}")
        print(f"Total patents: {len(df)}")
        print(f"Unique companies (applicants): {df['applicants'].nunique()}")
        print(f"Unique IPC codes: {df['ipc_codes'].nunique()}")
        print(f"Date range: {df['publication_date'].min()} - {df['publication_date'].max()}")
        print(f"Saved to: {output_path}")
        print(f"\nTop 15 applicants:")
        print(df['applicants'].value_counts().head(15).to_string())
        print(f"\nSample data:")
        print(df[['title', 'applicants', 'ipc_codes', 'publication_date']].head(5).to_string())
    else:
        print("\n[ERROR] No patents collected. Check errors above.")

    print("\nNext step: python 02_eda.py")
