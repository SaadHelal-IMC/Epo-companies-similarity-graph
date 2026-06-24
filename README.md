# Epo Companies Similarity Graph

**Capstone Project — IMC University of Applied Sciences, Summer 2026**

Analyzing technological similarity between companies through European patent data. This project builds a company similarity network by comparing patent abstracts using NLP techniques (TF-IDF and Sentence-BERT), then clusters companies and models topics to reveal industry structure.

**Authors:** Saad Helal & Enas Alaawaj

---

## Overview

We collected 3,799 European patent applications from the EPO Open Patent Services API across 34 major companies and 5 IPC technology areas (G06N – AI/ML, H04L – telecom, H04W – wireless, H01M – batteries, B60L – electric vehicles); 3,769 had usable abstracts and were carried into the analysis. The pipeline compares two text embedding methods — TF-IDF (keyword-based) and Sentence-BERT (semantic) — for measuring company similarity based on patent abstracts. Key findings:

- SBERT produces significantly better-separated clusters (silhouette score 0.1305 vs 0.0627 for TF-IDF)
- Translation of 321 non-English abstracts (mostly German) eliminates language-based clustering artifacts
- The resulting network reveals intuitive industry groupings: automotive, tech/telecom, electronics, healthcare, and a pharma outlier (Novartis)
- Bridge companies like IBM, Roche, and Philips connect different technological communities

## Project Structure

```
├── 01_data_collection.py       # EPO OPS API data collection
├── 02_eda.py                   # Exploratory data analysis & visualizations
├── 02b_translate.py            # Language detection & translation pipeline
├── 03_nlp_similarity.py        # NLP similarity, clustering & topic modeling
├── requirements.txt            # Python dependencies (Python 3.10)
├── .gitignore
├── data/                       #  datasets
│   ├── epo_patents_all.csv           # Raw API output
│   ├── epo_patents_cleaned.csv       # After name cleaning
│   ├── epo_patents_translated.csv    # After translation
│   ├── company_similarity_tfidf.csv  # TF-IDF similarity matrix
│   ├── company_similarity_sbert.csv  # SBERT similarity matrix
│   ├── company_clusters.csv          # Cluster assignments
│   ├── company_topic_distribution.csv
│   ├── company_graph.gexf            # TF-IDF network graph
│   └── company_graph_sbert.gexf      # SBERT network graph
└── plots/                      # Generated visualizations
    ├── 01_patents_per_company.png
    ├── 03_company_ipc_heatmap.png
    ├── 08_similarity_heatmap_sbert.png
    ├── 11_dendrogram_sbert.png
    ├── 12_tsne_sbert.png
    ├── 13_company_graph_sbert.png
    ├── 14_topic_distribution.png

    └── ...
```

## Pipeline

The scripts are numbered in execution order:

### 1. Data Collection (`01_data_collection.py`)

Queries the EPO Open Patent Services (OPS) REST API for patent bibliographic data. Searches by company name, IPC section, and publication year (2023-2024). Extracts titles, abstracts, IPC codes, applicant names, and publication dates. Handles API rate limiting and pagination automatically.

### 2. Exploratory Data Analysis (`02_eda.py`)

Cleans and standardizes applicant names (e.g., mapping "SAMSUNG ELECTRONICS CO LTD" variants to "SAMSUNG"). Generates visualizations: patents per company, IPC distribution, company-IPC heatmap, abstract length distribution, and filing trends over time.

### 3. Translation (`02b_translate.py`)

Detects language of each patent abstract using `langdetect`. Translates non-English abstracts to English via Google Translate (`deep-translator`). Of 3,769 patents, 321 were non-English — 243 German, 33 French, 27 Korean, 15 Spanish, and others. Saves the translated dataset with original text preserved.

### 4. NLP Similarity & Clustering (`03_nlp_similarity.py`)

The main analysis pipeline:

- **TF-IDF embeddings**: 5,000-dimensional vectors with bigrams and sublinear TF scaling
- **SBERT embeddings**: 384-dimensional semantic vectors from `all-MiniLM-L6-v2`
- **Cosine similarity matrices** for both methods
- **K-Means clustering** (k=5) with elbow and silhouette analysis
- **Hierarchical clustering** with Ward linkage dendrograms
- **t-SNE visualization** of company clusters
- **Network graph** construction (SBERT threshold=0.85) with centrality analysis
- **LDA topic modeling** (8 topics) showing technology themes per company

## Setup

### Prerequisites

- Python 3.9+
- EPO OPS API credentials ([register here](https://developers.epo.org/))

### Installation

```bash
git clone https://github.com/SaadHelal-IMC/Capstone-Project.git
cd Capstone-Project

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
EPO_API_KEY=your_consumer_key
EPO_API_SECRET=your_consumer_secret
```

### Running the Pipeline

Run the scripts in order:

```bash
# Step 1: Collect patent data from EPO API (~15-20 min)
python 01_data_collection.py

# Step 2: Clean names and generate EDA plots
python 02_eda.py

# Step 3: Translate non-English abstracts (~10 min)
python 02b_translate.py

# Step 4: Run NLP analysis, clustering, and graph construction
python 03_nlp_similarity.py
```

## Results

### Company Clusters (SBERT, k=5)

| Cluster | Label                   | Companies                                                                                                          |
| ------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------ |
| 0       | Industrial / Automotive | ABB, Airbus, BMW, Bosch, General Electric, Hitachi, Honda, Huawei, Schneider Electric, Siemens, Toyota, Volkswagen |
| 1       | Technology / Telecom    | Apple, Ericsson, Google, Honeywell, IBM, Intel, Microsoft, Nokia, Qualcomm, SAP                                    |
| 2       | Pharma Outlier          | Novartis                                                                                                           |
| 3       | Electronics / Materials | BASF, Canon, Infineon, LG Electronics, Panasonic, Samsung, Sony, Toshiba                                           |
| 4       | Healthcare              | Bayer, Philips, Roche                                                                                              |

### Method Comparison

| Metric                   | TF-IDF | SBERT  |
| ------------------------ | ------ | ------ |
| Embedding dimensions     | 5,000  | 384    |
| Silhouette score (k=5)   | 0.0627 | 0.1305 |
| Mean pairwise similarity | 0.430  | 0.796  |

## Technologies

- **Data Collection**: EPO OPS REST API, `requests`
- **NLP**: scikit-learn (TF-IDF, LDA, K-Means), sentence-transformers (SBERT)
- **Translation**: langdetect, deep-translator (Google Translate)
- **Graph Analysis**: NetworkX
- **Visualization**: matplotlib, seaborn

## License

This project was developed as an academic capstone at IMC University of Applied Sciences.
