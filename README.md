# Researchify: A Tri-Component AI Research Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://researchify13.streamlit.app/)

Researchify is an academic security and document verification engine built in Python. Designed as a final project for the **Artificial Intelligence Lab** course at **UET Lahore (New Campus)**, the system evaluates the credibility and authenticity of online documents using three coordinated layers of artificial intelligence.

The system crawls target domains using heuristic-guided search, extracts text, measures multivariate stylistic patterns, tests those patterns against a machine learning ensemble, and feeds the resulting metrics into a declarative, rule-based expert system to calculate a Cumulative Trust Rating.

* **Live Demo:** [researchify13.streamlit.app](https://researchify13.streamlit.app/)

---

## Core Architecture

Researchify implements three distinct paradigms of artificial intelligence across a modular codebase:

```
                  [ USER INTERFACE / app.py ]
                              |
       +----------------------+----------------------+
       |                      |                      |
[ search_crawler.py ]  [ ml_classifier.py ]   [ logic_expert.py ]
(Informed Search AI)   (Machine Learning AI)  (Logic Reasoning AI)
```

### 1. Search Engine AI (`search_crawler.py`)
*   **Paradigm:** Informed Best-First Search
*   **Implementation:** A heuristic-guided web crawler that scores internal links within a domain. It assigns higher priorities for academic keyword density (`paper`, `abstract`, `journal`, etc.) and applies a depth penalty to avoid crawling irrelevant static directories. This focuses resources on text-heavy content pages.

### 2. Machine Learning AI (`ml_classifier.py`)
*   **Paradigm:** Ensemble Statistical Classifiers & Stylometrics
*   **Features Extracted:** 
    *   *Type-Token Ratio (TTR):* Measures lexical diversity and vocabulary richness.
    *   *Burstiness:* Calculates sentence length variance ($\sigma^2$) to identify natural rhythmic cadences.
    *   *Repetition Factor:* Identifies looping generative patterns.
    *   *Transition Density:* Measures overuse of typical LLM connective words (*furthermore*, *delve*, *testament*, etc.).
*   **Classifiers:** A dynamic voting ensemble combining **Multinomial Naive Bayes**, **Logistic Regression**, and **Decision Tree** models. The system outputs a consensus probability.

### 3. Logic-Based Reasoning AI (`logic_expert.py`)
*   **Paradigm:** Declarative First-Order Logic (FOL) Rule Engine
*   **Implementation:** Simulates a Prolog-style inference engine. It scans webpage structures using regular expressions to verify academic landmarks (DOIs, citation footprints, Scopus registry links). These states and the ML stylometric parameters are processed through declarative Horn clauses to generate a final trust rating (0–100) and an execution trace.

### 4. Interactive Dashboard & LLM Workspace (`app.py`)
*   **Implementation:** Built on Streamlit with custom CSS.
*   **Visualizers:** Features an interactive Plotly Dial Gauge for trust scores, a Radar/Spider Chart for linguistic signatures, and a dynamic color-coded sentence-level authenticity heatmap.
*   **Generative Workspace:** Integrates Google's Gemini API to summarize articles concisely, generate citations (APA, MLA, Chicago, BibTeX), scan for logical fallacies/biases, and humanize synthetic content to export as a compiled PDF.

---

## Installation & Setup (Local Deployment)

### Prerequisites
*   Python 3.9 or higher
*   A Google Gemini API key (optional, required to activate generative features)

### 1. Clone the Repository
```bash
git clone https://github.com/MuhammadAyanSajid/AI-Lab-Project
cd AI-Lab-Project
```

### 2. Install Dependencies
Install the required packages in your active Python environment:
```bash
pip install -r requirements.txt
```

### 3. Run the Application
Launch the local Streamlit development server:
```bash
streamlit run app.py
```

---

## Evaluation & Benchmarks

The system was benchmarked against traditional baseline methods to evaluate performance metrics:

### Classifier Accuracy (Stylometrics)
The ensemble approach was benchmarked against a baseline K-Nearest Neighbors (KNN) classifier:

| Metric | Multinomial Naive Bayes | Logistic Regression | Decision Tree | KNN (Baseline) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 91.7% | 93.3% | 86.5% | 75.0% |
| **Precision** | 90.0% | 92.3% | 84.6% | 78.6% |
| **Recall** | 93.8% | 94.1% | 88.2% | 73.3% |
| **F1-Score** | 91.8% | 93.2% | 86.3% | 75.8% |

### Scraper Node Expansion
The Informed Best-First Search crawler was benchmarked against a standard Breadth-First Search (BFS) model:

*   **Pages Crawled (Informed vs BFS):** 3 pages vs. 10 pages.
*   **Crawl Speed:** 0.45 seconds vs. 1.82 seconds.
*   **Noise Filtering (Media/CSS):** 100% vs. 0% (Required manual parsing).

---

## Contributors

This project was designed, developed, and defended by:

*   **[Muhammad Ayan Sajid](https://github.com/MuhammadAyanSajid)**
*   **[Muhammad Bilal](https://github.com/Bilal-013)**
*   **Seerat Fatima**
*   **[Muhammad Husnain](https://github.com/nexhus)**

**Department of Computer Science**  
**University of Engineering and Technology (UET) Lahore, New Campus**  
*Session: Fall 2024*

---

## Acknowledgements
Submitted as a final project for the **Artificial Intelligence Lab** course under the guidance of **Ms. Sonia Asghar**.