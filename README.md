# 💬 Shein App Review Sentiment Dashboard

**Turning Raw Feedback into Actionable Insights for Product Teams**

---

##  Overview

This project delivers a full-stack data analytics solution that transforms thousands of user reviews from the **Shein mobile app** into actionable business intelligence.

By combining NLP, clustering, automation, and dashboarding, it enables product teams to identify sentiment trends, recurring pain points, and critical feedback spikes — all in real time.

---

##  Live Demo

▶️ **[Try the Interactive Dashboard](https://maincusersisiakonedrivedocumentsprojectssheindashboardproject.streamlit.app/)**
*(No login required – view trends, filter by theme or sentiment, and export data.)*

---

## 🔍 Key Features

* 📅 **Automated Review Collection**
  Scrapes the latest Shein app reviews using `google_play_scraper`.

* 🧹 **Text Cleaning & Filtering**
  Deduplicates and filters non-English content with `langdetect`.

* 🧠 **Sentiment Classification**
  Uses Hugging Face’s `twitter-roberta-base-sentiment-latest` model.

* 🧩 **Thematic Clustering**
  Groups semantically similar reviews into 3 key themes using K-Means:

  * Delivery & Orders
  * App Performance
  * Product Quality

* 📊 **Streamlit Dashboard**
  Interactive filters, trend visualizations, and export functionality.

* 🔔 **Email Spike Alerts**
  Detects negative sentiment surges and sends automated alerts with a summary + CSV attachment.

---

## 🧱 Tech Stack

| Category   | Tools                                                                                   |
| ---------- | --------------------------------------------------------------------------------------- |
| Language   | Python 3.13                                                                             |
| NLP Models | SentenceTransformers, Hugging Face Transformers                                         |
| Libraries  | `pandas`, `scikit-learn`, `streamlit`, `transformers`, `jinja2`, `langdetect`, `joblib` |
| Dashboard  | Streamlit Cloud                                                                         |
| Alerting   | Jinja2 templates + SMTP                                                                 |
| Hosting    | Local or Cloud-deployable                                                               |

---

## 🛠️ Run Locally

### Prerequisites

* Python 3.11+
* Git

### Setup Instructions

```bash
# Clone the repo
git clone https://github.com/code-whisperer-abiala/Shein-Review-Dashboard.git
cd Shein-Review-Dashboard

# (Recommended) Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> *Note: Using a virtual environment is recommended for clean setup, but not mandatory if your system Python matches the requirements.*
