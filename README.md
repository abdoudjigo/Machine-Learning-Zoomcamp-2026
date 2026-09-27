<div align="center">

# 🤖 Machine Learning Zoomcamp 2026

### From Data to Deployed Model — A Structured ML Engineering Journey

[![Course](https://img.shields.io/badge/Course-DataTalks.Club-8A2BE2?style=for-the-badge)](https://github.com/DataTalksClub/machine-learning-zoomcamp)
[![Cohort](https://img.shields.io/badge/Cohort-2026-blue?style=for-the-badge)](https://courses.datatalks.club/ml-zoomcamp-2026/)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)]()

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)]()
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)]()
[![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=flat-square&logo=kubernetes&logoColor=white)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)]()
[![AWS](https://img.shields.io/badge/AWS%20Lambda-FF9900?style=flat-square&logo=awslambda&logoColor=white)]()

</div>

---

## 👋 À propos

Ce repo documente mon parcours complet dans le **[Machine Learning Zoomcamp 2026](https://github.com/DataTalksClub/machine-learning-zoomcamp)** de DataTalks.Club : un cursus gratuit et pratique qui va de l'introduction au Machine Learning jusqu'au déploiement de modèles en production (Docker, Kubernetes, Serverless).

Chaque module contient :
- 📓 **Mes notes** — résumé personnel des concepts (pas un copier-coller des slides)
- 💻 **Mon homework** — notebooks résolus
- 🔗 **Mes liens "learning in public"**

> **Objectif certification :** valider 2 projets sur 3 (Midterm + Capstone, ou Capstone 1 + 2), chacun avec un **modèle réellement déployé** et 3 peer-reviews complétées.

---

## 🗺️ Roadmap du cursus

```mermaid
flowchart TD
    A["01 · Introduction to ML<br/>CRISP-DM · Supervised Learning"] --> B["02 · Regression<br/>Car Price Prediction"]
    B --> C["03 · Classification<br/>Customer Churn"]
    C --> D["04 · Evaluation Metrics<br/>ROC · AUC · Cross-Val"]
    D --> E["05 · Deployment<br/>Flask/FastAPI · Docker"]
    E --> F["06 · Trees & Ensembles<br/>Random Forest · XGBoost"]
    F --> M{{"🎯 Midterm Project"}}
    M --> G["08 · Deep Learning<br/>CNN · Transfer Learning"]
    G --> H["09 · Serverless<br/>AWS Lambda"]
    H --> I["10 · Kubernetes<br/>TF Serving"]
    I --> C1{{"🏆 Capstone 1"}}
    C1 --> C2{{"🏆 Capstone 2 (optionnel)"}}

    style M fill:#8A2BE2,color:#fff
    style C1 fill:#2E8B57,color:#fff
    style C2 fill:#2E8B57,color:#fff
```

---

## 📊 Progression

<!-- Mets à jour les emojis au fil de l'eau : ⏳ à faire · 🔄 en cours · ✅ terminé -->

| # | Module | Notes | Homework | Statut |
|:-:|---|:-:|:-:|:-:|
| 01 | Introduction to Machine Learning | [📓](01-intro/notes.md) | [💻](01-intro/homework/) | 🔄 |
| 02 | Machine Learning for Regression | [📓](02-regression/notes.md) | [💻](02-regression/homework/) | ⏳ |
| 03 | Machine Learning for Classification | [📓](03-classification/notes.md) | [💻](03-classification/homework/) | ⏳ |
| 04 | Evaluation Metrics for Classification | [📓](04-evaluation/notes.md) | [💻](04-evaluation/homework/) | ⏳ |
| 05 | Deploying Machine Learning Models | [📓](05-deployment/notes.md) | [💻](05-deployment/homework/) | ⏳ |
| 06 | Decision Trees & Ensemble Learning | [📓](06-trees/notes.md) | [💻](06-trees/homework/) | ⏳ |
| 🎯 | **Midterm Project** | — | [🔗](midterm-project/) | ⏳ |
| 08 | Neural Networks & Deep Learning | [📓](08-deep-learning/notes.md) | [💻](08-deep-learning/homework/) | ⏳ |
| 09 | Serverless Deep Learning | [📓](09-serverless/notes.md) | [💻](09-serverless/homework/) | ⏳ |
| 10 | Kubernetes & TensorFlow Serving | [📓](10-kubernetes/notes.md) | [💻](10-kubernetes/homework/) | ⏳ |
| 🏆 | **Capstone Project 1** | — | [🔗](capstone-1/) | ⏳ |
| 🏆 | **Capstone Project 2** *(optionnel)* | — | [🔗](capstone-2/) | ⏳ |

**Avancement global :** ▓░░░░░░░░░░░ `1/12`

---

## 📁 Structure du repo

```
ml-zoomcamp-2026/
├── 01-intro/
│   ├── notes.md
│   └── homework/
│       └── homework_01.ipynb
├── 02-regression/
├── 03-classification/
├── 04-evaluation/
├── 05-deployment/
├── 06-trees/
├── 08-deep-learning/
├── 09-serverless/
├── 10-kubernetes/
├── midterm-project/        → lien vers repo dédié
├── capstone-1/              → lien vers repo dédié
├── capstone-2/              → lien vers repo dédié
└── README.md
```

---

## 🚀 Projets

| Projet | Description | Dataset | Déploiement | Repo |
|---|---|---|---|---|
| Midterm | *à définir* | *à définir* | *à définir* | — |
| Capstone 1 | *à définir* | *à définir* | *à définir* | — |
| Capstone 2 | *à définir* | *à définir* | *à définir* | — |

> ⚠️ Règle du cours : dataset **inédit** (jamais utilisé en cours), public, >100 lignes — modèle **obligatoirement déployé** (FastAPI+Docker, AWS Lambda, Streamlit/Gradio ou Kubernetes).

---

## 📢 Learning in Public

- 🔗 *lien 1*
- 🔗 *lien 2*

---

## 🔗 Liens utiles

- 📚 [Matériel du cours](https://github.com/DataTalksClub/machine-learning-zoomcamp)
- 🖥️ [Plateforme de soumission](https://courses.datatalks.club/ml-zoomcamp-2026/)
- 💬 [Slack DataTalks.Club](https://datatalks.club/slack.html)
- ❓ [FAQ officielle](https://datatalks.club/faq/machine-learning-zoomcamp.html)

---

<div align="center">

**Abdoulaye "Bou" DJIGO** · Data Engineering Student @ Orange Digital Center, Dakar
*Formé via Sonatel Académie — DEV DATA P8*

</div>