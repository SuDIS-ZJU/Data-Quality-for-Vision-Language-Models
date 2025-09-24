# 📚 Awesome Data Quality for Large Vision–Language Models (ARC-LVLM)

*A living hub for data quality in LVLMs — taxonomy, diagnosis, and curated resources.*

[![Awesome](https://img.shields.io/badge/Awesome-yes-ffd700.svg)](https://awesome.re)
![Status](https://img.shields.io/badge/status-updating-brightgreen)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository accompanies the survey **“Data Quality Management for Large Vision–Language Models: Issues, Techniques, and Prospects.”**  
We organize the materials into four parts:

1. **Introduction**
2. **ARC Framework and Data Quality Issues**
3. **Diagnosis Framework**
4. **Relevant Papers**

> We will continuously maintain this repo with newly published papers, diagrams, and practical checklists.

---

## 1. Introduction

Large vision–language models (LVLMs) have rapidly advanced multimodal reasoning, generation, and interaction. Beyond architecture, **data scale and quality are decisive for LVLM reliability and capability**. This repo builds around a survey that frames data quality as both a **theoretical taxonomy** and a **practical diagnostic tool**:

- **ARC Framework** — a three-dimensional lens:
  - **Availability**: can we train at all (sufficiency, balance, and usability of multimodal corpora)?
  - **Reliability**: can we learn correctly (semantic fidelity, consistency, and cross-modal alignment)?
  - **Credibility**: can we trust and deploy (safety, ethics, privacy, and integrity)?
- **11 Representative Issues** across ARC, from scarcity/overload/imbalance to redundancy/mismatch to toxicity/poisoning/privacy.
- **Diagnosis Framework** that links **observable symptoms** (training failure, uneven performance, risky outputs) to **root-cause data flaws** and **targeted remedies**.

**Goal of this repo.** Serve as a practical companion to the survey:
- A clear **index of problems → methods → references**.
- **Diagrams** (ARC & Diagnosis) for quick onboarding.
- **Actionable checklists** and **paper digests** for engineers and researchers.

> If you find missing papers or have better categorizations, feel free to send a PR!

---

## 2. ARC Framework and Data Quality Issues
> *Coming up next.* We will add an at-a-glance chart of 11 issues (with “Appears: P/A/I”, key characteristics, and potential consequences), plus links to representative techniques and datasets.

---

## 3. Diagnosis Framework
> *Coming up next.* A step-by-step troubleshooting flow: from “can it train?” → “why unstable/uneven?” → “is it safe/credible?”, with playbook-style guidance.

---

## 4. Relevant Papers
> *Coming up next.* Curated list with concise tags: **[Availability] [Reliability] [Credibility] [Filtering] [Selection] [Alignment] [Privacy] [Safety] [Poisoning] …**  
> We will support fast lookup (by issue, method family, stage P/A/I) and BibTeX exports.

---

### How to Contribute
- Open an issue with the paper title, venue, year, and a one-sentence contribution.
- Or submit a PR editing `papers/relevant-papers.md`.
- Keep tags concise and consistent (we will provide a template).

### License
MIT for code/text in this repo. Figures follow their original licenses if stated.

### Citation
If you find this repository useful, please cite the accompanying survey (BibTeX coming soon).
