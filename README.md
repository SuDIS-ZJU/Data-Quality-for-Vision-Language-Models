# 📚 Awesome Data Quality for Large Vision–Language Models (ARC-LVLM)

*A living hub for data quality in LVLMs — taxonomy, diagnosis, and curated resources.*

[![Awesome](https://img.shields.io/badge/Awesome-yes-ffd700.svg)](https://awesome.re)  
![Status](https://img.shields.io/badge/status-updating-brightgreen)  
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository accompanies the survey **“Data Quality Management for Large Vision–Language Models: Issues, Techniques, and Prospects.”**  
We organize the materials into four parts:

- [1. Introduction](#1-introduction)  
- [2. ARC Framework and Data Quality Issues](#2-arc-framework-and-data-quality-issues)  
- [3. Diagnosis Framework](#3-diagnosis-framework)  
- [4. Relevant Papers](#4-relevant-papers)  

> We will continuously maintain this repo with newly published papers, diagrams, and practical checklists.

---

## 1. Introduction

Large vision–language models (LVLMs) have rapidly advanced multimodal reasoning, generation, and interaction. Beyond architecture, **data scale and quality are decisive for LVLM reliability and capability**. This repo builds around a survey that frames data quality as both a **theoretical taxonomy** and a **practical diagnostic tool**:

- **ARC Framework** — a three-dimensional lens:
  - **Availability**: can we train at all (sufficiency, balance, and usability of multimodal corpora)?
  - **Reliability**: can we learn correctly (semantic fidelity, consistency, and cross-modal alignment)?
  - **Credibility**: can we trust and deploy (safety, ethics, privacy, integrity)?
- **11 Representative Issues** across ARC, from scarcity/overload/imbalance to redundancy/mismatch to toxicity/poisoning/privacy.
- **Diagnosis Framework** that links **observable symptoms** (e.g. training failures, uneven performance, risky outputs) to **root-cause data flaws** and **targeted remedies**.

**Goal of this repo.** Serve as a practical companion to the survey:
- A clear **index of problems → methods → references**.
- **Diagrams** (ARC & Diagnosis) for quick onboarding.
- **Actionable checklists** and **paper digests** for engineers and researchers.

> If you find missing papers or have better categorizations, feel free to send a PR!

---

## 2. ARC Framework and Data Quality Issues

> *Coming up soon.*  
We will add:
- 概览表：11 个数据质量问题 + 它们所属的 ARC 维度 + 简要描述  
- 每个问题对应的代表方法与关键挑战  
- 图像：ARC 框架示意图  

---

## 3. Diagnosis Framework

> *Coming up soon.*  
内容包括：
- 诊断流程图（训练 / 适应 / 推理三阶段）  
- 各阶段可观察的异常类型  
- 根因定位至数据缺陷（ARC 视角）＋修复建议  

---

## 4. Relevant Papers

> *Coming up soon.*  
我们将准备一个按主题 / ARC 维度 / 方法类别索引的论文清单（Markdown 格式），每条带标签、摘要、BibTeX、链接。  

---

### How to Contribute

- 提交 Issue：提供论文题目、出版年份、所属问题 / 维度标签、简单一句话贡献  
- 提交 PR：编辑 `papers/relevant-papers.md` 或补充图示 / 诊断 checklist  
- 请遵循标签统一规范（例如 `[Availability]`, `[Reliability]`, `[Credibility]`）

### License

MIT for repository content (text, structure). Figures 保持其原始版权／许可。如无特别说明，按 CC BY 4.0。

### Citation

如果你觉得本仓库对你有帮助，请引用我们对应的 survey（BibTeX 稍后补充）。

