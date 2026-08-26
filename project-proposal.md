---
geometry:
  - top=2.5cm
  - bottom=2.5cm
  - left=4cm
  - right=4cm
colorlinks: true
linkcolor: blue
urlcolor: blue
---

# Target-model Agnostic Evaluation Compression for image-classification Models: Project Proposal

## Problem
Deep neural networks are commonly evaluated on fixed labeled test datasets, where every input is processed to estimate model performance and reveal failures. As datasets and models grow, this evaluation and labeling effort can become resource intensive. Classical software testing addresses a similar problem through test-suite reduction. Here, a smaller subset of tests is selected while preserving the suite’s ability to expose faults. This project will investigate whether reduction methods, such as FAST++ and FAST-CS can be adapted to construct target-model-agnostic evaluation coresets for image-classification models. The aim is to determine whether a subset selected using only image representations can be labeled once and reused to reliably evaluate multiple present and future models, without depending on their predictions or internal representations.

## Method
The project will review literature on test-suite reduction, DNN testing, coreset selection, dataset diversity, and efficient model evaluation. FAST++, FAST-CS, and baselines such as random sampling will be adapted to operate on embeddings produced by an independent image encoder. Reduced evaluation sets of different sizes will be compared by how they impact dataset performance metrics, model rankings, and distinct failure modes across multiple classifiers and checkpoints. Since reduction methods may not preserve the original data distribution, approaches with weighting the samples may also be investigated. A second setting where labels are hidden during reduction can simulate label-efficient benchmark construction allowing the potential reduction in annotation cost to be quantified. Depending on scope, subgroup and fairness-metric preservation may also be studied.

Some relevant studies that the study is taking inspiration from are:

- ["Scalable Approaches for Test Suite Reduction", Cruciani et al. (ICSE 2019)](https://ieeexplore.ieee.org/document/8812048)
- ["Black-Box Testing of Deep Neural Networks through Test Case Diversity", Aghababaeyan et al. (IEEE TSE. 2023)](https://www.computer.org/csdl/journal/ts/2023/05/10041782/1KCLU6a4CSk)
- ["DeepSample: DNN Sampling-Based Testing for Operational Accuracy Assessment", Guerriero, Pietrantuono & Russo (ICSE 2024)](https://dl.acm.org/doi/10.1145/3597503.3639584)

## Dataset
The experiments will take it's offset in publicly available retinal image-classification datasets, with [Harvard-FairVision](https://github.com/Harvard-Ophthalmology-AI-Lab/FairVision) and [Harvard-GF](https://github.com/Harvard-Ophthalmology-AI-Lab/Harvard-GF). These provide supervised glaucoma, diabetic-retinopathy, or AMD classification tasks including demographic information suitable for subgroup evaluation. Their complete labeled test sets will serve as reference benchmarks, while the proposed methods will simulate selecting and labeling smaller reusable evaluation sets. The project will quantify the trade-off between evaluation fidelity, failure coverage, annotation cost, and repeated inference cost.

The deliverable will be a project report and a Github repository.
