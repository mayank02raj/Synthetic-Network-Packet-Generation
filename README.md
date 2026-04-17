# Synthetic Network Packet Generation through Statistical Learning and Genetic Algorithms

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f7931e.svg)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/numpy-1.24%2B-013243.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/license-see%20below-lightgrey.svg)](#license-and-release-terms)
[![Paper](https://img.shields.io/badge/paper-under%20review%20%40%20ICCCN%202026-b31b1b.svg)](#paper)
[![Dataset](https://img.shields.io/badge/dataset-ACI--IoT--2023-purple.svg)](#dataset)

A constraint-enforcing synthetic IoT network packet generation framework. Two complementary methods — a statistical learning pipeline (PCA + dual OCSVM/Isolation Forest gating) and a genetic algorithm with multi-objective fitness — both embed hard validity constraints directly into the synthesis pipeline rather than hoping a generative model learns them implicitly. Evaluated on the complete ACI-IoT-2023 dataset (1,231,411 packets, 12 attack categories, class imbalance up to 175,805:1), both methods achieve PASS status across all 12 categories under independently trained validators, with the statistical method reaching 1.20% average anomaly rate at ~1,091 packets/sec and the GA reaching 0.62% at ~5.7 packets/sec — a **~190:1 throughput ratio** that turns into evidence-based method selection for rapid augmentation versus adversarial-robustness testing.

Supports the paper:

> **Raj, M.**, Bastian, N. D., Kul, G., Fiondella, L. *Synthetic Network Packet Generation through Statistical Learning and Genetic Algorithms.* Under review at ICCCN 2026.

## The problem with existing generators

Every published GAN, VAE, and tabular-diffusion approach for synthetic network traffic has the same structural gap: quality control is an **emergent property** of the adversarial training process, not an enforceable constraint. There is no mechanism to reject an individual synthetic packet that violates physical feature bounds or fails anomaly-model acceptance during synthesis. Post-hoc evaluation via downstream classifier performance cannot fix individual broken packets — it can only tell you, statistically, that some fraction of your synthetic set is invalid.

At the same time, classical techniques like SMOTE fail on extreme class imbalance. SMOTE interpolates between k-nearest neighbors, which requires sufficient local density. At **n = 5 samples** (ARP Spoofing in ACI-IoT-2023), the neighborhood structure is degenerate and any interpolation produces physically implausible feature combinations. The categories where defenders most urgently need synthetic data are precisely the ones where standard oversampling fails.

**This repository closes that gap with constraint-enforcing generation:** every synthetic packet must pass concurrent OCSVM and Isolation Forest acceptance, and every feature must land within observed class-specific bounds, before the packet is admitted to the output set. The validity criteria are hard gates, not soft training signals.

## Two methods, one evaluation framework

|Method                  |Architecture                                                                                                              |Throughput         |Avg. anomaly (worst tier)|Best for                                                                      |
|------------------------|--------------------------------------------------------------------------------------------------------------------------|-------------------|-------------------------|------------------------------------------------------------------------------|
|**Statistical Learning**|PCA latent sampling + Gaussian mixture + dual OCSVM/IF binary gate                                                        |**~1,091 pkts/sec**|1.20%                    |Rapid dataset augmentation, overnight IDS training expansion                  |
|**Genetic Algorithm**   |P=200 population, composite fitness, tournament selection, 3 crossover + 3 mutation operators, elitism, stagnation restart|~5.7 pkts/sec      |**0.62%**                |Adversarial robustness testing, red-team traffic diversity, IDS stress testing|

Both methods hit PASS (anomaly rate < τ = 0.30) across all 12 categories under independently trained validators that share no parameters with the generation-phase models.

**The GA’s trade-off:** lower worst-tier anomaly rate (0.62% vs. 1.20%) and organic per-class variance (0.00%–2.50%) better suited for downstream adversarial evaluation, at a 190× computational cost. The statistical method’s trade-off: near-uniform anomaly rates and two orders of magnitude higher throughput at the cost of reduced intra-class variability.

Neither method is universally superior — they are complementary tools for different operational contexts.

## Architecture

```
  ┌────────────────────────────────────────────────────────────┐
  │ Layer 1: Input Data                                         │
  │   ACI-IoT-2023 (1.23M pkts, 12 classes, 85 → 75 features    │
  │   after preprocessing)                                      │
  └─────────────────────────┬──────────────────────────────────┘
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
  ┌────────────────────────┐  ┌────────────────────────┐
  │ Layer 2a: Statistical   │  │ Layer 2b: Genetic       │
  │                         │  │                         │
  │ PCA latent sampling +   │  │ P=200 population        │
  │ Gaussian mixture        │  │ Composite fitness       │
  │        │                │  │ (OCSVM 0.4 + IF 0.4     │
  │        ▼                │  │  + e^-D 0.2)            │
  │ Dual OCSVM + IF gate    │  │ Tournament selection    │
  │ (binary accept/reject)  │  │ Crossover + mutation    │
  │                         │  │ Elitism + restart       │
  │ Feature clipping to     │  │ Feature clipping to     │
  │ [l_c, u_c]              │  │ [l_c, u_c]              │
  └────────────┬────────────┘  └────────────┬────────────┘
               │                            │
               └──────────────┬─────────────┘
                              ▼
  ┌────────────────────────────────────────────────────────────┐
  │ Layer 3: Independent Validation                             │
  │   OCSVM + IF models trained on ORIGINAL data only           │
  │   (no shared parameters with generation-phase models)       │
  │   4 tiers for SA · 2 tiers for GA · τ = 0.30                │
  └─────────────────────────┬──────────────────────────────────┘
                            ▼
  ┌────────────────────────────────────────────────────────────┐
  │ Layer 4: Analysis & Output                                  │
  │   Anomaly rate A_f · Boundary compliance B_c ·              │
  │   Distributional fidelity D · Per-class + aggregate reports │
  └────────────────────────────────────────────────────────────┘
```

The strict separation between generation and validation layers is what makes the quality claims defensible. Validators train on the original ACI-IoT-2023 data and never see the generator’s hyperparameters, seeds, or model weights. Passing validation therefore reflects generalization to independent decision boundaries, not self-assessment.

## Three research questions, three clean findings

**RQ1 — Does embedding dual anomaly-detection gating and feature-range clamping directly into the pipeline produce packets that independently trained validators consistently accept?**
Yes, across all 12 attack categories. The statistical method averages 0.00% anomaly under global OCSVM, 1.20% under global IF, 0.07% under class-specific OCSVM, and 0.00% under class-specific IF. The GA averages 0.62% OCSVM and 0.02% IF on its two validation tiers. The only elevated rate is 13.20% for Benign under the global IF tier — explained by Benign’s extreme intra-class variance (n = 879,027 across heterogeneous IoT device types) and not observed in any of the class-specific or GA tiers.

**RQ2 — Can constraint-enforcing generation amplify extremely scarce categories (n = 5) while maintaining anomaly rates below threshold?**
Yes, by **200×**. Both methods generate 1,000 validated ARP Spoofing packets from only 5 original samples. Statistical method: 0.00% max anomaly rate across all four tiers. GA: 2.50% max anomaly rate on OCSVM, 0.00% on IF — both well below τ = 0.30. The adaptive regularization νc = min(0.1, 1/(n+1)) capped at 0.1 prevents OCSVM collapse onto sparse points, while the GA’s three-source initialization (25% seeded + 50% Gaussian + 25% uniform) ensures genetic diversity from only 5 seeds.

**RQ3 — What are the quantitative trade-offs between statistical and evolutionary approaches?**
A ~190:1 throughput advantage for the statistical method (~1,091 vs. ~5.7 packets/sec), counterbalanced by the GA’s lower worst-tier anomaly (0.62% vs. 1.20%) and wider organic per-class variance (0.00%–2.50%) — making each method better suited to different operational contexts: rapid augmentation versus adversarial-diversity generation.

## What’s in this repository

|Component                    |Purpose                                                                                                                                                                                                                      |
|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|Statistical learning pipeline|PCA-based latent-space sampling + Gaussian mixture (α=0.7 PCA, 0.3 Gaussian), dual OCSVM+IF binary gate, feature clipping, batch rejection sampling                                                                          |
|Genetic algorithm pipeline   |P=200 population, composite fitness (Eq. 7 in paper), tournament-k3 selection, uniform/single-point/blend crossover, Gaussian/uniform/boundary-reset mutation (pm=0.05), top-20 elitism, 3-gen stagnation restart, 50-gen max|
|Preprocessing module         |Feature-wise imputation for Flow Bytes/s and Flow Packets/s (3,848 missing values), constant-variance column removal, one-hot encoding of Connection Type, StandardScaler normalization (85 → 75 features)                   |
|Adaptive parameter selection |νc = min(0.1, 1/(n+1)) for OCSVM regularization; ncomp = min(0.95, max(0.1, 1 − 10/n)) for PCA variance retention — both scale gracefully from n=5 to n=879,027                                                              |
|Independent validator        |Fresh OCSVM + IF trained on original ACI-IoT-2023 features only; global tier (10,000-sample cross-class, ν=0.01, 100 trees) and class-specific tier (adaptive νc, up to 5,000 samples per class)                             |
|Quality metric computation   |Anomaly rate A_f (Eq. 1), boundary compliance B_c (Eq. 2, always = 1.0 by construction), distributional fidelity D (Eq. 3, z-score normalized Euclidean)                                                                     |
|ACI-IoT-2023 data loader     |12-class stratified loader with class mappings for Benign, Port Scan, ICMP Flood, Ping Sweep, DNS Flood, Vulnerability Scan, OS Scan, Dictionary Attack, Slowloris, UDP Flood, SYN Flood, ARP Spoofing                       |

## Quick start

```bash
git clone https://github.com/mayank02raj/Synthetic-Network-Packet-Generation.git
cd Synthetic-Network-Packet-Generation

# Environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Dataset — ACI-IoT-2023 from the Army Cyber Institute at West Point
# See: Nack, McKenzie, Bastian (MILCOM 2024)
# Place CSV under ./data/ACI-IoT-2023/

# Generate synthetic packets (both methods, all 12 categories, 1,000 per class)
python generate_statistical.py --target 1000 --output synthetic_sa/
python generate_ga.py --target 1000 --output synthetic_ga/

# Run independent validation
python validate.py --synthetic synthetic_sa/ --method sa
python validate.py --synthetic synthetic_ga/ --method ga
```

Statistical method runs in ~11 seconds for all 12,000 synthetic packets on the hardware below. GA takes ~35 minutes for identical output.

## Reproducibility

|Setting                             |Value                                                                                                                       |
|------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
|Python                              |3.10+                                                                                                                       |
|scikit-learn                        |1.3+                                                                                                                        |
|NumPy / Pandas                      |1.24 / 2.0                                                                                                                  |
|Hardware (paper results)            |MacBook Pro 16-inch 2024, Apple M4 Max (16-core CPU, 40-core GPU, 16-core Neural Engine), 128 GB unified memory, macOS Tahoe|
|Random seed                         |Fixed (set in both generation scripts)                                                                                      |
|Dataset                             |ACI-IoT-2023, 1,231,411 packets, 12 classes, 5 categories (Benign, Recon, DoS, Brute Force, Spoofing)                       |
|Active features after preprocessing |d = 75 (reduced from 85 by removing constant-variance columns and one-hot encoding Connection Type)                         |
|Acceptance threshold                |τ = 0.30 on all validation tiers                                                                                            |
|Target synthetic packets per class  |M = 1,000                                                                                                                   |
|Max attempts per class (statistical)|5M = 5,000                                                                                                                  |
|Max GA generations                  |50 per class                                                                                                                |

### Statistical method hyperparameters

|Parameter                    |Symbol|Value                        |
|-----------------------------|------|-----------------------------|
|OCSVM regularization         |νc    |min(0.1, 1/(n+1))            |
|OCSVM kernel                 |K     |RBF, γ=scale                 |
|IF contamination             |cf    |= νc                         |
|IF ensemble size             |T     |100 trees                    |
|PCA variance retained        |vc    |min(0.95, max(0.1, 1 − 10/n))|
|PCA weight in hybrid sampling|α     |0.7                          |
|Gaussian weight              |1 − α |0.3                          |
|Noise scaling                |β     |0.5                          |
|Latent noise std             |σz    |0.1                          |
|Batch size                   |b     |100                          |

### Genetic algorithm hyperparameters

|Parameter               |Symbol                                 |Value                   |
|------------------------|---------------------------------------|------------------------|
|Population size         |P                                      |200                     |
|Elite count             |e                                      |20 individuals          |
|Mutation rate           |pm                                     |0.05                    |
|Tournament size         |k                                      |3                       |
|Max generations         |Gmax                                   |50                      |
|Target fitness          |f*                                     |0.9                     |
|Stagnation restart      |sr                                     |3 generations           |
|Noise (seeded init)     |σs                                     |0.01                    |
|Noise (Gaussian init)   |σg                                     |0.3                     |
|Mutation noise          |σm                                     |0.1 · rj (feature range)|
|Initial population split|25% seeded + 50% Gaussian + 25% uniform|                        |

## Complete results table

Condensed from Tables IV–VII of the paper:

### Statistical learning — independent validation (Table IV)

|Attack Category   |Global OCSVM (%)|Global IF (%)|Class OCSVM (%)|Class IF (%)|
|------------------|---------------:|------------:|--------------:|-----------:|
|Benign            |0.00            |**13.20**    |0.00           |0.00        |
|Port Scan         |0.00            |0.00         |0.00           |0.00        |
|ICMP Flood        |0.00            |0.00         |0.00           |0.00        |
|Ping Sweep        |0.00            |0.00         |0.00           |0.00        |
|DNS Flood         |0.00            |0.00         |0.00           |0.00        |
|Vulnerability Scan|0.00            |0.00         |0.00           |0.00        |
|OS Scan           |0.00            |0.00         |0.80           |0.00        |
|Dictionary Attack |0.00            |0.00         |0.00           |0.00        |
|Slowloris         |0.00            |0.00         |0.00           |0.00        |
|UDP Flood         |0.00            |0.00         |0.00           |0.00        |
|SYN Flood         |0.00            |0.00         |0.00           |0.00        |
|ARP Spoofing (n=5)|0.00            |0.00         |0.00           |0.00        |
|**Average**       |**0.00**        |**1.20**     |**0.07**       |**0.00**    |

### Genetic algorithm — independent validation (Table V)

|Attack Category   |OCSVM (%)|IF (%)  |Status        |
|------------------|--------:|-------:|--------------|
|Benign            |0.90     |0.00    |PASS          |
|Port Scan         |1.00     |0.00    |PASS          |
|ICMP Flood        |0.00     |0.00    |PASS          |
|Ping Sweep        |0.00     |0.00    |PASS          |
|DNS Flood         |0.60     |0.00    |PASS          |
|Vulnerability Scan|0.20     |0.00    |PASS          |
|OS Scan           |2.00     |0.00    |PASS          |
|Dictionary Attack |0.00     |0.20    |PASS          |
|Slowloris         |0.10     |0.00    |PASS          |
|UDP Flood         |0.00     |0.00    |PASS          |
|SYN Flood         |0.10     |0.00    |PASS          |
|ARP Spoofing (n=5)|**2.50** |0.00    |**PASS**      |
|**Average**       |**0.62** |**0.02**|**12/12 PASS**|

### Computational comparison (Table VII)

|Metric                             |Statistical      |GA                 |
|-----------------------------------|-----------------|-------------------|
|Total generation time (12K packets)|**~11 s**        |~35 min            |
|Throughput (packets/s)             |**~1,091**       |~5.7               |
|Throughput ratio                   |                 |**~190 : 1**       |
|Model predictions per packet       |2 (accept/reject)|~400 per generation|
|PASS rate (categories)             |12/12            |12/12              |
|Avg. anomaly (worst tier)          |1.20%            |**0.62%**          |
|Max single-class Af                |13.20%           |**2.50%**          |
|ARP Spoofing (n=5) max Af          |**0.00%**        |2.50%              |
|Boundary compliance Bc             |1.0              |1.0                |

## Why this work matters

Adversarial ML for network intrusion detection has a data problem that compounds every other problem. You cannot robustly evaluate IDS against novel or rare attacks if you only have 5 training samples of them. You cannot build adversarially trained detectors if your training distribution does not reflect the tail. And you cannot trust the output of a GAN or VAE unless you have an independent mechanism to reject the packets it generates that are physically impossible.

This repository proposes one answer: **make validity a hard constraint in the generation pipeline, not a hopeful property of the loss function.** Dual anomaly-detection gating plus feature clamping plus independent validation gives you synthetic data whose validity can be verified before you use it downstream, which is a different kind of data than what existing generators produce.

## Skills demonstrated

Synthetic data generation, anomaly detection (OCSVM, Isolation Forest), genetic algorithms / multi-objective optimization, PCA-based latent sampling, adaptive regularization under class imbalance, independent-validation experimental design, reproducibility practices, scientific Python tooling, IoT network security, IDS training-data engineering.

## Skills mapped to job postings

- **“Synthetic data generation”** — two complementary methods implemented end-to-end on a production-scale IoT security dataset
- **“Class imbalance handling”** — 200× amplification of a 5-sample category with validator acceptance
- **“Anomaly detection at scale”** — OCSVM + Isolation Forest dual-gating with adaptive regularization across five orders of magnitude of class size
- **“Genetic algorithms / evolutionary computation”** — P=200 multi-objective GA with composite fitness, three crossover and three mutation operators, elitism and stagnation restart
- **“Experimental rigor”** — independent validation with strict parameter separation between generation and assessment phases
- **“Reproducible ML”** — fixed seeds, pinned dependencies, complete hyperparameter tables, documented hardware
- **“Collaboration with government / defense research”** — DoD Cooperative Agreement, collaboration with U.S. Military Academy at West Point

## Related work in my portfolio

This repository is part of a coherent research program across adversarial ML for network security:

- [`Robustness-of-NIDS`](https://github.com/mayank02raj/Robustness-of-NIDS) — the adversarial-robustness side: three-architecture comparison (CNN, LSTM, Random Forest) under FGSM/PGD/CLEVER, formalizing the **False Champion Problem**. IEEE Access submission.
- [`SOC-home-lab`](https://github.com/mayank02raj/SOC-home-lab) — detection infrastructure: 11-service Dockerized SOC with Sigma rules, threat hunting, and ATT&CK-mapped adversary emulation
- [`ATTACK-Coverage-Dashboard`](https://github.com/mayank02raj/ATTACK-Coverage-Dashboard) — MITRE ATT&CK detection-coverage analytics with weighted scoring across 130+ threat actors
- [`Phishing-URL-Detector`](https://github.com/mayank02raj/Phishing-URL-Detector) — production-shaped ML service with SHAP explainability and PSI drift monitoring

The synthetic-generation work here provides the data that the robustness work uses: if Robustness-of-NIDS measures where detectors fail, this repository gives you the training data to fix them for the rare attack classes standard oversampling cannot touch.

## Paper

**Synthetic Network Packet Generation through Statistical Learning and Genetic Algorithms.**
Raj, M., Bastian, N. D., Kul, G., Fiondella, L.
*Under review at ICCCN 2026.*

Preprint available on request — please reach out via the contact details below. If you use this work, please cite:

```bibtex
@inproceedings{raj2026synthetic,
  title     = {Synthetic Network Packet Generation through Statistical Learning
               and Genetic Algorithms},
  author    = {Raj, Mayank and Bastian, Nathaniel D. and Kul, Gokhan and
               Fiondella, Lance},
  booktitle = {IEEE International Conference on Computer Communications and
               Networks (ICCCN) (under review)},
  year      = {2026},
  note      = {Preprint available on request}
}
```

## Funding and disclaimer

This work was supported by the U.S. Military Academy (USMA) under Cooperative Agreement No. W911NF-22-2-0160.

The views and conclusions expressed in this paper are those of the authors and do not reflect the official policy or position of the U.S. Military Academy or U.S. Army.

## License and release terms

License status is **pending institutional review** prior to open-source release.

This repository contains research software produced under a U.S. Department of Defense Cooperative Agreement (W911NF-22-2-0160). Data rights, release terms, and applicable open-source licensing are being confirmed with:

- The Principal Investigator (Dr. Gokhan Kul, UMass Dartmouth)
- UMass Dartmouth’s Office of Research Administration
- Co-investigator institutions (U.S. Military Academy at West Point)

Until a formal license is posted, please contact the corresponding author before using, redistributing, or building on this code. For academic evaluation and reference in review of the paper’s results, the code is provided as-is. A formal license file (`LICENSE`) will be added to this repository once release terms are finalized.

## Contact

**Mayank Raj** — M.S. Data Science (Thesis Track), UMass Dartmouth · Graduating May 2026

- Portfolio: [mayank02raj.github.io](https://mayank02raj.github.io)
- LinkedIn: [linkedin.com/in/mayank02raj](https://linkedin.com/in/mayank02raj)
- GitHub: [github.com/mayank02raj](https://github.com/mayank02raj)
- Email: mraj1@umassd.edu

Open to full-time cybersecurity and ML-security roles in the US. F-1 STEM OPT eligible — no sponsorship required through August 2029.

## Limitations and honest caveats

The paper is transparent about the following, and so is this README:

1. **Individual packet generation only.** Both methods generate packets independently without modeling the sequential dependencies inherent in real IoT traffic flows — device schedules, protocol state machines, multi-stage attack progressions. LSTM-based or temporal-GA approaches for sequence-aware generation are identified as future work.
1. **No protocol-semantic validation.** Boundary clamping ensures no feature exceeds observed ranges, but does not guarantee that resulting feature combinations correspond to valid MQTT, CoAP, Zigbee, or BLE protocol states. Protocol-aware constraint layers would strengthen semantic validity for protocol-level security evaluation.
1. **Single dataset.** All experiments use ACI-IoT-2023. The methodology is dataset-agnostic (requires only labeled numerical network features), but empirical validation on Bot-IoT, TON IoT, and Edge-IIoTset would strengthen generalizability claims.
1. **No downstream IDS evaluation yet.** The current work measures whether synthetic packets pass independent anomaly validators. Whether augmenting IDS training sets with these packets improves classifier precision/recall/F1 is a natural downstream study not included here.
1. **The GA’s 2.50% OCSVM anomaly rate on ARP Spoofing** is the highest single-class failure rate and reflects the difficulty of synthesizing from only 5 seed samples; still well under τ = 0.30 but worth acknowledging as the practical hard case.
1. **Statistical method’s Benign 13.20% under global IF** is driven by Benign’s extreme intra-class variance across diverse IoT device types. Not observed in any of the three other validation tiers. Documented rather than hidden.

## Extension ideas

- Extend both pipelines to sequence-aware generation (LSTM-based or temporal GA) for synthesizing realistic multi-step attack campaigns rather than independent packets
- Integrate protocol-aware constraint layers (MQTT, CoAP, Zigbee, BLE state-machine validation) for semantic-level validity
- Port evaluation to Bot-IoT, TON IoT, and Edge-IIoTset for cross-dataset validation of the constraint-enforcing approach
- Downstream IDS evaluation: augment training sets of detectors from [`Robustness-of-NIDS`](https://github.com/mayank02raj/Robustness-of-NIDS) with synthetic packets generated here, measure precision/recall/F1 impact on rare attack classes
- Hybrid method: use the statistical pipeline for fast bulk generation, feed outputs as seeds into the GA for targeted diversity injection — potentially capturing the throughput of SA with the diversity of GA
- Compare against recent LLM-based generators (Liu et al. MILCOM 2025) on the same constraint-enforcement evaluation
- Adversarial-training data pipeline: use GA outputs specifically for adversarial-training augmentation in IDS models and measure robustness improvement under FGSM/PGD

## Co-authors

- **Dr. Nathaniel D. Bastian** — Deputy Director of Robotics Research Center, U.S. Military Academy at West Point
- **Dr. Lance Fiondella** — Director of Cybersecurity Center, UMass Dartmouth (NSA/DHS-designated CAE-R)
- **Dr. Gokhan Kul** (advisor) — Associate Director of Cybersecurity Center, UMass Dartmouth 
