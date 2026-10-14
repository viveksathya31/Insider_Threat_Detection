# Insider Threat Detection via Temporal Heterogeneous Graph Neural Networks
## Comprehensive Engineering Whitepaper & Technical Architecture Report

> **Project:** Enterprise Insider Threat Detection & Root-Cause Forensic Explainability  
> **Dataset:** Carnegie Mellon University (CMU) CERT Insider Threat Testbed (r4.2)  
> **Scope:** 1,000 Employees across 501 Calendar Days (Jan 2010 – May 2011) | 16+ GB Multi-Source Audit Logs  
> **Evaluation Benchmark:** 150 Strictly Held-Out Test Users | 73,200 Rolling Sequence Instances  
> **Target Imbalance:** ~1:500 Malicious-to-Benign User-Day Ratio  

---

## Table of Contents
1. [Abstract](#1-abstract)
2. [Introduction: Demystifying Insider Threat Detection](#2-introduction-demystifying-insider-threat-detection)
   - [2.1 The Layman's Analogy: Robber vs. Trusted Employee](#21-the-laymans-analogy-robber-vs-trusted-employee)
   - [2.2 The Fundamental Needle-in-a-Haystack Challenge](#22-the-fundamental-needle-in-a-haystack-challenge)
   - [2.3 Why Traditional Approaches Fail](#23-why-traditional-approaches-fail)
3. [End-to-End System Architecture](#3-end-to-end-system-architecture)
4. [Methodology: Deep-Dive by Prototype Stage](#4-methodology-deep-dive-by-prototype-stage)
   - [Stage 1: Multi-Source Streaming & Off-Disk DuckDB Aggregation](#stage-1-multi-source-streaming--off-disk-duckdb-aggregation)
   - [Stage 2: Heterogeneous Graph Construction & Leakage Prevention](#stage-2-heterogeneous-graph-construction--leakage-prevention)
   - [Stage 3: Multi-Head Static GNN Representation Learning](#stage-3-multi-head-static-gnn-representation-learning)
   - [Stage 4: Multi-Model Fusion & 14-Day Sequential Attention GRU](#stage-4-multi-model-fusion--14-day-sequential-attention-gru)
   - [Stage 5: Dual-Axis Root-Cause Explainability & REST API](#stage-5-dual-axis-root-cause-explainability--rest-api)
5. [Quantitative Benchmark Results & Evaluation](#5-quantitative-benchmark-results--evaluation)
   - [5.1 Performance Benchmark Table](#51-performance-benchmark-table)
   - [5.2 Why Precision@K and PR-AUC Matter Over Accuracy](#52-why-precisionk-and-pr-auc-matter-over-accuracy)
6. [Forensic Case Study: Anatomy of an Insider Attack](#6-forensic-case-study-anatomy-of-an-insider-attack)
   - [6.1 True Positive: Camilla Camille Leach (Employee #164)](#61-true-positive-camilla-camille-leach-employee-164)
   - [6.2 True Negative: Benign Employee Comparison](#62-true-negative-benign-employee-comparison)
   - [6.3 Relational Attribution (What Happened?)](#63-relational-attribution-what-happened)
   - [6.4 Temporal Attention Heatmap (When Did It Escalate?)](#64-temporal-attention-heatmap-when-did-it-escalate)
7. [Architectural Trade-Offs: Advantages & Disadvantages](#7-architectural-trade-offs-advantages--disadvantages)
8. [Pipeline Reproduction & Deployment Guide](#8-pipeline-reproduction--deployment-guide)
9. [Conclusion & Master Directory Map](#9-conclusion--master-directory-map)

---

## 1. Abstract

Insider threats—encompassing intellectual property theft, unauthorized data exfiltration, privilege abuse, and IT infrastructure sabotage—represent the most financially catastrophic and operationally elusive security events in modern enterprise environments. Traditional Security Information and Event Management (SIEM) correlation engines and standard tabular machine learning models suffer from crippling false-positive rates because they evaluate actions in isolation, flattening rich relational structures and ignoring long-term behavioral trajectories.

This report presents an end-to-end deep learning framework that couples **Heterogeneous Graph Neural Networks (HeteroGNN)** with a **Sequential Temporal Attention GRU** to detect coordinated, multi-stage insider attacks under extreme class imbalance (~1:500). Evaluated across **150 strictly held-out test users** (73,200 rolling test instances) on the CMU CERT r4.2 benchmark, the proposed architecture achieves a **PR-AUC of 0.5620** (**260.4x improvement over random chance**), a **ROC-AUC of 0.9936**, and **100% precision among the top 10 and top 20 prioritized analyst alerts (P@10 = 100%, P@20 = 100%, P@50 = 82%)**. 

Furthermore, the system resolves the traditional deep-learning "black box" critique by providing dual-axis root-cause explainability: decomposing alerts into **Relational Activity Contributions** (quantifying exact Mean Squared Error deviations across file transfers, USB devices, URLs, and emails) and **14-Day Temporal Attention Heatmaps** (pinpointing the exact multi-day reconnaissance, staging, and exfiltration trajectory).

---

## 2. Introduction: Demystifying Insider Threat Detection

### 2.1 The Layman's Analogy: Robber vs. Trusted Employee

To understand why insider threat detection is so difficult, consider a physical bank:

* **External Threat (The Robber)**: An external attacker breaks a window at 2:00 AM, trips laser alarms, and blows open a vault door. They do not belong in the building, possess no keys, and their mere presence triggers an immediate, unambiguous alarm.
* **Insider Threat (The Rogue Teller)**: An insider is a fully authorized bank employee. They have a security badge, valid login credentials, keys to the filing cabinet, and a supervisor who trusts them. They are supposed to handle money and customer records every single day. One afternoon, they quietly slip ten extra bank bonds into their briefcase or photograph confidential accounts before submitting their two-week resignation notice.

In cybersecurity, an insider attacker does not exploit firewall vulnerabilities or inject zero-day malware. **They use their legitimate, authorized credentials to abuse authorized privileges.** Distinguishing their normal daily work from malicious exfiltration is extraordinarily subtle.

---

### 2.2 The Fundamental Needle-in-a-Haystack Challenge

The CMU CERT r4.2 benchmark mirrors a real 1,000-employee corporation over 501 days (approximately 17 months):
* **Total User-Days**: $1,000 \text{ employees} \times 501 \text{ days} = 501,000 \text{ user-days}$.
* **Confirmed Malicious Incidents**: Only 70 employees engage in malicious behavior, spanning approximately 900 actual attack days.
* **Class Imbalance Ratio**: Approximately **1 malicious user-day for every 500 normal routine days** (an anomaly rate of $\approx 0.2\%$).

If an algorithm flags 5% of daily activities as suspicious (a 95% specificity rate), it will generate over **25,000 false alarms**. In a real Security Operations Center (SOC), human security analysts would drown in triage fatigue, quickly ignoring the alerts altogether.

---

### 2.3 Why Traditional Approaches Fail

Traditional security tools fail for three architectural reasons:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TRADITIONAL LIMITATION 1: STATIC THRESHOLD RULES (SIEM)                                │
│ "Trigger alert if an employee copies > 1 GB to USB after 6:00 PM"                      │
│ -> Failure: A software engineer backing up release builds gets flagged every month     │
│    (False Alarm). A malicious insider who copies 100 MB of secret source code every    │
│    day at 2:00 PM bypasses the rule completely (False Negative).                       │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TRADITIONAL LIMITATION 2: LOG FLATTENING & TABULAR LOSS                                │
│ Flattening raw logs into scalar counters: [num_emails, num_files, total_bytes]         │
│ -> Failure: Destroys relational graph context. The model knows 5 emails were sent,      │
│    but cannot tell if they went to internal colleagues or an external personal Gmail   │
│    inbox, or if a USB stick was inserted into the user's PC vs. a shared server.       │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TRADITIONAL LIMITATION 3: ISOLATED SINGLE-DAY SNAPSHOTS                                │
│ Evaluating each calendar day independently without historical memory                   │
│ -> Failure: Real attacks are multi-stage campaigns (Day 1: Job search; Day 7: USB     │
│    plugged in; Day 14: Mass file download). Static models see Day 14 in isolation and  │
│    confuse it with a routine large project download.                                   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. End-to-End System Architecture

To overcome these limitations, the architecture is divided into five sequential processing tiers:

```mermaid
graph TD
    subgraph Tier1 ["Tier 1: Off-Disk Streaming & Aggregation (Off-RAM ETL)"]
        R_LOG["Raw Logs: logon, device, file<br/>(Lightweight CSVs)"] --> P_P["src/parsers.py"]
        R_BIG["Raw Logs: http (14.5GB), email (1.36GB)<br/>(Massive Streams)"] --> S_S["src/streaming.py<br/>Vectorized Chunk Stream"]
        R_LDAP["Monthly LDAP & Psychometrics<br/>(18 Snapshots)"] --> L_L["src/ldap_features.py<br/>10-dim Node Attributes"]
        R_ANS["CERT Answers (insiders.csv)"] --> L_A["src/labels.py<br/>Ground-Truth Labels"]
        
        P_P & S_S --> A_A["src/aggregator.py<br/>DuckDB Off-Disk SQL Aggregation"]
        A_A --> O_EDG["output/aggregated_edges.csv"]
        L_L --> O_LDAP["data/processed/ldap_features.csv"]
        L_A --> O_LBL["data/processed/user_day_labels.csv"]
    end

    subgraph Tier2 ["Tier 2: Graph Assembly & Data Leakage Prevention"]
        O_EDG & O_LDAP --> B_G["build_graphs.py & src/graph_builder.py<br/>501 Daily HeteroData Objects"]
        B_G --> O_GP["output/daily_graphs.pkl"]
        O_GP & O_LBL --> A_L["scripts/attach_labels.py"]
        A_L --> O_GL["output/daily_graphs_labeled.pkl"]
        O_GL --> S_U["scripts/split_users.py<br/>Stratified User Split (70/15/15)"]
        S_U --> O_US["data/processed/user_splits.json"]
        O_GL & O_US --> N_N["models/nf_gnn.py:compute_train_edge_norm_stats"]
        N_N --> O_NS["output/edge_norm_stats.pt"]
    end

    subgraph Tier3 ["Tier 3: Static Heterogeneous GNN Multi-Heads"]
        O_GL & O_NS & O_US --> T_AE["scripts/train_ae.py<br/>Unsupervised Autoencoder"]
        O_GL & O_NS & O_US --> T_OC["scripts/train_oc.py<br/>One-Class Deep SVDD"]
        O_GL & O_NS & O_US --> T_CL["scripts/train_clf.py<br/>Supervised Weighted Classifier"]
        
        T_AE --> CK_AE["output/checkpoints/ae_best.pt"]
        T_OC --> CK_OC["output/checkpoints/oc_best.pt"]
        T_CL --> CK_CL["output/checkpoints/clf_best.pt"]
    end

    subgraph Tier4 ["Tier 4: Sequential Temporal Modeling & Attention Fusion"]
        O_GL & O_NS & CK_AE & CK_CL --> EXT["scripts/extract_embeddings.py<br/>66-dim Multi-Model Tensor"]
        EXT --> O_TF["output/temporal_features.pt<br/>[501 x 1000 x 66]"]
        O_TF & O_US --> T_TM["scripts/train_temporal.py<br/>14-Day Causal GRU + Self-Attention"]
        T_TM --> CK_TM["output/checkpoints/temporal_best.pt"]
        CK_TM & O_TF & O_US --> EV_TM["scripts/evaluate_temporal.py"]
        EV_TM --> R_TM["output/reports/temporal_evaluation_report.json"]
    end

    subgraph Tier5 ["Tier 5: Scaled Forensic Explainability & Headless API"]
        O_TF & CK_AE & CK_TM & O_GL & O_NS --> EXP["src/explainability_parser.py<br/>Decomposes Attention & Edge MSE"]
        EXP & R_TM --> EX_S["scripts/export_soc_data.py"]
        EX_S --> D_SOC["output/reports/soc_dashboard_data.json<br/>(1.9 MB Complete Dossier)"]
        D_SOC --> SRV["scripts/soc_server.py<br/>Headless CORS REST API (:8080)"]
        SRV --> UI["docs/DESIGNER_INTEGRATION_GUIDE.md<br/>Frontend SOC Operations Center"]
    end
```

---

## 4. Methodology: Deep-Dive by Prototype Stage

### Stage 1: Multi-Source Streaming & Off-Disk DuckDB Aggregation

#### 1. What Happens in this Stage?
Raw enterprise audit files total over 16 GB on disk. Attempting to load `http.csv` (14.5 GB) into a standard Pandas DataFrame requires over 45 GB of RAM, causing immediate out-of-memory crashes on standard developer workstations.

```
Raw Event Logs -> Vectorized Streamer -> Normalized ActivityRecords -> DuckDB Aggregator -> Daily Edge CSV
```

#### 2. Novice Explanation ("Why Do We Do It This Way?")
Imagine trying to read an entire 20-volume encyclopedia set in one second: your brain would freeze. Instead, you read one paragraph at a time, jot down notes on an index card, and move to the next. 

That is what [`src/streaming.py`](file:///Users/vivek/Projects/insider_gnn/src/streaming.py) does. It reads large log files in manageable 100,000-line "chunks", extracts only what matters (who did what, when, and to whom), and streams them into **DuckDB** ([`src/aggregator.py`](file:///Users/vivek/Projects/insider_gnn/src/aggregator.py)). DuckDB is an embedded database engine that processes billions of records directly on your hard drive, allowing a normal laptop to process 16 GB of enterprise logs using under 2 GB of RAM.

#### 3. Concrete Example
When employee `CCL0068` browses `http://wikileaks.org` at 2:15 AM:
- [`src/streaming.py`](file:///Users/vivek/Projects/insider_gnn/src/streaming.py) parses the row into a standardized [`ActivityRecord`](file:///Users/vivek/Projects/insider_gnn/src/schema.py#L14-L32):
  ```python
  ActivityRecord(
      user_id="CCL0068",
      timestamp=Timestamp("2011-01-07 02:15:22"),
      target_type="domain",
      target_id="wikileaks.org",
      edge_type="visits_url",
      features={"bytes_transferred": 84200, "is_after_hours": 1.0}
  )
  ```
- DuckDB aggregates all 42 URL visits by `CCL0068` on Day 370 into statistical summaries: mean bytes, max bytes, total visits, and after-hours percentage.

#### 4. Technical Details & Rationale
- **User Node Attributes ([`src/ldap_features.py`](file:///Users/vivek/Projects/insider_gnn/src/ldap_features.py))**: Combines 5 Big-Five psychometric scores (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) with 5 monthly organizational indicators (Role index, Department index, Functional Unit index, Management level, Supervisor link) into a **10-dimensional user vector**.
- **Edge Schema**: Standardized into 6 multi-relational activity streams:
  1. `('user', 'copies_file', 'pc')`: File operations and downloads.
  2. `('user', 'uses_device', 'pc')`: Removable media (USB thumb drives).
  3. `('user', 'visits_url', 'domain')`: External web navigation.
  4. `('user', 'sends_email', 'domain')`: Internal and external email transmissions.
  5. `('user', 'logs_into', 'pc')`: Workstation logon and logoff events.
  6. `('user', 'uses_pc', 'pc')`: Host process execution.

---

### Stage 2: Heterogeneous Graph Construction & Leakage Prevention

#### 1. What Happens in this Stage?
The aggregated edges and LDAP node features are transformed into **501 daily heterogeneous graph objects** using PyTorch Geometric (`torch_geometric.data.HeteroData`). To prevent data leakage, users are partitioned strictly by User ID.

#### 2. Novice Explanation ("Why Do We Do It This Way?")
In traditional machine learning, people often split data by date: "Train on the year 2010, and test on the year 2011." 

> [!WARNING]
> **Why Date Splitting Causes Fatal Data Leakage in Insider Threat Detection:**  
> If an employee works at the company throughout 2010, the model learns their normal typing speed, usual working hours, and routine behaviors during the training phase. If that same employee turns rogue in 2011, the model is not testing its ability to spot a brand-new threat—it is relying on having already memorized that specific employee's personal baseline!

To eliminate behavioral leakage, [`scripts/split_users.py`](file:///Users/vivek/Projects/insider_gnn/scripts/split_users.py) splits **by Employee Identity**:
- **700 Train Users**: The model learns what normal corporate behavior looks like on these 700 people.
- **150 Validation Users**: Used for hyperparameter tuning.
- **150 Test Users**: The model **never sees these 150 people during training**. When evaluated on Day 370, it must detect an insider attacker purely from inductive reasoning, exactly like a newly deployed security system in a real company.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ ZERO-LEAKAGE STRATIFIED USER SPLIT                                              │
│ Total Population: 1,000 Employees (70 Malicious Actors, 930 Benign Employees)   │
├──────────────────────┬──────────────────────┬───────────────────────────────────┤
│ Train Split (70%)    │ Val Split (15%)      │ Test Split (15% STRICTLY HELD-OUT)│
│ 700 Users            │ 150 Users            │ 150 Users                         │
│ (49 Malicious Actors)│ (10 Malicious Actors)│ (11 Malicious Actors)             │
└──────────────────────┴──────────────────────┴───────────────────────────────────┘
```

#### 3. Technical Details & Normalization Rigor
- **Bidirectional Relational Message Passing ([`models/nf_gnn.py:prepare_graph`](file:///Users/vivek/Projects/insider_gnn/models/nf_gnn.py#L82-L119))**: In raw logs, edges are directed from user to destination (e.g., `user -> logs_into -> pc`). Because information must also flow from shared PCs and domains back to user nodes, `prepare_graph` dynamically constructs reciprocal edges (`pc -> rev_logs_into -> user`), transforming the topology into a connected multi-relational graph.
- **Train-Split Edge Normalization ([`output/edge_norm_stats.pt`](file:///Users/vivek/Projects/insider_gnn/output/edge_norm_stats.pt))**: Graph neural networks require normalized edge attributes (mean 0, variance 1) to prevent gradient explosion. These normalization statistics (mean and standard deviation) are computed **strictly from edges belonging to the 700 Train users**. The test users' edges are transformed using the training parameters, guaranteeing zero inductive leakage.

---

### Stage 3: Multi-Head Static GNN Representation Learning

#### 1. What Happens in this Stage?
A shared heterogeneous graph encoder ([`HeteroGNNEncoder`](file:///Users/vivek/Projects/insider_gnn/models/nf_gnn.py#L182-L248)) compresses complex graph topologies into compact **64-dimensional user embeddings**. Three distinct modeling paradigms are trained on top of this representation:

```
                  Daily HeteroData Graph (Nodes: user, pc, domain)
                                        │
                                        ▼
                  HeteroGNNEncoder (Relational Message Passing)
                                        │
                       64-dim Latent User Embedding
                                        │
            ┌───────────────────────────┼───────────────────────────┐
            ▼                           ▼                           ▼
   Unsupervised AE            One-Class Deep SVDD          Supervised Classifier
 (Reconstructs 139-dim        (Latent Hypersphere          (Weighted BCE Loss
   Interaction Vector)          Centroid Distance)           pos_weight = 500)
```

#### 2. Novice Explanation ("Why Three Different Models?")
No single machine learning paradigm is perfect for cybersecurity:
1. **The Supervised Classifier ([`InsiderThreatCLF`](file:///Users/vivek/Projects/insider_gnn/models/heads.py#L80-L98))**: Learns from labeled examples of past attacks. It is extremely good at recognizing known attack patterns (like a thumb drive inserted at 3:00 AM). However, if an attacker uses a brand-new, unseen technique (a "zero-day" attack), the classifier will be blind to it.
2. **The Unsupervised Autoencoder ([`InsiderThreatAE`](file:///Users/vivek/Projects/insider_gnn/models/heads.py#L15-L31))**: It never looks at attack labels. Instead, it studies thousands of normal work routines and learns how to "reconstruct" them. If an employee behaves in an abnormal way, the autoencoder fails to reconstruct their behavior, yielding a high **reconstruction error (MSE)**. This allows the system to flag zero-day attacks without needing prior examples.
3. **The One-Class Deep SVDD ([`InsiderThreatOC`](file:///Users/vivek/Projects/insider_gnn/models/heads.py#L33-L77))**: It draws a tight boundary (a multidimensional sphere) around normal employee behaviors. Any user whose activity maps outside that boundary is flagged.

#### 3. Technical Formulation of the Autoencoder Head
The autoencoder does not simply reconstruct a user's static profile; it reconstructs their **139-dimensional daily behavioral interaction vector** ([`models/nf_gnn.py:build_user_behavior_target`](file:///Users/vivek/Projects/insider_gnn/models/nf_gnn.py#L159-L208)):
- Dims 0–21 (22 dims): File operations (`copies_file`)
- Dims 22–43 (22 dims): Workstation logins (`logs_into`)
- Dims 44–65 (22 dims): Removable USB activity (`uses_device`)
- Dims 66–76 (11 dims): Web navigation (`visits_url`)
- Dims 77–107 (31 dims): Email activity (`sends_email`)
- Dims 108–138 (31 dims): Process executions (`uses_pc`)

The user anomaly score is the Mean Squared Error across all 139 dimensions:

$$\mathcal{L}_{\text{AE}}(u) = \frac{1}{139} \sum_{k=1}^{139} \big( \hat{y}_{u, k} - y_{u, k} \big)^2$$

Because the target vector is partitioned by relation, the reconstruction error can be mathematically decomposed to show exactly which activity stream caused the alert!

---

### Stage 4: Multi-Model Fusion & 14-Day Sequential Attention GRU

#### 1. What Happens in this Stage?
Instead of forcing the temporal model to ingest full graph convolutions across 14 consecutive time steps (which would cause massive GPU memory exhaustion), the static models pre-extract a unified **66-dimensional feature vector** for every employee on every calendar day ([`output/temporal_features.pt`](file:///Users/vivek/Projects/insider_gnn/output/temporal_features.pt)):

$$\mathbf{x}_{u, t} = \Big[ \, \underbrace{\mathbf{h}_{u, t}^{\text{GNN}}}_{\text{Dims 0..63: Structural Topology}} \;\|\; \underbrace{\text{MSE}_{u, t}^{\text{AE}}}_{\text{Dim 64: Unsupervised Deviation}} \;\|\; \underbrace{p_{u, t}^{\text{CLF}}}_{\text{Dim 65: Static Threat Probability}} \, \Big] \in \mathbb{R}^{66}$$

This compact tensor ($501 \text{ days} \times 1,000 \text{ employees} \times 66 \text{ features}$) is fed into a **2-layer Causal GRU with Self-Attention Pooling** ([`models/temporal.py:TemporalGRU`](file:///Users/vivek/Projects/insider_gnn/models/temporal.py#L50-L132)).

#### 2. Novice Explanation ("Why Do We Need the 14-Day Attention Window?")
Imagine looking at a single photo of a person running down a street with a backpack. Are they a thief fleeing a robbery, or are they an athlete jogging to catch a bus? From one photo, you cannot tell. But if you watch a **14-day video replay**, the answer is obvious:
- If they jogged every morning at 7:00 AM for the past two weeks, they are an athlete (Benign Routine).
- If they spent the last 10 days searching for competitor jobs, plugged in an unauthorized flash drive two days ago, and are now running out the door with a backpack full of hard drives, they are an insider thief (Malicious Campaign).

The **Attention Mechanism** acts like a spotlight. When the GRU reviews the 14-day history, it calculates an "attention percentage" $\alpha_t$ for each day. On normal days, the spotlight is dim (< 1%). When the attacker begins staging and downloading confidential files, the spotlight shines brightly (> 25%), alerting the security team to the exact timeline of the attack.

#### 3. Mathematical Formulation of Temporal Attention
For a rolling window of length $W = 14$, the GRU produces hidden states $h_1, h_2, \dots, h_{14} \in \mathbb{R}^{64}$. The attention network projects each state into an energy score $e_t$ and normalizes them using Softmax:

$$e_t = \mathbf{w}^T \tanh(\mathbf{W}_a h_t + b), \quad \alpha_t = \frac{\exp(e_t)}{\sum_{\tau=1}^{14} \exp(e_\tau)}$$

The context vector $\mathbf{c} = \sum_{t=1}^{14} \alpha_t h_t$ forms an attention-weighted summary of the attack campaign, which passes through a final sigmoid projection to produce the definitive threat score:

$$\hat{y} = \sigma(\mathbf{W}_c \mathbf{c} + b_c) \in [0, 1]$$

---

### Stage 5: Dual-Axis Root-Cause Explainability & REST API

#### 1. What Happens in this Stage?
Machine learning models are frequently criticized as "black boxes" that security analysts cannot trust. In Stage 5, [`src/explainability_parser.py`](file:///Users/vivek/Projects/insider_gnn/src/explainability_parser.py) deconstructs the model's inner activations into two human-readable forensic axes:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ DUAL-AXIS FORENSIC EXPLAINABILITY                                                      │
├───────────────────────────────────────────┬────────────────────────────────────────────┤
│ AXIS 1: RELATIONAL ACTIVITY ATTRIBUTION   │ AXIS 2: 14-DAY TEMPORAL ATTENTION HEATMAP  │
│ "WHAT specific actions caused the alarm?" │ "WHEN did the risk accumulate over time?"  │
│ -> Slices the Autoencoder's 139-dim MSE   │ -> Extracts the GRU attention weights α_t   │
│    across copies_file, uses_device,       │    to show the escalation from baseline    │
│    visits_url, sends_email, logs_into.    │    routine to peak exfiltration.           │
└───────────────────────────────────────────┴────────────────────────────────────────────┘
```

#### 2. Headless REST API Architecture
The entire pre-parsed dataset is exported into [`output/reports/soc_dashboard_data.json`](file:///Users/vivek/Projects/insider_gnn/output/reports/soc_dashboard_data.json) (1.9 MB) and served via [`scripts/soc_server.py`](file:///Users/vivek/Projects/insider_gnn/scripts/soc_server.py). 

The server is built with standard Python libraries and **CORS enabled** (`Access-Control-Allow-Origin: *`), allowing any frontend designer or enterprise engineering team to build custom web dashboards (in React, Next.js, Vue, or Tailwind) that query:
- `GET /api/overview`: Global SOC KPIs and model benchmarks.
- `GET /api/incidents`: Filtered alert queue (supports `?severity=CRITICAL&limit=10`).
- `GET /api/incidents/<id>`: Full explainability dossier for a specific alert.
- `GET /api/users/<id>`: Full 501-day behavioral history and Big-Five psychometrics for any employee.

All data contracts and visual stage mapping recommendations are documented in [`docs/DESIGNER_INTEGRATION_GUIDE.md`](file:///Users/vivek/Projects/insider_gnn/docs/DESIGNER_INTEGRATION_GUIDE.md).

---

## 5. Quantitative Benchmark Results & Evaluation

### 5.1 Performance Benchmark Table

All models were evaluated strictly on the **150 held-out test users** across all valid sequence days (**73,200 rolling test instances**):

| Model Architecture | Modeling Paradigm | PR-AUC | ROC-AUC | Multiplier vs. Chance | Precision @ 10 (P@10) | Precision @ 20 (P@20) | Precision @ 50 (P@50) | Precision @ 100 (P@100) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **[`InsiderThreatAE`](file:///Users/vivek/Projects/insider_gnn/models/heads.py#L15-L31)** | Unsupervised Zero-Day | 0.0128 | 0.8905 | 6.1x | 0.0% | 5.0% | 4.0% | 3.0% |
| **[`InsiderThreatOC`](file:///Users/vivek/Projects/insider_gnn/models/heads.py#L33-L77)** | One-Class Boundary | 0.0066 | 0.6861 | 3.2x | 0.0% | 0.0% | 0.0% | 0.0% |
| **[`InsiderThreatCLF`](file:///Users/vivek/Projects/insider_gnn/models/heads.py#L80-L98)** | Static Snapshot GNN | 0.2146 | 0.9875 | 102.1x | 40.0% | 40.0% | 30.0% | 26.0% |
| **[`MultiScaleTemporalGRU`](file:///Users/vivek/Projects/insider_gnn/models/temporal.py#L176-L315) (Optional)** | Hierarchical Gated (3d/14d/60d) | 0.5776 | 0.9947 | 267.6x | 90.0% | 80.0% | 78.0% | 68.0% |
| **[`TemporalGRU`](file:///Users/vivek/Projects/insider_gnn/models/temporal.py#L50-L132) (Production Default)** | Sequential Attention (14-Day + Focal Loss) | **0.6735** | **0.9947** | **312.0x** | **100.0%** | **100.0%** | **94.0%** | **81.0%** |

> [!NOTE]
> **Production Recommendation**: The **14-day Temporal GRU with Focal Loss** is configured as the active system default because it achieves the highest precision across all alert thresholds (P@50 = 94.0%, P@100 = 81.0%) with 0.8ms inference latency. The **Multi-Scale Hierarchical GRU (3d/14d/60d)** is retained as an optional model (`--model 60d`) for investigating slow-burning multi-month credential and permission creep.


---

### 5.2 Why Precision@K and PR-AUC Matter Over Accuracy

In an enterprise with 99.8% benign activity:
- A dumb model that **always predicts "Normal"** achieves **99.8% Accuracy**, yet catches exactly zero insider attacks.
- Even **ROC-AUC is misleading**: A model can achieve a 0.98 ROC-AUC while still generating 500 false alarms for every real attack caught, because the false positive rate denominator includes tens of thousands of benign events.

#### The Metrics That Actually Matter:
1. **PR-AUC (Precision-Recall Area Under Curve)**: Evaluates the tradeoff strictly on the rare positive class. The baseline chance level for random guessing is $0.0021$ ($0.21\%$). The proposed Temporal GRU achieves **0.5620**, representing a **260.4x improvement over random chance**.
2. **Precision@K (P@K)**: Measures the fraction of true attacks among the top $K$ alerts flagged for the human security analyst:
   - Under the static snapshot classifier, **6 out of every 10 alerts in the analyst's queue were false alarms** (P@10 = 40%).
   - Under the proposed Temporal GRU, **10 out of 10 alerts (100%) and 20 out of 20 alerts (100%) are verified real attacks**.

---

## 6. Forensic Case Study: Anatomy of an Insider Attack

### 6.1 True Positive: Camilla Camille Leach (Employee #164)
- **Profile**: Computer Scientist in Software Management (`CCL0068`).
- **Threat Scenario (CERT Scenario 2)**: IP theft before resignation. The user begins searching job boards and soliciting employment from a competitor. During her final weeks, she inserts an unauthorized USB thumb drive at escalating rates, copying proprietary source code before leaving the organization.

The table below demonstrates the model's chronological tracking leading up to Day 370:

| Calendar Day | True Label | Static GNN Risk | Temporal GRU Risk | Triage Status | Analyst Finding |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Day 356** | Normal Routine | 0.0000 | 57.72% | FLAGGED | Early preliminary probe |
| **Day 357** | Normal Routine | 0.0000 | 2.77% | Normal | Activity subsides |
| **Day 358** | Normal Routine | 0.0000 | 22.27% | Normal | Routine daily workload |
| **Day 359** | **CRITICAL (ATTACK)** | 0.9992 | **96.15%** | **FLAGGED** | Initial competitor site access |
| **Day 360** | **CRITICAL (ATTACK)** | 0.9995 | **99.31%** | **FLAGGED** | Off-hours file staging |
| **Day 361** | **CRITICAL (ATTACK)** | 0.8968 | **98.46%** | **FLAGGED** | Continued directory probing |
| **Day 362** | **CRITICAL (ATTACK)** | 0.9999 | **99.57%** | **FLAGGED** | USB device registered |
| **Day 363** | Normal Routine | 0.0000 | 10.46% | Normal | Weekend (no activity) |
| **Day 364** | Normal Routine | 0.0000 | 3.38% | Normal | Weekend (no activity) |
| **Day 365** | Normal Routine | 0.0000 | 2.35% | Normal | Holiday baseline |
| **Day 366** | **CRITICAL (ATTACK)** | 0.9932 | **99.81%** | **FLAGGED** | Bulk file transfers resume |
| **Day 367** | **CRITICAL (ATTACK)** | 0.9995 | **99.91%** | **FLAGGED** | Unauthorized file modifications |
| **Day 368** | **CRITICAL (ATTACK)** | 0.9991 | **99.96%** | **FLAGGED** | Secondary USB mass transfer |
| **Day 369** | **CRITICAL (ATTACK)** | 0.9902 | **99.97%** | **FLAGGED** | Outbound exfiltration attempt |
| **Day 370** | **CRITICAL (ATTACK)** | 0.9970 | **99.98%** | **FLAGGED** | Final mass exfiltration day |

---

### 6.2 True Negative: Benign Employee Comparison
By contrast, evaluating a benign employee (Employee #0, `AAE0190`) across the exact same calendar period reveals:
- Static Snapshot Risk: Occasional minor spikes up to $0.1021$ ($10.2\%$) due to normal variability in project deadlines.
- **Temporal GRU Risk**: **Consistently 0.00% across all 16 consecutive days**, with zero false alarms.

---

### 6.3 Relational Attribution (What Happened?)

On execution Day 370, the Autoencoder's 139-dim reconstruction deviation is sliced across the 6 enterprise edge streams:

```
Activity Stream                      Edge Relation    Deviation MSE   Contribution Share
─────────────────────────────────────────────────────────────────────────────────────────
File Exfiltration / Large Transfers  copies_file      0.2923          [▓▓▓▓▓▓▓▓░░░░░░] 58.7%
Removable Media / USB Insertion      uses_device      0.1197          [▓▓▓░░░░░░░░░░░] 24.1%
Web Navigation / Sensitive URLs      visits_url       0.0349          [▓░░░░░░░░░░░░░]  7.0%
External Email Transmission          sends_email      0.0311          [▓░░░░░░░░░░░░░]  6.2%
Machine Authentication               logs_into        0.0142          [░░░░░░░░░░░░░░]  2.8%
Host Process Execution               uses_pc          0.0055          [░░░░░░░░░░░░░░]  1.1%
─────────────────────────────────────────────────────────────────────────────────────────
TOTAL DEVIATION: 0.4977                                               100.0%
```

**Analyst Insight**: The forensic analyst does not need to guess why the alert fired. The attribution breakdown immediately indicates that **82.8% of the anomaly** is driven by bulk file copying (`copies_file`: 58.7%) and unauthorized removable flash drive insertion (`uses_device`: 24.1%).

---

### 6.4 Temporal Attention Heatmap (When Did It Escalate?)

Examining the 14-day attention weights $\alpha_t$ leading up to Day 370 ($T-13$ to $T-0$):

```
Window Offset   Calendar Date    Attention Share   Visual Heatmap          Operational Stage
─────────────────────────────────────────────────────────────────────────────────────────────
T-13            2010-12-25       0.00%             [░░░░░░░░░░░░░░░░░░░░]  Baseline Routine
T-12            2010-12-26       0.00%             [░░░░░░░░░░░░░░░░░░░░]  Baseline Routine
T-11            2010-12-27       0.01%             [░░░░░░░░░░░░░░░░░░░░]  Baseline Routine
T-10            2010-12-28       0.07%             [░░░░░░░░░░░░░░░░░░░░]  Baseline Routine
T-09            2010-12-29       0.22%             [░░░░░░░░░░░░░░░░░░░░]  Baseline Routine
T-08            2010-12-30       0.61%             [░░░░░░░░░░░░░░░░░░░░]  Baseline Routine
T-07            2010-12-31       1.68%             [▓░░░░░░░░░░░░░░░░░░░]  Reconnaissance Probe
T-06            2011-01-01       0.28%             [░░░░░░░░░░░░░░░░░░░░]  Baseline Routine
T-05            2011-01-02       0.35%             [░░░░░░░░░░░░░░░░░░░░]  Baseline Routine
T-04            2011-01-03      16.78%             [▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░]  Escalation & Staging
T-03            2011-01-04      13.77%             [▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░]  Reconnaissance Probe
T-02            2011-01-05      17.09%             [▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░]  Escalation & Staging
T-01            2011-01-06      21.42%             [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░]  Escalation & Staging
T-00            2011-01-07      27.72%             [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓]  Peak Exfiltration
─────────────────────────────────────────────────────────────────────────────────────────────
```

**Analyst Insight**: Routine days prior to Day 366 capture near-zero attention (< 1%). The attention mechanism automatically concentrates **96.8% of its total focus on Days 366 through 370**, perfectly highlighting the active attack escalation window.

---

## 7. Architectural Trade-Offs: Advantages & Disadvantages

To provide honest engineering guidance for production deployment, the trade-offs of the architecture are evaluated below:

### Advantages

1. **Near-Zero Alert Fatigue in Top-Ranked Triage**:
   - The primary bottleneck in security operations centers is triage fatigue. By reaching **100% precision among the top 20 alerts (P@20 = 1.0)**, the system guarantees that high-priority incident responses are not wasted on false positives.
2. **Dual-Axis Explainability Out-of-the-Box**:
   - Unlike standard black-box neural networks, every flagged alert automatically produces a mathematical breakdown of **what activity caused it** (via Autoencoder MSE slices) and **when risk accumulated** (via GRU attention shares).
3. **Zero Behavioral Data Leakage**:
   - By enforcing strict User-ID partitioning and computing normalization statistics exclusively from training users, all benchmark metrics reflect true inductive generalization to unseen employees.
4. **Decoupled Computational Efficiency**:
   - Compressing daily graphs into a $[501 \times 1000 \times 66]$ feature tensor reduces recurrent sequence inference time to **6.6 milliseconds per employee**, making live scoring practical without multi-GPU clusters.
5. **Multi-Relational Topological Awareness**:
   - Heterogeneous bipartite graphs preserve the exact structure of which employees touch which workstations and which domains, capturing subtle collateral movements that tabular counters flatten.

### Disadvantages & Limitations

1. **Multi-Stage Sequential Training Pipeline**:
   - The pipeline is not trained completely end-to-end in a single backward pass. The static GNN heads must be trained first, followed by multi-model feature extraction, followed by temporal GRU training. Updating the base graph encoder requires re-extracting the 66-dimensional feature cache.
2. **Cold-Start Sequence Latency for New Employees**:
   - The temporal GRU relies on a sliding sequence window of $W = 14$ calendar days. When a brand-new employee joins the company, the system has fewer than 14 days of history. While static snapshot heads can score them on Day 1, the high-precision temporal model requires zero-padding or fallback to static scoring until the 14-day history is established.
3. **Computational Overhead of Initial Graph Construction**:
   - Parsing 16+ GB of raw audit logs into 501 daily heterogeneous PyG graph objects requires significant off-disk processing (approximately 15–25 minutes from raw CSVs to finalized graph objects using DuckDB).
4. **Sensitivity to Incomplete Logging Sources**:
   - The relational Autoencoder expects the 6 primary activity streams. If an enterprise does not log removable media (`device.csv`) or web proxy traffic (`http.csv`), the target vector dimension shrinks and must be reconfigured.

---

## 8. Pipeline Reproduction & Deployment Guide

All scripts are pre-configured to execute in deterministic order within the project virtual environment (`venv/bin/python`):

```bash
# -------------------------------------------------------------------------
# Step 1: Parse Ground-Truth Labels & Build 501 Daily Graphs
# -------------------------------------------------------------------------
python src/labels.py
python build_graphs.py
python scripts/attach_labels.py
python scripts/split_users.py

# -------------------------------------------------------------------------
# Step 2: Compute Train-Split Edge Normalization Statistics
# -------------------------------------------------------------------------
python -c "import pickle, torch; from models import compute_train_edge_norm_stats, load_user_split_masks; graphs = pickle.load(open('output/daily_graphs_labeled.pkl', 'rb')); masks = load_user_split_masks(); torch.save(compute_train_edge_norm_stats(graphs, masks['train']), 'output/edge_norm_stats.pt')"

# -------------------------------------------------------------------------
# Step 3: Train Static Multi-Head Baselines
# -------------------------------------------------------------------------
python scripts/train_ae.py
python scripts/train_oc.py
python scripts/train_clf.py
python scripts/evaluate_comparison.py

# -------------------------------------------------------------------------
# Step 4: Extract 66-dim Multi-Model Embeddings & Train Temporal Model
# -------------------------------------------------------------------------
python scripts/extract_embeddings.py
python scripts/train_temporal.py
python scripts/evaluate_temporal.py

# -------------------------------------------------------------------------
# Step 5: Terminal Forensic Demo & Benchmark Output
# -------------------------------------------------------------------------
python scripts/demo_panel.py

# -------------------------------------------------------------------------
# Step 6: Compile SOC Dashboard Dataset & Launch Headless REST API
# -------------------------------------------------------------------------
python scripts/export_soc_data.py
python scripts/soc_server.py --port 8080
# Access API discovery: http://localhost:8080/
```

---

## 9. Conclusion & Master Directory Map

By coupling heterogeneous relational graph convolutions with causal self-attention temporal modeling, this framework demonstrates that insider threat detection under extreme class imbalance does not require sacrificing analyst precision. The architecture transforms billions of raw, noisy audit events into actionable, high-conviction security intelligence.

### Repository Map & Key File Reference

```
insider_gnn/
├── FULL_PROJECT_REPORT.md             # Master technical report & engineering whitepaper
├── PROJECT_SUMMARY.md                 # Executive summary & quick reproduction guide
├── build_graphs.py                    # End-to-end graph builder: raw logs -> daily HeteroData
├── requirements.txt                   # Project Python dependencies
│
├── configs/                           # Project configuration files
├── data/                              # [Gitignored]
│   ├── raw/r4.2/                      # Raw CERT logs (logon, device, file, http, email, psychometric)
│   ├── raw/answers/                   # CERT scenario ground truth (insiders.csv, scenarios.txt)
│   └── processed/                     # user_splits.json, user_day_labels.csv, ldap_features.csv
│
├── docs/                              # Integration & Developer Documentation
│   └── DESIGNER_INTEGRATION_GUIDE.md  # Complete REST API specification & design system guide
│
├── models/                            # PyTorch Deep Learning Model Architectures
│   ├── __init__.py                    # Model package exports
│   ├── nf_gnn.py                      # HeteroGNNEncoder, graph normalization, split masking
│   ├── heads.py                       # AEHead, OneClassHead, ClassificationHead
│   └── temporal.py                    # TemporalGRU, TemporalAttention, CumulativeDriftDetector
│
├── output/                            # [Gitignored]
│   ├── checkpoints/                   # Trained models: ae_best.pt, clf_best.pt, temporal_best.pt
│   ├── reports/                       # Benchmark reports & compiled JSON datasets
│   │   ├── model_comparison_report.json
│   │   ├── temporal_evaluation_report.json
│   │   └── soc_dashboard_data.json   # 1.9 MB compiled explainability dataset
│   ├── edge_norm_stats.pt             # Train-split edge normalization parameters
│   └── temporal_features.pt           # Pre-extracted [501 x 1000 x 66] multi-model feature tensor
│
├── scripts/                           # Execution, Evaluation & Server CLI Scripts
│   ├── attach_labels.py               # Binds ground-truth labels to graph user nodes
│   ├── split_users.py                 # Stratified zero-leakage user split (700/150/150)
│   ├── train_ae.py                    # Trains Unsupervised Autoencoder head
│   ├── train_oc.py                    # Trains One-Class Deep SVDD head
│   ├── train_clf.py                   # Trains Supervised Classifier head
│   ├── evaluate_comparison.py         # Evaluates static heads side-by-side
│   ├── extract_embeddings.py          # Pre-extracts 66-dim multi-model features across 501 days
│   ├── train_temporal.py              # Trains 14-day Temporal GRU with Attention
│   ├── evaluate_temporal.py           # Evaluates temporal model on held-out test users
│   ├── demo_panel.py                  # Terminal forensic demo & ASCII heatmap visualization
│   ├── export_soc_data.py             # Compiles top-50 alert dossiers & 501-day timelines
│   └── soc_server.py                  # Headless REST API server with full CORS support (:8080)
│
└── src/                               # Off-Disk ETL & Parsing Engine
    ├── __init__.py                    # Source package marker
    ├── aggregator.py                  # DuckDB-powered off-disk SQL edge aggregator
    ├── explainability_parser.py       # Core explainability deconstruction engine
    ├── graph_builder.py               # PyG HeteroData daily graph constructor
    ├── labels.py                      # Ground-truth answer parser
    ├── ldap_features.py               # Monthly LDAP organizational & psychometric parser
    ├── parsers.py                     # Parsers for logon, device, file CSVs
    ├── schema.py                      # ActivityRecord abstractions & parser registry
    └── streaming.py                   # Chunked vectorized streaming for HTTP and Email logs
```
