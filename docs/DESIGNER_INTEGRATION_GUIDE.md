# Frontend Designer & UI Developer Integration Guide

> **Project:** CYBER-GNN Forensic SOC Dashboard  
> **Backend Architecture:** Heterogeneous Graph Neural Network + Sequential Attention GRU  
> **Dataset:** CMU CERT Insider Threat Testbed (r4.2) | 1,000 Employees across 501 Days  
> **Data File Location:** [`output/reports/soc_dashboard_data.json`](file:///Users/vivek/Projects/insider_gnn/output/reports/soc_dashboard_data.json)

---

## 1. Quick Start for Designers

The backend provides two ways to consume forensic explainability data:

### Option A: Local REST API Server (Live or Mock)
The server runs on Python's built-in standard library with **CORS enabled** (`Access-Control-Allow-Origin: *`), so you can build with any modern frontend stack (React, Next.js, Vite, Vue, Svelte, Tailwind, or Vanilla JS) on any port (`localhost:3000`, `localhost:5173`, etc.):

```bash
# Start backend API server on port 8080
venv/bin/python scripts/soc_server.py --port 8080
```
Base URL: `http://localhost:8080`

### Option B: Direct Static JSON Import
If you prefer building offline or bundling data directly into your frontend build, import:
```
output/reports/soc_dashboard_data.json (1.9 MB)
```
This file contains the complete pre-extracted dataset (top-50 alert dossiers, 21 full 501-day employee timelines, and benchmark metrics).

---

## 2. REST API Endpoints Specification

### 2.1 Overview & SOC KPIs
- **Endpoint**: `GET /api/overview`
- **Description**: High-level SOC telemetry, verification benchmarks, and scope summary.
- **Example Response**:
```json
{
  "kpi": {
    "total_monitored_users": 1000,
    "evaluation_days": 501,
    "test_users": 150,
    "test_instances_evaluated": 73200,
    "temporal_pr_auc": 0.562,
    "temporal_roc_auc": 0.9936,
    "temporal_multiplier": 260.4,
    "precision_at_10": 100.0,
    "precision_at_20": 100.0,
    "precision_at_50": 82.0,
    "false_positive_reduction": "6.5x vs Static CLF"
  },
  "metadata": {
    "generated_at": "2026-10-14 06:59:02 UTC",
    "dataset": "CMU CERT Insider Threat Testbed (r4.2)",
    "pipeline": "Temporal Heterogeneous GNN + Attention GRU"
  }
}
```

---

### 2.2 Alerts Triage Queue
- **Endpoint**: `GET /api/incidents`
- **Query Parameters**:
  - `severity`: Filter by `CRITICAL` (>=95%), `HIGH` (>=80%), `MEDIUM` (>=50%).
  - `scenario`: Filter by scenario ID (`1`, `2`, `3`).
  - `user`: Filter by CERT User ID (e.g., `CCL0068`).
  - `limit`: Integer limit (e.g., `limit=20`).
- **Example Response**:
```json
{
  "total": 50,
  "incidents": [
    {
      "incident_id": "INC-370-CCL0068",
      "rank": 1,
      "user_idx": 164,
      "user_id": "CCL0068",
      "day": 370,
      "calendar_date": "2011-01-07",
      "severity": "CRITICAL",
      "temporal_risk": 0.9998,
      "static_risk": 0.9970,
      "ae_error": 0.0784,
      "is_malicious": 1,
      "split": "test",
      "profile": {
        "employee_name": "Camilla Camille Leach",
        "email": "Camilla.Camille.Leach@dtaa.com",
        "role": "ComputerScientist",
        "functional_unit": "2 - ResearchAndEngineering",
        "department": "2 - SoftwareManagement",
        "team": "1 - Software",
        "supervisor": "Vance Ira Alston"
      },
      "psychometrics": {
        "name": "Camilla Camille Leach",
        "O": 13,
        "C": 16,
        "E": 23,
        "A": 19,
        "N": 23
      },
      "scenario": {
        "scenario_id": 2,
        "title": "IP Theft & Solicit Competitor Employment Prior to Resignation",
        "pattern": "Browsing job boards and competitor domains, escalating thumb drive transfers to copy trade secrets before resignation.",
        "mitre_tactics": ["Collection (TA0009)", "Data Staged (T1074)", "Exfiltration via Removable Media (T1052)"],
        "start": "12/27/2010 09:20:16",
        "end": "02/21/2011 08:25:17"
      },
      "attention_heatmap": [ ... 14 daily objects ... ],
      "relational_attribution": [ ... 6 edge breakdown objects ... ]
    }
  ]
}
```

---

### 2.3 Single Incident Forensic Dossier
- **Endpoint**: `GET /api/incidents/<incident_id>`
- **Description**: Returns the comprehensive explainability dossier for an alert (e.g. `INC-370-CCL0068`).
- **Explainability Axis 1: 14-Day Temporal Attention Heatmap (`attention_heatmap`)**:
  Array of 14 daily objects ($T-13$ to $T-0$) representing the model's sliding sequence window:
  ```json
  {
    "offset": "T-00",
    "day": 370,
    "date": "2011-01-07",
    "attention_share": 27.72,
    "stage": "Peak Attack Exfiltration",
    "stage_color": "crimson",
    "static_risk": 0.9970,
    "is_malicious": 1
  }
  ```
- **Explainability Axis 2: Relational Activity Attribution (`relational_attribution`)**:
  Array of 6 activity streams showing why the Autoencoder flagged the anomaly:
  ```json
  [
    {
      "relation": "copies_file",
      "stream_name": "File Exfiltration & Large Transfers",
      "mse": 0.2923,
      "edge_count": 1,
      "share_pct": 58.7
    },
    {
      "relation": "uses_device",
      "stream_name": "Removable Media & USB Insertion",
      "mse": 0.1197,
      "edge_count": 1,
      "share_pct": 24.1
    },
    {
      "relation": "visits_url",
      "stream_name": "Web Navigation & External URLs",
      "mse": 0.0349,
      "edge_count": 42,
      "share_pct": 7.0
    },
    {
      "relation": "sends_email",
      "stream_name": "Outbound Email Transmission",
      "mse": 0.0311,
      "edge_count": 8,
      "share_pct": 6.2
    },
    {
      "relation": "logs_into",
      "stream_name": "Machine Authentication & Logon Activity",
      "mse": 0.0142,
      "edge_count": 1,
      "share_pct": 2.8
    },
    {
      "relation": "uses_pc",
      "stream_name": "Host Process & Workstation Usage",
      "mse": 0.0055,
      "edge_count": 1,
      "share_pct": 1.1
    }
  ]
  ```

---

### 2.4 Monitored Users & 501-Day Histories
- **Endpoint**: `GET /api/users` & `GET /api/users/<user_id>`
- **Description**: Returns monitored employee profiles and their full 501-day behavioral timelines (`timeline` array containing daily `temporal_risk`, `static_risk`, `ae_error`, `is_malicious`).
- **Included Employees**:
  - All 11 confirmed malicious actors in the held-out test split (e.g., `CCL0068`, `KRL0501`, `CEJ0109`, `HXL0968`, `IUB0565`).
  - Representative benign employees across departments (to showcase 0.0% false alarm baseline).

---

### 2.5 Model Comparison Benchmarks
- **Endpoint**: `GET /api/benchmarks`
- **Description**: Quantitative benchmark metrics across all four modeling paradigms evaluated on test users:
```json
{
  "benchmarks": [
    {
      "name": "Unsupervised AE",
      "paradigm": "Zero-Day Behavioral",
      "pr_auc": 0.0128,
      "roc_auc": 0.8905,
      "multiplier": 6.1,
      "p10": 0.0,
      "p20": 0.05,
      "p50": 0.04,
      "p100": 0.03
    },
    {
      "name": "One-Class Deep SVDD",
      "paradigm": "Single-Sphere Latent",
      "pr_auc": 0.0066,
      "roc_auc": 0.6861,
      "multiplier": 3.2,
      "p10": 0.0,
      "p20": 0.0,
      "p50": 0.0,
      "p100": 0.0
    },
    {
      "name": "Supervised Snapshot CLF",
      "paradigm": "Static GNN Snapshot",
      "pr_auc": 0.2146,
      "roc_auc": 0.9875,
      "multiplier": 102.1,
      "p10": 0.40,
      "p20": 0.40,
      "p50": 0.30,
      "p100": 0.26
    },
    {
      "name": "Sequential Temporal GRU",
      "paradigm": "Temporal Attention (14-Day)",
      "pr_auc": 0.5620,
      "roc_auc": 0.9936,
      "multiplier": 260.4,
      "p10": 1.0,
      "p20": 1.0,
      "p50": 0.82,
      "p100": 0.67
    }
  ]
}
```

---

## 3. Recommended Design System & Visualization Guidelines

1. **Color Palette Semantics**:
   - **Critical / High Risk**: Crimson / Coral (`#ff3366`, `#f43f5e`)
   - **Moderate / Staging**: Amber / Orange (`#ffaa00`, `#f59e0b`)
   - **Reconnaissance / Probe**: Tech Cyan (`#00d8f6`, `#06b6d4`)
   - **Benign Baseline**: Terminal Emerald (`#00e599`, `#10b981`)
   - **Neutral / Backdrop**: Dark Slate / Carbon (`#070b14`, `#0d1424`)
2. **Key Visual Components to Design**:
   - **Triage Queue**: Sortable, filterable alert feed with quick severity badges and threat vector tags.
   - **14-Day Attention Heatmap**: Sequential 14-cell timeline representing $T-13$ to $T-0$, highlighting when risk spiked.
   - **Relational Activity Attribution**: Horizontal stacked or grouped bar chart displaying which operations (USB vs. File Copy vs. URL) drove the alert.
   - **Dual-Stream Progression Chart**: Dual-line graph comparing single-day static risk (showing noisy spikes) vs. sequential temporal risk (showing clean, sustained attack escalation).
   - **Psychometric Radar / Gauges**: Visualization for Big Five (O, C, E, A, N) personality scores (scores range from 10 to 50).
