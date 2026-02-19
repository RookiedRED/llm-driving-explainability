# 🚗 LLM-Powered Explainable Autonomous Driving

An end-to-end system that bridges autonomous driving perception outputs
with Large Language Model (LLM) reasoning to generate structured,
grounded, and safety-aware driving explanations.

------------------------------------------------------------------------

## 🎯 Motivation

Modern autonomous driving systems operate as complex
perception--planning--control pipelines.\
However, these systems often lack interpretable reasoning for their
decisions.

This project explores:

> How can we transform structured driving state representations into
> grounded, safety-aware natural language explanations using LLMs?

Instead of feeding raw images to an LLM, we design a **structured
driving state abstraction layer** that serves as a reliable interface
between perception outputs and reasoning.

------------------------------------------------------------------------

## 🏗️ System Architecture

Dataset / Simulator (nuScenes / CARLA) ↓ Perception Layer - Object
detection - Tracking - Ego pose ↓ Driving State Abstraction - Distance
computation - Relative velocity - TTC (Time-to-collision) - Risk scoring
↓ LLM Reasoning Layer - Structured prompt - JSON output schema - Safety
constraints ↓ Evaluation Layer - Groundedness check - Consistency
analysis - Latency benchmark

------------------------------------------------------------------------

## 📦 Features

-   Structured driving state schema
-   Distance & risk feature extraction
-   TTC estimation
-   LLM-based explainability
-   Grounded JSON outputs
-   Hallucination detection
-   Latency benchmarking

------------------------------------------------------------------------

## 🧠 Design Principles

### 1. Simulator-Agnostic

The system does not depend on a specific simulator.\
It can ingest:

-   nuScenes dataset
-   CARLA
-   Synthetic scenarios

------------------------------------------------------------------------

### 2. Structured Interface (No Raw Vision to LLM)

Example state input:

``` json
{
  "ego_speed_kmh": 28,
  "objects": [
    {"type": "pedestrian", "distance_m": 8.2},
    {"type": "vehicle", "distance_m": 14.5}
  ],
  "traffic_light": "red",
  "ttc_s": 1.1,
  "risk_level": "high"
}
```

This enforces:

-   Reduced hallucination
-   Numerical grounding
-   Safer reasoning

------------------------------------------------------------------------

### 3. Evaluation-First Design

We evaluate:

-   Explanation groundedness
-   Action consistency
-   Safety rule alignment
-   Latency

------------------------------------------------------------------------

## 📂 Project Structure

    llm-driving-explainability/
    ├── src/
    │   ├── ingest/
    │   ├── perception/
    │   ├── state/
    │   ├── reasoning/
    │   ├── eval/
    │   ├── app/
    │   └── utils/
    ├── scripts/
    ├── data/
    ├── tests/
    └── README.md

------------------------------------------------------------------------

## 🚀 Getting Started

### 1️⃣ Install

``` bash
pip install -r requirements.txt
```

### 2️⃣ Download Dataset

Download:

nuScenes v1.0-mini

Place under:

data/nuscenes/

Expected structure:

data/nuscenes/ samples/ sweeps/ maps/ v1.0-mini/

### 3️⃣ Export Structured Driving States

``` bash
python scripts/export_states.py
```

------------------------------------------------------------------------

## 📊 Example Output

``` json
{
  "action": "brake",
  "explanation": [
    "Pedestrian detected ahead at 8.2m.",
    "Time-to-collision is 1.1 seconds.",
    "Collision risk classified as HIGH."
  ],
  "evidence": {
    "ttc_s": 1.1,
    "ego_speed_kmh": 28
  },
  "confidence": 0.87
}
```

------------------------------------------------------------------------

## 📈 Evaluation Metrics

-   Groundedness Score
-   Consistency Score
-   Rule Agreement Rate
-   Mean Latency (ms)

------------------------------------------------------------------------

## 🛣️ Roadmap

-   [ ] nuScenes integration
-   [ ] Risk feature engineering
-   [ ] LLM structured reasoning
-   [ ] Evaluation framework
-   [ ] Streamlit demo
-   [ ] CARLA integration
-   [ ] Edge case benchmark

------------------------------------------------------------------------

## ⚠️ Disclaimer

This project is a research prototype and not intended for real-world
autonomous vehicle deployment.
