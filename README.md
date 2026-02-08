# Intelligent Causal Bidding Agent 

A **Causal Inference** system that optimizes Real-Time Bidding (RTB) by estimating the **Conditional Average Treatment Effect (CATE)** of ads. Unlike traditional propensity models that target users *most likely to buy* (often wasting budget on "Sure Things"), this agent targets users *most likely to be persuaded*, maximizing incremental ROAS.

## Business Impact & Results

Simulated performance in a Second-Price Auction environment against a random bidding baseline:

| Metric | Random Bidding (Baseline) | T-Learner (Causal Baseline) | X-Learner (Champion) | Improvement |
| --- | --- | --- | --- | --- |
| **ROAS** | 0.05x | 0.32x | **0.38x** | **+613%** Return on Ad Spend |
| **CPA** | $189.78 | $30.96 | **$26.61** | **-86%** Cost Per Acquisition |
| **Lift** | N/A | +0.65% | **+0.75%** | **15%** Lift vs Baseline Model |

### Key Wins

* **Identified Wasteful Spending:** The model successfully segmented the bottom 10% of users who had a **negative response to advertising** (-0.2% lift). Not wasting bids on this group saved ~15% of the budget.
* **Budget Efficiency:** The X-Learner captured **70% of total conversions** while spending only **10% of the budget**, effectively identified the most potential conversions from the data.
* **Control:** Implemented a **PID Controller** class (`src/agents/agent.py`) to smooth bid responses and prevent budget exhaustion in volatile live environments.

---

## Technical Architecture

### 1. Causal Uplift Models

* Standard ML predicts `P(Buy|Ad)`, which targets users who would buy anyway.
* Implemented **X-Learner** (Meta-Learner) using `XGBoost` to estimate `P(Buy|Ad) - P(Buy|No Ad)`.
* Used **Isotonic Calibration** to correct probability drift, ensuring bid prices match real-world conversion probabilities.
* Verified Causal Assumptions via **Propensity Score Matching (AUC ~0.50)** and **Placebo Tests** to ensure Common Support.

### 2. PID Bidding Agent

* A Proportional-Integral-Derivative (PID) controller adjusts the bid multiplier (`alpha`) in real-time.
* `Bid = Uplift * Alpha`. The agent raises `alpha` when underspending and lowers it when overspending to hit a daily budget target exactly.

---

## Project Structure

```text
├── data/               # Parquet files (Gitignored)
├── models/             # Serialized X-Learner artifacts
├── src/
│   ├── agents/         # PID Bidding Logic
│   ├── 01_data_prep.py     # Data Ingestion
│   ├── 02_train_t_learner.py   # T-Learner Training
│   ├── 03_inference_evaluation.py       # Quick inference evaluation
│   ├── 04_refutation_test.py       #Placebo Treatement Test
│   └── 05_train_x_learner.py #X-Learner Training  
├── notebooks/          # Analysis: Qini Curves, Decile Lift, Common Support
│   ├── 01_EDA.ipynb     # Explore Features Interaction and Test Causal Assumptions
│   ├── 02_Model_Evaluation.ipynb    # Evaluate important metrics of T-learner
│   ├── 03_Model_Comparison.ipynb       # T-learner vs X-learner
│   ├── 04_Bidding_Simulation_Agent.ipynb   # Simuation of Agent using T-learner
│   └── 05_Bidding_Simulation_Xlearner.py #Simulation of Xlearner on CPA, ROAS, Spend  
└── README.md

```
---

## How to Reproduce

### 1. Environment Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/uplift-bidding-agent.git
cd uplift-bidding-agent

# Install dependencies
pip install -r requirements.txt

```

### 2. Data Pipeline & Training

```bash
# Generate synthetic data and engineer features
python src/01_data_prep.py

# Train the Baseline T-Learner
python src/02_train_t_learner.py

# Train the Champion X-Learner
python src/05_train_x_learner.py

```

---

## Visualizations

### 1. Targeting Efficiency (Decile Analysis)

*The model correctly identifies the top 10% of "Persuadables" (Decile 9) while filtering out "Lost Causes" (Decile 0-5).*

### 2. Model Performance (Qini Curve)

*The X-Learner consistently outperforms the T-Learner, achieving a higher Area Under Uplift Curve (AUUC).*

---

## References

* **Methodology:** Kunzel et al. (2019), "Metalearners for estimating heterogeneous treatment effects using machine learning."
* **Metrics:** Radcliffe, N. J. (2007). "Using control groups to target on predicted lift: Building and assessing uplift models."