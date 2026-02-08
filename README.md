# Intelligent Causal Bidding Agent 🎯

A **Causal Inference** system that optimizes Real Time Bidding (RTB) by estimating the **Conditional Average Treatment Effect (CATE)** of ads.

Unlike traditional propensity models that target users *most likely to buy* (often wasting budget on "Sure Things"), this agent targets users *most likely to be persuaded*, maximizing incremental Return on Ad Spend (ROAS).

## The Problem: "Sure Things" vs. "Persuadables"

Traditional churn and conversion models predict . In advertising, this creates a **budget inefficiency trap**:

* **Wasted Spend:** Bidding on users who would have converted anyway ("Sure Things").
* **Negative ROI:** Spamming users who react negatively to ads ("Sleeping Dogs").
* **Missed Revenue:** Ignoring the people who need an ad to convert ("Persuadables").

**The Solution:** Instead of predicting *Outcome*, we predict *Uplift*:



## Business Impact & Results

### Offline Simulation Performance

Evaluated on a 20% holdout of the **Criteo Uplift v2 Dataset** using a Second-Price Auction simulation.

| Strategy | Total Spend | Conversions | CPA (Cost/Conv) | ROAS | Lift vs Random |
| --- | --- | --- | --- | --- | --- |
| **Random Bidding** (Saturation) | $1,384,256 | 7,294 | $189.78 | 0.05x | Baseline |
| **T-Learner** (Causal Baseline) | $162,148 | 5,237 | $30.96 | 0.32x | +540% |
| **X-Learner** (Champion) | **$135,082** | **5,076** | **$26.61** | **0.38x** | **+660%** |

### 💡 Key Strategic Insights

* **7x Efficiency Gain:** The X-Learner captured **70% of the total market conversions** (5,076 vs 7,294) while spending only **10% of the budget** ($135k vs $1.38M).
* **"Sleeping Dog" Suppression:** identified ~12% of users with **negative uplift** scores. Suppressing bids on this segment prevented wasted spend and potential brand damage.
* **CPA Reduction:** Reduced Cost Per Acquisition from $189 (Random) to **$26.61**, making the campaign profitable (assuming $30 target CPA).

---

## Technical Architecture

### 1. Causal Uplift Models

* **Why X-Learner?** The dataset has a heavy class imbalance (Control group < Treatment group). T-Learners often struggle here. The X-Learner uses a two-stage estimation process to regularize the treatment effect, reducing variance in the control arm.
* **Algorithm:** `XGBoost` used as the base learner for propensity and outcome models.
* **Calibration:** Applied **Isotonic Regression** to correct probability drift, ensuring predicted lift matches empirical lift.

### 2. PID Bidding Agent

* **Problem:** In live production, raw model scores fluctuate, causing budget to be exhausted too early or under utilized.
* **Solution:** Implemented a **Proportional Integral-Derivative (PID) Controller** (`src/agents/agent.py`).
* **Logic:** The agent monitors the "Spend Velocity" (dollars/minute).
* If velocity > target, it lowers the bid multiplier ( error correction).
* If velocity < target, it raises the multiplier to capture cheaper inventory.

## Model Validation

Before simulation, the model was validated using the following causal metrics:

* **Qini Coefficient (AUUC):**
* **X-Learner:** 0.0234 (Best rank ordering)
* **T-Learner:** 0.0198
* *Result:* X-Learner is 18% better at sorting users from high to low lift.


* **Common Support (Overlap):**
* Propensity Score AUC: **0.502** (Ideal is 0.50).
* *Interpretation:* Treatment assignment was effectively random, satisfying the "Ignorability" assumption.


* **Calibration Error:**
* Mean Absolute Error (MAE) on Lift: **< 0.002** in top 3 deciles.


---

## Limitations & Constraints

* **Simulation vs. Reality:** Results are based on an offline simulation (Counterfactual Evaluation). Real-world performance is subject to feedback loops and competitive bid density not captured here.
* **Auction Dynamics:** Assumes a standard **Second-Price Auction** (Vickrey). First-Price auctions (common in header bidding) would require a different bid shading strategy.
* **Data Bias:** The Criteo dataset is pre-collected; we assume **SUTVA** (no interference between users) holds, which is generally true for RTB but not for social network effects.


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

# Build Docker Container (Recommended)
docker build -t uplift-agent .

```

### 2. Download Data

This project uses the **Criteo Uplift Modeling Dataset v2**.

1. Download from [Criteo AI Lab](https://ailab.criteo.com/criteo-uplift-prediction-dataset/).
2. Extract `criteo-uplift-v2.1.csv` into `data/raw/`.

### 3. Run the Pipeline

```bash
# 1. Prepare Data & Engineer Features
docker run -v $(pwd)/data:/app/data uplift-agent python src/01_data_prep.py

# 2. Train Champion Model (X-Learner)
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models uplift-agent python src/05_train_x_learner.py

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

