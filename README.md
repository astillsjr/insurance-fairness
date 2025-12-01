# Fairness and Bias Analysis in Vehicle Insurance Risk Prediction

This repository contains the implementation and analysis for examining fairness,
bias, and predictive performance in ML-based auto insurance claim prediction 
systems. The project evaluates baseline models, measures group-level disparities,
and applies fairness-aware techniques to study accuracy–fairness tradeoffs.

## Repository Structure
- `notebooks/` — Google Colab notebooks for data processing, modeling, and fairness analysis
  - `00_exploratory_data_analysis.ipynb` — Initial data exploration, demographic distributions, bias pattern identification
  - `01_preprocessing.ipynb` — Data cleaning, encoding, balancing, train/test splits
  - `02_baseline_models.ipynb` — Training Logistic Regression and Random Forest models, accuracy evaluation
  - `03_fairness_analysis.ipynb` — Fairness metrics evaluation (demographic parity, equalized odds, precision parity)
  - `04_mitigation.ipynb` — Applying fairness-aware techniques and comparing tradeoffs
  - `05_comparison.ipynb` — Final results comparison, fairness-accuracy tradeoff analysis
- `data/` — Dataset files (AutoInsurance.csv)
- `results/` — Metrics, charts, and exported artifacts generated during experiments
- `report/` — Final writeup, slides, or paper for the course project

## Tech Stack
All code runs in **Google Colab**, using:
- Python 3.10+
- numpy
- pandas
- scikit-learn
- fairlearn (fairness metrics + mitigation algorithms)
- matplotlib & seaborn (visualization)

## Usage

### 1. Open a Notebook in Google Colab
Each notebook is designed to run independently in a Colab environment.
Click “Open in Colab” (if badges are added) or upload the `.ipynb` files manually.

### 2. Install Dependencies (Colab)
Every notebook begins with dependency installation:
```python
!pip install fairlearn seaborn -q
```

### 3. Run Notebooks in Order
The notebooks are designed to run sequentially:
1. **00_exploratory_data_analysis.ipynb** — Start here to understand the data
2. **01_preprocessing.ipynb** — Prepare and clean the data
3. **02_baseline_models.ipynb** — Train baseline models
4. **03_fairness_analysis.ipynb** — Evaluate fairness metrics
5. **04_mitigation.ipynb** — Apply fairness mitigation techniques
6. **05_comparison.ipynb** — Compare all results and draw conclusions

Note: Each notebook saves intermediate results to the `results/` directory for use in subsequent notebooks.