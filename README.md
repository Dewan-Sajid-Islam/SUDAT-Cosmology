````markdown
# SUDAT: Scale-Dependent Unified Dark Sector Model

This repository contains the full analysis pipeline for the SUDAT cosmological model, including mock data generation, likelihood evaluation, and Bayesian model comparison against ΛCDM.

---

## 📌 Overview

SUDAT introduces a scale-dependent modification to the matter power spectrum, producing localized features near:

k ~ 0.1 h/Mpc

These features are not reproducible within ΛCDM, enabling observational distinguishability.

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
````

---

## 🚀 Running the Pipeline

### 1. Generate mock data

```bash
python src/generate_mock_data.py
```

### 2. Fit models

```bash
python src/lcdm_failure_test.py
```

### 3. Run Bayesian evidence

```bash
python src/run_nested_sampling.py
```

---

## 📊 Outputs

* `results/pk_comparison.pdf` → Power spectrum comparison
* `results/delta_pk.pdf` → Residual differences
* `results/corner.png` → Posterior constraints

---

## 🔬 Key Result

* SUDAT remains competitive with ΛCDM on standard data
* Produces distinct scale-dependent features
* Achieves strong Bayesian preference when such features are present

---

## 📄 Paper

See `paper/main.tex` for full manuscript.

---

## ⚠️ Note

This repository uses mock data for controlled model comparison. Future work will incorporate real observational datasets.

---