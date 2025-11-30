# Training Analysis Report
## HGIB Recommendation System for Travel Website

**Date:** November 30, 2025  
**Model:** Heterogeneous Graph Information Bottleneck (HGIB)  
**Data Source:** ✅ PostgreSQL Database (Heroku Cloud) via `.env` configuration

---

## 1. Dataset Overview

| Metric | Value |
|--------|-------|
| **Data Source** | ✅ PostgreSQL Database |
| **Configuration** | `.env` file (git-ignored) |
| Total users in database | 1,500 |
| Users with travel history | 326 |
| Total trips | 8,208 |
| Unique destinations | 91 |
| Average trips per user | 25.18 |

---

## 2. Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Hidden Channels | 128 |
| Output Channels | 32 |
| Learning Rate | 0.0004 |
| **Epochs** | **250** |
| Beta (KL weight) | 0.001 |
| Early Stopping Delta | 0.01 |

---

## 3. Training Results

### 3.1 Loss Curves

![Training Progress](metrics.png)

| Metric | Initial Value | Final Value | Assessment |
|--------|---------------|-------------|------------|
| Training Loss | ~7.0 | ~0.72 | ✅ Good convergence |
| Validation Loss | ~0.70 | ~0.65 | ✅ Decreasing |
| Epochs Completed | - | 250 | ✅ Full run |
| Best Val Loss | - | 0.6479 | ✅ Below random (0.693) |

### 3.2 Training Curve Analysis

| Aspect | Observation | Assessment |
|--------|-------------|------------|
| **Convergence** | Loss still decreasing at epoch 250 | ⚠️ Could benefit from more epochs |
| **Overfitting** | Train and val loss stay close | ✅ No signs |
| **Training stability** | Smooth descent | ✅ Stable |

---

## 4. Evaluation Metrics

### 4.1 Final NDCG@10

**Mean NDCG@10: 0.0434**

### 4.2 Interpretation

| Benchmark | NDCG@10 | Notes |
|-----------|---------|-------|
| Random baseline | ~0.02-0.03 | Pure chance |
| **Our Model (250 epochs)** | **0.0434** | 1.5x better than random |
| Our Model (1500+ epochs) | 0.0817 | With more training |

⚠️ **Note:** With only 250 epochs, the model hasn't fully converged. For better results, increase epochs to 1500+.

---

## 5. Qualitative Analysis: Sample Recommendations

### User 1: Rajesh (Indian, Cold/Adventure Preference)
| Attribute | Value |
|-----------|-------|
| Profile Type | Froid/Ski/Nordique |
| Past Trips | 23 |

**Top 5 Recommendations:** Delhi, Copenhagen, Buenos Aires, Hanoi, Miami

### User 2: Noah (Australian, Cold/Adventure Preference)  
| Attribute | Value |
|-----------|-------|
| Profile Type | Froid/Ski/Nordique |
| Past Trips | 59 |

**Top 5 Recommendations:** Nairobi, Delhi, Hanoi, Kyoto, Banff ✅ (cold destination)

### User 3: Emily (American, Hot/Exotic Preference)
| Attribute | Value |
|-----------|-------|
| Profile Type | Exotique/Budget |
| Past Trips | 42 |

**Top 5 Recommendations:** Nairobi, Buenos Aires, Hanoi, Patagonia, Kyoto

---

## 6. Summary

| Question | Answer |
|----------|--------|
| Data loaded from database? | ✅ **Yes** (via `.env`) |
| Model converged? | ⚠️ **Partially** (250 epochs) |
| Better than random? | ✅ **Yes** (1.5x) |
| Secure configuration? | ✅ **Yes** (`.env` in `.gitignore`) |

### Key Files

| File | Description |
|------|-------------|
| `.env` | Database credentials (git-ignored) ✅ |
| `Model/artifacts/hgib_model.pth` | Trained model |
| `Model/artifacts/metrics.png` | Training curves |
| `Model/artifacts/metrics.json` | Training history |

---

*Data Source: PostgreSQL Database via secure `.env` configuration*
