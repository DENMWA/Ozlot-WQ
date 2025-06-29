
# Ozlotter-smartpro-ML-Core

🎯 **Ozlotter-smartpro-ML-Core** is the foundational build of an intelligent, feature-rich prediction engine designed to model and score Oz Lotto draw sets using advanced statistical, gap-theoretic, and machine learning layers.

---

## 🚀 Features

- ✅ **Ξ-Matrix Feature Engine** (λ₁–λ₁₅)
- ✅ Machine Learning scoring via Random Forest (λ₁₁)
- ✅ Customizable weight configuration via `weights.json`
- ✅ Upload-and-retrain ML model from historical data
- ✅ Predict 70 ranked draw sets per execution
- ✅ Dual-tab Streamlit UI for user control

---

## 🧠 Feature Layers (λ₁–λ₁₅)

| Lambda | Description |
|--------|-------------|
| λ₁ | Variance of drawn numbers |
| λ₂ | Standard deviation |
| λ₃ | Normalized mean |
| λ₄ | Sum of gaps |
| λ₅ | Redundancy penalty |
| λ₆ | Max gap (momentum) |
| λ₇ | Normalized total |
| λ₈–λ₁₀ | Controlled entropy injections |
| λ₁₁ | ML score: predicted Division 1–3 potential |
| λ₁₂–λ₁₅ | Quantum-like modulation layers |

> Note: λ₁₆ (Hilbert embedding) excluded in this version for stability.

---

## 📁 File Structure

- `app.py` — Main UI for predictions and ML retraining
- `xi_matrix_engine.py` — Core logic for prediction and scoring
- `reflexive_trainer.py` — ML model training from historical draws
- `Oz_Lotto_Historical_Draws.csv` — Dataset for training and simulation
- `Ξ_Matrix_Engine_Layers.txt` — Full feature description

---

## ⚙️ Getting Started

1. Install dependencies:
   ```bash
   pip install streamlit scikit-learn pandas numpy joblib
   ```

2. Run the app:
   ```bash
   streamlit run app.py
   ```

3. Upload your draw data and retrain ML model live.

---

## 🔭 Roadmap

- Add λ₁₆: Hilbert Embedding
- Integrate heatmaps and division-match visualizations
- Expand to support Saturday Lotto and Powerball
- Mobile app wrapper (Q4 target)

---

## 🧬 License
MIT — for experimental and research use only.

---

**Built for predictive insight — refined for precision.**
