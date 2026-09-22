# HealthQuest Data-Cleaning Engine (v15.0)

A high-performance, vectorized CLI data cleaning engine built with Python, Pandas, and NumPy. This engine processes raw hospital records into a precision-grade clinical dataset tailored for 30-day diabetic readmission risk modeling.

---

## 🛠️ Key Engine Architecture & Methodology

### 1. Data Deduplication & Null Normalization
* **Action:** Standardized irregular null representations (`?`, `??`, `null`, `None`, empty strings) to `np.nan` and dropped duplicate `(encounter_id, patient_nbr)` pairs.
* **Rationale:** Removes artificial event inflation, ensuring each patient encounter is uniquely represented.

### 2. Demographic & Clinical Survival Purge
* **Action:** 
  * Isolated Male cohort records aged 40–80.
  * Extracted specific minority demographic subgroups.
  * Filtered out unmedicated records (`num_medications == 0`).
  * Removed deceased/hospice outcomes using specific discharge disposition IDs (`11, 13, 14, 19, 20, 21`).
* **Rationale:** Eliminates non-readmittable outcomes and focuses strictly on active, viable treatment cohorts.

### 3. Domain-Preserving Imputation
* **Action:** Preserved missing biomarker lab tests (`max_glu_serum`, `A1Cresult`) by tagging them explicitly as `'not_recorded'`. Applied median/mode imputation strictly to non-biomarker features.
* **Rationale:** Prevents unsafe biological hallucination on unrecorded medical tests while maintaining dataframe completeness.

### 4. Vectorized Feature Reconstruction & Severity Scoring
* **Action:** Constructed automated risk metrics (`critical_percentage` and `needs_attention`) using matrix operations across 23 distinct diabetes medications:
  * **+5%** score per active medication (`steady`, `up`, `down`).
  * **+20%** penalty if the medication plan changed (`ch`).
  * **+10%** penalty if actively prescribed diabetes medication.
  * **Strict Floor Override:** Applied a mandatory minimum 50% severity floor for acute readmissions (`<30` days).
  * Clamped output scores to `[0, 100]%` and flagged high-risk records (`score >= 65%`).

---

## 🚀 Execution & Requirements

### Requirements
```text
pandas
numpy
tqdm
```

### How to Run
1. Download `healthquest_cleaner.py`.
2. Place your raw hospital `.csv` dataset in the **same folder** as the script.
3. Open your terminal/command prompt in that folder and run:
   ```bash
   python healthquest_cleaner.py
   ```
