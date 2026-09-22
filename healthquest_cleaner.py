import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import time
import warnings
import os

# --- ANSI COLOR CODES ---
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)

def healthquest_cli_cleaner():
    # 0. FETCH CSV
    files = [f for f in glob.glob("*.csv") if "cleaned" not in f.lower()]
    if not files: 
        print(f"\n{YELLOW}[ERROR]{RESET} No source CSV found.")
        return
    
    df = pd.read_csv(files[0], low_memory=False)
    rows_before, cols_before = df.shape
    
    os.system('cls' if os.name == 'nt' else 'clear')

    print(f"{CYAN}{BOLD}" + "="*65)
    print("           HEALTHQUEST DATA-CLEANING ENGINE v15.0")
    print("           (MILESTONE 3: PRECISION CLINICAL LOGIC)")
    print("="*65 + f"{RESET}")

    def run_step(description, info_line):
        with tqdm(total=100, desc=f"{BOLD}{description}{RESET}", bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
            for i in range(100):
                time.sleep(0.003)
                pbar.update(1)
            pbar.bar_format = f"{BOLD}{description}{RESET} {GREEN}" + "{bar:30}" + f"{RESET} {{r_bar}}"
        print(f"{info_line}\n")

    # --- STEP 1: DATADEDUPLICATION ---
    run_step("DATADEDUPLICATION      ", 
             f"{GREEN}✔ SUCCESS:{RESET} Removed redundant records for same encounter and patient.")
    df = df.replace(['?', '??', 'null', 'None', ' '], np.nan)
    df = df.dropna(how='all').drop_duplicates(subset=['encounter_id', 'patient_nbr'], keep='first')

    # --- STEP 2: LABEL HARMONIZATION ---
    run_step("LABEL HARMONIZATION     ", 
             f"{GREEN}✔ SUCCESS:{RESET} Standardized case for ethnicity and gender columns.")
    if 'race' in df.columns: df['race'] = df['race'].astype(str).str.lower().str.strip()
    if 'gender' in df.columns: df['gender'] = df['gender'].astype(str).str.lower().str.strip()

   # --- STEP 3: TEMPORAL RANGE & PRECISION CUTS ---
    run_step("TEMPORAL RANGE          ", "Executing Demographic and Survival Purge.")
    
    # A. Gender Isolation (Male Only)
    if 'gender' in df.columns:
        df = df[df['gender'].isin(['male', 'm'])]
    
    # B. ETHNICITY PURGE: Remove Caucasians AND Nulls/Unknowns
    if 'race' in df.columns:
        # Step 2 already lowercased this. We remove 'nan', 'unknown', '?', and 'other'
        # to ensure we only have confirmed Minority data.
        df = df[df['race'].notna()]
        invalid_race = ['nan', 'unknown', '?', 'other', 'caucasian']
        df = df[~df['race'].isin(invalid_race)]
    
    # C. Age Stratification (40-80)
    if 'age' in df.columns:
        allowed = ['[40-50)', '[50-60)', '[60-70)', '[70-80)']
        df = df[df['age'].isin(allowed)]
        df['age'] = df['age'].astype(str).str.replace(r'[\[\)]', '', regex=True)
    
    # D. Survival Censoring (Removing Expired/Hospice)
    if 'discharge_disposition_id' in df.columns:
        # 11, 13, 14, 19, 20, 21 are IDs for death/hospice
        df = df[~df['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])]
        
    # E. Treatment Filter (Remove Unmedicated)
    if 'num_medications' in df.columns:
        df = df[df['num_medications'] > 0]
    
    # --- STEP 4: IMPUTATION ---
    run_step("IMPUTATION              ", 
             f"{GREEN}✔ SUCCESS:{RESET} Mode used for demographics; Lab results kept as 'not_recorded'.")
    lab_cols = ['max_glu_serum', 'A1Cresult']
    for col in lab_cols:
        if col in df.columns:
            df[col] = df[col].fillna('not_recorded')
    
    for col in [c for c in df.columns if c not in lab_cols]:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())
        else:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])

# --- STEP 5: FEATURE RECONSTRUCTION (OPTIMIZED) ---
    run_step("FEATURE RECONSTRUCTION  ", "2 New columns created named critical_attention & needs_attention.")
    
    med_list = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 
                'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 
                'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 
                'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 
                'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
    
    existing_meds = [m for m in med_list if m in df.columns]
    
    # 1. VECTORIZED MEDICATION COUNT (Fast matrix check)
    med_matrix = df[existing_meds].astype(str).apply(lambda x: x.str.lower())
    active_med_count = med_matrix.isin(['steady', 'up', 'down']).sum(axis=1)
    
    # 2. VECTORIZED BASE SCORE
    score = active_med_count * 5
    
    # 3. VECTORIZED PENALTIES (No loops)
    if 'change' in df.columns:
        score += np.where(df['change'].astype(str).str.lower() == 'ch', 20, 0)
    
    if 'diabetesMed' in df.columns:
        score += np.where(df['diabetesMed'].astype(str).str.lower() == 'yes', 10, 0)

    # 4. THE STRICT FLOOR OVERRIDE (Vectorized)
    if 'readmitted' in df.columns:
        # If <30, add 30 and ensure floor of 50
        is_readmit_30 = (df['readmitted'] == '<30')
        score = np.where(is_readmit_30, np.maximum(50, score + 30), score)
        
        # If >30, add 15
        score = np.where(df['readmitted'] == '>30', score + 15, score)

    # 5. FINAL CLAMPING & EXPORT
    final_score = np.clip(score, 0, 100).astype(int)
    df['critical_percentage'] = final_score.astype(str) + "%"
    df['needs_attention'] = np.where(final_score >= 65, 'yes', 'no')

    # SAVE TO DISK
    df.to_csv("HealthQuest_M3_Cleaned.csv", index=False)
    
    print(f"{GREEN}{BOLD}✔ ENGINE TASKS COMPLETE. TARGET DATA SECURED.{RESET}\n")
    print(f"{CYAN}{BOLD}" + "="*65)
    print(f"{'ENGINE METRIC':<35} | {'BEFORE':<10} | {'AFTER':<10}")
    print("-" * 65 + f"{RESET}")
    print(f"{'Total Patient Rows':<35} | {rows_before:<10} | {len(df):<10}")
    print(f"{'Total Feature Columns':<35} | {cols_before:<10} | {len(df.columns):<10}")
    print(f"{CYAN}{BOLD}" + "="*65 + f"{RESET}\n")

if __name__ == "__main__":
    healthquest_cli_cleaner()