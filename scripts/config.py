from pathlib import Path

# ---------------- Paths ----------------
# The only line to edit: folder of MIMIC-IV v3.1 as downloaded from PhysioNet (it contains hosp/)
MIMIC_DIR = Path(r"path/to/mimic-iv-3.1")

LABEVENTS = MIMIC_DIR / "hosp" / "labevents.csv.gz"
OMR = MIMIC_DIR / "hosp" / "omr.csv.gz"
DIAGNOSES = MIMIC_DIR / "hosp" / "diagnoses_icd.csv.gz"
ADMISSIONS = MIMIC_DIR / "hosp" / "admissions.csv.gz"
PATIENTS = MIMIC_DIR / "hosp" / "patients.csv.gz"

# Outputs go to output/ in the repository root (excluded from version control)
OUTPUT = Path(__file__).resolve().parent.parent / "output"
OUTPUT.mkdir(exist_ok=True)
LABS_PARQUET = OUTPUT / "labs.parquet"
BP_PARQUET = OUTPUT / "bp.parquet"
COHORT = OUTPUT / "cohort.parquet"                 # all eligible patients, before plausibility exclusions
COHORT_CLEAN = OUTPUT / "cohort_clean.xlsx"        # after plausibility exclusions; input to the analyses
AVAILABILITY = OUTPUT / "availability.parquet"     # >=3 determinations per analyte, all patients with lab data
TABLE1_DATA = OUTPUT / "table1_data.parquet"       # age, sex and number of determinations
SELECTION_LASSO = OUTPUT / "selection_lasso.xlsx"
SELECTION_ELASTICNET = OUTPUT / "selection_elasticnet.xlsx"
SELECTION_CATBOOST = OUTPUT / "selection_catboost.xlsx"
LASSO_RANKING_FOLDS = OUTPUT / "lasso_ranking_folds.xlsx"
KCURVE_LASSO = OUTPUT / "kcurve_lasso.xlsx"
BENCHMARK_DIR = OUTPUT / "benchmark"
BENCHMARK_DIR.mkdir(exist_ok=True)
BENCHMARK = BENCHMARK_DIR / "benchmark_results.xlsx"
FIGURES = OUTPUT / "figures"
FIGURES.mkdir(exist_ok=True)

# ---------------- Analytes (Supplementary Table 5; identical item IDs in v3.0 and v3.1) ----------------
ITEMIDS = {
    "Sodium": [50983, 50824, 52623], "Potassium": [50971, 50822, 52610], "Chloride": [50902, 50806, 52535],
    "Bicarbonate": [50882, 50803], "Anion GAP": [50868, 52500], "Urea Nitrogen": [51006],
    "Creatinine": [50912, 52024, 52546], "Glucose": [50931, 50809, 52569],
    "Hemoglobin": [51222, 50811, 51645], "Hematocrit": [51221, 50810], "Red Blood Cells": [51279],
    "MCV": [51250], "MCH": [51248], "MCHC": [51249], "RDW": [51277],
    "White Blood Cells": [51301, 51755], "Platelet Count": [51265],
}
MIN_DETERMINATIONS = 3

# ---------------- Hypertension codes (as used in the analysis; stored without the decimal point) ----------------
# A: essential hypertension
ICD9_A, ICD10_A = ["4019"], ["I10"]
# B: remaining codes of Supplementary Table 4 (excluding 4010 and 4011); in the cohort, B is set only without A
ICD9_B = ["40200", "40201", "40210", "40211", "40290", "40291",
          "40300", "40301", "40310", "40311", "40390", "40391",
          "40400", "40401", "40402", "40403", "40410", "40411", "40412", "40413",
          "40490", "40491", "40492", "40493",
          "40501", "40509", "40511", "40519", "40591", "40599"]
ICD10_B = ["I11", "I110", "I119", "I12", "I120", "I129",
           "I13", "I130", "I131", "I1310", "I1311", "I132",
           "I15", "I150", "I151", "I152", "I158", "I159", "I16", "I160", "I161", "I169"]
TARGET = "Hypertension A+B"   # A or B

# ---------------- Plausibility limits (Supplementary Table 3; MIMIC-IV units) ----------------
# A patient is excluded if _min < lower or _max > upper. H = Harutyunyan (MIMIC-Extract), R = ricu, P = own criterion
LIMITS = {
    "Anion GAP": (0, 55),  # H
    "Bicarbonate": (0, 66),  # H
    "Chloride": (0, 200),  # H
    "Creatinine": (0, 66),  # H
    "Glucose": (0, 2200),  # H
    "Hematocrit": (0, 100),  # H
    "Hemoglobin": (0, 30),  # H
    "Potassium": (0, 15),  # H
    "Sodium": (0, 250),  # H
    "Urea Nitrogen": (0, 275),  # H
    "MCH": (0, 60),  # P
    "MCHC": (20, 50),  # R
    "MCV": (50, 150),  # R
    "Platelet Count": (0, 2200),  # H
    "RDW": (0, 100),  # R
    "Red Blood Cells": (0, 20),  # R
    "White Blood Cells": (0, 1100),  # H
    "bp_s": (0, 375),  # H
    "bp_d": (0, 375),  # H
}
# Zero rule (an exact 0 marks missing data) for all variables except the anion gap, which can legitimately be 0
NO_ZERO_RULE = ["Anion GAP"]
NON_PREDICTORS = ["Hypertension A", "Hypertension B", "Admission"]

# ---------------- Column order of the cohort file ----------------
COLUMNS = ['Anion GAP_mean', 'Anion GAP_variance', 'Anion GAP_sd', 'Anion GAP_min', 'Anion GAP_max', 'Anion GAP_q1',
 'Anion GAP_q3', 'Bicarbonate_mean', 'Bicarbonate_variance', 'Bicarbonate_sd', 'Bicarbonate_min',
 'Bicarbonate_max', 'Bicarbonate_q1', 'Bicarbonate_q3', 'Chloride_mean', 'Chloride_variance', 'Chloride_sd',
 'Chloride_min', 'Chloride_max', 'Chloride_q1', 'Chloride_q3', 'Creatinine_mean', 'Creatinine_variance',
 'Creatinine_sd', 'Creatinine_min', 'Creatinine_max', 'Creatinine_q1', 'Creatinine_q3', 'Glucose_mean',
 'Glucose_variance', 'Glucose_sd', 'Glucose_min', 'Glucose_max', 'Glucose_q1', 'Glucose_q3',
 'Hematocrit_mean', 'Hematocrit_variance', 'Hematocrit_sd', 'Hematocrit_min', 'Hematocrit_max',
 'Hematocrit_q1', 'Hematocrit_q3', 'Hemoglobin_mean', 'Hemoglobin_variance', 'Hemoglobin_sd',
 'Hemoglobin_min', 'Hemoglobin_max', 'Hemoglobin_q1', 'Hemoglobin_q3', 'Potassium_mean', 'Potassium_variance',
 'Potassium_sd', 'Potassium_min', 'Potassium_max', 'Potassium_q1', 'Potassium_q3', 'Sodium_mean',
 'Sodium_variance', 'Sodium_sd', 'Sodium_min', 'Sodium_max', 'Sodium_q1', 'Sodium_q3', 'Urea Nitrogen_mean',
 'Urea Nitrogen_variance', 'Urea Nitrogen_sd', 'Urea Nitrogen_min', 'Urea Nitrogen_max', 'Urea Nitrogen_q1',
 'Urea Nitrogen_q3', 'MCH_mean', 'MCH_variance', 'MCH_sd', 'MCH_min', 'MCH_max', 'MCH_q1', 'MCH_q3',
 'MCHC_mean', 'MCHC_variance', 'MCHC_sd', 'MCHC_min', 'MCHC_max', 'MCHC_q1', 'MCHC_q3', 'MCV_mean',
 'MCV_variance', 'MCV_sd', 'MCV_min', 'MCV_max', 'MCV_q1', 'MCV_q3', 'Platelet Count_mean',
 'Platelet Count_variance', 'Platelet Count_sd', 'Platelet Count_min', 'Platelet Count_max',
 'Platelet Count_q1', 'Platelet Count_q3', 'RDW_mean', 'RDW_variance', 'RDW_sd', 'RDW_min', 'RDW_max',
 'RDW_q1', 'RDW_q3', 'Red Blood Cells_mean', 'Red Blood Cells_variance', 'Red Blood Cells_sd',
 'Red Blood Cells_min', 'Red Blood Cells_max', 'Red Blood Cells_q1', 'Red Blood Cells_q3',
 'White Blood Cells_mean', 'White Blood Cells_variance', 'White Blood Cells_sd', 'White Blood Cells_min',
 'White Blood Cells_max', 'White Blood Cells_q1', 'White Blood Cells_q3', 'Hypertension A', 'bp_s_mean',
 'bp_s_variance', 'bp_s_sd', 'bp_s_min', 'bp_s_max', 'bp_s_q1', 'bp_s_q3', 'bp_d_mean', 'bp_d_variance',
 'bp_d_sd', 'bp_d_min', 'bp_d_max', 'bp_d_q1', 'bp_d_q3', 'Hypertension B', 'Hypertension A+B', 'Admission']
