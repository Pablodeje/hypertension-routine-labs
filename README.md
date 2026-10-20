# Identifying hypertension from routine laboratory data

Code for the paper *Identifying Hypertension from Routine Laboratory Data: A Machine Learning Analysis with Potential for Clinical Translation* (de Jesús Fieiro P, Muñiz Castro B, González-Juanatey JR, Alvarez-Lorenzo C, Goyanes A).

The scripts reproduce, from the raw MIMIC-IV files, the analytical cohort, the predictor selection, the benchmark of five algorithms on four predictor sets, and the comparisons between predictor sets and analytes reported in the paper. They also train the minimalist model used in the demonstration web page (see below). The scripts that format the tables and draw the figures of the paper are not included.

## Data

The analysis uses **MIMIC-IV version 3.1** (Johnson et al., PhysioNet, doi:10.13026/kpb9-mt58). Access requires credentialed PhysioNet access, the required training and a signed data use agreement. No data, derived files or trained models are included in this repository.

Run on MIMIC-IV 3.1 with the package versions in `requirements.txt`, the code reproduces the numbers reported in the paper.

Four files from the `hosp` module are read: `labevents`, `omr`, `diagnoses_icd` and `admissions` (`.csv.gz`, as distributed by PhysioNet; no need to decompress).

## Installation

Python 3.14 (the version used for the paper; 3.12 also works). From the repository root:

```
pip install -r requirements.txt
```

## Configuration

Edit a single line in `scripts/config.py`:

```python
MIMIC_DIR = Path(r"path/to/mimic-iv-3.1")   # folder that contains hosp/
```

All outputs are written to `output/` in the repository root, created automatically and excluded from version control.

## Running

The analysis is split into numbered scripts in `scripts/`, run in numerical order:

| Script | Does | Main output |
|---|---|---|
| `01_labs.py` | Extracts the 17 analytes (36 item identifiers) from `labevents` | `labs.parquet` |
| `02_bp.py` | Extracts blood pressure readings from `omr` | `bp.parquet` |
| `03_cohort.py` | Eligibility (≥3 determinations of each analyte and of blood pressure), six summary statistics per series, outcome and admission flag | `cohort.parquet` |
| `04_exclusions.py` | Plausibility exclusions | `cohort_clean.xlsx` (analytical cohort) |
| `05_availability.py` | Analyte availability among all patients with laboratory data (missing-data analysis) | `availability.parquet` |
| `10_bolasso.py` | BP+Lasso selection | `selection_lasso.xlsx` |
| `11_elastic_net.py` | Elastic-net sensitivity analysis | `selection_elasticnet.xlsx` |
| `12_catboost_shap.py` | BP+SHAP selection and k curve | `selection_catboost.xlsx` |
| `13_lasso_ranking_folds.py` | Lasso ranking within each cross-validation fold | `lasso_ranking_folds.xlsx` |
| `14_kcurve_lasso.py` | k curve in lasso order | `kcurve_lasso.xlsx` |
| `20_benchmark.py` | Benchmark of all models (settings, metrics and export in `benchmark_common.py`) | `benchmark/` |
| `38_set_comparison.py` | Paired comparison between the laboratory predictor sets | printed |
| `39_analyte_contribution.py` | Contribution of each analyte and analyte group | printed |
| `50_minimal_model.py` | Minimalist model for the demonstration web page | `benchmark/` |

`20_benchmark.py` takes several hours; it saves the results of each algorithm as soon as it finishes. `01_labs.py` reads `labevents` (about 17 GB uncompressed) in chunks.

On Windows, set `PYTHONUTF8=1` before running (in Git Bash: `export PYTHONUTF8=1`), so that non-ASCII characters print correctly when the output is redirected to a file.

### Make goals

The `Makefile` groups the scripts into goals. From the repository root, run the goals in this order:

```
make cohort
make selection
make benchmark
make results
make minimal
```

or everything at once with `make all`. If your Python command is not `python`, pass it explicitly, e.g. `make all PYTHON=python3`. `make -n <goal>` lists the commands of a goal without running them.

| Goal | Runs | Does | Output (in `output/`) |
|---|---|---|---|
| `cohort` | `01` to `05` | Extracts laboratory and blood pressure data, applies eligibility, computes the six summary statistics per series, defines the outcome, applies the plausibility exclusions and computes analyte availability | `cohort_clean.xlsx` (analytical cohort), `availability.parquet` |
| `lasso` | `10` | BP+Lasso: lasso with the 1-SE penalty and bootstrap selection (soft-thresholded Bolasso) | `selection_lasso.xlsx` |
| `elasticnet` | `11` | Elastic-net sensitivity analysis | `selection_elasticnet.xlsx` |
| `shap` | `12` | BP+SHAP: per-fold SHAP ranking (CatBoost) and cross-validated k curve | `selection_catboost.xlsx` |
| `kcurve-lasso` | `13` to `14` | k curve adding analytes in each fold's lasso order (after `shap`) | `kcurve_lasso.xlsx` |
| `selection` | `10` to `14` | The four goals above | |
| `benchmark` | `20` | 5 algorithms × 4 predictor sets: tuning, test-set evaluation, bootstrap CIs and paired differences (several hours) | `benchmark/` |
| `results` | `38` to `39` | Paired comparison between the laboratory predictor sets and contribution of each analyte, printed to the terminal (after `benchmark`) | |
| `minimal` | `50` | Minimalist model for the demonstration web page (after `benchmark`) | `benchmark/` |
| `all` | all | Everything, in order | |

Without `make`, open the `Makefile` and run the commands listed under each goal, in the same order, from the repository root. `make` is available by default on Linux and macOS; on Windows it can be installed, for example, with `choco install make`.

## Minimalist model (demonstration web page)

`50_minimal_model.py` trains a post hoc model for the demonstration web page: the 12 blood pressure predictors plus the first four laboratory predictors of the SHAP ranking (`MINIMAL_K = 4`). On MIMIC-IV 3.1 these are maximum urea nitrogen, minimum chloride, maximum glucose and minimum potassium, all from the basic metabolic panel, so the model needs no blood count; k = 4 was chosen after seeing the ranking for that reason. It follows the same procedure as the benchmark (same split, algorithms, grids, metrics and bootstrap), keeps the algorithm with the best cross-validated AUC, and reports its test AUROC and the paired difference against the best blood-pressure-only model. It is not part of the analyses reported in the paper.

## Notes on reproducibility

- **Blood pressure** is taken from every `omr` row whose `result_name` starts with "Blood Pressure", including position-specific records, without removing duplicates.
- **Standard deviations** use `ddof = 0` for blood pressure and `ddof = 1` for the analytes, as in the original analysis.
- **`_variance` columns** are generated for compatibility with the original cohort file but are not used by any analysis.
- **Outcome**: `Hypertension A+B` (the outcome of all models) is any qualifying code at any admission. `Hypertension A` flags codes 401.9 and I10; `Hypertension B` flags the remaining qualifying codes in patients without an A code. Codes 401.0 and 401.1 are not included.
- **Plausibility limits** are evaluated among patients with an admission record, and excluded patients are removed from the cohort file; patients without an admission record are kept and flagged, and are excluded by the analysis scripts.
- **Predictor sets**: `20_benchmark.py` reads the selected predictors from the outputs of `10_bolasso.py` and `12_catboost_shap.py`, and prints a warning if they differ from those reported in the paper.
- **Exported models**: for the best model of each set, `20_benchmark.py` and `50_minimal_model.py` save the fitted pipeline (`.joblib`) and an ONNX version (`.onnx`), and record the maximum difference between the ONNX and Python probabilities on the test set. ONNX works in float32, so for k-nearest neighbours the difference can be larger, because of ties between distances.
- All random seeds are fixed in the scripts.

## Citation

If you use this code, please cite the paper (reference to be added on publication).

## Licence

MIT (see `LICENSE`).
