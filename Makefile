# Reproduces the analysis. Usage: make <goal>   (e.g. make cohort). Run the goals in the order listed.
# Without make, copy the commands under each goal and run them from the repository root.

PYTHON ?= python
S = scripts

.PHONY: all cohort lasso elasticnet shap kcurve-lasso selection benchmark results minimal

# Everything, in order (several hours, mostly the benchmark)
all: cohort selection benchmark results minimal

# Analytical cohort from the raw MIMIC-IV files -> output/cohort_clean.xlsx
cohort:
	$(PYTHON) $(S)/01_labs.py
	$(PYTHON) $(S)/02_bp.py
	$(PYTHON) $(S)/03_cohort.py
	$(PYTHON) $(S)/04_exclusions.py
	$(PYTHON) $(S)/05_availability.py

# BP+Lasso (Bolasso) -> output/selection_lasso.xlsx
lasso:
	$(PYTHON) $(S)/10_bolasso.py

# Elastic-net sensitivity analysis -> output/selection_elasticnet.xlsx
elasticnet:
	$(PYTHON) $(S)/11_elastic_net.py

# BP+SHAP (CatBoost SHAP ranking and k curve) -> output/selection_catboost.xlsx
shap:
	$(PYTHON) $(S)/12_catboost_shap.py

# k curve in lasso order (needs shap) -> output/kcurve_lasso.xlsx
kcurve-lasso:
	$(PYTHON) $(S)/13_lasso_ranking_folds.py
	$(PYTHON) $(S)/14_kcurve_lasso.py

# All predictor selection
selection: lasso elasticnet shap kcurve-lasso

# 5 algorithms x 4 predictor sets (several hours) -> output/benchmark/ (best model of each set: predictions, .joblib, .onnx)
benchmark:
	$(PYTHON) $(S)/20_benchmark.py

# Comparison between predictor sets and contribution of each analyte, printed to the terminal
results:
	$(PYTHON) $(S)/38_set_comparison.py
	$(PYTHON) $(S)/39_analyte_contribution.py

# Minimalist model for the demonstration web page (post hoc; after benchmark) -> output/benchmark/
minimal:
	$(PYTHON) $(S)/50_minimal_model.py
