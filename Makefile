# Reproduces the analysis. Usage: make <goal>   (e.g. make cohort). Run the goals in the order listed.
# Without make, copy the commands under each goal and run them from the repository root.

PYTHON ?= python
S = scripts

.PHONY: all cohort lasso elasticnet shap kcurve-lasso selection benchmark tables figures minimal

# Everything, in order (several hours, mostly the benchmark)
all: cohort selection benchmark tables figures minimal

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

# Tables 1-2 and Supplementary Tables 3, 7, 9-11, printed to the terminal
tables:
	$(PYTHON) $(S)/30_table1_data.py
	$(PYTHON) $(S)/31_table1.py
	$(PYTHON) $(S)/32_table2.py
	$(PYTHON) $(S)/33_supp_table3.py
	$(PYTHON) $(S)/34_supp_table7.py
	$(PYTHON) $(S)/35_supp_table9.py
	$(PYTHON) $(S)/36_supp_table10.py
	$(PYTHON) $(S)/37_supp_table11.py

# Figures 1-4, Supplementary Figures 1-2, radar chart and graphical abstract -> output/figures/
figures:
	$(PYTHON) $(S)/40_fig1_cohort_flow.py
	$(PYTHON) $(S)/41_fig2_k_curve.py
	$(PYTHON) $(S)/42_fig3_forest.py
	$(PYTHON) $(S)/43_fig4_shap_analytes.py
	$(PYTHON) $(S)/44_supp_fig1_availability.py
	$(PYTHON) $(S)/45_supp_fig2_calibration.py
	$(PYTHON) $(S)/46_supp_spider.py
	$(PYTHON) $(S)/47_graphical_abstract.py

# Minimalist model for the demonstration web page (post hoc; after benchmark) -> output/benchmark/
minimal:
	$(PYTHON) $(S)/50_minimal_model.py
