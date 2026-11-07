# DSCI 552 - Homework 6

Haresh Sambandamoorthy

## Final documents

- `DSCI_552_HW6_Report.pdf`: 18-page full report, with larger readable Times-style font, clean section flow, black text and headings, and **colored charts**.
- `DSCI_552_HW6_ISLR_Solutions.pdf`: the same required and optional ISLR answers extracted into a separate reference PDF.
- `DSCI_552_HW6_Reworked.ipynb`: executed Jupyter notebook with the original verified code and outputs; the latest change was formatting and restoring plot color only.

## Supporting files

- `data/`: the original APS train/test CSV files.
- `figures/`: all 18 plots in color for clarity, including comparison plots and textbook diagrams.
- `results/`: all original numeric outputs, cross-validation tables, confusion matrix/ROC result arrays, and feature variations.
- `make_report.py`, `run_original_baselines.py`, `l1_logistic_model_tree.py`, `run_model_trees.py`, `finalize_model_trees.py`, and other scripts support regeneration.

To re-render just the reports from the supplied results: `python make_report.py`.
To rebuild the model runs and reports (computationally expensive): `python run_all.py`.

**Method note:** The submitted custom model tree uses L1-penalized logistic regression at the internal splits. Standard XGBoost is reported separately; the two are not the same algorithm. The assignment names XGBoost while describing custom multivariable splits, so the report openly explains the difference.

## AI use acknowledgment

AI tools were used to assist with organizing the code, compiling the report, and checking formatting. All reported numerical results were generated from the included APS datasets using the accompanying reproducible code.
