# Exploring Kolmogorov–Arnold Networks for Unsupervised Anomaly Detection in Industrial Processes

This repository contains the code and results associated with the manuscript:

**Exploring Kolmogorov–Arnold Networks for Unsupervised Anomaly Detection in Industrial Processes**  
Enrique Luna-Villagómez and Vladimir Mahalec, 2025.

The repository includes the implementations, experiments, and output files used to generate all results and figures presented in the paper.

---

## Repository Contents

- **src/**  
  All source code for data preprocessing, model training, evaluation, and figure generation.

- **data/**  
  Raw TEP datasets. Includes the fault-free training file (`d00_tr.csv`) and the fault cases stored in `cases/`.

- **results/**  
  Numerical outputs used in the manuscript, including FDR/FAR metrics and other intermediate results.

- **tables/**  
  Monitoring results for each trained instance (model type–seed), provided as LaTeX tables or CSV files.

- **Bayesian Results/**  
  Outputs of the Bayesian signed-rank tests reported in Section 4.4.

- **Experiments/**  
  Scripts used to run the training procedures and evaluation routines.

- **logscal_profiles_final/**  
  Figures corresponding to Section 4.1 of the manuscript.

---

## Disclaimer

This codebase is provided **as is**, without warranty of any kind.  
It is intended solely to support transparency and reproducibility of the results reported in the associated manuscript.

---

## Citation

If you use this repository or build upon its contents, please cite the paper listed above.
