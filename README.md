# Property Tax Shift Simulation and Distributional Impacts

**MATH 6243: Statistical Learning — Final Project**  
**Author:** Ethan Go  
**Northeastern University — Spring 2026**  
**Professor:** Eric Gerber

---

## Problem Statement

This project investigates **Option A: Property Tax Shift Simulation and Distributional Impacts** from the MATH 6243 course project guidelines.

Currently, Cook County (which includes Chicago) taxes both land and the buildings on it. An alternative called a **land value tax (LVT)** would shift the tax burden onto land alone, while reducing or eliminating the tax on improvements (buildings). The key constraint is **revenue neutrality** — the county still collects the same total tax revenue.

### Core Research Questions

1. **Under a revenue-neutral shift from improvements-tax to land/site-tax, what fraction of Cook County homeowners would see a lower tax bill?**
   - *Improvement tax*: A tax on the value of anything built on land (house, garage, renovations, etc.)
   - *Land/site tax*: A tax on the value of the land itself, regardless of what is built on it
   - The assessed values (`board_bldg` and `board_land`) are already provided in the CCAO dataset. The goal is to estimate these values independently using ML models and compare how the distributional outcomes differ.

2. **How does Cook County compare to Philadelphia using a consistent pipeline?** *(time-permitting)*
   - Cook County uses a **classified assessment** system (10% residential, 25% commercial)
   - Philadelphia uses a **uniform assessment** system (100% for all property types)
   - Requires data harmonization across the two jurisdictions

3. **How do different ML methodologies compare in estimating the distributional shift?**
   - Evaluate whether the policy conclusion (who benefits, who loses) is sensitive to the choice of valuation model

---

## Data Sources

| Dataset | Source | Link |
|---|---|---|
| Cook County Assessed Values | Cook County Open Data Portal | [uzyt-m557](https://datacatalog.cookcountyil.gov/Property-Taxation/Assessor-Historic-Assessed-Values/uzyt-m557/data) |
| Cook County Building Characteristics | Cook County Open Data Portal | [x54s-btds](https://datacatalog.cookcountyil.gov/Property-Taxation/Assessor-Single-and-Multi-Family-Improvement-Chara/x54s-btds) |
| Cook County Parcel Universe | Cook County Open Data Portal | [nj4t-kc8j](https://datacatalog.cookcountyil.gov/Property-Taxation/Assessor-Parcel-Universe/nj4t-kc8j) |
| Cook County Parcel Sales | Cook County Open Data Portal | [wvhk-k5uv](https://datacatalog.cookcountyil.gov/Property-Taxation/Assessor-Parcel-Sales/wvhk-k5uv) |
| Cook County Data Wiki | GitHub | [ccao-data/wiki](https://github.com/ccao-data/wiki) |
| Philadelphia Properties & Assessment | OpenDataPhilly | [OPA Properties](https://opendataphilly.org/datasets/philadelphia-properties-and-assessment-history/) |

### Key Variables

**Cook County:**
- `board_land` — Board of Review certified land assessed value (L_i in simulation formula)
- `board_bldg` — Board of Review certified building assessed value (B_i in simulation formula)
- `board_tot` — Board of Review certified total assessed value
- `class` — Property class code (200s = residential, 500s = commercial)
- `township_name` — Township identifier for geographic analysis

**Philadelphia:**
- `market_value_land` — Land market value
- `market_value_improvement` — Building/improvement market value
- `market_value` — Total market value

**Important:** Cook County values are assessed values (10% of market value for residential, 25% for commercial). Philadelphia values are already at market value (100%). Harmonization requires dividing Cook County values by the assessment ratio.

---

## Methods

### ML Models for Property Valuation

The following regression methods are compared for predicting property values:

1. **Regularized Regression** (Ridge/Lasso/Elastic Net)
2. **Split Regression** (Piecewise/segmented regression)
3. **GAM** (Generalized Additive Model)
4. **Random Forest Regression**
5. **Double/Debiased Machine Learning** (Chernozhukov et al. 2018) — *time-permitting extension*

### From-Scratch Implementation

A custom regression function implemented from scratch in Python (NumPy only) that supports:
- Different regularization penalties (L1, L2, Elastic Net)
- Configurable loss functions
- Cross-validated hyperparameter selection

### Revenue-Neutral Simulation

Following the methodology of England and Zhao (2005), the simulation uses three assumptions:

1. **Tax burden shifts toward land:** τ_L > τ > τ_B ≥ 0, C ≥ 0
2. **Non-negativity:** Individual tax payments cannot be negative after reform
3. **Revenue neutrality:** Total revenue before = Total revenue after

The revenue-neutrality constraint:

```
Σ τ(L_i + B_i) = Σ (τ_L × L_i + τ_B × B_i − C) × I_i
```

Where:
- τ = current uniform tax rate
- τ_L = new land tax rate
- τ_B = new building tax rate  
- C = uniform credit (optional)
- L_i = land assessed value for parcel i
- B_i = building assessed value for parcel i
- I_i = indicator (1 if new tax > 0, else 0)

Reference implementation: [LVTShift toolkit](https://github.com/gregmiller00/LVTShift)

---

## Evaluation Plan

### Model Performance
- K-fold Cross-Validation RMSE comparison across models
- Spatial block CV following Roberts et al. (2017) to avoid inflated performance from spatial autocorrelation

### Distributional Comparison
- **Effect Size:** Fraction of homeowners paying less under each model's simulation
- **Distribution Characteristics:** Mean, median, range, SD, moments of ΔTax distributions
- **Formal Tests:** Kolmogorov-Smirnov test, Wasserstein distance between model distributions
- **Stratified Analysis:** Results by township, property class, and value decile
- **Visual:** Histograms, QQ plots, overlaid density plots

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| **Missingness** | Remove, impute, or add indicator features. Ensure consistent treatment across models. |
| **Confounding** | Appropriate regularization, cross-validation, and feature selection |
| **Data Access** | All data is publicly available online. No API keys required. |
| **Data Harmonization** | Build explicit crosswalk between Cook County and Philadelphia schemas |
| **Data Balancing** | Use weighting or matching if land vs. building distributions are imbalanced |

---

## Literature

- England, Richard W., and Min Qiang Zhao. "Assessing the Distributive Impact of a Revenue-Neutral Shift from a Uniform Property Tax to a Two-Rate Property Tax with a Uniform Credit." *National Tax Journal*, vol. 58, no. 2, 2005, pp. 247–260.

- Bowman, John H., and Michael E. Bell. "Distributional Consequences of Converting the Property Tax to a Land Value Tax: Replication and Extension of England and Zhao." *National Tax Journal*, vol. 61, no. 4, 2008, pp. 593–607.

- Mullainathan, S., and J. Spiess. "Machine Learning: An Applied Econometric Approach." *Journal of Economic Perspectives*, vol. 31, no. 2, 2017, pp. 87–106.

- Roberts, D. R., V. Bahn, S. Ciuti, M. S. Boyce, J. Elith, et al. "Cross-Validation Strategies for Data with Temporal, Spatial, Hierarchical, or Phylogenetic Structure." *Ecography*, vol. 40, no. 8, 2017, pp. 913–929.

- Chernozhukov, V., D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, W. Newey, and J. Robins. "Double/Debiased Machine Learning for Treatment and Structural Parameters." *Econometrics Journal*, vol. 21, no. 1, 2018, pp. C1–C68.

---

## Project Structure

```
├── README.md
├── data_raw/                  # Raw data downloads (not tracked in git)
│   ├── cook_county/
│   └── philadelphia/
├── data_processed/            # Cleaned, merged datasets
├── src/
│   ├── data_prep.py           # Data download, merge, preprocessing
│   ├── from_scratch.py        # From-scratch regression implementation
│   ├── models.py              # Model training (regularized, GAM, RF)
│   ├── tax_simulation.py      # Revenue-neutral LVT simulation
│   ├── distributional.py      # Distributional analysis and comparison
│   └── evaluation.py          # Metrics, spatial CV, distribution tests
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_modeling.ipynb      # Model training and comparison
│   ├── 03_simulation.ipynb    # Tax shift simulation
│   └── 04_results.ipynb       # Results and visualization
├── reports/
│   ├── proposal.pdf
│   ├── final_report.pdf
│   └── poster.pdf
└── requirements.txt
```

**Note:** Raw data files are not included in the repository due to size. See the Data Sources table above for download links. Place downloaded files in `data_raw/` before running the pipeline.

---

## Reproducibility

1. Clone this repository:
   ```
   git clone https://github.com/[username]/cook-county-lvt-simulation.git
   cd cook-county-lvt-simulation
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the data (see Data Sources above) and place in `data_raw/`

4. Run notebooks in order: `01_eda.ipynb` → `02_modeling.ipynb` → `03_simulation.ipynb` → `04_results.ipynb`

---

## AI Disclosure

Claude (Anthropic) was used to brainstorm research directions, discover and summarize relevant literature, and assist with structuring the proposal. All analysis, code, and writing in the final report are the author's own work.

---

## License

This project is for academic purposes as part of MATH 6243 at Northeastern University.
