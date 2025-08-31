# WHO Data Loading and SemTab Conversion

This section documents how raw WHO tables are converted into a SemTab-style dataset that downstream components (e.g., Sherlock preprocessing and training) can consume.

`sherlock_preprocessing/`

- **`sherlock_data_preprocesisng.ipynb`**  
  Runnable notebook that takes raw `.csv` tables as input and produces a SemTab-formatted dataset.  
  The generated dataset is then used by other data-preprocessing notebooks to prepare inputs for the Sherlock model.

---

## Input Expectations

- One or more raw table files in `.csv` format.
- Optional mapping files (e.g., dictionaries or label maps) if your workflow requires aligning raw column headers to semantic types.
- A designated output directory where the SemTab-style dataset will be written.