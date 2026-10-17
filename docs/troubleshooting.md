# Troubleshooting

Run commands from the project root. In these examples, `.venv` is the local environment; activate it first or call its interpreter directly. Do not routinely delete environments, reset Git history, or force-push to resolve errors.

## Python and environment

| Symptom | Likely cause and diagnosis | Solution | Verify |
|---|---|---|---|
| `python` is not recognized | Python is missing or the Windows App Execution Alias intercepts it. Check `py --version` and `where.exe py`. | Install Python from the official Python distribution or use the installed `py` launcher; reopen VS Code after PATH changes. | `py --version`, then `py -m venv .venv`. |
| VS Code imports differ from terminal | VS Code selected another interpreter. Check the status-bar interpreter and `python -c "import sys; print(sys.executable)"`. | Select `.\.venv\Scripts\python.exe` using **Python: Select Interpreter**. | Run `python -m pip show pandas` in VS Code's terminal. |
| PowerShell activation is blocked | Local script execution policy prevents `Activate.ps1`. | Use Command Prompt `.\.venv\Scripts\activate.bat`, or invoke `.\.venv\Scripts\python.exe` directly; avoid weakening machine-wide policy. | `python -c "import sys; print(sys.prefix)"` should point to `.venv`. |
| Package installation fails | Network, Python version, wheel availability, or interpreter mismatch. Diagnose with `py --version` and `py -m pip --version`. | Use the project interpreter and `python -m pip install -r requirements.txt`; read the first package error. Do not mix global and virtual-environment installs. | `python -m pip check` and `python -c "import sklearn, pandas, streamlit"`. |
| Dependency conflict | Existing environment has incompatible package versions. | Create a separate project venv and install the declared requirements there; keep the existing environment untouched. | `python -m pip check`. |

## Dataset and validation

| Symptom | Likely cause and diagnosis | Solution | Verify |
|---|---|---|---|
| CSV file not found | Command ran outside the project or generation has not run. Check `data\raw\synthetic_employee_data.csv`. | From the project root run `python src\data_generator.py`. | `Test-Path data\raw\synthetic_employee_data.csv`. |
| Missing/extra columns | Dataset schema is incompatible with this project. The validator reports names. | Regenerate the sample, or intentionally update the schema, model training inputs, tests, and docs together. Do not silently drop or rename fields. | `python -c "import pandas as pd; print(pd.read_csv('data/raw/synthetic_employee_data.csv').columns.tolist())"`. |
| Invalid target labels | Labels are not exactly High, Medium, or Low. | Correct the source/generator so the class meaning is explicit; do not coerce unknown labels to a known class. | `python -c "import pandas as pd; print(pd.read_csv('data/raw/synthetic_employee_data.csv').perf_band_next.value_counts())"`. |
| Empty or malformed CSV | File has no rows, broken quoting, or invalid encoding. | Preserve the original file; inspect a copy/source, then regenerate or correct the CSV formatting. | `python -c "from src.data_validation import load_and_validate_csv; print(load_and_validate_csv('data/raw/synthetic_employee_data.csv').shape)"`. |
| Missing values | Input has nulls or blank cells. The validator reports affected fields; model preprocessing can impute inference features, but project dataset validation does not silently approve nulls. | Identify why values are missing and choose/document an explicit cleaning or imputation decision. Do not remove rows without a stated reason. | Re-run the validator and inspect missing counts before training. |
| Unexpected categories | A category falls outside the documented generator values. | Verify spelling and schema meaning; either correct the source or intentionally document and test a supported category change. Inference-time unknown categories are ignored by the fitted encoder, but the training dataset schema remains validated. | Re-run validation and the unseen-category preprocessing test. |
| Wrong numeric type/range | A numeric field contains text, infinity, or a value outside documented bounds. | Correct the source value or fix its data-generation rule. Do not silently cast invalid strings or clip outliers. | Run `python src\data_generator.py` and `python -m pytest -q`. |

## Machine learning

| Symptom | Likely cause and diagnosis | Solution | Verify |
|---|---|---|---|
| Training fails | Dataset absent/invalid, too few records per class, package issue, or schema mismatch. Read the first exception and run validation/tests. | Generate and validate the CSV; ensure each class can support the stratified split and five CV folds. | `python src\data_generator.py`, then `python src\train_model.py`. |
| Class imbalance / CV split error | A class has fewer members than folds or split requirements. Check `value_counts()`. | For this synthetic dataset regenerate a sufficient number of rows; if using another dataset, choose split/fold counts based on class support and document changes. | Confirm class counts and rerun training. |
| Model artifact is missing | Training was not run or ran from a different project. | Run `python src\train_model.py` from the root and check `models\employee_performance_model.joblib`. | Load it using `python -c "import joblib; print(type(joblib.load('models/employee_performance_model.joblib')))"`. |
| Incompatible prediction features | Input is missing a trained feature or has incompatible columns/order. | Supply exactly the expected `feature_names_in_` schema; use the dashboard's generated example controls. Never include the target as an input. | Run `python src\predict.py` or the prediction tests. |
| Unexpectedly low scores | Noise, formula difficulty, data size, seed, features, or class errors can lower metrics. | Inspect the confusion matrix, class report, split, and data generator. Treat low metrics honestly; do not repeatedly tune the held-out test set. | Compare CV and fixed test metrics to the DummyClassifier in `outputs/reports/model_evaluation.json`. |
| Model is no better than baseline | Features may carry little signal, the baseline may be strong, or preprocessing/target may be wrong. | Verify target class construction and feature pipeline; report the result if it persists. Do not claim a better model without evidence. | Re-run the exact workflow and compare saved macro-F1/accuracy. |

## Dashboard

| Symptom | Likely cause and diagnosis | Solution | Verify |
|---|---|---|---|
| Streamlit does not launch | Wrong interpreter or missing package. | Use `python -m streamlit run app.py` in the project venv. | Confirm Streamlit prints a local URL and opens the overview tab. |
| Port conflict | Another service occupies Streamlit's port. | Use `python -m streamlit run app.py --server.port 8502` and open the printed URL. | Refresh the specified local URL. |
| Missing model/reports | Training/evaluation artifacts have not been created. | Run generation, EDA, training, then evaluation from the project root. The dashboard does not train automatically. | Check the generated files in `models/` and `outputs/reports/`. |
| Uploaded CSV rejected | Upload lacks required columns, has extra fields, invalid target values, data types, or ranges. | Use a generated-compatible CSV and address each validation message. Upload contents are treated only as tabular data, never code. | Confirm the schema checks pass and the dashboard says it uses the upload. |
| Charts do not appear | EDA/evaluation has not generated the PNGs or paths are wrong. | Run `python src\eda.py` and `python src\evaluate_model.py` from the root. | Check `outputs\figures\` and refresh the tab. |
| Prediction error | Model missing, malformed inputs, or unexpected artifact version. | Regenerate the model in the current environment; use exact controls/features and inspect the displayed error. | Run prediction tests and try the fictional sample. |

## GitHub and Git

This workspace was not a Git repository when checked. These instructions are for a future publication; do not initialize over any existing repository without first inspecting it.

| Symptom | Likely cause and diagnosis | Solution | Verify |
|---|---|---|---|
| Git is not recognized | Git for Windows is missing or PATH has not refreshed. Check `git --version`. | Install Git for Windows, restart the terminal, and use non-interactive Git commands. | `git --version`. |
| Authentication problem | No authorized GitHub login, expired credential, or wrong account. Diagnose with `gh auth status`; do not paste secrets into chat. | Use GitHub CLI's browser/device authentication or configured credential manager; request appropriate repository permissions. | `gh auth status` shows the intended account. |
| Incorrect remote URL | `git remote -v` points to the wrong repository or protocol. | Verify the destination URL in the GitHub UI and update only the intended remote with `git remote set-url origin <verified-url>`. | `git remote -v` shows the verified destination. |
| Push rejected | Remote has commits not present locally or permission/branch protections apply. | Inspect `git status`, fetch and review divergence, then integrate changes using an understood merge/rebase workflow. Ask a maintainer if protected. Do not force-push by default. | `git status` is clean and the expected commit is visible on the remote. |
| Sensitive/unwanted file staged | Broad add patterns can stage `.venv`, data, credentials, or large artifacts. | Inspect `git status --short` and `git diff --cached --name-only`; unstage a specific unwanted path with `git restore --staged -- path`. Update `.gitignore`; if a secret was published, revoke/rotate it immediately and follow incident procedures. | Recheck staged names and scan the final diff before committing. |

## Safe publication checklist

Before a first push: inspect Git history/status/remotes; review each staged file; confirm dataset provenance and license; exclude `.venv/`, caches, model binaries, credentials, and confidential data; verify README links and image paths; and never use a force push as a routine repair. This workspace currently has no repository, remote, commits, or verified publication.
