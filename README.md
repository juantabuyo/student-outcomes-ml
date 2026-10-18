# Student outcomes: supervised models and exploratory profiles

My Machine Learning final project compares models for students' final outcomes
and second-semester grades, then uses PCA and clustering to explore academic
profiles. It uses scikit-learn estimators; the focus is experimental design,
evaluation and interpretation.

## Run

Python 3.12; tested with 3.12.14. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py --data data/synthetic_students.csv --output results/synthetic_smoke --folds 2 --synthetic
```

On Windows, activate with `.venv\Scripts\activate`. The bundled 180-row CSV is
**entirely fictitious**: its scores demonstrate execution, not educational findings.
Regenerate it with `python synthetic_data.py --output data/synthetic_students.csv`.

The full experiment requires an authorised local copy of the course dataset:

```bash
python main.py --data /path/to/authorised_students.csv --output results/local_run
```

Replace the placeholder path with an existing CSV. The real dataset is absent
and must be obtained separately with its source, translation mapping and terms.
[data/README.md](data/README.md) describes the schema and source hash. There are no
automatic downloads. Full runs use five-fold cross-validation.

`student_outcomes.ipynb` calls the same modules and defaults to synthetic data.
Install its kernel/client dependencies with:

```bash
python -m pip install -r requirements-notebook.txt
```

Open the notebook in an existing Jupyter installation with this folder as the
working directory. It creates `results/notebook_demo/` when run and replaces
files there on subsequent runs. CLI and sequential cell execution have been
checked; execution through a real Jupyter kernel remains unverified. The bundled
notebook retains synthetic outputs, not results from real students.

## Methods and results

Each supervised task compares seven models and retains a five-configuration
Gradient Boosting search. Both use one outcome-stratified 80/20 split, seed 42;
EDA uses training rows, and encoding/scaling are fitted within each CV fold.
The restricted search does not establish the best model across all candidates.

The full run on 17 October 2026 used 4,424 rows: 3,539 training and 885 holdout.
[Saved results](results/corrected_full/README.md) include comparisons, plots,
settings and dataset/split hashes.

| Task | Holdout metrics | Selected Gradient Boosting configuration |
|---|---|---|
| Classification | Accuracy 0.7695; macro-F1 0.6914 | 100 trees; depth 3; learning rate 0.1 |
| Regression | R² 0.7859; RMSE 2.4358; MAE 1.3449 | 200 trees; depth 3; learning rate 0.05 |

![Normalised confusion matrix](results/corrected_full/classification_confusion_normalised.png)

PCA, K-means and Ward use selected numeric/binary fields, with and without
second-semester variables. Silhouette favours k=2 in both variants; the reported
k=3 profiles retain an interpretation informed by the three outcome classes.
K-means ARI is 0.1558 and 0.0961 respectively.

## Interpretation and limits

The **holdout was historically explored**. A corrected split does not make it
unseen, and regression's changed membership prevents an unchanged-test-set
comparison with earlier scores. Classification includes second-semester records,
so it is not an early-warning model. Regression excludes all six second-semester
fields and the outcome, but administrative predictor timestamps remain unverified.

Clustering describes the full dataset; outcome colouring and ARI are post-fit
comparisons. Neither clustering nor feature importance establishes causality.
Zero grades are retained without assuming their cause.

## Tests and version notes

```bash
python -m unittest discover -s tests -v
```

Seven tests check shared split membership, regression exclusions, fold-local
encoding/scaling, unseen categories, zero-grade handling, model settings and
input immutability. They do not validate educational predictions.
[Technical notes](docs/technical-notes.md) give further evaluation details.
Post-submission corrections introduced the shared split and fold-local
preprocessing while retaining the model settings and restricted searches.

## Academic context and credit

Individual Machine Learning submission, iMAT, ICAI,
Universidad Pontificia Comillas, 2025–2026. Scikit-learn supplies the learning
algorithms; PCA, K-means and Ward also use library implementations.
