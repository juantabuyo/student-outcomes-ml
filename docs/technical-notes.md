# Evaluation and preprocessing

The [README](../README.md) gives the run commands, saved metrics and main limits.
The real CSV is not included; [the data contract](../data/README.md) describes
its schema and integrity hash.

Both tasks share an outcome-stratified 80/20 split. Classification CV is stratified;
regression uses shuffled KFold with seed 42. Within each fold, training rows alone
fit the one-hot vocabulary and scaler. Unknown categories become all-zero dummies;
the fixed `si`/`no` mapping estimates no statistics.

The classification comparison baseline has depth 4, outside the restricted
search's depths 3 and 5. The regression depth-3 baseline and matching grid entry
both give CV R² 0.757912598 in the saved full run. Selection uses training CV,
not holdout metrics; it does not select among every possible model or setting.

Historical classification EDA contained 724 of the 885 original regression holdout
rows. The shared split makes membership consistent but cannot undo exploration.
It does not establish that the original fit used holdout targets. Changed
regression membership also prevents comparison on an unchanged independent test set.

The course data have 870 zero second-semester grades, including 469 with positive
evaluation counts. These values are retained without assigning a cause.
Administrative predictor observation times remain unresolved. No external cohort,
temporal validation or prospective service has been evaluated.

Clustering describes the full dataset. The variant without second-semester fields
still includes first-semester information. Outcome labels are not clustering
inputs; colouring and ARI are post-fit comparisons. The k=3 interpretation is
informed by the three known outcome classes although silhouette favours k=2.
Feature importance and clustering are descriptive, not causal. No trained model
or student-level prediction table is included.

The notebook calls `main.run`, which creates its output directory, including
missing parents. By default it writes synthetic results to `results/notebook_demo/`.
Its saved cell outputs are also synthetic. Changing to a real dataset requires
updating the path and synthetic flag, and using five folds for the full experiment.
Keep a distinct output directory to avoid overwriting results you want to retain.
