# Responsible AI

This repository is a controlled educational simulation, not an HR tool. Synthetic records avoid exposure of real employee data, but synthetic generation does not make the generated assumptions neutral or applicable to people.

## Risks and safeguards

- **Synthetic-data limitations:** model behavior tracks rule-generated labels. Never describe the target as observed performance.
- **Possible bias:** excluded protected attributes do not prevent proxy relationships. Group error-rate comparisons can reveal differences in this sample but cannot prove fairness or explain causes.
- **Uncertainty:** probabilities are classifier estimates, not calibrated guarantees. A label is not a measure of a person's worth or potential.
- **Data leakage:** keep the target and identifiers out of features; fit imputation, scaling, and encoding only using training folds/data. Avoid including future outcomes if adapting the example.
- **Privacy:** do not upload, paste, or enter confidential business or employee data. The demo CSV uploader accepts only rows passing this project's synthetic schema and never executes file content.
- **Human oversight:** any development suggestions are optional, non-punitive prompts requiring respectful human review; they are not findings about an individual.

## Prohibited consequential uses

Do not use this project to screen, hire, reject, rank, monitor, discipline, terminate, promote, or set compensation for people. Do not treat its predictions as appraisal recommendations. The model and generated data have not been validated for any real employment setting.

## Fairness interpretation

Reported group counts, predicted-High rates, macro metrics, High-class error rates, and confusion matrices are descriptive. Thresholds, sample size, missing group membership, class support, synthetic ground truth, and proxy variables influence them. Equal measured rates do not establish fairness; disparities do not establish discrimination or identify a remedy. No threshold mitigation is implemented.
