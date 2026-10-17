# Limitations

- Every record and label is synthetic; no external dataset provenance is claimed.
- Target labels are deterministic score quantiles plus random noise, not actual appraisal outcomes.
- Several input variables deliberately influence the generated target, so the model can learn the generator's formula.
- A single fixed-seed stratified random split does not establish temporal, organizational, or population generalization.
- Cross-validation and test metrics are specific to synthetic rows, class definitions, selected features, and this random seed.
- The model can reflect proxy effects even though gender and salary band are excluded from the predictive features.
- Group fairness statistics are descriptive; small group sizes, class absence, missing values, and group definitions affect them.
- Permutation importance is global, may be unstable with correlated features, and is not causal or person-specific.
- Probability values are not calibrated against real outcomes and are not guarantees.
- Work metrics and generated recommendations do not represent a person's ability, intentions, health, or circumstances.
- The demonstration has not been validated for real appraisals and must not be used for employment decisions.
