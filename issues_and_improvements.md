02_baseline_models.ipynb

1. Unused variables in model comparison (Cell 11)
- best_model_obj, best_val_pred, best_val_proba are defined but unused
- Recommendation: Remove or add a note that they’re for reference

2. Missing note about fairness analysis
- Header mentions fairness analysis but doesn’t note it’s moved to notebook 03
- Recommendation: Add a note in the header

3. Hardcoded threshold assumption
- Uses default 0.5 threshold; document this
- Recommendation: Add a comment about threshold selection

4. Feature importance only for Random Forest
- No feature importance for Logistic Regression
- Recommendation: Add coefficient analysis for LR

5. Missing validation on loaded data
- Limited validation after loading
- Recommendation: Add checks for class distribution, feature ranges

03_fairness_analysis.ipynb

1. Potential issue with group ordering
- In visualize_fairness_metrics, groups may not match selection_by_group keys
- Fix:
   # In visualize_fairness_metrics, ensure groups match:
   groups = sorted(by_group.index.tolist())  # Sort for consistency
   # Or better:
   groups = list(metrics_dict['selection_by_group'].keys())

2. Missing error handling for empty groups
- If a group has no samples, calculations may fail
- Recommendation: Add checks for group sample sizes

3. No statistical significance testing
- Differences are shown but not tested
- Recommendation: Add p-values or confidence intervals (optional)

4. Heatmap may fail with single model/attribute
- Pivot may fail if only one model or attribute
- Recommendation: Add conditional logic

5. Missing interpretation guidance
- No explanation of what values mean
- Recommendation: Add a brief interpretation section

6. y_proba parameter unused
- calculate_fairness_metrics accepts y_proba but doesn’t use it
- Recommendation: Remove or use for calibration analysis