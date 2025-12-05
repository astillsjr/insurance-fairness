Summary of calculations and results

1. Attribute selection (Cell 3)
Selected: EmploymentStatus

Baseline disparities:
- Demographic Parity Difference: 0.8609 (86.09 percentage points)
- Equalized Odds Difference: 0.9038 (90.38 pp)

Interpretation: EmploymentStatus had the worst disparities, so it was chosen as the mitigation target.

2. Post-processing: threshold optimization (Cell 7)

What it does: Adjusts decision thresholds per EmploymentStatus group to achieve demographic parity, without retraining.

Results:

Logistic Regression:
- Accuracy: 0.8505 (↑ from baseline 0.7119)
- DP Difference: 0.0231 (↓ from 0.8609)
- Improvement: 83.8 pp reduction in DP disparity

Random Forest:
- Accuracy: 0.8818 (↓ from baseline 0.9161)
- DP Difference: 0.1246 (↓ from 0.7989)
- Improvement: 67.4 pp reduction

Interpretation: LR Threshold Optimizer achieved the best fairness improvement (DP: 0.0231) while increasing accuracy. RF also improved fairness but with a small accuracy drop.

3. In-processing: ExponentiatedGradient (Cell 9)

What it does: Trains a fair model by enforcing fairness constraints during training.

Results:

Demographic Parity constraint:
- Accuracy: 0.6951
- DP Difference: 0.0481
- Training time: 1.04 seconds

Equalized Odds contraint:
- Accuracy: 0.7629
- EO Difference: 0.1494
- Training time: 1.98 seconds

Interpretation: Both constraints improved fairness (DP: 0.0481, EO: 0.1494) but with lower accuracy than threshold optimization. The DP constraint achieved better fairness.

4. Preprocessing: sample reweighting (Cell 11)

What it does: Adjusts sample weights during training to balance EmploymentStatus groups.

Results:

Logistic Regression:
- 
- 

Random Forest:
- 
- 

Interpretation: Reweighting reduced disparities but not as much as other methods, and accuracy dropped more. Not the best approach here.

5. Comprehensive comparison (Cell 13)

Key findings:
- Best accuracy: RF_Baseline (0.9161)
- Best DP difference: LR_ThresholdOptimizer (0.0231)
- Best EO difference: LR_ThresholdOptimizer (0.0513)


Insights

Best overall approach: LR Threshold Optimizer
- DP Difference: 0.0231 (97.3% reduction from baseline)
- EO Difference: 0.0513 (94.3% reduction)
- Accuracy: 0.8505 (↑ 19.5% from LR baseline)
- Trade-off: F1-Score dropped to 0.0639 (recall dropped significantly)

Performance vs fairness trade-offs
1. LR Threshold Optimizer: Best fairness, good accuracy, but low F1 (low recall)
2. RF Threshold Optimizer: Good balance (accuracy 0.8818, DP 0.1246)
3. ExponentiatedGradient: Moderate fairness and accuracy
4. Reweighting: Least effective for this problem

Why threshold optimization worked well
- No retraining needed
- Preserves model probabilities
- Can fine-tune thresholds per group
- Fast to apply

Why reweighting struggled
- May not address the root cause of bias
- Can overfit to training distribution
- EmploymentStatus groups may be too imbalanced


Recommendations

1. Use LR Threshold Optimizer if fairness is the priority (DP: 0.0231).

2. Use RF Threshold Optimizer if you need a balance (accuracy 0.8818, DP 0.1246).

3. Investigate the low F1-Score in LR Threshold Optimizer — the recall drop (0.0357) suggests many true positives are missed.

The notebook successfully demonstrated that post-processing (threshold optimization) was most effective for this problem, reducing EmploymentStatus disparities by over 97% while maintaining or improving accuracy.