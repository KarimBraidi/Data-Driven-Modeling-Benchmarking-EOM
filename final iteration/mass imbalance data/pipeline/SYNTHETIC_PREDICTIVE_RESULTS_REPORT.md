# Synthetic Predictive Results: MSE and R2

This section reports the predictive performance of the synthetic mass-imbalance equation-discovery experiments. For each algorithm, regime, and degree of freedom, the configuration with the lowest validation MSE was selected, and its held-out test MSE, test R2, ATE, and model size were reported. This selection procedure separates model selection from final evaluation and prevents the test set from being used to choose the best configuration.

The predictive metrics should be interpreted as complementary to the EOM recovery metrics. MSE and R2 measure how well the discovered equations reproduce the observed acceleration data, while structural recall and coefficient relative error measure whether the recovered equations match the true physical terms and coefficients. Therefore, a model with low MSE is not necessarily the best equation-recovery model if it uses spurious terms or misses the true EOM structure.

## Figures

- [synthetic_predictive_figures/synthetic_test_mse_heatmap_bestval.svg](synthetic_predictive_figures/synthetic_test_mse_heatmap_bestval.svg): mean test log10(MSE) by algorithm and regime.
- [synthetic_predictive_figures/synthetic_test_r2_heatmap_bestval.svg](synthetic_predictive_figures/synthetic_test_r2_heatmap_bestval.svg): mean test R2 by algorithm and regime.
- [synthetic_predictive_figures/synthetic_predictive_bars_bestval.svg](synthetic_predictive_figures/synthetic_predictive_bars_bestval.svg): side-by-side comparison of mean log10(MSE) and R2.
- [synthetic_predictive_figures/synthetic_dof_test_mse_heatmap_bestval.svg](synthetic_predictive_figures/synthetic_dof_test_mse_heatmap_bestval.svg): DOF-resolved test log10(MSE).
- [synthetic_predictive_figures/synthetic_dof_test_r2_heatmap_bestval.svg](synthetic_predictive_figures/synthetic_dof_test_r2_heatmap_bestval.svg): DOF-resolved test R2.
- [synthetic_predictive_figures/synthetic_ate_heatmap_bestval.svg](synthetic_predictive_figures/synthetic_ate_heatmap_bestval.svg): mean ATE by algorithm and regime.

## Overall Predictive Summary

| Algorithm | Mean test MSE | Median test MSE | Mean log10(test MSE) | Mean test R2 | Mean ATE | Mean terms |
|---|---:|---:|---:|---:|---:|---:|
| SINDy | 5.04e-05 | 4.72e-07 | -9.66 | 0.709 | 0.0145 | 29.78 |
| Hybrid SINDy | 1.18e-01 | 2.53e-08 | -7.16 | 0.841 | 0.0159 | 32.33 |
| Lagrangian SINDy | 1.80e-02 | 1.71e-02 | -3.57 | 0.556 | 0.0217 | 11.56 |

Overall, Hybrid SINDy achieved the highest mean test R2, indicating strong predictive agreement with the synthetic acceleration data. SINDy achieved the lowest mean and median test MSE overall, although the MSE comparison should be read on a log scale because the acceleration targets have different magnitudes and a few configurations can dominate raw averages. Lagrangian SINDy produced the weakest predictive metrics among the three methods in this acceleration-form evaluation, which is consistent with the fact that its learned model is not expressed in the same direct acceleration basis as SINDy and Hybrid SINDy.

## Regime-Wise Interpretation

In the bouncing regime, Hybrid SINDy performed best predictively, with a mean test MSE of 1.46e-07 and mean test R2 close to 1.0. This indicates that the regime-aware hybrid formulation was particularly effective when contact dynamics were active and the model could specialize to the bouncing mode. SINDy also produced a low ATE in bouncing, but its mean R2 was lower, suggesting that the single temporal-split fit did not capture all DOFs with the same consistency.

In the rolling regime, SINDy achieved the lowest test MSE, with values near numerical precision for the selected configurations. Hybrid SINDy also produced very low MSE in this regime, while Lagrangian SINDy was less accurate in direct acceleration prediction. The R2 values for SINDy and Hybrid SINDy were both around 0.667 when averaged across DOFs, suggesting that at least one rolling target had low variance or was difficult to score using R2 even when MSE was small. For this reason, rolling should be interpreted with both MSE and R2 rather than R2 alone.

In the flight regime, SINDy achieved the lowest mean test MSE, while Hybrid SINDy achieved the highest mean test R2. This shows that the two predictive metrics emphasize different aspects of performance: MSE rewards small absolute error, whereas R2 rewards explained variance relative to the target signal. Lagrangian SINDy performed reasonably in R2 for flight but had higher MSE than SINDy.

## Connection to EOM Recovery

These predictive results should be reported alongside structural recall and coefficient relative error. Hybrid SINDy can achieve better recall because its regime-aware fitting is better aligned with the hybrid physics of bouncing, rolling, and flight. However, SINDy can still produce better coefficient relative error because its temporal splits provide broader continuous data coverage for estimating the coefficients of the terms it does recover. This creates an important tradeoff: Hybrid SINDy is often better at identifying the correct EOM structure, while SINDy can be better at estimating coefficient magnitudes and minimizing absolute prediction error.

The main conclusion is that MSE and R2 are necessary but not sufficient for evaluating equation discovery. They show whether the learned equations reproduce the synthetic data, but they do not prove that the true physics was recovered. For this project, the strongest model is the one that balances low MSE, high R2, high structural recall, sparse equations, and low coefficient relative error.

## Recommended Caption Text

**Predictive-performance heatmaps for the synthetic mass-imbalance experiments.** For each algorithm, regime, and DOF, the configuration with the lowest validation MSE was selected and evaluated on the held-out test set. MSE is shown on a log10 scale to reduce the influence of large outliers and target-scale differences. R2 is reported as a normalized measure of explained variance. These metrics evaluate predictive accuracy and are interpreted together with structural recall and coefficient relative error to assess full equation-of-motion recovery.