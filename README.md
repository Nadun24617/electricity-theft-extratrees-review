# electricity-theft-extratrees-review

This is my Phase 2 (baseline implementation) and Phase 3 (technical review and stress tests) work for the Machine Learning and Related Applications module.

The paper I'm reviewing is Appiah et al. (2023), "Extremely randomised trees machine learning model for electricity theft detection", published in Machine Learning with Applications (vol. 12, 100458). They use the SGCC smart meter dataset. In Phase 1, while reading the paper, I noticed they run SMOTETomek on the whole dataset before the train/test split. That means synthetic theft samples end up in the test set, so this phase was mainly about checking how much that inflates their results.

## Data

After removing empty customers (all missing or all zero) and duplicate rows, I get the same counts as the paper: 40,256 customers, of which 3,579 are theft (about 8.9%), with 1,035 days of readings each.

## Experiments

- **E1 - the paper's pipeline.** Missing days filled with Eq. 1 from the paper, values capped at mean + 2 std, Min-Max scaling and SMOTETomek on all the data, then an 80/20 split of the resampled data. I used the paper's best settings (500 trees, max depth 70) rather than rerunning their grid search. I also checked 4-fold CV without shuffling, which is what their `GridSearchCV(cv=4)` would do.
- **E2 - split first.** Same preprocessing and same model, but the 80/20 split is done on the real customers first, and scaling and SMOTETomek are fitted on the training part only. The test set keeps the real theft rate.
- **E3 - my corrected pipeline from Phase 1.** Rule-based cleaning (readings over 10,000 kWh set to missing, gaps of up to 3 days interpolated, log scale, clipping at median + 3 MAD) plus 11 summary features per customer. I used class weights instead of SMOTE. Each model is tuned with nested CV (5 outer folds, 5 inner folds, random search over 8 settings, scored on PR-AUC). The decision threshold is the one with the best MCC on the out-of-fold training predictions. Models: Extra Trees, Random Forest, Logistic Regression, and Extra Trees using only the summary features.

## Results (test MCC)

| Setup | MCC |
|---|---|
| Reported in the paper | 0.9506 |
| E1, resampled 50/50 test (paper style) | 0.931 |
| E1, same model scored on real customers only | 0.834 |
| E2, split first | 0.295 |
| E3 Extra Trees | 0.348 |
| E3 Random Forest | 0.342 |
| E3 Logistic Regression | 0.271 |
| E3 Extra Trees, summary features only | 0.317 |

So the paper's number can be reproduced, but only with their leaky setup, where synthetic SMOTE samples end up in the test set. When the split is done first and the test set has the real theft rate, MCC drops to around 0.3. The notebook also reports PR-AUC, ROC-AUC, precision, recall, FPR and precision in the top 5%. The full discussion is in my Phase 2 report.

## What's in here

- `Phase2_baseline_implementation.ipynb` - the main notebook. I left the outputs in, so you can see the results without running anything.
- `figures/`
  - `fig_compare.png` - MCC for each setup, plus precision-recall curves on the real-ratio test set
  - `fig_confusion.png` - confusion matrices for E1, E2 and E3 Extra Trees
  - `fig_cv.png` - cross-validation vs test PR-AUC for the E3 models
- `p2out/` - the outputs saved by my run of the notebook
  - `results.json` - all the metrics, CV scores, chosen settings and thresholds
  - `comparison.csv` - the comparison table with the paper (all metrics, not just MCC)
  - `fig_compare.png`, `fig_confusion.png`, `fig_cv.png` - the same figures as in `figures/`
  - `p_E2.npy`, `p_E3_ET.npy`, `p_E3_RF.npy`, `p_E3_LR.npy`, `p_E3_ET_summary.npy` - predicted theft probabilities on the test set for each model
  - `y_test.npy` - true labels of the test set
  - `split_test_idx.npy` - row indices of the test customers, so the same split can be checked

  With the predictions and labels saved, the figures and metrics can be recalculated without rerunning the models.

## How to run it

I didn't upload the dataset because it's too big for GitHub. You can get it from Zheng et al.'s repo:
https://github.com/henryRDlab/ElectricityTheftDetection

Download `data.zip`, `data.z01` and `data.z02`, extract them together to get `data.csv`, and put that file in the same folder as the notebook.

Install the libraries. For Phase 2 I used scikit-learn 1.9.1, imbalanced-learn 0.14.2 and numpy 2.5.3:

```
pip install numpy pandas matplotlib scikit-learn imbalanced-learn
```

For Phase 3 see the section below (it has its own requirements.txt).

Then run all the cells from top to bottom. The notebook writes everything listed above into `p2out/` (overwriting what's there), and logs progress to `progress.txt`. I set `SEED = 42` everywhere, so the split and the tuned settings should come out the same each time. The whole run took me about 47 minutes, mostly because of the nested CV for Random Forest (around 26 minutes on its own).

## Phase 3 - alternative model and stress tests

`Phase3_StressTests.ipynb` is the Phase 3 notebook. It uses the same data, the same cleaning and the same 80/20 split (`SEED = 42`) as Phase 2, so the test customers are the same. It does three things:

- trains a HistGradientBoosting model on my corrected features and compares it with Extra Trees (accuracy and training time)
- stress-tests the authors' model and my corrected Extra Trees in four ways: removing 10-25 September 2016, adding noise to the meter readings, flipping 10% or 20% of the theft labels in training, and adding 1,000 or 4,000 useless features
- runs a paired bootstrap (1,000 resamples of the test customers) to check if each change is bigger than chance

Main results (test PR-AUC, a random model gets 0.089):

| Model | PR-AUC | ROC-AUC | Training time |
|---|---|---|---|
| Authors' model (split first, 200 trees) | 0.336 | 0.778 | 389 s |
| Corrected Extra Trees (200 trees) | 0.378 | 0.800 | 260 s |
| HistGradientBoosting (not tuned) | 0.455 | 0.825 | 86 s |

The gain of HistGradientBoosting over Extra Trees has a 95% bootstrap interval of +0.056 to +0.097. The results of all the stress tests, with their intervals, are in `p3out/stress_results.csv`.

`p3out/` has the outputs from my run:

- `results_phase3.json` - all the scores, training times and bootstrap intervals
- `stress_results.csv` - one row per stress test and model, with the change from the baseline and the 95% interval
- `fig_stress.png` - test PR-AUC in each stress test (the figure in my report)
- `test_probs.npz` - the test-set predictions of every model, so the bootstrap can be recalculated
- `progress.txt` - the time log of the run

To run it, put `data.csv` next to the notebook (same as Phase 2), then:

```
pip install -r requirements.txt
```

and run all the cells. I ran it in Google Colab (2 CPU cores, scikit-learn 1.6.1) and it took about 1 hour 40 minutes. Setting `QUICK = True` at the top runs a small version in a few minutes, just to check that the code works.