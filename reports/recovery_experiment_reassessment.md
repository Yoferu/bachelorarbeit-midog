# Reassessment of Pruned60 performance-recovery experiments

## Scope and decision rule

This report re-evaluates existing artifacts only; no training or inference was rerun. The common validation set contains 39 images and is disjoint from training, calibration, and final test data ([split metadata](../experiments/distillation/pruned60_frozen_backbone_recovery/runs/supervised_lr1e-4_seed42/metadata.json)). Unless explicitly marked otherwise, AP is validation AP at the fixed diagnostic threshold 0.584.

Run termination and checkpoint validity are treated separately. A checkpoint is counted as a valid observation when it predates the first recorded numerical-safety failure, exists with a nonzero recorded size, loaded through the model registry, and completed inference (`loading_status=registry`, `inference_status=ok`) in its validation-results CSV. A later failure invalidates the trajectory from the failure onward, not earlier checkpoints. The artifacts do not contain an independent per-checkpoint tensor-finiteness certificate; direct tensor scanning was unavailable in the audit environment. However, successful loading/inference and the absence of an earlier triggered safety event are direct evidence that the pre-failure checkpoints were usable. “Stable-selected” below describes the existing selector; `fallback_isolated_raw_max` is not a demonstrated plateau.

The directly measured Pruned60 start is **0.8665903211 AP** ([step-0 metrics](../experiments/distillation/pruned60_frozen_backbone_recovery/runs/supervised_lr1e-4_seed42/dense_validation_metrics/step_000000_metrics.json)). Every executed frozen run and every dense unfrozen sweep reproduces this value. Thus all recovery deltas use this baseline.

## Checkpoint-level finding for frozen supervised LR=1e-4

The LR=1e-4 run first triggered its numerical criterion at optimizer step 345, when gradient norm 1062.479 exceeded the threshold 1000 ([run status](../experiments/distillation/pruned60_frozen_backbone_recovery/runs/supervised_lr1e-4_seed42/run_status.json)). Its evaluated checkpoints end at step 325. Step 50 was written (62,572,933 bytes), loaded successfully, and evaluated successfully at **0.8715287447 AP** ([validation row](../experiments/distillation/pruned60_frozen_backbone_recovery/runs/supervised_lr1e-4_seed42/dense_validation_results.csv)). It predates the first detected failure by 295 steps and is therefore a **valid experimental observation**. The same reasoning applies to steps 0–325. The run status makes this learning rate unsuitable as a robust configuration; it does not erase its earlier observations.

The true full-grid maximum is actually step 25, **0.8765819669 AP**, not step 50. It is valid by the same artifact criteria, but isolated: step 0 was 0.866590, step 50 was 0.871529, and step 75 was 0.867571 ([selection diagnostics](../experiments/distillation/pruned60_frozen_backbone_recovery/runs/supervised_lr1e-4_seed42/stable_checkpoint_selection.json)).

## Table A – Recovery overview

Here, ΔAP is a unitless AP difference; “percentage points” is ΔAP × 100. Relative improvement is ΔAP/start AP × 100. “Best” means the maximum valid evaluated checkpoint, independent of termination.

| Recovery method | LR | Seed(s) | Start AP | Best valid AP | ΔAP | Δ percentage points | Relative improvement | Best step | Termination | Interpretation |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Unfrozen supervised, dense | 1e-4 | 42 | 0.866590 | 0.866590 | 0 | 0.000 | 0.000% | 0 | completed | No recovery; all updates worse |
| Unfrozen supervised, dense | 3e-5 | 42 | 0.866590 | 0.866590 | 0 | 0.000 | 0.000% | 0 | completed | No recovery; all updates worse |
| Unfrozen supervised, dense | 1e-5 | 42 | 0.866590 | 0.866801 | +0.000210 | +0.021 | +0.024% | 512 | completed | Negligible isolated gain |
| Unfrozen KD, clean | 1e-4 | 42 | 0.866590¹ | 0.799696 | −0.066895 | −6.689 | −7.719% | 512 | completed | Strong degradation |
| Frozen supervised | 1e-6 | 42 | 0.866590 | 0.868681 | +0.002091 | +0.209 | +0.241% | 450 | early stopped | Small gain; stable policy chooses step 200/AP 0.868279 |
| Frozen supervised | 3e-6 | 42 | 0.866590 | 0.869323 | +0.002733 | +0.273 | +0.315% | 75 | completed | Early isolated peak; ends below start |
| Frozen supervised | 1e-5 | 42 | 0.866590 | 0.870138 | +0.003548 | +0.355 | +0.409% | 50 | completed | Early isolated peak; prolonged degradation |
| Frozen supervised | 3e-5 | 42/43/44 | 0.866590 | 0.872421 / 0.870358 / 0.871113 | +0.005830 / +0.003768 / +0.004522 | +0.583 / +0.377 / +0.452 | +0.673% / +0.435% / +0.522% | 25 / 25 / 450 | completed / failed at 397 / completed | Gains in all seeds, but peak timing and robustness vary |
| Frozen supervised | 5e-5 | 42 | 0.866590 | 0.872203 | +0.005613 | +0.561 | +0.648% | 25 | completed | Early isolated peak; marked later degradation |
| Frozen supervised | 1e-4 | 42 | 0.866590 | **0.876582** | **+0.009992** | **+0.999** | **+1.153%** | 25 | failed at 345 | Highest valid observation; isolated and numerically non-robust |
| Frozen KD | 1e-6 | 42 | 0.866590 | 0.867591 | +0.001001 | +0.100 | +0.116% | 200 | early stopped | Small supported plateau; below supervised |
| Frozen KD | 1e-5 | 42 | 0.866590 | 0.870247 | +0.003657 | +0.366 | +0.422% | 100 | completed | Early isolated peak; later degradation |

¹ Reconstructed from the identical starting Pruned60 checkpoint evaluated by the other clean runs; the historical KD CSV begins at epoch 1 and contains no direct step-0 row. Sources: [complete current artifact inventory](../experiments/distillation/pruned60_frozen_backbone_recovery/reports/current_recovery_results_summary.csv), [unfrozen dense trajectories](../experiments/distillation/pruned60_clean_recovery/dense_checkpoint_lr_sweep/dense_checkpoint_results.csv), and [historical clean KD trajectory](../experiments/distillation/runs/fcos_x101_to_fcos18_pruned60_distillation_clean/validation_results.csv).

The earlier non-clean supervised and KD directories contain training checkpoints but no checkpoint-level validation trajectory; their comparison report remains `pending`. They are therefore pilot evidence, not numerically rankable recovery results ([inventory E13](thesis_experiment_inventory.md)).

## Table B – Frozen-backbone supervised LR sweep

| LR | Valid AP trajectory (step: AP) | First improvement | Peak | After peak / endpoint | Existing conservative selection | Numerical/run state |
| ---: | --- | --- | --- | --- | --- | --- |
| 1e-6 | 0:.866590, 50:.868411, 100:.866514, 150:.868418, 200:.868279, 250:.868489, 300:.868607, 350:.867429, 400:.867626, 450:.868681, 500:.867692, 550:.867810, 600:.867691, 650:.867523, 700:.867173, 750:.867248 | step 50 | .868681 @450 | .867248 @750; decreases | .868279 @200, supported plateau | no numerical failure; early stopped |
| 3e-6 | 0:.866590, 25:.866807, 50:.867353, 75:.869323, 100:.869222, 125:.868770, 150:.868467, 175:.867198, 200:.867510, 225:.867261, 250:.866898, 275:.867239, 300:.867049, 325:.864622, 350:.866426, 375:.866331, 400:.866466, 425:.865939, 450:.865986, 475:.866736, 500:.866109, 512:.866051 | step 25 | .869323 @75 | .866051 @512; below start | .869323 @75, isolated fallback | completed |
| 1e-5 | 0:.866590, 25:.869425, 50:.870138, 75:.869150, 100:.869733, 125:.869963, 150:.869519, 175:.869320, 200:.868790, 225:.865680; thereafter oscillates mostly .857–.866 through 2560 | step 25 | .870138 @50 | .859396 @2560; sustained degradation | .870138 @50, isolated fallback | completed |
| 3e-5, seed 42 | 0:.866590, 25:.872421, 50:.871147, 75:.865413, 100:.864687, 125:.866156, 150:.861673, 175:.862473, 200:.862732, 225:.862507, 250:.862759, 275:.866693, 300:.865005, 325:.858413, 350:.858572, 375:.858144, 400:.863212, 425:.865419, 450:.863863, 475:.865921, 500:.861028, 512:.860710 | step 25 | .872421 @25 | .860710 @512; below start | full grid .872421 @25; common-grid .871147 @50; both isolated fallbacks | completed |
| 5e-5 | 0:.866590, 25:.872203, 50:.870484, 75:.869195, 100:.867101, 125:.865844, 150:.864018, 175:.861882, 200:.858688, 225:.857809, 250:.859758, 275:.868135, 300:.862827, 325:.853509, 350:.857675, 375:.860023, 400:.855596, 425:.862752, 450:.864527, 475:.860983, 500:.857530, 512:.858001 | step 25 | .872203 @25 | .858001 @512; below start | .872203 @25, isolated fallback | completed; max recorded gradient 369.485 |
| 1e-4 | 0:.866590, 25:.876582, 50:.871529, 75:.867571, 100:.867629, 125:.862443, 150:.851974, 175:.858850, 200:.853854, 225:.854104, 250:.857068, 275:.868919, 300:.858038, 325:.850585 | step 25 | .876582 @25 | .850585 @325; below start | full grid .876582 @25; common-grid .871529 @50; isolated fallbacks | first safety failure @345; failed |
| 1e-3 | no full-run checkpoints | unavailable | unavailable | unavailable | none | prepared only; one-step smoke is not an LR result |

Every trajectory above is directly reconstructed from each run’s `dense_validation_results.csv`; the paths are indexed in the [artifact inventory](../experiments/distillation/pruned60_frozen_backbone_recovery/reports/current_recovery_results_summary.csv). The longer 1e-5 row is abbreviated after step 225 for readability; its complete 108-row trajectory is in the [source CSV](../experiments/distillation/pruned60_frozen_backbone_recovery/runs/supervised_lr1e-5_seed42_20260725T212156Z/dense_validation_results.csv). There is no observed basis for claiming completed runs would later become unstable.

## Optimization behaviour versus robust selection

**Optimization behaviour.** Frozen training can make small, rapid adjustments: the global valid maximum is 0.876582 at LR=1e-4 step 25 (+0.009992 AP, +0.999 percentage points). AP then drops by 0.005053 by step 50 and is below the start from step 125 onward except for a small rebound at step 275. The other moderate/high learning rates also peak at step 25 (3e-5 seed 42 and 5e-5), whereas 3e-6 peaks at step 75 and 1e-6 at step 450. This is direct evidence of early peaks and little useful optimization headroom after them. “Small local adjustment followed by degradation” is supported; overfitting is only a possible interpretation because no training-versus-validation generalization analysis isolates its mechanism.

**Robust checkpoint selection.** Only supervised 1e-6 and KD 1e-6 meet the existing three-checkpoint plateau rule. Their selected APs are 0.868279 at step 200 and 0.867514 at step 150, respectively ([supervised selection](../experiments/distillation/pruned60_frozen_backbone_recovery/runs/supervised_lr1e-6_seed42/stable_checkpoint_selection.json), [KD selection](../experiments/distillation/pruned60_frozen_backbone_recovery/runs/distillation_lr1e-6_seed42/stable_checkpoint_selection.json)). All stronger maxima are selector fallbacks to isolated peaks. If “conservative” permits the project’s fallback policy, the best completed configuration is seed-42 LR=3e-5: common-grid 0.871147 at step 50. If it requires an actual supported plateau, LR=1e-6 step 200 is the defensible choice. LR=1e-4 is optimization evidence, not a robust training recommendation.

## Table C – Stability and reproducibility

| Configuration | Seeds | Mean best AP | AP range / sample SD | Peak-step variation | Numerical failures | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Frozen supervised 3e-5, all valid pre-failure observations | 3 | **0.871297** | 0.870358–0.872421 / 0.001044 | 25–450 | 1/3 (seed 43 at 397) | All seeds improve, but magnitude and peak time vary; moderately repeatable AP gain, imperfect numerical robustness |
| Frozen supervised 3e-5, completed runs only (legacy report population) | 2 | 0.871767¹ | 0.871113–0.872421 / 0.000925¹ | 25–450 | excluded by definition | Selection-biased; not preferred for inference |

¹ Newly calculated from full-grid maxima. The existing seed report instead computes completed-seed **common-grid** mean 0.871130 and excludes seed 43 entirely ([legacy reproducibility report](../experiments/distillation/pruned60_frozen_backbone_recovery/reports/supervised_lr3e-5_seed_reproducibility.md)). Across all three seeds, gains are +0.005830, +0.003768, and +0.004522 AP; mean gain is +0.004707 AP (+0.471 percentage points). Therefore the configuration produces consistently small positive maxima, but it is not fully stable: one trajectory fails numerically and seed 44 peaks much later.

## Table D – Recovery relative to pruning loss

| Model | AP | Loss vs FCOS_18 | Recovery vs Pruned60 | Fraction of lost AP recovered |
| --- | ---: | ---: | ---: | ---: |
| FCOS_18 on the clean 39-image recovery validation split | unavailable | — | — | — |
| Pruned60 step 0 | 0.866590 | unavailable | 0 | unavailable |
| Best valid recovery (LR=1e-4 step 25) | 0.876582 | unavailable | +0.009992 | unavailable |
| Conservative completed candidate (LR=3e-5 seed 42, common step 50) | 0.871147 | unavailable | +0.004557 | unavailable |

A scientifically valid recovery fraction cannot be computed: no FCOS_18 evaluation on the same 39-image validation split was found. The older quick-set quality artifacts report FCOS_18 0.869288 and Pruned60 0.865398 ([FCOS_18](../experiments/distillation/recovery_quality/fcos18/quality_metrics.json), [Pruned60](../experiments/distillation/recovery_quality/pruned60_untuned/quality_metrics.json)), but combining that different dataset/evaluation with recovery-validation AP would be invalid and can even imply recovery above 100%. This is missing evidence, not evidence of full recovery.

## Audit of existing reports and scripts

1. The current summary correctly retains the LR=1e-4 full-grid maximum and explicitly calls it the “highest observed AP”; it does **not** wholly erase the failed run ([current summary](../experiments/distillation/pruned60_frozen_backbone_recovery/reports/current_recovery_results_summary.md)). Its candidate language nevertheless privileges “completed controlled” runs for deployment. That is appropriate for robustness, but must not be reused as the optimization ranking.
2. The LR comparison’s prose ranks common-grid AP, hiding the higher valid step-25 observations for 3e-5, 5e-5, and 1e-4 even though its CSV retains them ([comparison CSV](../experiments/distillation/pruned60_frozen_backbone_recovery/reports/supervised_learning_rate_comparison.csv)). Common-grid ranking answers comparability, not “highest AP observed.”
3. The seed-report generator defines a seed as comparable only when status is exactly `completed` and final step is 512 ([generator](../experiments/distillation/pruned60_frozen_backbone_recovery/scripts/generate_supervised_lr3e5_reproducibility_report.py)). Consequently it excludes seed 43’s 16 successfully evaluated pre-failure checkpoints from aggregate AP statistics. The report’s conclusion “incomplete” is fair for full-trajectory robustness, but its two-seed variance is not the variance of valid checkpoint maxima across all attempted seeds.
4. Stable selection labels an isolated raw maximum as `fallback_isolated_raw_max`; calling its output “stable AP” without the status is misleading. In particular LR=1e-4 step 25 is valid but not plateau-supported.
5. Historical clean unfrozen completion did not imply successful recovery: two learning rates select step 0 and the third gains only 0.000210 AP. Conversely, numerical failure did not make LR=1e-4’s earlier measurements invalid. These examples demonstrate why run status and checkpoint evidence must remain separate.

## Thesis-ready Results interpretation

Supervised recovery did improve Pruned60 at some frozen-backbone checkpoints. The highest valid observation was 0.876582 AP at LR=1e-4 after 25 optimizer steps, an increase of 0.009992 AP, equivalent to 0.999 percentage points or 1.153% relative to the 0.866590 starting AP. This peak was isolated and was followed by degradation. Under a more conservative completed-run/common-grid policy, LR=3e-5 seed 42 reached 0.871147 at step 50, a gain of 0.004557 AP (0.456 percentage points; 0.526% relative). Under the stricter requirement of a supported local plateau, LR=1e-6 selected only 0.868279 (+0.001689 AP; +0.169 percentage points).

Frozen-backbone training was more effective than the earlier unfrozen sweep, in which the best two configurations retained step 0 and the third improved by only 0.000210 AP. Knowledge distillation did not show a practically relevant advantage over matched supervised recovery: at LR=1e-5 its best AP exceeded supervised by only 0.000109, while at LR=1e-6 it was worse.

Learning rate affected both early gain and robustness. Larger rates generally produced larger early peaks, but also stronger subsequent degradation; LR=1e-4 eventually crossed the numerical-safety threshold at step 345. That failure means the full LR=1e-4 trajectory was not robust and the configuration should not be recommended for deployment. It does **not** invalidate step 25 or step 50, both of which were created and successfully evaluated before the first detected failure.

At LR=3e-5 all three seeds achieved a positive valid maximum, with a mean gain of 0.004707 AP (0.471 percentage points), but best AP ranged from 0.870358 to 0.872421, peak steps ranged from 25 to 450, and one of three runs later failed numerically. The appropriate characterization is therefore **moderately repeatable small gains with imperfect trajectory stability**, not robust convergence and not complete irreproducibility.

The data support the distinction between optimization success and practical recovery success. Optimization briefly increased validation AP, but the robust gains were well below one percentage point and generally appeared as early isolated peaks. Because the unpruned FCOS_18 was not evaluated on the same recovery-validation split, the fraction of pruning-induced AP loss recovered cannot be estimated without mixing incomparable artifacts. On the available evidence, recovery was **technically positive but practically negligible/unsuccessful as restoration of pruned accuracy**; this conclusion should remain qualified by the missing matched FCOS_18 baseline and the use of one repeatedly inspected validation split.

## Compact terminal summary

```text
Pruned60 starting AP:                 0.8665903211
Best valid AP (all recovery runs):   0.8765819669 (frozen supervised, LR=1e-4, seed 42, step 25)
Absolute AP gain:                    +0.0099916458
Percentage-point gain:               +0.9991645813 pp
Best robust completed configuration: frozen supervised LR=3e-5, seed 42, common-grid step 50, AP 0.8711468577
LR=1e-4 step 50 valid?:              Yes; AP 0.8715287447, successful evaluation, first safety failure at step 345
Seed variability (LR=3e-5):          mean best AP 0.8712971; range 0.8703579–0.8724206; sample SD 0.0010437; peaks 25–450
Fraction of pruning loss recovered:  unavailable (no matched FCOS_18 AP on the 39-image recovery validation split)
Practical assessment:                technically positive optimization, but negligible/non-robust practical recovery
```
