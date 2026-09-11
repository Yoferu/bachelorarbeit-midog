# Pruning and recovery

The canonical **Pruned60** artifact is
`experiments/pruning/models/FCOS_18_depgraph_fpn_head_60pct.pt`, created by
`create_depgraph_structural_pruned_model.py` with the corresponding 60% YAML.
It physically removes channels from FPN/internal detection-head convolutions;
60% is the configured pruning ratio, not a 60% reduction of the entire model.
The backbone and fixed output predictors are excluded as pruning roots.

See [the experiment map](experiments.md) for creation and evaluation commands.
`run_depgraph_structural_sweep.py` and `run_fpn_head_threshold_sweep.py` retain the
scope/ratio selection and diagnostic test-subset threshold experiments. Their
quick results do not establish held-out final accuracy or calibration.

The older `create_structured_pruned_model.py` / `finetune_pruned_model.py` pair
implements masked 10% head pruning and its recovery pilot. Tensor shapes remain
unchanged, so those artifacts cannot substantiate structural speedup claims.
They remain for the methodology recorded in the historical pruning summaries.

Supervised and distillation recovery of physical Pruned60 use
`train_fcos_distillation.py`. Clean and frozen-backbone configurations are separate
from the masked-pruning fine-tuning configs. `student.freeze` freezes the whole
student; `training.freeze_backbone` freezes its backbone only. Preserve the
configured seeds, learning rates, validation intervals, and selection policy.

August quantization's `pruned_recovered` label refers to a selected step-0
checkpoint. Later CPU scripts point to `student_stable_selected.pt` from frozen
recovery. Consult checkpoint hashes and selection records before using these as
interchangeable identities. The older `final_pruning_summary.md` describes the
state before the later full evaluations; use the September CPU report for the
corrected performance comparison.
