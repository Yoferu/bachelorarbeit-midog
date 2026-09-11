from __future__ import annotations

from experiments.distillation.pruned60_frozen_backbone_recovery.scripts.generate_comparison_report import recommendations


def test_report_recommends_retaining_1e5_when_1e6_does_not_exceed_it() -> None:
    rows = [
        {
            "Method": "supervised",
            "Backbone frozen": "True",
            "Learning rate": "1e-05",
            "Best AP": 0.8701,
            "Stable-selected AP": 0.8701,
            "Best step": 50,
        },
        {
            "Method": "distillation",
            "Backbone frozen": "True",
            "Learning rate": "1e-05",
            "Best AP": 0.8702,
            "Stable-selected AP": 0.8702,
            "Best step": 100,
        },
        {
            "Method": "supervised",
            "Backbone frozen": "True",
            "Learning rate": "1e-06",
            "Best AP": 0.8699,
            "Stable-selected AP": 0.8699,
            "Best step": 512,
            "Best checkpoint timing": "during epoch 2",
            "Improving near end": "False",
        },
        {
            "Method": "distillation",
            "Backbone frozen": "True",
            "Learning rate": "1e-06",
            "Best AP": 0.8700,
            "Stable-selected AP": 0.8700,
            "Best step": 512,
            "Best checkpoint timing": "during epoch 2",
            "Improving near end": "False",
        },
    ]

    text = "\n".join(recommendations(rows))
    assert "retain the best supported 1e-5 checkpoint" in text
    assert "practically equivalent" in text
    assert "does not currently justify" in text
