import json

import numpy as np
import torch

from src.benchmark.midog_guide_adapter import _jsonable


def test_jsonable_handles_nested_torch_and_numpy_values():
    payload = {
        "tensor": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "scalar_tensor": torch.tensor(7),
        "array": np.array([[1, 2], [3, 4]], dtype=np.int64),
        "numpy_scalar": np.float32(0.75),
        "nested": [
            {
                "scores": np.array([0.1, 0.9], dtype=np.float32),
                "keep": np.bool_(True),
                "count": np.int64(2),
            },
            (torch.tensor(1.5), np.array(3.5)),
        ],
    }

    converted = _jsonable(payload)

    assert converted["tensor"] == [[1.0, 2.0], [3.0, 4.0]]
    assert converted["scalar_tensor"] == 7
    assert converted["array"] == [[1, 2], [3, 4]]
    assert isinstance(converted["numpy_scalar"], float)
    assert converted["nested"][0]["scores"] == [np.float32(0.1).item(), np.float32(0.9).item()]
    assert converted["nested"][0]["keep"] is True
    assert converted["nested"][0]["count"] == 2
    assert converted["nested"][1] == [1.5, 3.5]

    json.dumps(converted)


def test_jsonable_preserves_plain_python_values():
    payload = {"text": "case", "none": None, "flag": False, "number": 3}

    assert _jsonable(payload) == payload
