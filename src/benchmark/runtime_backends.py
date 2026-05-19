from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


RuntimeBackendName = Literal[
    "pytorch_eager",
    "pytorch_compile",
    "onnxruntime_cpu",
    "openvino_cpu",
]


@dataclass
class RuntimeBackendResult:
    model: Any
    runtime_backend: RuntimeBackendName
    effective_runtime_backend: str
    compile_backend: str | None = None
    compile_mode: str | None = None
    onnx_opset: int | None = None
    openvino_device: str | None = None
    export_path: Path | None = None
    export_reused: bool | None = None
    runtime_warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "runtime_backend": self.runtime_backend,
            "effective_runtime_backend": self.effective_runtime_backend,
            "compile_backend": self.compile_backend,
            "compile_mode": self.compile_mode,
            "onnx_opset": self.onnx_opset,
            "openvino_device": self.openvino_device,
            "export_path": str(self.export_path) if self.export_path else None,
            "export_reused": self.export_reused,
            "runtime_warnings": self.runtime_warnings,
        }


class RuntimeBackendUnavailable(RuntimeError):
    pass


def configure_runtime_backend(
    *,
    model: Any,
    runtime_backend: RuntimeBackendName,
    compile_backend: str,
    compile_mode: str | None,
    onnx_opset: int,
    openvino_device: str,
    export_dir: Path,
    model_name: str,
) -> RuntimeBackendResult:
    if runtime_backend == "pytorch_eager":
        return RuntimeBackendResult(
            model=model,
            runtime_backend=runtime_backend,
            effective_runtime_backend="pytorch_eager",
        )

    if runtime_backend == "pytorch_compile":
        return _configure_torch_compile(
            model=model,
            compile_backend=compile_backend,
            compile_mode=compile_mode,
            requested_backend=runtime_backend,
        )

    if runtime_backend == "onnxruntime_cpu":
        export_path = export_dir / f"{model_name}_opset{onnx_opset}.onnx"
        raise RuntimeBackendUnavailable(
            "onnxruntime_cpu is a documented placeholder in this repository state. "
            "The current environment does not provide onnx/onnxruntime, and the FCOS "
            "detection model includes output structures and postprocessing that need a "
            "dedicated forward-pass export/validation implementation before results are valid. "
            f"Planned export path: {export_path}"
        )

    if runtime_backend == "openvino_cpu":
        export_path = export_dir / f"{model_name}_openvino"
        raise RuntimeBackendUnavailable(
            "openvino_cpu is a documented placeholder in this repository state. "
            "OpenVINO is not installed here, and OpenVINO export should be implemented after "
            "a validated ONNX forward-pass export exists. "
            f"Planned export path: {export_path}"
        )

    raise RuntimeBackendUnavailable(f"Unknown runtime backend: {runtime_backend}")


def _configure_torch_compile(
    *,
    model: Any,
    compile_backend: str,
    compile_mode: str | None,
    requested_backend: RuntimeBackendName,
) -> RuntimeBackendResult:
    warnings = []
    try:
        import torch
    except Exception as exc:
        warning = f"Could not import torch for torch.compile: {exc}. Falling back to pytorch_eager."
        print(f"Runtime warning: {warning}")
        return RuntimeBackendResult(
            model=model,
            runtime_backend=requested_backend,
            effective_runtime_backend="pytorch_eager",
            compile_backend=compile_backend,
            compile_mode=compile_mode,
            runtime_warnings=[warning],
        )

    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        warning = "torch.compile is unavailable in this PyTorch version. Falling back to pytorch_eager."
        print(f"Runtime warning: {warning}")
        return RuntimeBackendResult(
            model=model,
            runtime_backend=requested_backend,
            effective_runtime_backend="pytorch_eager",
            compile_backend=compile_backend,
            compile_mode=compile_mode,
            runtime_warnings=[warning],
        )

    try:
        kwargs = {"backend": compile_backend}
        if compile_mode:
            kwargs["mode"] = compile_mode
        compiled_model = compile_fn(model, **kwargs)
    except Exception as exc:
        warning = f"torch.compile failed during setup: {exc}. Falling back to pytorch_eager."
        print(f"Runtime warning: {warning}")
        return RuntimeBackendResult(
            model=model,
            runtime_backend=requested_backend,
            effective_runtime_backend="pytorch_eager",
            compile_backend=compile_backend,
            compile_mode=compile_mode,
            runtime_warnings=[warning],
        )

    wrapped_model = _TorchCompileFallbackWrapper(
        eager_model=model,
        compiled_model=compiled_model,
        runtime_warnings=warnings,
    )

    return RuntimeBackendResult(
        model=wrapped_model,
        runtime_backend=requested_backend,
        effective_runtime_backend="pytorch_compile",
        compile_backend=compile_backend,
        compile_mode=compile_mode,
        runtime_warnings=warnings,
    )


class _TorchCompileFallbackWrapper:
    def __init__(self, *, eager_model: Any, compiled_model: Any, runtime_warnings: list[str]) -> None:
        self.eager_model = eager_model
        self.compiled_model = compiled_model
        self.runtime_warnings = runtime_warnings
        self.use_eager = False

    def eval(self) -> Any:
        self.eager_model.eval()
        self.compiled_model.eval()
        return self

    def to(self, *args: Any, **kwargs: Any) -> Any:
        self.eager_model.to(*args, **kwargs)
        self.compiled_model.to(*args, **kwargs)
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.use_eager:
            return self.eager_model(*args, **kwargs)
        try:
            return self.compiled_model(*args, **kwargs)
        except Exception as exc:
            warning = f"torch.compile failed during forward pass: {exc}. Falling back to pytorch_eager."
            print(f"Runtime warning: {warning}")
            self.runtime_warnings.append(warning)
            self.use_eager = True
            return self.eager_model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.eager_model, name)
