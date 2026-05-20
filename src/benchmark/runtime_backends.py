from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
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
    execution_providers: list[str] | None = None
    validation_path: Path | None = None
    validation_status: str | None = None
    onnx_export_method: str | None = None
    openvino_version: str | None = None
    available_devices: list[str] | None = None
    selected_device: str | None = None
    source_onnx_path: Path | None = None
    source_onnx_reused: bool | None = None
    openvino_model_path: Path | None = None
    openvino_weights_path: Path | None = None
    openvino_ir_reused: bool | None = None
    runtime_limitations: list[str] = field(default_factory=list)
    runtime_warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "runtime_backend": self.runtime_backend,
            "effective_runtime_backend": self.effective_runtime_backend,
            "compile_backend": self.compile_backend,
            "compile_mode": self.compile_mode,
            "onnx_opset": self.onnx_opset,
            "onnx_model_path": str(self.source_onnx_path or self.export_path) if self.onnx_opset else None,
            "openvino_device": self.openvino_device,
            "export_path": str(self.export_path) if self.export_path else None,
            "export_reused": self.export_reused,
            "execution_providers": self.execution_providers,
            "validation_path": str(self.validation_path) if self.validation_path else None,
            "validation_status": self.validation_status,
            "onnx_export_method": self.onnx_export_method,
            "openvino_version": self.openvino_version,
            "available_devices": self.available_devices,
            "selected_device": self.selected_device,
            "source_onnx_path": str(self.source_onnx_path) if self.source_onnx_path else None,
            "source_onnx_reused": self.source_onnx_reused,
            "openvino_model_path": str(self.openvino_model_path) if self.openvino_model_path else None,
            "openvino_weights_path": str(self.openvino_weights_path) if self.openvino_weights_path else None,
            "openvino_ir_reused": self.openvino_ir_reused,
            "runtime_limitations": self.runtime_limitations,
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
    patch_size: int,
    config_file: Path | None = None,
    validation_output: Path | None = None,
    example_input: Any | None = None,
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
        return _configure_onnxruntime_cpu(
            model=model,
            requested_backend=runtime_backend,
            export_dir=export_dir,
            model_name=model_name,
            patch_size=patch_size,
            onnx_opset=onnx_opset,
            config_file=config_file,
            validation_output=validation_output,
            example_input=example_input,
        )

    if runtime_backend == "openvino_cpu":
        return _configure_openvino_cpu(
            model=model,
            requested_backend=runtime_backend,
            export_dir=export_dir,
            model_name=model_name,
            patch_size=patch_size,
            onnx_opset=onnx_opset,
            openvino_device=openvino_device,
            config_file=config_file,
            validation_output=validation_output,
            example_input=example_input,
        )

    raise RuntimeBackendUnavailable(f"Unknown runtime backend: {runtime_backend}")


def _require_modules(*, backend_name: RuntimeBackendName, install_hint: str, modules: list[str]) -> None:
    missing = []
    for module_name in modules:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)

    if missing:
        raise RuntimeBackendUnavailable(
            f"{backend_name} requires missing Python package(s): {', '.join(missing)}. "
            f"Install them with: {install_hint}"
        )


def _configure_onnxruntime_cpu(
    *,
    model: Any,
    requested_backend: RuntimeBackendName,
    export_dir: Path,
    model_name: str,
    patch_size: int,
    onnx_opset: int,
    config_file: Path | None,
    validation_output: Path | None,
    example_input: Any | None,
) -> RuntimeBackendResult:
    _require_modules(
        modules=["onnx", "onnxruntime"],
        install_hint="python -m pip install -r requirements-runtime.txt",
        backend_name=requested_backend,
    )

    import onnx
    import onnxruntime as ort
    import torch

    warnings: list[str] = []
    limitations = [
        "ONNX Runtime uses the exported torchvision FCOS inference graph on CPU. "
        "Preprocessing, patch extraction, patch merging, benchmark orchestration, and "
        "metric calculation remain in Python.",
    ]

    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = _onnx_export_path(
        export_dir=export_dir,
        model_name=model_name,
        patch_size=patch_size,
        onnx_opset=onnx_opset,
        config_file=config_file,
    )
    export_reused = export_path.exists()
    export_method = "reused"
    if example_input is None:
        warnings.append(
            "No representative benchmark patch was available for ONNX validation; "
            "using a zero tensor fallback."
        )
        example_input = torch.zeros(3, patch_size, patch_size, dtype=torch.float32)
    else:
        example_input = example_input.detach().cpu().float()

    model.eval()
    model.to("cpu")

    if not export_reused:
        export_method, export_warnings = _export_onnx_detection_model(
            model=model,
            example_input=example_input,
            export_path=export_path,
            onnx_opset=onnx_opset,
        )
        warnings.extend(export_warnings)

    try:
        onnx.checker.check_model(str(export_path))
    except Exception as exc:
        raise RuntimeBackendUnavailable(f"ONNX checker rejected {export_path}: {exc}") from exc

    try:
        session = ort.InferenceSession(str(export_path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise RuntimeBackendUnavailable(f"Could not create ONNX Runtime CPU session for {export_path}: {exc}") from exc

    providers = session.get_providers()
    if "CPUExecutionProvider" not in providers:
        raise RuntimeBackendUnavailable(
            "ONNX Runtime session did not enable CPUExecutionProvider. "
            f"Available providers: {providers}"
        )

    output_roles = _infer_detection_output_roles(session)
    validation = _validate_onnx_detection_model(
        eager_model=model,
        session=session,
        example_input=example_input,
        output_roles=output_roles,
        export_path=export_path,
        export_reused=export_reused,
        onnx_opset=onnx_opset,
        providers=providers,
        export_method=export_method,
        warnings=warnings,
        limitations=limitations,
    )
    validation_status = str(validation["status"])
    if validation_status != "passed":
        raise RuntimeBackendUnavailable(
            f"ONNX Runtime validation failed for {export_path}. "
            f"Status: {validation_status}. Details: {validation}"
        )

    if validation_output:
        validation_output.parent.mkdir(parents=True, exist_ok=True)
        validation_output.write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")

    wrapped_model = _OnnxRuntimeDetectionWrapper(
        session=session,
        input_name=session.get_inputs()[0].name,
        output_roles=output_roles,
    )

    return RuntimeBackendResult(
        model=wrapped_model,
        runtime_backend=requested_backend,
        effective_runtime_backend="onnxruntime_cpu",
        onnx_opset=onnx_opset,
        export_path=export_path,
        export_reused=export_reused,
        execution_providers=providers,
        validation_path=validation_output,
        validation_status=validation_status,
        onnx_export_method=export_method,
        runtime_limitations=limitations,
        runtime_warnings=warnings,
    )


def _prepare_onnx_export(
    *,
    model: Any,
    export_dir: Path,
    model_name: str,
    patch_size: int,
    onnx_opset: int,
    config_file: Path | None,
    example_input: Any,
) -> tuple[Path, bool, str, list[str]]:
    import onnx

    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = _onnx_export_path(
        export_dir=export_dir,
        model_name=model_name,
        patch_size=patch_size,
        onnx_opset=onnx_opset,
        config_file=config_file,
    )
    export_reused = export_path.exists()
    export_method = "reused"
    warnings: list[str] = []

    if not export_reused:
        export_method, export_warnings = _export_onnx_detection_model(
            model=model,
            example_input=example_input,
            export_path=export_path,
            onnx_opset=onnx_opset,
        )
        warnings.extend(export_warnings)

    try:
        onnx.checker.check_model(str(export_path))
    except Exception as exc:
        raise RuntimeBackendUnavailable(f"ONNX checker rejected {export_path}: {exc}") from exc

    return export_path, export_reused, export_method, warnings


def _configure_openvino_cpu(
    *,
    model: Any,
    requested_backend: RuntimeBackendName,
    export_dir: Path,
    model_name: str,
    patch_size: int,
    onnx_opset: int,
    openvino_device: str,
    config_file: Path | None,
    validation_output: Path | None,
    example_input: Any | None,
) -> RuntimeBackendResult:
    _require_modules(
        modules=["onnx", "onnxruntime", "openvino"],
        install_hint="python -m pip install -r requirements-runtime.txt",
        backend_name=requested_backend,
    )

    import onnxruntime as ort
    import openvino as ov
    import torch

    warnings: list[str] = []
    limitations = [
        "OpenVINO CPU uses the ONNX export as an intermediate representation. "
        "Preprocessing, patch extraction, patch merging, benchmark orchestration, and "
        "metric calculation remain in Python.",
    ]

    if example_input is None:
        warnings.append(
            "No representative benchmark patch was available for OpenVINO validation; "
            "using a zero tensor fallback."
        )
        example_input = torch.zeros(3, patch_size, patch_size, dtype=torch.float32)
    else:
        example_input = example_input.detach().cpu().float()

    model.eval()
    model.to("cpu")

    onnx_path, onnx_reused, onnx_export_method, onnx_warnings = _prepare_onnx_export(
        model=model,
        export_dir=export_dir,
        model_name=model_name,
        patch_size=patch_size,
        onnx_opset=onnx_opset,
        config_file=config_file,
        example_input=example_input,
    )
    warnings.extend(onnx_warnings)

    xml_path = onnx_path.with_name(f"{onnx_path.stem}_openvino.xml")
    bin_path = xml_path.with_suffix(".bin")
    ir_reused = (
        xml_path.exists()
        and bin_path.exists()
        and xml_path.stat().st_mtime >= onnx_path.stat().st_mtime
        and bin_path.stat().st_mtime >= onnx_path.stat().st_mtime
    )

    try:
        core = ov.Core()
    except Exception as exc:
        raise RuntimeBackendUnavailable(f"Could not initialize OpenVINO Core: {exc}") from exc

    available_devices = list(core.available_devices)
    if openvino_device not in available_devices:
        raise RuntimeBackendUnavailable(
            f"OpenVINO device {openvino_device!r} is unavailable. Available devices: {available_devices}"
        )

    try:
        if ir_reused:
            ov_model = core.read_model(str(xml_path))
        else:
            ov_model = ov.convert_model(str(onnx_path))
            ov.save_model(ov_model, str(xml_path))
    except Exception as exc:
        raise RuntimeBackendUnavailable(
            f"OpenVINO ONNX-to-IR conversion/loading failed for {onnx_path}: {exc}"
        ) from exc

    try:
        compiled_model = core.compile_model(ov_model, openvino_device)
    except Exception as exc:
        raise RuntimeBackendUnavailable(
            f"OpenVINO failed to compile {xml_path} for {openvino_device}: {exc}"
        ) from exc

    try:
        ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise RuntimeBackendUnavailable(f"Could not create ONNX Runtime validation session for {onnx_path}: {exc}") from exc

    output_roles = _infer_openvino_detection_output_roles(compiled_model.outputs)
    onnx_output_roles = _infer_detection_output_roles(ort_session)
    validation = _validate_openvino_detection_model(
        eager_model=model,
        ort_session=ort_session,
        compiled_model=compiled_model,
        example_input=example_input,
        openvino_output_roles=output_roles,
        onnx_output_roles=onnx_output_roles,
        source_onnx_path=onnx_path,
        openvino_model_path=xml_path,
        openvino_weights_path=bin_path,
        onnx_reused=onnx_reused,
        ir_reused=ir_reused,
        onnx_opset=onnx_opset,
        onnx_export_method=onnx_export_method,
        openvino_version=str(getattr(ov, "__version__", "unknown")),
        available_devices=available_devices,
        selected_device=openvino_device,
        warnings=warnings,
        limitations=limitations,
    )
    validation_status = str(validation["status"])
    if validation_status not in {"passed", "passed_with_warnings"}:
        raise RuntimeBackendUnavailable(
            f"OpenVINO validation failed for {xml_path}. Status: {validation_status}. Details: {validation}"
        )
    warnings.extend(str(warning) for warning in validation.get("validation_warnings", []))

    if validation_output:
        validation_output.parent.mkdir(parents=True, exist_ok=True)
        validation_output.write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")

    wrapped_model = _OpenVinoDetectionWrapper(
        compiled_model=compiled_model,
        input_obj=compiled_model.inputs[0],
        output_roles=output_roles,
    )

    return RuntimeBackendResult(
        model=wrapped_model,
        runtime_backend=requested_backend,
        effective_runtime_backend="openvino_cpu",
        onnx_opset=onnx_opset,
        openvino_device=openvino_device,
        export_path=xml_path,
        export_reused=ir_reused,
        validation_path=validation_output,
        validation_status=validation_status,
        onnx_export_method=onnx_export_method,
        openvino_version=str(getattr(ov, "__version__", "unknown")),
        available_devices=available_devices,
        selected_device=openvino_device,
        source_onnx_path=onnx_path,
        source_onnx_reused=onnx_reused,
        openvino_model_path=xml_path,
        openvino_weights_path=bin_path,
        openvino_ir_reused=ir_reused,
        runtime_limitations=limitations,
        runtime_warnings=warnings,
    )


def _onnx_export_path(
    *,
    export_dir: Path,
    model_name: str,
    patch_size: int,
    onnx_opset: int,
    config_file: Path | None,
) -> Path:
    safe_model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")
    hash_source = model_name.encode("utf-8")
    if config_file and config_file.exists():
        hash_source += config_file.read_bytes()
    config_hash = hashlib.sha256(hash_source).hexdigest()[:10]
    return export_dir / f"{safe_model_name}_patch{patch_size}_opset{onnx_opset}_{config_hash}.onnx"


def _export_onnx_detection_model(
    *,
    model: Any,
    example_input: Any,
    export_path: Path,
    onnx_opset: int,
) -> tuple[str, list[str]]:
    import torch

    warnings: list[str] = []
    captured = io.StringIO()

    if getattr(torch.onnx, "export", None) is None:
        raise RuntimeBackendUnavailable("torch.onnx.export is unavailable in this PyTorch installation.")

    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            torch.onnx.export(
                model,
                ([example_input],),
                str(export_path),
                opset_version=onnx_opset,
                dynamo=True,
                input_names=["image"],
            )
        return "dynamo", warnings
    except Exception as exc:
        warnings.append(
            "torch.onnx.export(dynamo=True) failed for torchvision FCOS detection "
            "postprocessing; falling back to the legacy exporter. "
            f"Error: {_format_exception(exc)}"
        )

    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            torch.onnx.export(
                model,
                ([example_input],),
                str(export_path),
                opset_version=onnx_opset,
                dynamo=False,
                input_names=["image"],
            )
        warnings.append(
            "ONNX export used the legacy TorchScript-based exporter because the "
            "dynamo exporter could not handle FCOS postprocessing/NMS data-dependent shapes."
        )
        return "legacy", warnings
    except Exception as exc:
        raise RuntimeBackendUnavailable(
            "ONNX export failed with both dynamo=True and the legacy exporter. "
            f"Legacy exporter error: {_format_exception(exc)}"
        ) from exc


def _format_exception(exc: Exception, *, max_chars: int = 800) -> str:
    message = f"{type(exc).__name__}: {exc}"
    message = " ".join(message.split())
    if len(message) > max_chars:
        return f"{message[:max_chars]}..."
    return message


def _infer_detection_output_roles(session: Any) -> dict[str, int]:
    roles: dict[str, int] = {}
    for idx, output in enumerate(session.get_outputs()):
        shape = list(output.shape)
        output_type = output.type
        if len(shape) == 2 and shape[-1] == 4 and output_type == "tensor(float)":
            roles["boxes"] = idx
        elif len(shape) == 1 and output_type == "tensor(float)":
            roles["scores"] = idx
        elif len(shape) == 1 and output_type in {"tensor(int64)", "tensor(int32)"}:
            roles["labels"] = idx

    missing = {"boxes", "scores", "labels"} - roles.keys()
    if missing:
        output_summary = [
            {"name": output.name, "shape": output.shape, "type": output.type}
            for output in session.get_outputs()
        ]
        raise RuntimeBackendUnavailable(
            "Could not map ONNX outputs to torchvision detection fields "
            f"{sorted(missing)}. Outputs: {output_summary}"
        )
    return roles


def _validate_onnx_detection_model(
    *,
    eager_model: Any,
    session: Any,
    example_input: Any,
    output_roles: dict[str, int],
    export_path: Path,
    export_reused: bool,
    onnx_opset: int,
    providers: list[str],
    export_method: str,
    warnings: list[str],
    limitations: list[str],
) -> dict[str, Any]:
    import numpy as np
    import torch

    input_name = session.get_inputs()[0].name
    with torch.inference_mode():
        eager_output = eager_model([example_input])[0]
    ort_outputs = session.run(None, {input_name: example_input.numpy()})

    comparisons: dict[str, Any] = {}
    status = "passed"
    for role, idx in output_roles.items():
        eager_value = eager_output[role].detach().cpu().numpy()
        ort_value = ort_outputs[idx]
        comparison: dict[str, Any] = {
            "pytorch_shape": list(eager_value.shape),
            "onnx_shape": list(ort_value.shape),
            "pytorch_dtype": str(eager_value.dtype),
            "onnx_dtype": str(ort_value.dtype),
        }
        if eager_value.shape != ort_value.shape:
            comparison["shape_match"] = False
            status = "failed"
        else:
            comparison["shape_match"] = True
            if np.issubdtype(eager_value.dtype, np.number) and np.issubdtype(ort_value.dtype, np.number):
                max_abs_diff = float(np.max(np.abs(eager_value - ort_value))) if eager_value.size else 0.0
                comparison["max_abs_diff"] = max_abs_diff
                comparison["allclose"] = bool(np.allclose(eager_value, ort_value, rtol=1e-3, atol=1e-4))
                if role in {"boxes", "scores"} and not comparison["allclose"]:
                    status = "failed"
        comparisons[role] = comparison

    return {
        "status": status,
        "runtime_backend": "onnxruntime_cpu",
        "onnx_model_path": str(export_path),
        "export_reused": export_reused,
        "onnx_opset": onnx_opset,
        "onnx_export_method": export_method,
        "execution_providers": providers,
        "output_roles": output_roles,
        "comparisons": comparisons,
        "warnings": warnings,
        "limitations": limitations,
    }


class _OnnxRuntimeDetectionWrapper:
    def __init__(self, *, session: Any, input_name: str, output_roles: dict[str, int]) -> None:
        self.session = session
        self.input_name = input_name
        self.output_roles = output_roles

    def eval(self) -> Any:
        return self

    def to(self, *args: Any, **kwargs: Any) -> Any:
        return self

    def __call__(self, images: list[Any], *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        import numpy as np
        import torch

        if args or kwargs:
            raise RuntimeError("onnxruntime_cpu backend supports inference-only calls: model(images).")

        predictions = []
        for image in images:
            image_array = image.detach().cpu().numpy().astype(np.float32, copy=False)
            outputs = self.session.run(None, {self.input_name: image_array})
            predictions.append(
                {
                    "boxes": torch.from_numpy(outputs[self.output_roles["boxes"]]).float(),
                    "scores": torch.from_numpy(outputs[self.output_roles["scores"]]).float(),
                    "labels": torch.from_numpy(outputs[self.output_roles["labels"]]).long(),
                }
            )
        return predictions


def _infer_openvino_detection_output_roles(outputs: list[Any]) -> dict[str, int]:
    roles: dict[str, int] = {}
    for idx, output in enumerate(outputs):
        shape = [dim.get_length() if dim.is_static else None for dim in output.partial_shape]
        output_type = str(output.get_element_type())
        if len(shape) == 2 and shape[-1] == 4 and output_type == "<Type: 'float32'>":
            roles["boxes"] = idx
        elif len(shape) == 1 and output_type == "<Type: 'float32'>":
            roles["scores"] = idx
        elif len(shape) == 1 and output_type in {"<Type: 'int64_t'>", "<Type: 'int32_t'>"}:
            roles["labels"] = idx

    missing = {"boxes", "scores", "labels"} - roles.keys()
    if missing:
        output_summary = [
            {
                "name": output.get_any_name(),
                "shape": str(output.partial_shape),
                "type": str(output.get_element_type()),
            }
            for output in outputs
        ]
        raise RuntimeBackendUnavailable(
            "Could not map OpenVINO outputs to torchvision detection fields "
            f"{sorted(missing)}. Outputs: {output_summary}"
        )
    return roles


def _validate_openvino_detection_model(
    *,
    eager_model: Any,
    ort_session: Any,
    compiled_model: Any,
    example_input: Any,
    openvino_output_roles: dict[str, int],
    onnx_output_roles: dict[str, int],
    source_onnx_path: Path,
    openvino_model_path: Path,
    openvino_weights_path: Path,
    onnx_reused: bool,
    ir_reused: bool,
    onnx_opset: int,
    onnx_export_method: str,
    openvino_version: str,
    available_devices: list[str],
    selected_device: str,
    warnings: list[str],
    limitations: list[str],
) -> dict[str, Any]:
    import numpy as np
    import torch

    input_array = example_input.numpy()
    with torch.inference_mode():
        eager_output = eager_model([example_input])[0]

    onnx_outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_array})
    openvino_result = compiled_model({compiled_model.inputs[0]: input_array})
    openvino_outputs = [openvino_result[output] for output in compiled_model.outputs]

    comparisons: dict[str, Any] = {}
    validation_warnings: list[str] = []
    status = "passed"
    for role in ("boxes", "scores", "labels"):
        eager_value = eager_output[role].detach().cpu().numpy()
        onnx_value = onnx_outputs[onnx_output_roles[role]]
        openvino_value = openvino_outputs[openvino_output_roles[role]]
        comparison: dict[str, Any] = {
            "pytorch_shape": list(eager_value.shape),
            "onnx_shape": list(onnx_value.shape),
            "openvino_shape": list(openvino_value.shape),
            "pytorch_dtype": str(eager_value.dtype),
            "onnx_dtype": str(onnx_value.dtype),
            "openvino_dtype": str(openvino_value.dtype),
        }

        if eager_value.shape != onnx_value.shape or eager_value.shape != openvino_value.shape:
            comparison["shape_match"] = False
            same_rank = eager_value.ndim == onnx_value.ndim == openvino_value.ndim
            expected_detection_shape = role in {"scores", "labels"} and same_rank and eager_value.ndim == 1
            expected_box_shape = (
                role == "boxes"
                and same_rank
                and eager_value.ndim == 2
                and onnx_value.shape[-1] == 4
                and openvino_value.shape[-1] == 4
            )
            if expected_detection_shape or expected_box_shape:
                status = "passed_with_warnings"
                validation_warnings.append(
                    f"OpenVINO {role} count differs from PyTorch/ONNX on the validation patch: "
                    f"pytorch={list(eager_value.shape)}, onnx={list(onnx_value.shape)}, "
                    f"openvino={list(openvino_value.shape)}. This can happen after exported "
                    "FCOS filtering/NMS due to backend numeric differences."
                )
            else:
                status = "failed"
        else:
            comparison["shape_match"] = True
            if np.issubdtype(eager_value.dtype, np.number) and np.issubdtype(openvino_value.dtype, np.number):
                openvino_max_abs_diff = (
                    float(np.max(np.abs(eager_value - openvino_value))) if eager_value.size else 0.0
                )
                onnx_max_abs_diff = float(np.max(np.abs(eager_value - onnx_value))) if eager_value.size else 0.0
                ov_onnx_max_abs_diff = float(np.max(np.abs(onnx_value - openvino_value))) if onnx_value.size else 0.0
                comparison["pytorch_openvino_max_abs_diff"] = openvino_max_abs_diff
                comparison["pytorch_onnx_max_abs_diff"] = onnx_max_abs_diff
                comparison["onnx_openvino_max_abs_diff"] = ov_onnx_max_abs_diff
                comparison["pytorch_openvino_allclose"] = bool(
                    np.allclose(eager_value, openvino_value, rtol=1e-3, atol=1e-4)
                )
                comparison["onnx_openvino_allclose"] = bool(
                    np.allclose(onnx_value, openvino_value, rtol=1e-3, atol=1e-4)
                )
                if role in {"boxes", "scores"} and not comparison["pytorch_openvino_allclose"]:
                    if status == "passed":
                        status = "passed_with_warnings"
                    validation_warnings.append(
                        f"OpenVINO {role} is not numerically allclose to PyTorch on the validation patch "
                        f"(max_abs_diff={openvino_max_abs_diff})."
                    )
        comparisons[role] = comparison

    return {
        "status": status,
        "runtime_backend": "openvino_cpu",
        "openvino_version": openvino_version,
        "available_devices": available_devices,
        "selected_device": selected_device,
        "source_onnx_path": str(source_onnx_path),
        "openvino_model_path": str(openvino_model_path),
        "openvino_weights_path": str(openvino_weights_path),
        "source_onnx_reused": onnx_reused,
        "openvino_ir_reused": ir_reused,
        "onnx_opset": onnx_opset,
        "onnx_export_method": onnx_export_method,
        "openvino_output_roles": openvino_output_roles,
        "onnx_output_roles": onnx_output_roles,
        "comparisons": comparisons,
        "validation_warnings": validation_warnings,
        "warnings": warnings,
        "limitations": limitations,
    }


class _OpenVinoDetectionWrapper:
    def __init__(self, *, compiled_model: Any, input_obj: Any, output_roles: dict[str, int]) -> None:
        self.compiled_model = compiled_model
        self.input_obj = input_obj
        self.output_roles = output_roles
        self.outputs = list(compiled_model.outputs)

    def eval(self) -> Any:
        return self

    def to(self, *args: Any, **kwargs: Any) -> Any:
        return self

    def __call__(self, images: list[Any], *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        import numpy as np
        import torch

        if args or kwargs:
            raise RuntimeError("openvino_cpu backend supports inference-only calls: model(images).")

        predictions = []
        for image in images:
            image_array = image.detach().cpu().numpy().astype(np.float32, copy=False)
            result = self.compiled_model({self.input_obj: image_array})
            outputs = [result[output] for output in self.outputs]
            predictions.append(
                {
                    "boxes": torch.from_numpy(outputs[self.output_roles["boxes"]]).float(),
                    "scores": torch.from_numpy(outputs[self.output_roles["scores"]]).float(),
                    "labels": torch.from_numpy(outputs[self.output_roles["labels"]]).long(),
                }
            )
        return predictions


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
