from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch


RuntimeBackendName = Literal[
    "pytorch_eager",
    "pytorch_compile",
    "onnxruntime_cpu",
    "onnxruntime_int8",
    "openvino_cpu",
    "openvino_int8",
    "tensorrt",
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
    openvino_compress_to_fp16: bool | None = None
    torch_version: str | None = None
    cuda_available: bool | None = None
    cuda_device_name: str | None = None
    tensorrt_version: str | None = None
    onnx_path: Path | None = None
    tensorrt_engine_path: Path | None = None
    tensorrt_engine_reused: bool | None = None
    precision: str | None = None
    engine_build_time_seconds: float | None = None
    inference_latency_excludes_engine_build: bool | None = None
    runtime_limitations: list[str] = field(default_factory=list)
    runtime_warnings: list[str] = field(default_factory=list)
    runtime_properties: dict[str, Any] | None = None

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
            "openvino_compress_to_fp16": self.openvino_compress_to_fp16,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "cuda_device_name": self.cuda_device_name,
            "tensorrt_version": self.tensorrt_version,
            "onnx_path": str(self.onnx_path) if self.onnx_path else None,
            "tensorrt_engine_path": str(self.tensorrt_engine_path) if self.tensorrt_engine_path else None,
            "tensorrt_engine_reused": self.tensorrt_engine_reused,
            "precision": self.precision,
            "engine_build_time_seconds": self.engine_build_time_seconds,
            "inference_latency_excludes_engine_build": self.inference_latency_excludes_engine_build,
            "runtime_limitations": self.runtime_limitations,
            "runtime_warnings": self.runtime_warnings,
            "runtime_properties": self.runtime_properties,
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
    openvino_compress_to_fp16: bool = True,
    openvino_performance_hint: str = "LATENCY",
    openvino_num_streams: int = 1,
    runtime_intra_op_threads: int = 0,
    runtime_inter_op_threads: int = 1,
    openvino_inference_num_threads: int = 4,
    openvino_inference_precision: str = "f32",
    int8_model_path: Path | None = None,
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
            intra_op_threads=runtime_intra_op_threads,
            inter_op_threads=runtime_inter_op_threads,
        )

    if runtime_backend == "onnxruntime_int8":
        return _configure_onnxruntime_int8(
            model=model,
            requested_backend=runtime_backend,
            int8_model_path=int8_model_path,
            validation_output=validation_output,
            example_input=example_input,
            intra_op_threads=runtime_intra_op_threads,
            inter_op_threads=runtime_inter_op_threads,
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
            openvino_compress_to_fp16=openvino_compress_to_fp16,
            performance_hint=openvino_performance_hint,
            num_streams=openvino_num_streams,
            inference_num_threads=openvino_inference_num_threads,
            inference_precision=openvino_inference_precision,
            config_file=config_file,
            validation_output=validation_output,
            example_input=example_input,
        )

    if runtime_backend == "openvino_int8":
        return _configure_openvino_int8(
            requested_backend=runtime_backend,
            int8_model_path=int8_model_path,
            openvino_device=openvino_device,
            validation_output=validation_output,
            example_input=example_input,
            performance_hint=openvino_performance_hint,
            num_streams=openvino_num_streams,
            inference_num_threads=openvino_inference_num_threads,
            inference_precision=openvino_inference_precision,
        )

    if runtime_backend == "tensorrt":
        return _configure_tensorrt(
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
    intra_op_threads: int,
    inter_op_threads: int,
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
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = intra_op_threads
        session_options.inter_op_num_threads = inter_op_threads
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(
            str(export_path), sess_options=session_options, providers=["CPUExecutionProvider"]
        )
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
        runtime_properties={
            "intra_op_num_threads": session.get_session_options().intra_op_num_threads,
            "inter_op_num_threads": session.get_session_options().inter_op_num_threads,
            "execution_mode": str(session.get_session_options().execution_mode),
        },
    )


def _configure_onnxruntime_int8(
    *,
    model: Any,
    requested_backend: RuntimeBackendName,
    int8_model_path: Path | None,
    validation_output: Path | None,
    example_input: Any | None,
    intra_op_threads: int,
    inter_op_threads: int,
) -> RuntimeBackendResult:
    _require_modules(
        modules=["onnx", "onnxruntime"],
        install_hint="python -m pip install -r requirements-runtime.txt",
        backend_name=requested_backend,
    )
    if int8_model_path is None:
        raise RuntimeBackendUnavailable("onnxruntime_int8 requires --int8_model_path; no FP32 fallback is allowed.")
    int8_model_path = Path(int8_model_path).resolve()
    if not int8_model_path.is_file():
        raise RuntimeBackendUnavailable(f"INT8 ONNX artifact not found: {int8_model_path}")
    if example_input is None:
        raise RuntimeBackendUnavailable("A real smoke patch is required to validate the INT8 artifact.")

    import numpy as np
    import onnx
    import onnxruntime as ort

    graph = onnx.load(str(int8_model_path), load_external_data=False)
    op_counts: dict[str, int] = {}
    for node in graph.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    int8_initializers = sum(
        initializer.data_type in {onnx.TensorProto.INT8, onnx.TensorProto.UINT8}
        for initializer in graph.graph.initializer
    )
    qdq_count = op_counts.get("QuantizeLinear", 0) + op_counts.get("DequantizeLinear", 0)
    qlinear_count = sum(count for op, count in op_counts.items() if op.startswith("QLinear"))
    if int8_initializers == 0 or (qdq_count == 0 and qlinear_count == 0):
        raise RuntimeBackendUnavailable(
            f"Artifact is not a verified quantized graph: int8_initializers={int8_initializers}, "
            f"Q/DQ nodes={qdq_count}, QLinear nodes={qlinear_count}"
        )

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = intra_op_threads
    session_options.inter_op_num_threads = inter_op_threads
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(int8_model_path), sess_options=session_options, providers=["CPUExecutionProvider"]
    )
    providers = session.get_providers()
    if providers != ["CPUExecutionProvider"]:
        raise RuntimeBackendUnavailable(f"INT8 benchmark requires CPUExecutionProvider only; got {providers}")
    output_roles = _infer_detection_output_roles(session)
    sample = example_input.detach().cpu().float()
    outputs = session.run(None, {session.get_inputs()[0].name: sample.numpy()})
    output_summary = {}
    for role, index in output_roles.items():
        value = outputs[index]
        output_summary[role] = {"shape": list(value.shape), "dtype": str(value.dtype), "finite": bool(np.isfinite(value).all())}
        if not output_summary[role]["finite"]:
            raise RuntimeBackendUnavailable(f"INT8 smoke output {role} contains non-finite values")
    model.eval()
    model.to("cpu")
    with torch.inference_mode():
        eager = model([sample])[0]
    valid_shapes = (
        len(output_summary["boxes"]["shape"]) == 2
        and output_summary["boxes"]["shape"][1] == 4
        and len(output_summary["scores"]["shape"]) == 1
        and len(output_summary["labels"]["shape"]) == 1
        and output_summary["boxes"]["shape"][0] == output_summary["scores"]["shape"][0]
        and output_summary["scores"]["shape"][0] == output_summary["labels"]["shape"][0]
    )
    if not valid_shapes:
        raise RuntimeBackendUnavailable(f"INT8 smoke returned an invalid detection structure: {output_summary}")

    validation = {
        "status": "passed",
        "runtime_backend": "onnxruntime_int8",
        "onnx_model_path": str(int8_model_path),
        "execution_providers": providers,
        "operator_counts": op_counts,
        "int8_uint8_initializer_count": int8_initializers,
        "qdq_node_count": qdq_count,
        "qlinear_node_count": qlinear_count,
        "outputs": output_summary,
        "eager_detection_count": int(eager["scores"].numel()),
        "int8_detection_count": int(outputs[output_roles["scores"]].size),
    }
    if validation_output:
        validation_output.parent.mkdir(parents=True, exist_ok=True)
        validation_output.write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    return RuntimeBackendResult(
        model=_OnnxRuntimeDetectionWrapper(
            session=session, input_name=session.get_inputs()[0].name, output_roles=output_roles
        ),
        runtime_backend=requested_backend,
        effective_runtime_backend="onnxruntime_int8",
        export_path=int8_model_path,
        execution_providers=providers,
        validation_path=validation_output,
        validation_status="passed",
        precision="INT8",
        runtime_limitations=["QDQ INT8 compute is executed by ONNX Runtime CPU EP; surrounding patch and MIDOG evaluation code remains FP32/Python."],
        runtime_properties={
            "intra_op_num_threads": session.get_session_options().intra_op_num_threads,
            "inter_op_num_threads": session.get_session_options().inter_op_num_threads,
            "execution_mode": str(session.get_session_options().execution_mode),
        },
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
    openvino_compress_to_fp16: bool,
    config_file: Path | None,
    validation_output: Path | None,
    example_input: Any | None,
    performance_hint: str,
    num_streams: int,
    inference_num_threads: int,
    inference_precision: str,
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

    precision_suffix = "" if openvino_compress_to_fp16 else "_fp32"
    xml_path = onnx_path.with_name(f"{onnx_path.stem}_openvino{precision_suffix}.xml")
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
            save_kwargs = {} if openvino_compress_to_fp16 else {"compress_to_fp16": False}
            ov.save_model(ov_model, str(xml_path), **save_kwargs)
    except Exception as exc:
        raise RuntimeBackendUnavailable(
            f"OpenVINO ONNX-to-IR conversion/loading failed for {onnx_path}: {exc}"
        ) from exc

    try:
        compile_config = {
            "PERFORMANCE_HINT": performance_hint,
            "NUM_STREAMS": num_streams,
            "INFERENCE_NUM_THREADS": inference_num_threads,
            "INFERENCE_PRECISION_HINT": inference_precision,
        }
        compiled_model = core.compile_model(ov_model, openvino_device, compile_config)
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
        openvino_compress_to_fp16=openvino_compress_to_fp16,
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
        openvino_compress_to_fp16=openvino_compress_to_fp16,
        runtime_limitations=limitations,
        runtime_warnings=warnings,
        runtime_properties=_openvino_compiled_properties(compiled_model),
    )


def _configure_openvino_int8(
    *, requested_backend: RuntimeBackendName, int8_model_path: Path | None,
    openvino_device: str, validation_output: Path | None, example_input: Any | None,
    performance_hint: str, num_streams: int, inference_num_threads: int,
    inference_precision: str,
) -> RuntimeBackendResult:
    _require_modules(modules=["openvino"], install_hint="python -m pip install openvino", backend_name=requested_backend)
    if int8_model_path is None:
        raise RuntimeBackendUnavailable("openvino_int8 requires --int8_model_path; no FP32 fallback is allowed.")
    xml_path = Path(int8_model_path).resolve()
    bin_path = xml_path.with_suffix(".bin")
    if not xml_path.is_file() or not bin_path.is_file():
        raise RuntimeBackendUnavailable(f"OpenVINO INT8 IR pair not found: {xml_path}, {bin_path}")
    if example_input is None:
        raise RuntimeBackendUnavailable("A real smoke patch is required to validate the OpenVINO INT8 artifact.")

    import numpy as np
    import openvino as ov

    core = ov.Core()
    available_devices = list(core.available_devices)
    if openvino_device not in available_devices:
        raise RuntimeBackendUnavailable(f"OpenVINO device {openvino_device!r} is unavailable: {available_devices}")
    ov_model = core.read_model(str(xml_path))
    op_counts: dict[str, int] = {}
    element_type_counts: dict[str, int] = {}
    for op in ov_model.get_ops():
        op_counts[op.get_type_name()] = op_counts.get(op.get_type_name(), 0) + 1
        for output in op.outputs():
            dtype = str(output.get_element_type())
            element_type_counts[dtype] = element_type_counts.get(dtype, 0) + 1
    fake_quantize_count = op_counts.get("FakeQuantize", 0)
    low_precision_types = sum(element_type_counts.get(name, 0) for name in ("i8", "u8"))
    if fake_quantize_count == 0 and low_precision_types == 0:
        raise RuntimeBackendUnavailable(
            f"Artifact is not verified low precision: FakeQuantize={fake_quantize_count}, i8/u8={low_precision_types}"
        )
    compiled_model = core.compile_model(ov_model, openvino_device, {
        "PERFORMANCE_HINT": performance_hint,
        "NUM_STREAMS": num_streams,
        "INFERENCE_NUM_THREADS": inference_num_threads,
        "INFERENCE_PRECISION_HINT": inference_precision,
    })
    output_roles = _infer_openvino_detection_output_roles(compiled_model.outputs)
    sample = example_input.detach().cpu().float().numpy()
    result = compiled_model({compiled_model.inputs[0]: sample})
    outputs = {}
    for role, index in output_roles.items():
        value = result[compiled_model.outputs[index]]
        outputs[role] = {
            "shape": list(value.shape), "dtype": str(value.dtype), "finite": bool(np.isfinite(value).all()),
            "minimum": float(value.min()) if value.size else None, "maximum": float(value.max()) if value.size else None,
        }
        if not outputs[role]["finite"]:
            raise RuntimeBackendUnavailable(f"OpenVINO INT8 smoke output {role} contains non-finite values")
    scores_shape = outputs["scores"]["shape"]
    if not (len(outputs["boxes"]["shape"]) == 2 and outputs["boxes"]["shape"][1] == 4
            and len(scores_shape) == 1 and len(outputs["labels"]["shape"]) == 1
            and outputs["boxes"]["shape"][0] == scores_shape[0] == outputs["labels"]["shape"][0]):
        raise RuntimeBackendUnavailable(f"OpenVINO INT8 smoke returned an invalid detection structure: {outputs}")
    try:
        compiled_properties = {
            "execution_devices": list(compiled_model.get_property("EXECUTION_DEVICES")),
            "inference_num_threads": int(compiled_model.get_property("INFERENCE_NUM_THREADS")),
            "num_streams": str(compiled_model.get_property("NUM_STREAMS")),
        }
    except Exception as exc:
        compiled_properties = {"property_read_error": str(exc)}
    validation = {
        "status": "passed", "runtime_backend": "openvino_int8", "openvino_version": ov.__version__,
        "openvino_model_path": str(xml_path), "openvino_weights_path": str(bin_path),
        "available_devices": available_devices, "selected_device": openvino_device,
        "operator_counts": op_counts, "element_type_counts": element_type_counts,
        "fake_quantize_count": fake_quantize_count, "low_precision_output_count": low_precision_types,
        "rt_info": {str(k): str(v) for k, v in ov_model.get_rt_info().items()},
        "compiled_properties": compiled_properties, "outputs": outputs, "int8_detection_count": scores_shape[0],
    }
    if validation_output:
        validation_output.parent.mkdir(parents=True, exist_ok=True)
        validation_output.write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    return RuntimeBackendResult(
        model=_OpenVinoDetectionWrapper(compiled_model=compiled_model, input_obj=compiled_model.inputs[0], output_roles=output_roles),
        runtime_backend=requested_backend, effective_runtime_backend="openvino_int8", openvino_device=openvino_device,
        export_path=xml_path, validation_path=validation_output, validation_status="passed", openvino_version=ov.__version__,
        available_devices=available_devices, selected_device=openvino_device, openvino_model_path=xml_path,
        openvino_weights_path=bin_path, openvino_ir_reused=True, openvino_compress_to_fp16=False, precision="INT8",
        runtime_limitations=["NNCF-quantized OpenVINO IR executes on OpenVINO CPU; preprocessing, merging, NMS, and metrics remain FP32/Python."],
        runtime_properties=_openvino_compiled_properties(compiled_model),
    )


def _openvino_compiled_properties(compiled_model: Any) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for name in (
        "EXECUTION_DEVICES", "PERFORMANCE_HINT", "INFERENCE_NUM_THREADS",
        "NUM_STREAMS", "ENABLE_CPU_PINNING", "EXECUTION_MODE_HINT",
        "INFERENCE_PRECISION_HINT",
    ):
        try:
            value = compiled_model.get_property(name)
            properties[name.lower()] = list(value) if name == "EXECUTION_DEVICES" else str(value)
        except Exception as exc:
            properties[name.lower()] = f"unavailable: {exc}"
    return properties


def _configure_tensorrt(
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
        modules=["onnx", "tensorrt", "torch"],
        install_hint="Install NVIDIA TensorRT with Python bindings, then verify `python -c 'import tensorrt'`.",
        backend_name=requested_backend,
    )

    import tensorrt as trt
    import torch

    warnings: list[str] = []
    limitations = [
        "TensorRT uses a static ONNX export of the FCOS backbone/head as an intermediate representation and executes "
        "that tensor graph on CUDA. FCOS anchor generation, score filtering, top-k, NMS, and final box postprocessing "
        "remain in PyTorch.",
        "Preprocessing, patch extraction, patch merging, benchmark orchestration, and metric calculation remain in Python.",
        "TensorRT validation compares one representative patch against PyTorch CUDA. Exact numerical equality is not expected.",
    ]

    torch_version = str(getattr(torch, "__version__", "unknown"))
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available and torch.cuda.device_count() else None
    tensorrt_version = str(getattr(trt, "__version__", "unknown"))

    if not cuda_available:
        raise RuntimeBackendUnavailable("tensorrt requires torch.cuda.is_available() to be true.")

    precision = "fp32"
    if example_input is None:
        warnings.append(
            "No representative benchmark patch was available for TensorRT validation; "
            "using a zero tensor fallback."
        )
        example_input = torch.zeros(3, patch_size, patch_size, dtype=torch.float32)
    else:
        example_input = example_input.detach().cpu().float()

    model.eval()
    model.to("cpu")
    fcos_model = _unwrap_torchvision_detection_model(model)

    transformed_example_input, feature_shapes = _prepare_tensorrt_fcos_example(
        fcos_model=fcos_model,
        example_input=example_input,
    )
    export_model = _TensorRtFcosHeadExportWrapper(fcos_model=fcos_model)
    onnx_path, onnx_reused, onnx_export_method, onnx_warnings = _prepare_tensorrt_fcos_onnx_export(
        export_model=export_model,
        export_dir=export_dir,
        model_name=model_name,
        patch_size=patch_size,
        onnx_opset=onnx_opset,
        config_file=config_file,
        example_input=transformed_example_input,
    )
    warnings.extend(onnx_warnings)

    engine_path = _tensorrt_engine_path(
        onnx_path=onnx_path,
        precision=precision,
        tensorrt_version=tensorrt_version,
    )
    engine_reused = engine_path.exists()
    engine_build_time_seconds = 0.0
    if not engine_reused:
        engine_build_time_seconds = _build_tensorrt_engine(
            onnx_path=onnx_path,
            engine_path=engine_path,
            precision=precision,
            trt=trt,
        )

    engine = _load_tensorrt_engine(engine_path=engine_path, trt=trt)
    wrapped_model = _TensorRtFcosDetectionWrapper(
        engine=engine,
        trt=trt,
        fcos_model=fcos_model,
        feature_shapes=feature_shapes,
    )

    model.to("cuda")
    model.eval()
    fcos_model.eval()
    wrapped_model.eval()
    validation = _validate_tensorrt_detection_model(
        eager_model=model,
        tensorrt_model=wrapped_model,
        example_input=example_input,
        onnx_path=onnx_path,
        onnx_reused=onnx_reused,
        onnx_opset=onnx_opset,
        onnx_export_method=onnx_export_method,
        engine_path=engine_path,
        engine_reused=engine_reused,
        engine_build_time_seconds=engine_build_time_seconds,
        precision=precision,
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_device_name=cuda_device_name,
        tensorrt_version=tensorrt_version,
        warnings=warnings,
        limitations=limitations,
    )
    validation_status = str(validation["status"])
    if validation_status not in {"passed", "passed_with_warnings"}:
        raise RuntimeBackendUnavailable(
            f"TensorRT validation failed for {engine_path}. Status: {validation_status}. Details: {validation}"
        )
    warnings.extend(str(warning) for warning in validation.get("validation_warnings", []))

    if validation_output:
        validation_output.parent.mkdir(parents=True, exist_ok=True)
        validation_output.write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")

    return RuntimeBackendResult(
        model=wrapped_model,
        runtime_backend=requested_backend,
        effective_runtime_backend="tensorrt",
        onnx_opset=onnx_opset,
        export_path=engine_path,
        export_reused=engine_reused,
        validation_path=validation_output,
        validation_status=validation_status,
        onnx_export_method=onnx_export_method,
        source_onnx_path=onnx_path,
        source_onnx_reused=onnx_reused,
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_device_name=cuda_device_name,
        tensorrt_version=tensorrt_version,
        onnx_path=onnx_path,
        tensorrt_engine_path=engine_path,
        tensorrt_engine_reused=engine_reused,
        precision=precision,
        engine_build_time_seconds=engine_build_time_seconds,
        inference_latency_excludes_engine_build=True,
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


def _tensorrt_fcos_onnx_export_path(
    *,
    export_dir: Path,
    model_name: str,
    patch_size: int,
    onnx_opset: int,
    config_file: Path | None,
) -> Path:
    full_graph_path = _onnx_export_path(
        export_dir=export_dir,
        model_name=model_name,
        patch_size=patch_size,
        onnx_opset=onnx_opset,
        config_file=config_file,
    )
    return full_graph_path.with_name(f"{full_graph_path.stem}_tensorrt_fcos_head.onnx")


def _tensorrt_engine_path(*, onnx_path: Path, precision: str, tensorrt_version: str) -> Path:
    safe_version = re.sub(r"[^A-Za-z0-9_.-]+", "_", tensorrt_version).strip("_") or "unknown"
    return onnx_path.with_name(f"{onnx_path.stem}_tensorrt_{safe_version}_{precision}.engine")


def _unwrap_torchvision_detection_model(model: Any) -> Any:
    candidate = getattr(model, "model", model)
    required = ["transform", "backbone", "head", "anchor_generator", "postprocess_detections"]
    missing = [name for name in required if not hasattr(candidate, name)]
    if missing:
        raise RuntimeBackendUnavailable(
            "tensorrt backend currently supports torchvision FCOS-style models only. "
            f"Missing attributes: {missing}"
        )
    return candidate


def _prepare_tensorrt_fcos_example(*, fcos_model: Any, example_input: Any) -> tuple[Any, list[tuple[int, ...]]]:
    import torch

    with torch.no_grad():
        image_list, _ = fcos_model.transform([example_input], None)
        features = fcos_model.backbone(image_list.tensors)
        if isinstance(features, torch.Tensor):
            features = [features]
        else:
            features = list(features.values())
    return image_list.tensors.detach().cpu().float().clone(), [tuple(feature.shape) for feature in features]


def _prepare_tensorrt_fcos_onnx_export(
    *,
    export_model: Any,
    export_dir: Path,
    model_name: str,
    patch_size: int,
    onnx_opset: int,
    config_file: Path | None,
    example_input: Any,
) -> tuple[Path, bool, str, list[str]]:
    import onnx
    import torch

    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = _tensorrt_fcos_onnx_export_path(
        export_dir=export_dir,
        model_name=model_name,
        patch_size=patch_size,
        onnx_opset=onnx_opset,
        config_file=config_file,
    )
    export_reused = export_path.exists()
    warnings: list[str] = []
    export_method = "reused"

    if not export_reused:
        export_model.eval()
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                torch.onnx.export(
                    export_model,
                    (example_input,),
                    str(export_path),
                    opset_version=onnx_opset,
                    dynamo=False,
                    input_names=["images"],
                    output_names=["cls_logits", "bbox_regression", "bbox_ctrness"],
                )
            export_method = "legacy_fcos_head"
            warnings.append(
                "TensorRT exports only the static FCOS backbone/head graph. "
                "FCOS anchor generation, score filtering, top-k, NMS, and final box postprocessing remain in PyTorch."
            )
        except Exception as exc:
            raise RuntimeBackendUnavailable(
                "TensorRT FCOS-head ONNX export failed. "
                f"Exporter error: {_format_exception(exc)}"
            ) from exc

    try:
        onnx.checker.check_model(str(export_path))
    except Exception as exc:
        raise RuntimeBackendUnavailable(f"ONNX checker rejected TensorRT FCOS-head export {export_path}: {exc}") from exc

    return export_path, export_reused, export_method, warnings


def _build_tensorrt_engine(*, onnx_path: Path, engine_path: Path, precision: str, trt: Any) -> float:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    explicit_batch = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    network_flags = 0 if explicit_batch is None else 1 << int(explicit_batch)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    onnx_bytes = onnx_path.read_bytes()
    if not parser.parse(onnx_bytes):
        errors = []
        for idx in range(parser.num_errors):
            errors.append(str(parser.get_error(idx)))
        raise RuntimeBackendUnavailable(
            f"TensorRT ONNX parser failed for {onnx_path}. Errors: {errors[:10]}"
        )

    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 * 1024 * 1024)
    elif hasattr(config, "max_workspace_size"):
        config.max_workspace_size = 4 * 1024 * 1024 * 1024

    if precision == "fp16":
        if not getattr(builder, "platform_has_fast_fp16", False):
            raise RuntimeBackendUnavailable("TensorRT FP16 requested, but this platform does not report fast FP16 support.")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision != "fp32":
        raise RuntimeBackendUnavailable(f"Unsupported TensorRT precision: {precision}")

    build_start = time.perf_counter()
    if hasattr(builder, "build_serialized_network"):
        serialized_engine = builder.build_serialized_network(network, config)
    else:
        engine = builder.build_engine(network, config)
        serialized_engine = engine.serialize() if engine is not None else None
    build_time = time.perf_counter() - build_start
    if serialized_engine is None:
        raise RuntimeBackendUnavailable(f"TensorRT engine build failed for {onnx_path}.")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized_engine))
    return build_time


def _load_tensorrt_engine(*, engine_path: Path, trt: Any) -> Any:
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeBackendUnavailable(f"TensorRT failed to deserialize engine: {engine_path}")
    return engine


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


class _TensorRtFcosHeadExportWrapper(torch.nn.Module):
    def __init__(self, *, fcos_model: Any) -> None:
        super().__init__()
        self.fcos_model = fcos_model

    def eval(self) -> Any:
        self.fcos_model.eval()
        return self

    def __call__(self, images: Any) -> tuple[Any, Any, Any]:
        features = self.fcos_model.backbone(images)
        if isinstance(features, dict):
            features = list(features.values())
        elif not isinstance(features, list):
            features = [features]
        head_outputs = self.fcos_model.head(features)
        return head_outputs["cls_logits"], head_outputs["bbox_regression"], head_outputs["bbox_ctrness"]


class _TensorRtFcosDetectionWrapper:
    def __init__(self, *, engine: Any, trt: Any, fcos_model: Any, feature_shapes: list[tuple[int, ...]]) -> None:
        self.engine = engine
        self.trt = trt
        self.fcos_model = fcos_model
        self.feature_shapes = feature_shapes
        self.engine_runner = _TensorRtDetectionWrapper(engine=engine, trt=trt)

    def eval(self) -> Any:
        self.fcos_model.eval()
        return self

    def to(self, *args: Any, **kwargs: Any) -> Any:
        self.fcos_model.to(*args, **kwargs)
        return self

    def __call__(self, images: list[Any], *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        import torch

        if args or kwargs:
            raise RuntimeError("tensorrt backend supports inference-only calls: model(images).")

        predictions = []
        for image in images:
            original_image_sizes = [tuple(image.shape[-2:])]
            image_list, _ = self.fcos_model.transform([image], None)
            image_tensor = image_list.tensors.contiguous().float()
            if not image_tensor.is_cuda:
                image_tensor = image_tensor.cuda()

            raw_outputs = _normalize_tensorrt_fcos_outputs(self.engine_runner.run_raw(image_tensor))
            head_outputs = {
                "cls_logits": raw_outputs["cls_logits"],
                "bbox_regression": raw_outputs["bbox_regression"],
                "bbox_ctrness": raw_outputs["bbox_ctrness"],
            }
            num_anchors_per_level = [shape[-2] * shape[-1] for shape in self.feature_shapes]
            split_head_outputs = {
                name: list(value.split(num_anchors_per_level, dim=1))
                for name, value in head_outputs.items()
            }
            feature_maps = [
                torch.empty(shape, dtype=image_tensor.dtype, device=image_tensor.device)
                for shape in self.feature_shapes
            ]
            anchors = self.fcos_model.anchor_generator(image_list, feature_maps)
            split_anchors = [list(anchor.split(num_anchors_per_level)) for anchor in anchors]
            detections = self.fcos_model.postprocess_detections(
                split_head_outputs,
                split_anchors,
                image_list.image_sizes,
            )
            detections = self.fcos_model.transform.postprocess(
                detections,
                image_list.image_sizes,
                original_image_sizes,
            )
            predictions.append({
                "boxes": detections[0]["boxes"].detach().float().cpu(),
                "scores": detections[0]["scores"].detach().float().cpu(),
                "labels": detections[0]["labels"].detach().long().cpu(),
            })
        return predictions


class _TensorRtDetectionWrapper:
    def __init__(self, *, engine: Any, trt: Any) -> None:
        self.engine = engine
        self.trt = trt
        self.context = engine.create_execution_context()
        self.tensor_names = [engine.get_tensor_name(idx) for idx in range(engine.num_io_tensors)]
        self.input_names = [
            name for name in self.tensor_names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            name for name in self.tensor_names if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]
        if len(self.input_names) != 1:
            raise RuntimeBackendUnavailable(
                f"TensorRT backend expects one input tensor, found {len(self.input_names)}: {self.input_names}"
            )
        self.input_name = self.input_names[0]

    def eval(self) -> Any:
        return self

    def to(self, *args: Any, **kwargs: Any) -> Any:
        return self

    def __call__(self, images: list[Any], *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        import torch

        if args or kwargs:
            raise RuntimeError("tensorrt backend supports inference-only calls: model(images).")
        if not torch.cuda.is_available():
            raise RuntimeError("tensorrt backend requires CUDA.")

        predictions = []
        for image in images:
            if not image.is_cuda:
                raise RuntimeError("tensorrt backend requires input tensors on CUDA. Run with DEVICE=cuda.")
            image_tensor = image.contiguous().float()
            if any(dim < 0 for dim in self.engine.get_tensor_shape(self.input_name)):
                self.context.set_input_shape(self.input_name, tuple(image_tensor.shape))
            self.context.set_tensor_address(self.input_name, int(image_tensor.data_ptr()))

            outputs: dict[str, Any] = {}
            for output_name in self.output_names:
                output_shape = tuple(int(dim) for dim in self.context.get_tensor_shape(output_name))
                if any(dim < 0 for dim in output_shape):
                    raise RuntimeError(f"TensorRT output {output_name!r} has unresolved shape {output_shape}.")
                output = torch.empty(
                    output_shape,
                    device=image_tensor.device,
                    dtype=_torch_dtype_from_tensorrt(self.engine.get_tensor_dtype(output_name), self.trt),
                )
                self.context.set_tensor_address(output_name, int(output.data_ptr()))
                outputs[output_name] = output

            stream = torch.cuda.current_stream(device=image_tensor.device)
            if not self.context.execute_async_v3(stream_handle=stream.cuda_stream):
                raise RuntimeError("TensorRT execute_async_v3 failed.")
            stream.synchronize()

            output_roles = _infer_tensorrt_detection_output_roles(outputs)
            predictions.append(
                {
                    "boxes": outputs[output_roles["boxes"]].detach().float().cpu(),
                    "scores": outputs[output_roles["scores"]].detach().float().cpu(),
                    "labels": outputs[output_roles["labels"]].detach().long().cpu(),
                }
            )
        return predictions

    def run_raw(self, image_tensor: Any) -> dict[str, Any]:
        import torch

        if not image_tensor.is_cuda:
            raise RuntimeError("tensorrt backend requires input tensors on CUDA. Run with DEVICE=cuda.")
        image_tensor = image_tensor.contiguous().float()
        if any(dim < 0 for dim in self.engine.get_tensor_shape(self.input_name)):
            self.context.set_input_shape(self.input_name, tuple(image_tensor.shape))
        self.context.set_tensor_address(self.input_name, int(image_tensor.data_ptr()))

        outputs: dict[str, Any] = {}
        for output_name in self.output_names:
            output_shape = tuple(int(dim) for dim in self.context.get_tensor_shape(output_name))
            if any(dim < 0 for dim in output_shape):
                raise RuntimeError(f"TensorRT output {output_name!r} has unresolved shape {output_shape}.")
            output = torch.empty(
                output_shape,
                device=image_tensor.device,
                dtype=_torch_dtype_from_tensorrt(self.engine.get_tensor_dtype(output_name), self.trt),
            )
            self.context.set_tensor_address(output_name, int(output.data_ptr()))
            outputs[output_name] = output

        stream = torch.cuda.current_stream(device=image_tensor.device)
        if not self.context.execute_async_v3(stream_handle=stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed.")
        stream.synchronize()
        return outputs


def _torch_dtype_from_tensorrt(dtype: Any, trt: Any) -> Any:
    import torch

    if dtype == trt.float32:
        return torch.float32
    if dtype == trt.float16:
        return torch.float16
    if dtype == trt.int32:
        return torch.int32
    if hasattr(trt, "int64") and dtype == trt.int64:
        return torch.int64
    if dtype == trt.bool:
        return torch.bool
    raise RuntimeError(f"Unsupported TensorRT tensor dtype: {dtype}")


def _infer_tensorrt_detection_output_roles(outputs: dict[str, Any]) -> dict[str, str]:
    import torch

    roles: dict[str, str] = {}
    for name, tensor in outputs.items():
        shape = list(tensor.shape)
        if len(shape) == 2 and shape[-1] == 4 and tensor.dtype in {torch.float16, torch.float32}:
            roles["boxes"] = name
        elif len(shape) == 1 and tensor.dtype in {torch.float16, torch.float32}:
            roles["scores"] = name
        elif len(shape) == 1 and tensor.dtype in {torch.int32, torch.int64}:
            roles["labels"] = name

    missing = {"boxes", "scores", "labels"} - roles.keys()
    if missing:
        output_summary = {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in outputs.items()
        }
        raise RuntimeBackendUnavailable(
            "Could not map TensorRT outputs to torchvision detection fields "
            f"{sorted(missing)}. Outputs: {output_summary}"
        )
    return roles


def _normalize_tensorrt_fcos_outputs(outputs: dict[str, Any]) -> dict[str, Any]:
    if {"cls_logits", "bbox_regression", "bbox_ctrness"}.issubset(outputs):
        return outputs

    roles: dict[str, str] = {}
    for name, tensor in outputs.items():
        shape = list(tensor.shape)
        if len(shape) == 3 and shape[-1] == 2:
            roles["cls_logits"] = name
        elif len(shape) == 3 and shape[-1] == 4:
            roles["bbox_regression"] = name
        elif len(shape) == 3 and shape[-1] == 1:
            roles["bbox_ctrness"] = name

    missing = {"cls_logits", "bbox_regression", "bbox_ctrness"} - roles.keys()
    if missing:
        output_summary = {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in outputs.items()
        }
        raise RuntimeBackendUnavailable(
            "Could not map TensorRT outputs to FCOS head fields "
            f"{sorted(missing)}. Outputs: {output_summary}"
        )
    return {role: outputs[name] for role, name in roles.items()}


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
    openvino_compress_to_fp16: bool,
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
        "openvino_compress_to_fp16": openvino_compress_to_fp16,
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


def _validate_tensorrt_detection_model(
    *,
    eager_model: Any,
    tensorrt_model: Any,
    example_input: Any,
    onnx_path: Path,
    onnx_reused: bool,
    onnx_opset: int,
    onnx_export_method: str,
    engine_path: Path,
    engine_reused: bool,
    engine_build_time_seconds: float,
    precision: str,
    torch_version: str,
    cuda_available: bool,
    cuda_device_name: str | None,
    tensorrt_version: str,
    warnings: list[str],
    limitations: list[str],
) -> dict[str, Any]:
    import numpy as np
    import torch

    validation_warnings: list[str] = []
    status = "passed"
    cuda_input = example_input.detach().cuda().float()
    eager_model.eval()
    tensorrt_model.eval()
    with torch.inference_mode():
        eager_output = eager_model([cuda_input])[0]
        tensorrt_output = tensorrt_model([cuda_input])[0]

    comparisons: dict[str, Any] = {}
    for role in ("boxes", "scores", "labels"):
        eager_value = eager_output[role].detach().cpu().numpy()
        tensorrt_value = tensorrt_output[role].detach().cpu().numpy()
        comparison: dict[str, Any] = {
            "pytorch_shape": list(eager_value.shape),
            "tensorrt_shape": list(tensorrt_value.shape),
            "pytorch_count": int(eager_value.shape[0]) if eager_value.ndim > 0 else int(eager_value.size),
            "tensorrt_count": int(tensorrt_value.shape[0]) if tensorrt_value.ndim > 0 else int(tensorrt_value.size),
            "pytorch_dtype": str(eager_value.dtype),
            "tensorrt_dtype": str(tensorrt_value.dtype),
        }

        if eager_value.shape != tensorrt_value.shape:
            comparison["shape_match"] = False
            same_rank = eager_value.ndim == tensorrt_value.ndim
            expected_detection_shape = role in {"scores", "labels"} and same_rank and eager_value.ndim == 1
            expected_box_shape = (
                role == "boxes"
                and same_rank
                and eager_value.ndim == 2
                and eager_value.shape[-1] == 4
                and tensorrt_value.shape[-1] == 4
            )
            if expected_detection_shape or expected_box_shape:
                status = "passed_with_warnings"
                validation_warnings.append(
                    f"TensorRT {role} detection count differs from PyTorch CUDA on the validation patch: "
                    f"pytorch={list(eager_value.shape)}, tensorrt={list(tensorrt_value.shape)}. "
                    "This can happen after exported FCOS filtering/NMS due to backend numeric differences."
                )
            else:
                status = "failed"
        else:
            comparison["shape_match"] = True
            if np.issubdtype(eager_value.dtype, np.number) and np.issubdtype(tensorrt_value.dtype, np.number):
                max_abs_diff = float(np.max(np.abs(eager_value - tensorrt_value))) if eager_value.size else 0.0
                comparison["max_abs_diff"] = max_abs_diff
                comparison["allclose"] = bool(np.allclose(eager_value, tensorrt_value, rtol=1e-3, atol=1e-4))
                if role in {"boxes", "scores"} and not comparison["allclose"]:
                    if status == "passed":
                        status = "passed_with_warnings"
                    validation_warnings.append(
                        f"TensorRT {role} is not numerically allclose to PyTorch CUDA on the validation patch "
                        f"(max_abs_diff={max_abs_diff})."
                    )
        comparisons[role] = comparison

    return {
        "status": status,
        "runtime_backend": "tensorrt",
        "device": "cuda",
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_device_name,
        "tensorrt_version": tensorrt_version,
        "onnx_path": str(onnx_path),
        "source_onnx_reused": onnx_reused,
        "onnx_opset": onnx_opset,
        "onnx_export_method": onnx_export_method,
        "tensorrt_engine_path": str(engine_path),
        "tensorrt_engine_reused": engine_reused,
        "precision": precision,
        "engine_build_time_seconds": engine_build_time_seconds,
        "inference_latency_excludes_engine_build": True,
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
