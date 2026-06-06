#!/usr/bin/env python
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str]) -> tuple[int | None, str]:
    executable = shutil.which(command[0])
    if executable is None:
        return None, f"{command[0]} not found on PATH"
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:  # pragma: no cover - diagnostic script
        return None, f"{type(exc).__name__}: {exc}"
    output = "\n".join(part for part in (completed.stdout.strip(), completed.stderr.strip()) if part)
    return completed.returncode, output


def libcuda_paths() -> list[Path]:
    return sorted(Path("/usr/lib/wsl/lib").glob("libcuda.so*"))


def suggest_diagnosis(
    nvidia_smi_code: int | None,
    nvidia_smi_output: str,
    dxg_exists: bool,
    libcuda_exists: bool,
    torch_cuda_available: bool,
    torch_cuda_build: str | None,
) -> str:
    if torch_cuda_available:
        return "CUDA is available to PyTorch."
    smi_ok = nvidia_smi_code == 0
    if not smi_ok or not dxg_exists or not libcuda_exists:
        if "blocked by the operating system" in nvidia_smi_output.lower():
            return (
                "C. GPU access appears blocked in this execution context. "
                "If nvidia-smi and /dev/dxg work in a normal WSL terminal, run GPU fine-tuning there."
            )
        return (
            "A. Windows/WSL is not exposing the GPU here. Do not install Linux NVIDIA display drivers. "
            "Update the Windows NVIDIA driver with WSL CUDA support, run wsl --shutdown and wsl --update, "
            "then retest inside WSL."
        )
    if smi_ok and not torch_cuda_available:
        if torch_cuda_build is None:
            return (
                "D. PyTorch appears to be CPU-only even though WSL sees the GPU. "
                "Install the official PyTorch CUDA wheel for this environment."
            )
        return (
            "B or D. WSL sees the GPU, but PyTorch cannot use CUDA. Check LD_LIBRARY_PATH/PATH for "
            "/usr/lib/wsl/lib, verify the venv Python, and verify the installed PyTorch CUDA wheel."
        )
    return "Unable to classify from collected signals."


def main() -> None:
    print(f"python executable: {sys.executable}")
    print(f"python version: {sys.version.replace(os.linesep, ' ')}")
    print(f"WSL_DISTRO_NAME: {os.environ.get('WSL_DISTRO_NAME')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
    print(f"PATH contains /usr/lib/wsl/lib: {'/usr/lib/wsl/lib' in os.environ.get('PATH', '').split(':')}")

    nvidia_smi_code, nvidia_smi_output = run_command(["nvidia-smi"])
    print(f"nvidia-smi returncode: {nvidia_smi_code}")
    print("nvidia-smi output:")
    print(nvidia_smi_output or "<empty>")

    dxg = Path("/dev/dxg")
    print(f"/dev/dxg exists: {dxg.exists()}")
    if dxg.exists():
        print(f"/dev/dxg: {dxg}")

    paths = libcuda_paths()
    print(f"/usr/lib/wsl/lib/libcuda.so* count: {len(paths)}")
    for path in paths:
        print(f"  {path}")

    try:
        import torch
    except Exception as exc:
        print(f"torch import error: {type(exc).__name__}: {exc}")
        print("suggested diagnosis: PyTorch could not be imported in this Python environment.")
        return

    cuda_available = torch.cuda.is_available()
    print(f"torch version: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {cuda_available}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    if cuda_available:
        for idx in range(torch.cuda.device_count()):
            print(f"torch.cuda device {idx}: {torch.cuda.get_device_name(idx)}")

    print(
        "suggested diagnosis: "
        + suggest_diagnosis(
            nvidia_smi_code=nvidia_smi_code,
            nvidia_smi_output=nvidia_smi_output,
            dxg_exists=dxg.exists(),
            libcuda_exists=bool(paths),
            torch_cuda_available=cuda_available,
            torch_cuda_build=torch.version.cuda,
        )
    )


if __name__ == "__main__":
    main()
