"""
export_gguf.py
================

Convert a HuggingFace model directory or model id into the GGUF format
and quantize it for use with the llama.cpp runtime.  This script
wraps the `convert_hf_to_gguf.py` converter and the `llama-quantize`
binary provided by the llama.cpp project.  It produces a directory
containing the quantized `.gguf` file, tokenizer and prompt cache.

Usage::

    python export_gguf.py --model path/to/model --outdir gguf_artifacts --qtype q4_k_m

You must have llama.cpp cloned at `~/llama.cpp` with its build
artifacts available.  The script looks for `convert_hf_to_gguf.py`
and `llama-quantize` in that directory.  If the converter is not
found, it will emit an error.

"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def run_cmd(cmd: list[str], check: bool = True) -> None:
    """Execute a shell command and raise on failure."""
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command {' '.join(cmd)} failed: {result.stderr}")


def export_gguf(model: str, outdir: Path, qtype: str = "q4_k_m", converter_dir: Optional[Path] = None) -> None:
    """
    Convert a HuggingFace model to GGUF and quantize it.

    :param model: Path or HF model id of the fine tuned model.
    :param outdir: Directory where the GGUF and artifacts will be saved.
    :param qtype: Quantization type, e.g. q4_k_m, q3_k_m, q5_k_m.
    :param converter_dir: Optional directory containing llama.cpp and its tools.
    """
    if converter_dir is None:
        converter_dir = Path.home() / "llama.cpp"
    convert_script = converter_dir / "convert_hf_to_gguf.py"
    quant_bin = converter_dir / "build/bin/llama-quantize"
    if not convert_script.exists():
        raise FileNotFoundError(f"Converter script not found: {convert_script}")
    if not quant_bin.exists():
        raise FileNotFoundError(f"Quantize binary not found: {quant_bin}")
    outdir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Step 1: convert HF model to f16 GGUF
        f16_path = tmp_path / "model-f16.gguf"
        cmd_convert = [
            "python", str(convert_script),
            "--outtype", "f16",
            "--outfile", str(f16_path),
            model,
        ]
        run_cmd(cmd_convert)
        # Step 2: quantize the f16 GGUF to the requested type
        q_path = outdir / f"model-{qtype}.gguf"
        cmd_quant = [
            str(quant_bin),
            str(f16_path),
            str(q_path),
            qtype,
        ]
        run_cmd(cmd_quant)
        # Copy tokenizer files if present
        model_path = Path(model)
        for name in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json", "tokenizer_cache.json"]:
            src = model_path / name
            if src.exists():
                shutil.copy(src, outdir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a HF model to GGUF and quantize it")
    parser.add_argument("--model", type=str, required=True, help="Path or HF model id of the model to export")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for the GGUF artifact")
    parser.add_argument("--qtype", type=str, default="q4_k_m", help="Quantization type (default: q4_k_m)")
    parser.add_argument("--converter-dir", type=Path, default=None, help="Directory containing llama.cpp")
    args = parser.parse_args()

    export_gguf(args.model, args.outdir, args.qtype, args.converter_dir)
    print(f"Exported GGUF to {args.outdir} (quant={args.qtype})")


if __name__ == "__main__":
    main()