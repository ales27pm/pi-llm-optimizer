import json, os, subprocess, tempfile, time
from pathlib import Path

LLAMA_CONVERT = Path.home()/"llama.cpp/convert_hf_to_gguf.py"
LLAMA_QUANT   = Path.home()/"llama.cpp/build/bin/llama-quantize"

def sh(*args: str) -> None:
    subprocess.run(list(args), check=True)

def convert_and_quant(hf_model: str, out_dir: Path, outtype: str="f16", qtype: str="q4_k_m", revision: str|None=None, token: str|None=None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir/"model-f16.gguf"
    args = ["python3", str(LLAMA_CONVERT), "--outtype", outtype, "--outfile", str(base), hf_model]
    if revision: args += ["--revision", revision]
    if token:    os.environ["HF_TOKEN"] = token
    sh(*args)
    gguf_q = out_dir/f"model-{qtype}.gguf"
    sh(str(LLAMA_QUANT), str(base), str(gguf_q), qtype)
    return gguf_q

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--qtype", default="q4_k_m")
    ap.add_argument("--revision")
    ap.add_argument("--hf-token")
    args = ap.parse_args()

    out_dir = Path(args.out)
    gguf = convert_and_quant(args.model, out_dir, qtype=args.qtype, revision=args.revision, token=args.hf_token)
    print(json.dumps({"gguf": str(gguf)}, indent=2))
