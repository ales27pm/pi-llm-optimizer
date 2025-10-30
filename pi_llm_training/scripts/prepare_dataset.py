# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path

from prompt_templates import extract_fields, llama_inst


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", required=True)
    parser.add_argument("--out-file", required=True)
    parser.add_argument(
        "--drop-unlabeled",
        action="store_true",
        help="skip samples without assistant text",
    )
    args = parser.parse_args()

    src = Path(args.in_file)
    dst = Path(args.out_file)
    dst.parent.mkdir(parents=True, exist_ok=True)

    count_in = count_out = 0
    with dst.open("w", encoding="utf-8") as w:
        for ex in iter_jsonl(src):
            count_in += 1
            fields = extract_fields(ex)
            if args.drop_unlabeled and not fields["assistant"]:
                continue
            prompt_pack = llama_inst(fields["system"], fields["user"], fields["assistant"])
            w.write(json.dumps({"text": prompt_pack["prompt"]}, ensure_ascii=False) + "\n")
            count_out += 1

    print(
        json.dumps(
            {
                "input": args.in_file,
                "output": args.out_file,
                "n_in": count_in,
                "n_out": count_out,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
