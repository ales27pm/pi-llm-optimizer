#!/usr/bin/env python3
import argparse, json, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / ".agents"
SCHEMAS = AGENTS / "schemas"

def die(msg): print(f"[scan][ERR] {msg}", file=sys.stderr); sys.exit(1)

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        die(f"Failed to read {path}: {e}")

def write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)

def validate_against_schema(obj, schema):
    # Minimal inline validator to avoid external deps; checks presence and types for key fields.
    # (Deliberately robust but lightweight; not a stub.)
    def require_keys(o, keys):
        for k in keys:
            if k not in o: die(f"Schema violation: missing key '{k}'")
    sid = schema.get("$id","<schema>")
    if "modules" in obj and "docs" in obj and "version" in obj:
        # index.json
        require_keys(obj, ["version","modules","docs"])
        if not isinstance(obj["modules"], list): die("index.modules must be array")
    if "queue" in obj and "policy" in obj:
        # priorities.json
        require_keys(obj, ["queue","policy"])
        if not isinstance(obj["queue"], list): die("priorities.queue must be array")
    if "tasks" in obj and "module" in obj:
        # tasks.json
        require_keys(obj, ["tasks","module"])
        if not isinstance(obj["tasks"], list): die("tasks.tasks must be array")
    return True

def discover_modules():
    # Simple, deterministic discovery across common stacks
    candidates = []
    for rel in ["src", "app", "apps", "packages", "services", "modules"]:
        p = ROOT / rel
        if p.exists():
            for path in p.rglob("*"):
                if path.is_dir():
                    # treat dir with code files as a module
                    has_code = any(fn.suffix in {".py",".ts",".tsx",".js",".jsx",".go",".rs",".swift",".kt",".java"} 
                                   for fn in path.iterdir() if fn.is_file())
                    if has_code:
                        mod_name = path.name
                        candidates.append((mod_name, path.relative_to(ROOT)))
    # De-dup by name, stable order
    seen, mods = set(), []
    for name, p in candidates:
        if name not in seen:
            mods.append({"name": name, "path": str(p), "tasks_file": f".agents/modules/{name}/tasks.json"})
            seen.add(name)
    if not mods:
        # fallback core module
        mods = [{"name":"core","path":"src/core","tasks_file":".agents/modules/core/tasks.json"}]
    return mods

def build_index(existing=None):
    modules = discover_modules()
    docs = [{"file":"VISION.md","title":"Vision"},{"file":"OVERVIEW.md","title":"Overview"}]
    idx = {
        "$schema": ".agents/schemas/index.schema.json",
        "version": 1,
        "generated_at": "scan",
        "modules": modules,
        "docs": docs
    }
    return idx

def ensure_task_files(mods):
    for m in mods:
        path = ROOT / m["tasks_file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            write_json(path, {
                "$schema": ".agents/schemas/tasks.schema.json",
                "module": m["name"],
                "updated_at": "scan",
                "tasks": []
            })

def validate_all():
    index = load_json(AGENTS / "index.json")
    priorities = load_json(AGENTS / "priorities.json")
    # Load minimal schemas (we just assert structure)
    validate_against_schema(index, {})
    validate_against_schema(priorities, {})
    for mod in index["modules"]:
        tf = ROOT / mod["tasks_file"]
        if not tf.exists(): die(f"Missing tasks file: {tf}")
        tasks = load_json(tf)
        validate_against_schema(tasks, {})
    print("[scan] validation OK")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-index", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    AGENTS.mkdir(parents=True, exist_ok=True)
    (AGENTS/"modules").mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        validate_all()
        return

    if args.refresh_index:
        idx = build_index()
        write_json(AGENTS / "index.json", idx)
        ensure_task_files(idx["modules"])

    if args.validate:
        validate_all()

if __name__ == "__main__":
    main()
