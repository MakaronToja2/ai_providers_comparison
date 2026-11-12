#!/usr/bin/env python3
"""
Pull the prebuilt SWE-bench evaluation images for the study's instances once, so that
no evaluation depends on a pull succeeding mid-run. Run `docker login` first: anonymous
Docker Hub pulls are rate limited.

Run with the evaluation environment (it needs the swebench package for image names):
    .venv-eval/bin/python scripts/pull_images.py --subset smoke
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INSTANCES = PROJECT_ROOT / "data" / "instances_seed20261007.json"
DATASET = "SWE-bench/SWE-bench_Lite"
SPLIT = "test"
NAMESPACE = "swebench"


def local_images() -> set:
    output = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                            capture_output=True, text=True, check=True).stdout
    return set(output.split())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", required=True, choices=["smoke", "full", "all"])
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument("--attempts", type=int, default=4)
    args = parser.parse_args()

    from datasets import load_dataset
    from swebench.harness.test_spec.test_spec import make_test_spec

    selection = json.loads(args.instances.read_text())
    wanted = []
    if args.subset in ("smoke", "all"):
        wanted += selection["smoke_instance_ids"]
    if args.subset in ("full", "all"):
        wanted += selection["instance_ids"]

    dataset = load_dataset(DATASET, split=SPLIT, revision=selection["dataset_revision"])
    by_id = {row["instance_id"]: row for row in dataset}
    images = {i: make_test_spec(by_id[i], namespace=NAMESPACE).instance_image_key for i in wanted}

    present = local_images()
    failed = []
    for n, (instance_id, image) in enumerate(images.items(), 1):
        if image in present:
            print(f"[{n}/{len(images)}] present  {image}", flush=True)
            continue
        for attempt in range(1, args.attempts + 1):
            result = subprocess.run(["docker", "pull", "--quiet", image], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[{n}/{len(images)}] pulled   {image}", flush=True)
                break
            message = (result.stderr or result.stdout).strip().splitlines()[-1:] or ["unknown error"]
            print(f"[{n}/{len(images)}] attempt {attempt} failed for {image}: {message[0]}", flush=True)
            time.sleep(min(300, 30 * attempt))
        else:
            failed.append(instance_id)

    print(f"images wanted {len(images)}, missing after pulls: {len(failed)}")
    if failed:
        print("missing:", failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
