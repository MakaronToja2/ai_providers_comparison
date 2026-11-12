#!/usr/bin/env python3
"""
Select the evaluation instances for the study: a seeded, repository-stratified sample
of the SWE-bench Lite test split, plus a small smoke set that is disjoint from it.

Method (also written into the output file):
1. Population: all instances of the Lite test split at a pinned dataset revision.
2. Strata: repositories. Each repository gets seats in proportion to its share of the
   population (integer arithmetic), at least one seat, and the seats left over go to the
   largest remainders; ties are broken by larger stratum, then repository name.
3. Within a repository, instance ids are sorted and drawn without replacement with
   random.Random(f"{seed}:{repo}").
4. Smoke set: drawn the same way (random.Random(f"{seed}:smoke:{repo}")) from the
   instances NOT selected in step 3, so harness debugging never touches an evaluated
   instance.

Usage:
    .venv/bin/python scripts/select_instances.py --seed 20261007 --n 100 --out data/instances_seed20261007.json
"""
import argparse
import hashlib
import json
import platform
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

DATASET = "SWE-bench/SWE-bench_Lite"
SPLIT = "test"

# Smoke-set composition: the six largest repositories, so image and environment problems
# of different projects surface before the full runs.
SMOKE_COMPOSITION = {
    "django/django": 2,
    "sympy/sympy": 2,
    "matplotlib/matplotlib": 1,
    "scikit-learn/scikit-learn": 1,
    "pytest-dev/pytest": 1,
    "sphinx-doc/sphinx": 1,
}


def allocate(population: Dict[str, int], n: int) -> Dict[str, int]:
    """Proportional allocation with a minimum of one seat and largest remainders."""
    total = sum(population.values())
    if n > total:
        raise ValueError(f"Cannot draw {n} from a population of {total}")

    seats = {repo: max(1, (n * size) // total) for repo, size in population.items()}
    remaining = n - sum(seats.values())
    if remaining < 0:
        raise ValueError("Minimum of one seat per repository exceeds the sample size")

    # Remainders as exact integers (numerators over `total`); a repository lifted to the
    # one-seat minimum has already received more than its quota and gets no extra seat.
    def remainder(repo: str) -> int:
        size = population[repo]
        return (n * size) % total if (n * size) // total >= 1 else -1

    order = sorted(population, key=lambda repo: (-remainder(repo), -population[repo], repo))
    for repo in order[:remaining]:
        seats[repo] += 1

    for repo, k in seats.items():
        if k > population[repo]:
            raise ValueError(f"Allocation for {repo} exceeds its population")
    return seats


def draw(ids_by_repo: Dict[str, List[str]], seats: Dict[str, int], seed: int, label: str = "") -> List[str]:
    """Seeded draw within each repository. Output is sorted by repository, then id."""
    selected: List[str] = []
    for repo in sorted(seats):
        pool = sorted(ids_by_repo[repo])
        key = f"{seed}:{label}{repo}"
        selected.extend(sorted(random.Random(key).sample(pool, seats[repo])))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--revision", default=None, help="Dataset revision (commit hash). Default: the current head, which is then recorded.")
    args = parser.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi

    revision = args.revision or HfApi().dataset_info(DATASET).sha
    dataset = load_dataset(DATASET, split=SPLIT, revision=revision)

    ids_by_repo: Dict[str, List[str]] = defaultdict(list)
    for instance_id, repo in zip(dataset["instance_id"], dataset["repo"]):
        ids_by_repo[repo].append(instance_id)
    population = {repo: len(ids) for repo, ids in ids_by_repo.items()}

    seats = allocate(population, args.n)
    instance_ids = draw(ids_by_repo, seats, args.seed)

    chosen = set(instance_ids)
    remaining_by_repo = {repo: [i for i in ids if i not in chosen] for repo, ids in ids_by_repo.items()}
    smoke_ids = draw(remaining_by_repo, SMOKE_COMPOSITION, args.seed, label="smoke:")
    assert not chosen & set(smoke_ids)

    output = {
        "dataset": DATASET,
        "split": SPLIT,
        "dataset_revision": revision,
        "seed": args.seed,
        "n": args.n,
        "method": (
            "Stratified by repository over the full Lite test split. Seats per repository are "
            "proportional to its share of the split (floor, at least one), with leftover seats "
            "assigned by largest remainder (ties: larger repository, then name). Within a "
            "repository, sorted instance ids are sampled without replacement with "
            "random.Random(f'{seed}:{repo}')."
        ),
        "population": dict(sorted(population.items(), key=lambda kv: (-kv[1], kv[0]))),
        "allocation": dict(sorted(seats.items(), key=lambda kv: (-kv[1], kv[0]))),
        "instance_ids": instance_ids,
        "instance_ids_sha256": hashlib.sha256("\n".join(instance_ids).encode()).hexdigest(),
        "smoke_method": (
            "Disjoint from instance_ids: drawn from the instances not selected above with "
            "random.Random(f'{seed}:smoke:{repo}'), fixed composition per repository."
        ),
        "smoke_composition": SMOKE_COMPOSITION,
        "smoke_instance_ids": smoke_ids,
        "python_version": platform.python_version(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")

    print(f"dataset revision : {revision}")
    print(f"population       : {sum(population.values())} instances in {len(population)} repositories")
    for repo, k in output["allocation"].items():
        print(f"  {repo:32s} {population[repo]:4d} -> {k}")
    print(f"selected         : {len(instance_ids)} (sha256 {output['instance_ids_sha256'][:16]}...)")
    print(f"smoke set        : {smoke_ids}")
    print(f"written          : {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
