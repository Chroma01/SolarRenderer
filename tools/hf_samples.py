"""Turn the released training shards into the sample list tools/omni_preencode.py reads.

    python tools/hf_samples.py --data ./SolarRenderer-Data --out ./samples [--sources magpie,carla,unrealzoo]
                               [--per-source N]

Reads the WebDataset shards of the Hugging Face dataset (train/<source>/*.tar), writes the files each
clip needs under <out>/<source>/<clip>/ and one sample per clip to <out>/samples.jsonl. A clip's observations follow
its ``training_input``: the structure laid over every frame, the first frame on latent frame 0, and the prompt is the
one written for that combination. --per-source keeps the first N clips of each source (a quick test). A rerun skips files that are already written.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import tarfile

INPUTS = {  # training_input -> (structure, first frame)
    "structure + first frame + text": (True, True),
    "structure + text": (True, False),
    "first frame + text": (False, True),
    "text only": (False, False),
}
TRAIN = {"target": "render.mp4", "structure": "condition.mp4", "first": "first_frame.png"}


def clips(shard: str, names: dict[str, str]):
    """(key, {role: bytes}, meta) for every clip of a shard, keeping only the members named in ``names``."""
    by: dict[str, dict] = {}
    with tarfile.open(shard) as tar:
        for m in tar:
            for role, suffix in [*names.items(), ("meta", "json")]:
                if m.isfile() and m.name.endswith("." + suffix):
                    by.setdefault(m.name[: -len(suffix) - 1], {})[role] = tar.extractfile(m).read()
    for key, files in sorted(by.items()):
        yield key, files, json.loads(files.pop("meta"))


def write(path: str, data: bytes) -> str:
    if not os.path.exists(path):
        with open(path + ".part", "wb") as fh:
            fh.write(data)
        os.replace(path + ".part", path)
    return os.path.abspath(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="local copy of Enderfga/SolarRenderer-Data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", default="magpie,carla,unrealzoo")
    ap.add_argument("--per-source", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n = 0
    with open(os.path.join(a.out, "samples.jsonl"), "w") as out:
        for source in a.sources.split(","):
            pattern = f"{a.data}/train/{source}/*.tar"
            shards = sorted(glob.glob(pattern))
            if not shards:
                raise SystemExit(f"no shards at {pattern}; download them first (see docs/training.md)")
            done = 0
            for shard in shards:
                for key, files, meta in clips(shard, TRAIN):
                    if a.per_source and done >= a.per_source:
                        break
                    use_structure, use_first = INPUTS[meta["training_input"]]
                    d = os.path.join(a.out, source, key)
                    os.makedirs(d, exist_ok=True)
                    target = write(f"{d}/render.mp4", files["target"])
                    obs = []
                    if use_structure:
                        obs.append({"media": write(f"{d}/structure.mp4", files["structure"]), "frames": "all"})
                    if use_first:
                        obs.append({"media": write(f"{d}/first_frame.png", files["first"]), "frames": [0]})
                    out.write(json.dumps({"id": f"{source}/{key}", "target": target, "text": meta["prompt"],
                                          "observations": obs, "source": source}, ensure_ascii=False) + "\n")
                    done += 1
                if a.per_source and done >= a.per_source:
                    break
            print(f"{source}: {done} samples", flush=True)
            n += done
    print(f"{n} samples -> {os.path.join(a.out, 'samples.jsonl')}")


if __name__ == "__main__":
    main()
