# gemma-eval-kit

A repaired, diagnosed, documented-model evaluator for the Gemma 4 developer agent competition,
plus our own single-run labels for the 129 public dev tasks.

**Limits, stated first.** This is not the competition's scorer. It follows the scorer's
documented environment model, and we do not claim that it matches the scorer task by task: the
scorer returns only an aggregate score, so a per-task match cannot be checked. The Kaggle-route
outcomes discussed in our paper are a third non-scorer environment, reported as an environment,
not as ground truth.

## What is in this folder

- `src/factory/`: our repair and diagnostic layer (`local_env.py`, `local_eval.py` with the
  root-cause classifier, `docker_kit.py` for the public Docker kit, `rescore.py`, `c5.py`), the
  `factory eval` command line (`eval_cli.py`) and the paper's number code (`paper/`).
- `harness/`: our drivers and the Docker kit's runner image definition.
- `labels/dev129_labels.csv`: one row per (task, route), 129 tasks x 3 routes = 387 rows.
- `pins.json`: sha256 pins for the shipped drivers and for the competition wheels you download.

What is **not** in this folder: the competition's task files, patches, problem statements, test
patches or wheels, and the `swegemma` / `adk-eval-core` packages. You download those from Kaggle
yourself, under the competition's terms.

## Labels

Columns: `task_id,repo,route,gold_pass,empty_fail,root_cause,runs`.

- `route` is the harness: `as_shipped` and `diagnostic` (a WSL1 subprocess setup) and
  `public_docker` (the competition's public Docker kit).
- `gold_pass`: the task's gold patch passed. `empty_fail`: the empty patch failed.
- `root_cause`: our classifier's cause for a failed gold run (empty when the gold patch passed).
- `runs` is always 1. **These labels are single-run**: each comes from one run, and there is no
  stability (3 of 3) label in this release.

## Quickstart (public_docker route)

You need Docker, [uv](https://docs.astral.sh/uv/) and a Kaggle account that has joined the
competition. Pick a data folder and set `FACTORY_DATA_ROOT` to it (it is required; there is no
default). Then download the two Kaggle inputs into it:

- the competition data into `$FACTORY_DATA_ROOT/competition` (it must hold `tasks.jsonl`,
  `wheels/` and `docker/`):
  `kaggle competitions download -c gemma-4-developer-agent`, then unzip it there;
- the wheelhouse dataset into `$FACTORY_DATA_ROOT/wheelhouse`:
  `kaggle datasets download metric/gemma-4-developer-agent-wheelhouse --unzip -p "$FACTORY_DATA_ROOT/wheelhouse"`.

Then three commands give one task's gold verdict:

```
uv sync
uv run factory eval kit-build
uv run factory eval local --harness public_docker --mode gold --tasks one_task.txt
```

`kit-build` builds the sandbox and runner images. `eval local` prints a `TASK` line per task and
ends with `EVAL-OK`. For all 129 tasks use `--tasks all`; use `--mode empty` for the empty-patch
run. The `as_shipped` and `diagnostic` routes need a private WSL1 distro that is not part of this
release; their code is included so the repair layer can be read.

## Reproduce the paper's numbers

The Kaggle dataset `bscode/gemma-eval-kit-results` holds the labels above, the E2 re-score rows
(no log paths) and `paper_numbers.json`. The Kaggle notebook `bscode/gemma-eval-kit-reproduce`
installs this repository with pip, recomputes every number with `factory.paper.numbers` and
prints `REPRODUCED` with the key count. It needs Python 3.12 or newer and internet access (for
the pip install), and no GPU, Docker or competition download. To run it locally, download the
dataset (`kaggle datasets download bscode/gemma-eval-kit-results --unzip -p <dataset folder>`),
then run `uv sync` and `uv run python reproduce.py <dataset folder>` in this folder. It needs no
`FACTORY_DATA_ROOT`.

## Security note

The runner container mounts `/var/run/docker.sock`, so it can control your Docker host: that
access is root-equivalent on the host. Run the kit only on a machine where you accept that. The
task sandbox itself runs with `network_mode="none"`, as the competition's harness documents.

## AI assistance

This code and the accompanying paper were drafted with an AI coding assistant (Claude) under the author's direction. The author chose the experiments, reviewed the results and edited the text.

## License

Apache License 2.0 (see `LICENSE` and `NOTICE`).
