PY := .venv/bin/python

.PHONY: setup demo real synthetic smoke test clean

setup:
	bash scripts/setup.sh

# reproduce results from the committed hand tracks in data/tracks (no MediaPipe)
demo:
	$(PY) run_all.py --tracks data/tracks

# full run from phone clips in data/raw (MediaPipe -> data/tracks + data/clips)
real:
	$(PY) run_all.py data/raw

# pipeline check on generated hand tracks
synthetic:
	$(PY) run_all.py --synthetic

smoke test:
	$(PY) -m pytest -q tests

clean:
	rm -rf results/*/bc_policy.pt .pytest_cache
