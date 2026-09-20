# Notebooks

Scripts with `# %%` cell markers: they open as notebooks in VS Code or
Jupyter (via jupytext) and also run headless, so the training pipeline is
reproducible from a shell rather than by clicking through cells in order.

## No data yet?

`00_synthetic_data.py` generates a Lending-Club-shaped file so the pipeline
runs end to end before you have the real thing:

```
python notebooks/00_synthetic_data.py --rows 240000
python notebooks/01_load_and_target.py --raw data/raw/synthetic_loans.csv.gz
python notebooks/02_eda.py
python notebooks/03_features.py
python notebooks/05_train.py --train-end 2012 --calib-end 2013 --choose logistic --dataset-label synthetic
python notebooks/06_bands.py --calib-end 2013
python notebooks/07_population_shift.py
```

A model trained this way is **circular** — it rediscovers the coefficients
written at the top of `00_synthetic_data.py`. Passing `--dataset-label
synthetic` keeps `is_placeholder` true in the artefact, so the app goes on
showing its warning banner and the metrics can never be mistaken for real
ones. Do not put them in the README.

## Run order (real data)

```
python notebooks/01_load_and_target.py --raw data/raw/accepted_2007_to_2018Q4.csv.gz
python notebooks/02_eda.py
python notebooks/03_features.py
python notebooks/05_train.py --train-end 2014 --calib-end 2015 --choose logistic
python notebooks/06_bands.py --apply
python notebooks/07_population_shift.py --apply
```

Intermediate files land in `notebooks/_work/` (gitignored). The only outputs
that get committed are `models/pd_model.joblib` and the config changes from
06 and 07.

Supporting modules, imported rather than run:

- `leakage.py` — the exclusion list with a reason per column. `python
  notebooks/leakage.py` prints the markdown table for the README model card.
- `population_grid.py` — the synthetic reference grid standing in for the
  applicant population, shared by 06 and 07 so band cutoffs and the
  population shift come from the same distribution.

`00_synthetic_data.py` is the synthetic generator described above. There is
no `04`. It was the leakage audit, which became `leakage.py` when it
turned out to be needed by 01 and 03 as well as by the README.

`--apply` on 06 and 07 rewrites `config/config.yaml` through the YAML parser,
which drops comments. Reinstate them from git before committing: the
provenance comments are the reason anyone can audit the numbers.
