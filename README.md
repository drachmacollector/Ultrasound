# Anatomy or Acquisition?

**Auditing shortcut reliance and anatomical grounding of deep-learning fetal standard-plane
classifiers, with multi-device external validation.**

This is a research study, not a product or a clinical tool. It makes no clinical or
regulatory claims.

## Question

Fetal standard-plane classifiers trained on FETAL_PLANES_DB (FPDB) report 90–99% accuracy, but
they degrade on unseen ultrasound machines. In FPDB the class label is entangled with *how* an
image was acquired. For example, 65% of Aloka training images are "Other" against 14% of
Voluson E6 images, and image size alone almost identifies the machine. This study measures:

1. **RQ1, confounding:** is acquisition information (device, operator, image geometry,
   on-screen overlays) predictive of the label, and is it encoded in trained models?
2. **RQ2, reliance:** do models *use* non-anatomical content?
3. **RQ3, grounding:** does saliency fall on fetal anatomy (scored against expert biometry
   landmarks), and does that predict success on new devices?
4. **RQ4, mitigation:** does a simple anatomy-focused training protocol (sector masking,
   overlay handling, acquisition-balanced sampling, device-style augmentation) narrow the
   external-validity gap? The primary endpoint is pre-registered.

Full rationale: [docs/research/01_RESEARCH_PROPOSAL.md](docs/research/01_RESEARCH_PROPOSAL.md).
Execution plans: [docs/research/plans/](docs/research/plans/). Progress log:
[docs/research/RESEARCH_LOG.md](docs/research/RESEARCH_LOG.md).

## Data (all public and de-identified; no new human data collected)

| Dataset | Role | Licence |
|---|---|---|
| FETAL_PLANES_DB (Burgos-Artizzu et al., *Sci Rep* 2020) | train / val / in-distribution test (frozen patient-disjoint official split) | per Zenodo record |
| Landmark biometry benchmark, FP subset (Di Vece et al., 2025) | expert landmarks on FPDB images; localisation ground truth only, never trained on | CC BY-NC-SA 4.0 |
| HC18 and UCL subsets of the same benchmark | external test sets + landmarks | CC BY-NC-SA 4.0 / challenge terms |
| African fetal planes (Sendra-Balcells et al., *Sci Rep* 2023; Zenodo 7540448) | external low-resource test set (5 countries, 5 devices) | CC BY 4.0 |

Raw data lives under `data/raw/` (not in git). Dataset facts and quirks:
[00_INDEX_AND_VERIFIED_FACTS.md §3](docs/research/plans/00_INDEX_AND_VERIFIED_FACTS.md).

## Environment

Conda env `fetalplane` (Python 3.11, CUDA 12.8, one 8 GB GPU). Exact pins are in
`requirements-lock.txt`.

```bash
python -m pip install -r requirements.txt
python -m pytest                     # run from the repo root
```

## Layout

```
src/data         datasets, transforms, labels        src/preprocess  sector, overlays, pipeline
src/models       backbones, loading, saliency         src/localisation anatomy masks, metrics
src/eval         prediction, statistics, scoring      src/train       training loop
src/audit        confounding / reliance helpers       scripts/research one script per experiment
configs/         B0 baselines                         configs/research protocol arms
results/         tables, json, figures (preds/ local) runs/           training outputs (local)
docs/research/   proposal, plans, log, gates, results archive/        frozen legacy demo
```

## Status

Phase-by-phase status is in `docs/research/RESEARCH_LOG.md`, with walkthroughs in
`docs/research/walkthroughs/`.

## History

This repository began as **FetScan**, a real-time plane-classification demo. That code is
frozen at tag `legacy-fetscan-demo-v1` and kept under `archive/` for provenance.
