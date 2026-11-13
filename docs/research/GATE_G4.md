# Gate G4: pre-registration before any R5 training run

`scripts/research/r5_run_matrix.py` refuses to run the 24-run matrix until both lines below are filled in:
- a line `OSF: https://...` giving the OSF registration;
- a line `PREREGISTRATION commit: <hash>` naming a commit that is an ancestor of HEAD.

One-epoch smoke runs (`--smoke`, into `runs/_smoke/`) are allowed before G4. They are not evaluated on test or external sets.

## Status: waiting for the PI

1. ✅ `docs/research/PREREGISTRATION.md` is written and committed. The commit hash is below.
2. ⬜ **PI action:** lodge PREREGISTRATION.md on OSF. The step-by-step guide is `docs/research/OSF_REGISTRATION_GUIDE.md`. Short version: (osf.io → Registrations → "OSF Preregistration"). Attach or paste the file, and quote the commit hash. Then replace the placeholder line below with the registration URL.
3. ⬜ Commit this file. The matrix then starts (`python scripts/research/r5_run_matrix.py`, about 24–36 GPU-hours).

PREREGISTRATION commit: b81bf61ed3
OSF registration: (pending — the PI pastes the URL here as a line starting with "OSF: ")

**Commit history of the registered plan.**
- `91a54aedbc` (R5 part 1) first committed the document.
- `b81bf61ed3` (R5 part 2) is the commit to quote. It adds code for §6 items 3, 7, 8 and 9, which the plan listed but part 1 had not implemented, and it adds clarifications in §8.
- Neither commit trained or evaluated any R5 model beyond 1-epoch smoke runs on validation data.

## Pre-G4 checks done

- Unit tests for the sampler, caliper augmentation, pipeline modes and transform parity, and for the primary-endpoint script on synthetic data: see `R5_WALKTHROUGH.md`.
- Letterbox caches built and keyed: `data/processed/cache/sector_letterbox*/cache_manifest.json`.
- Smoke runs: see `R5_WALKTHROUGH.md`.
