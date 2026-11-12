"""
src/localisation/runner.py  (R4.6 core, shared by the localisation run and the G3 sanity check)

One checkpoint x one dataset x one saliency method -> one per-image table:

    results/preds/<run_id>/<dataset>__saliency-<method>.parquet   (+ .maps.npz for the figure subset)

Per image, saliency is computed for the **true** target and the **predicted** target (collapsed
classes; a Head target is the set of the three brain sub-planes) and, for Head images, the
single-sub-plane variant (``truesub`` = argmax brain sub-plane logit). For each target:

* GT-free (every image): ``out_sector_frac`` (energy outside the R1 ``sector_clean`` mask, the
  same region E5 perturbed) and ``overlay_frac`` (energy on the R1 ``ui`` mask; the caliper
  ``marker`` mask failed Gate G1 and is not used). Their uniform-map baselines are the per-image
  ``out_sector_area`` and ``ui_area``.
* Landmark images only: EPG and its area baseline, sector-aware EPG, pointing hit, CAM-box IoU;
  Femur also at capsule widths 0.05 and 0.15 of |FL|.

All masks are projected into model-input space with ``to_model_space`` (soft, INTER_LINEAR).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from src.audit.metadata import load_masks, mask_path
from src.data.dataset import CANONICAL_CLASSES
from src.data.labels import COLLAPSED_MEMBERS
from src.data.transforms import gray_to_rgb, load_grayscale
from src.eval.datasets import load_eval_set
from src.eval.predict import PREDS_ROOT, git_hash
from src.localisation import metrics as LM
from src.localisation.anatomy_masks import FEMUR_REL_WIDTHS, anatomy_mask_for_row, to_model_space
from src.models.loading import LoadedModel
from src.models.saliency import SaliencyEngine, normalised_fill

LANDMARKS = {"fpdb": Path("data/processed/landmarks/fpdb_landmarks.parquet"),
             "external": Path("data/processed/landmarks/external_landmarks.parquet")}
HEAD_SUB = COLLAPSED_MEMBERS["Head"]          # fine indices of the three brain sub-planes
PRIMARY_FEMUR_WIDTH = 0.10
SEL_SEED = 20261109

#: legacy seed-42 checkpoints (frozen): backbone -> short name used in run ids
LEGACY = {"convnext_tiny": "convnext", "tf_efficientnetv2_s": "effv2s", "repvgg_a1": "repvgga1",
          "repvgg_a2": "repvgga2", "mobilenetv3_large_100": "mnv3l", "efficientnet_lite0": "efflite0"}


def legacy_ckpt(bb: str) -> Path:
    return Path("checkpoints") / bb / "best.pt"


def legacy_run_id(bb: str) -> str:
    return f"legacy_{LEGACY[bb]}_s42"


def saliency_path(run_id: str, dataset: str, method: str, root: Path = PREDS_ROOT) -> Path:
    return root / run_id / f"{dataset}__saliency-{method}.parquet"


# ----------------------------------------------------------------------------- data
def load_landmarks() -> pd.DataFrame:
    lm = pd.concat([pd.read_parquet(p) for p in LANDMARKS.values()], ignore_index=True)
    if lm["image_id"].duplicated().any():
        raise ValueError("duplicate landmark image_id")
    return lm.set_index("image_id")


def load_frame(dataset: str, run_dir: Path, landmark_only: bool = False) -> pd.DataFrame:
    """Standard frame + the run's R2 predictions + landmark flags, sorted by image_id."""
    es = load_eval_set(dataset)
    f = es.frame.copy()
    preds = pd.read_parquet(run_dir / f"{dataset}.parquet",
                            columns=["image_id", "pred_fine", "pred_full", "correct_full", "conf_full_summed"])
    preds["image_id"] = preds["image_id"].astype(str)
    for c in ("pred_fine", "pred_full"):
        preds[c] = preds[c].astype(str)
    f = f.merge(preds, on="image_id", how="left", validate="1:1")
    if f["pred_full"].isna().any():
        raise ValueError(f"{dataset}: R2 predictions missing for some images in {run_dir}")
    lm = load_landmarks()
    f["has_landmarks"] = f["image_id"].isin(lm.index)
    f["anatomy"] = f["image_id"].map(lm["anatomy"])
    f["qc_flag"] = f["image_id"].map(lm["qc_flag"]).astype("boolean")
    f["lm_clipped"] = f["image_id"].map(lm["lm_clipped"]).astype("boolean")
    if landmark_only:
        f = f[f["has_landmarks"]]
    return f.sort_values("image_id").reset_index(drop=True)


class SaliencyInputs(Dataset):
    """Model input + model-space masks (sector, ui, anatomy at each femur width) + fill grey."""

    def __init__(self, frame: pd.DataFrame, lm: LoadedModel, landmarks: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)
        self.tf = lm.eval_transform()
        self.mode, self.S = lm.preprocess_mode, lm.img_size
        self.lm = landmarks

    def __len__(self) -> int:
        return len(self.frame)

    def _ms(self, m: np.ndarray) -> np.ndarray:
        return to_model_space(m, preprocess_mode=self.mode, img_size=self.S)

    def __getitem__(self, i: int) -> dict:
        row = self.frame.iloc[i]
        gray = load_grayscale(row["image_path"])
        mp = mask_path(row["image_id"], row["dataset"])
        has_r1 = mp.exists()
        if has_r1:
            masks = load_masks(mp)
            sector, ui = masks["sector_clean"], masks["ui"]
        else:
            sector, ui = np.ones_like(gray), np.zeros_like(gray)
        inside = gray[sector.astype(bool)]
        fill_grey = float(np.median(inside if inside.size else gray))
        anat = np.zeros((len(FEMUR_REL_WIDTHS), self.S, self.S), np.float32)
        if row["has_landmarks"]:
            lr = self.lm.loc[row["image_id"]]
            if (int(lr["height"]), int(lr["width"])) != gray.shape:
                raise ValueError(f"{row['image_id']}: landmark frame size differs from the image")
            for k, w in enumerate(FEMUR_REL_WIDTHS):
                if k == 0 or lr["anatomy"] == "Femur":
                    anat[k] = self._ms(anatomy_mask_for_row(lr, gray.shape, femur_rel_width=w))
            if lr["anatomy"] != "Femur":
                anat[1] = anat[2] = anat[0]
        x = self.tf(image=gray_to_rgb(gray))["image"]
        return {"x": x, "idx": i, "sector": self._ms(sector), "ui": self._ms(ui), "anat": anat,
                "fill_grey": fill_grey, "has_r1_mask": has_r1}


# ----------------------------------------------------------------------------- metrics
def target_sets(frame: pd.DataFrame) -> tuple[list[list[int]], list[list[int]]]:
    true = [COLLAPSED_MEMBERS[c] for c in frame["true_collapsed"]]
    pred = [COLLAPSED_MEMBERS[c] for c in frame["pred_full"]]
    return true, pred


def map_metrics(sal: np.ndarray, sector: np.ndarray, ui: np.ndarray, anat: np.ndarray | None,
                femur: bool) -> dict[str, float]:
    """All per-target metrics for one map; ``anat`` = (3, S, S) masks at FEMUR_REL_WIDTHS or None."""
    out = {"out_sector_frac": LM.fraction_in(sal, 1.0 - sector), "overlay_frac": LM.fraction_in(sal, ui),
           "sal_energy": float(np.maximum(sal, 0).sum())}
    if anat is None:
        return out
    widx = FEMUR_REL_WIDTHS.index(PRIMARY_FEMUR_WIDTH)
    r = LM.epg_relative(sal, anat[widx], sector)
    out.update({k: r[k] for k in ("epg", "epg_minus_area", "epg_over_area", "epg_in_sector",
                                  "epg_in_sector_minus_area")})
    out["hit"] = float(LM.pointing_hit(sal, anat[widx]))
    out["iou"] = LM.cam_box_iou(sal, anat[widx])
    if femur:
        for k, w in enumerate(FEMUR_REL_WIDTHS):
            if k != widx:
                out[f"epg_minus_area_w{int(round(w * 100)):02d}"] = LM.epg_relative(sal, anat[k])["epg_minus_area"]
    return out


def area_columns(anat: np.ndarray, sector: np.ndarray) -> dict[str, float]:
    widx = FEMUR_REL_WIDTHS.index(PRIMARY_FEMUR_WIDTH)
    r = LM.epg_relative(np.ones_like(sector), anat[widx], sector)
    return {"area_frac": r["area_frac"], "area_frac_in_sector": r["area_frac_in_sector"]}


@dataclass
class RunSpec:
    run_id: str
    dataset: str
    method: str
    frame: pd.DataFrame
    keep_maps: frozenset = frozenset()
    batch_size: int = 32
    num_workers: int = 4


def compute(lm: LoadedModel, spec: RunSpec, engine: SaliencyEngine | None = None,
            log_every: int = 0) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Per-image saliency metrics for ``spec.frame``; also the raw maps of ``spec.keep_maps`` ids."""
    frame = spec.frame.reset_index(drop=True)
    true_t, pred_t = target_sets(frame)
    landmarks = load_landmarks()
    loader = DataLoader(SaliencyInputs(frame, lm, landmarks), batch_size=spec.batch_size, shuffle=False,
                        num_workers=spec.num_workers, pin_memory=lm.device.type == "cuda")
    own = engine is None
    eng = engine or SaliencyEngine(lm.model, lm.backbone_name, spec.method)
    rows, maps_keep = [], {}
    try:
        for bi, batch in enumerate(loader):
            idx = batch["idx"].numpy()
            x = batch["x"]
            tt, pt = [true_t[i] for i in idx], [pred_t[i] for i in idx]
            head = [k for k, i in enumerate(idx) if frame.at[i, "true_collapsed"] == "Head"]
            if spec.method == "occlusion":
                fill = normalised_fill(batch["fill_grey"].numpy(), lm.normalize_mean, lm.normalize_std)
                cmaps, p0 = eng.occlusion_all(x, fill)
                m_true, m_pred = eng.combine(cmaps, tt), eng.combine(cmaps, pt)
                sub = [HEAD_SUB[int(np.argmax(p0[k, HEAD_SUB]))] for k in head]
                m_sub = eng.combine(cmaps[head], sub) if head else None
                pred_fine = p0.argmax(1)
            else:
                m_true, logits = eng.cam_maps(x, tt)
                same = all(a == b for a, b in zip(tt, pt))
                m_pred = m_true if same else eng.cam_maps(x, pt)[0]
                sub = [HEAD_SUB[int(np.argmax(logits[k, HEAD_SUB]))] for k in head]
                m_sub = eng.cam_maps(x[head], sub)[0] if head else None
                pred_fine = logits.argmax(1)
            sub_of = {k: j for j, k in enumerate(head)}
            sec, ui, anat = batch["sector"].numpy(), batch["ui"].numpy(), batch["anat"].numpy()
            for k, i in enumerate(idx):
                r = frame.iloc[i]
                has_lm = bool(r["has_landmarks"])
                a = anat[k] if has_lm else None
                femur = r["anatomy"] == "Femur"
                rec = {"image_id": r["image_id"], "pred_fine_idx_saliency": int(pred_fine[k]),
                       "fill_grey": float(batch["fill_grey"][k]), "has_r1_mask": bool(batch["has_r1_mask"][k]),
                       "out_sector_area": float(1.0 - sec[k].mean()), "ui_area": float(ui[k].mean())}
                if has_lm:
                    rec.update(area_columns(a, sec[k]))
                for name, m in (("true", m_true[k]), ("pred", m_pred[k])):
                    rec.update({f"{c}_{name}": v for c, v in map_metrics(m, sec[k], ui[k], a, femur).items()})
                if k in sub_of:
                    ms = m_sub[sub_of[k]]
                    rec["truesub_idx"] = sub[sub_of[k]]
                    rec.update({f"{c}_truesub": v for c, v in map_metrics(ms, sec[k], ui[k], a, False).items()})
                rows.append(rec)
                if r["image_id"] in spec.keep_maps:
                    maps_keep[f"{r['image_id']}|true"] = m_true[k].astype(np.float16)
                    maps_keep[f"{r['image_id']}|pred"] = m_pred[k].astype(np.float16)
                    maps_keep[f"{r['image_id']}|anat"] = (anat[k][0] > 0.5) if has_lm else np.zeros(1, bool)
                    maps_keep[f"{r['image_id']}|sector"] = sec[k] > 0.5
            if log_every and (bi + 1) % log_every == 0:
                print(f"    {spec.run_id} {spec.dataset} {spec.method}: {len(rows)}/{len(frame)}", flush=True)
    finally:
        if own:
            eng.close()
    met = pd.DataFrame(rows)
    keep = ["image_id", "patient_id", "dataset", "source_machine", "operator", "centre", "true_label",
            "true_collapsed", "pred_fine", "pred_full", "correct_full", "conf_full_summed", "has_landmarks",
            "anatomy", "qc_flag", "lm_clipped"]
    out = frame[keep].merge(met, on="image_id", how="left", validate="1:1")
    out["pred_agrees_r2"] = [CANONICAL_CLASSES[i] == p for i, p in zip(out["pred_fine_idx_saliency"], out["pred_fine"])]
    out["head_target"] = np.where(out["true_collapsed"] == "Head", "set", "")
    out["method"] = spec.method
    out["run_id"] = spec.run_id
    out["git_hash"] = git_hash()
    return out, maps_keep


def write(out: pd.DataFrame, maps: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    if maps:
        np.savez_compressed(path.with_suffix(".maps.npz"), **{k.replace("/", "__"): v for k, v in maps.items()})


# ----------------------------------------------------------------------------- subsets
def figure_subset(conv_preds: dict[str, pd.DataFrame], seed: int = SEL_SEED) -> list[str]:
    """48 test images chosen by correctness of the reference model (never by saliency values).

    FPDB-test: 4 correct + 4 incorrect per anatomy; HC18: 4 + 4; UCL: 3 + 2/3 per anatomy (16).
    Only non-QC landmark images. Shortfalls of incorrect images are filled with correct ones.
    """
    rng = np.random.default_rng(seed)
    plan = [("fpdb_test", a, 4, 4) for a in ("Head", "Abdomen", "Femur")] + [("hc18", "Head", 4, 4)]
    plan += [("ucl", "Head", 3, 3), ("ucl", "Abdomen", 3, 2), ("ucl", "Femur", 3, 2)]
    pick: list[str] = []
    for ds, anat, n_ok, n_bad in plan:
        f = conv_preds[ds]
        f = f[f["has_landmarks"] & (f["anatomy"] == anat) & ~f["qc_flag"].fillna(False).astype(bool)]
        ok = sorted(f.loc[f["correct_full"].astype(bool), "image_id"])
        bad = sorted(f.loc[~f["correct_full"].astype(bool), "image_id"])
        b = list(rng.choice(bad, size=min(n_bad, len(bad)), replace=False)) if bad else []
        g = list(rng.choice(ok, size=n_ok + n_bad - len(b), replace=False))
        pick += b + g
    return sorted(pick)


def occlusion_subset(frames: dict[str, pd.DataFrame], figure_ids: list[str], seed: int = SEL_SEED) -> dict[str, list[str]]:
    """600 landmark images: 200 per anatomy (FPDB-test), 100 HC18, 100 UCL (by anatomy share).

    Starts from the figure subset so every figure image also has an occlusion map. Non-QC only.
    """
    rng = np.random.default_rng(seed)
    want = {("fpdb_test", "Head"): 200, ("fpdb_test", "Abdomen"): 200, ("fpdb_test", "Femur"): 200,
            ("hc18", "Head"): 100}
    u = frames["ucl"]
    u = u[u["has_landmarks"] & ~u["qc_flag"].fillna(False).astype(bool)]
    share = u["anatomy"].value_counts()
    alloc = (share / share.sum() * 100).round().astype(int)
    alloc.iloc[0] += 100 - alloc.sum()
    want.update({("ucl", a): int(n) for a, n in alloc.items()})
    out: dict[str, list[str]] = {}
    fig = set(figure_ids)
    for (ds, anat), n in sorted(want.items()):
        f = frames[ds]
        f = f[f["has_landmarks"] & (f["anatomy"] == anat) & ~f["qc_flag"].fillna(False).astype(bool)]
        ids = sorted(f["image_id"])
        chosen = [i for i in ids if i in fig]
        pool = [i for i in ids if i not in fig]
        chosen += list(rng.choice(pool, size=n - len(chosen), replace=False))
        out.setdefault(ds, []).extend(chosen)
    return {k: sorted(v) for k, v in out.items()}
