"""R7 E10: rescued / new shares and the both-wrong delta, hand-computed."""
from __future__ import annotations

import sys

import pandas as pd
import pytest

sys.path.insert(0, "scripts/research")
import e10_shared_failures as E  # noqa: E402


def test_contrasts_hand_computed(monkeypatch):
    # 6 images (6 patients). B0 both-wrong on images 0,1,2; arm both-wrong on images 2,3.
    # delta = 2/6 - 3/6 = -1/6; rescued = images 0,1 of 3 = 2/3; new = image 3 of 2 = 1/2
    ids = [f"i{k}" for k in range(6)]
    b0 = pd.DataFrame({"image_id": ids, "patient_id": [f"p{k}" for k in range(6)], "both": [1, 1, 1, 0, 0, 0]})
    arm = pd.DataFrame({"image_id": ids, "patient_id": b0["patient_id"], "both": [0, 0, 1, 1, 0, 0]})
    import src.eval.stats as S
    monkeypatch.setattr(S, "B_DEFAULT", 50)
    r = E.contrasts(b0, arm)
    assert r["both_wrong_delta"]["point"] == pytest.approx(-1 / 6)
    assert r["rescued_share_of_b0_shared"]["point"] == pytest.approx(2 / 3)
    assert r["new_share_of_arm_shared"]["point"] == pytest.approx(1 / 2)


def test_both_wrong_and_ratio():
    d = pd.DataFrame({"w_convnext": [1, 1, 0, 0], "w_effv2s": [1, 0, 1, 0], "stratum": ["a"] * 4})
    d["both"] = d["w_convnext"] & d["w_effv2s"]
    assert E.both_wrong(d) == pytest.approx(0.25)
    # independence: 0.5 * 0.5 = 0.25 -> ratio 1
    assert E.ratio(d) == pytest.approx(1.0)
