# 07 — Stretch Goals & Future Roadmap

None of this blocks the v1 deliverable (Phases in files 01-06). Pursue only after the primary system is working and evaluated. Ordered roughly by value-to-effort ratio.

---

## Stretch Goal 1 — Detection-informed multi-task head (highest value, was our preferred design if data allowed)

Per the FAUSP-NET pattern: add a structure-detection head alongside the plane classifier, so the system can say *"Trans-thalamic plane — cavum septum pellucidum ✓, thalami ✓, midline falx not clearly visible"* instead of just a bare confidence score. This is what actually makes the tool clinically actionable rather than a black box.

**What's needed:**
- Structure-level bounding-box annotations. The UCL/HC18 dataset's landmark annotations (BPD/OFD, TAD/APAD, FL points) are a partial starting point for head/abdomen/femur — landmarks aren't boxes, but a box could be derived around each landmark cluster with a reasonable margin as a first pass.
- For thorax/cervix/other brain sub-planes, no existing annotation source was identified during planning — would require either manual annotation of a subset (time-intensive) or finding another public dataset with structure-level labels for these specific planes (worth a fresh search when this stretch goal is picked up, literature moves fast in this space).
- Architecture: shared backbone, two heads — a lightweight detection head (single-scale is probably sufficient given our anatomies are usually large in-frame, unlike FAUSP-NET's small structures like IVC) and the existing classification head, with the plane label optionally derivable from which structures are jointly detected (á la FAUSP-NET) rather than a separate softmax.

## Stretch Goal 2 — ONNX / TensorRT export and quantization

Prove the model *could* run on constrained hardware even though our deployment target didn't require it:
- Export the (re-parameterized, if RepVGG) model to ONNX, verify numerical parity with the PyTorch model on a validation batch.
- Run through `onnxruntime-gpu` and compare latency against native PyTorch.
- Try INT8 post-training quantization via TensorRT (requires an NVIDIA toolkit setup pass) and measure the accuracy/latency trade-off — this is the artifact that would let this system's design story extend credibly to an edge/Jetson deployment target, even without actually owning that hardware.

## Stretch Goal 3 — Tier-2 learned temporal module

Only relevant if Tier-1 smoothing (file 05) proved empirically insufficient. A small causal GRU/1D-conv over buffered per-frame backbone embeddings, trained on sequence data with plane-transition labels — needs either the IUGC video re-purposed purely for its motion characteristics (still not its labels) with synthetic plane-transition labels overlaid, or genuinely new sequence-labeled data.

## Stretch Goal 4 — Web UI polish (Streamlit/Gradio)

Once the `cv2.imshow` core loop is solid (file 05, Part B), wrap it in a small web app for easier sharing/demoing — video upload, live annotated playback, downloadable stability-metric report per clip.

## Stretch Goal 5 — Domain-specific self-supervised pretraining, done properly

If Phase 4's pretraining ablation (file 04, Step 2) found no usable public checkpoint, this becomes a real project: run SimCLR or a DINOv2-style self-supervised pass over the union of all unlabeled/labeled ultrasound frames we have access to (FETAL_PLANES_DB + UCL + IUGC frames, ignoring labels), producing our own domain-pretrained backbone checkpoint, then fine-tune on the labeled plane-classification task. This is a legitimate, portfolio-worthy sub-project on its own, referencing the label-efficient learning literature surfaced during planning (SimCLR for fetal planes, federated contrastive learning, DINOv2 fetal-US foundation models).

## Stretch Goal 6 — Second clinical track (intrapartum monitoring)

Explicitly deprioritized during scoping (see `00_PROJECT_OVERVIEW.md` §2) but not ruled out forever — if the primary system is complete and working well, and there's appetite to demonstrate the architecture's reusability across a second real clinical task, the IUGC dataset (already downloaded per file 02) is sitting there ready to train a second classification head + reuse the entire real-time serving pipeline (file 05, Part B) for PS/FH visibility classification and AoP/HSD measurement. Treat this as "build the second product," not "extend the first."

---

## Notes for whoever picks this up

Each stretch goal above assumes files 01-06 are fully complete, evaluated, and documented first (`EVAL_REPORT.md` exists and the primary system demonstrably works end-to-end on video). Don't let stretch-goal scope creep delay getting the core v1 system working — that's the actual deliverable.
