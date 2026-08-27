# FetScan — UI/UX Remediation Brief

**Context:** The newsprint/editorial design system from `Newsprint_design_prompt.md` was implemented into `app_streamlit.py` and `assets/style.css` per `implementation_plan_streamlit_styling.md`. The CSS design language (fonts, colors, borders, zero-radius) is largely correct and should be **preserved**. What's broken is **information architecture, content quality, and a handful of incomplete style overrides** that let raw Streamlit chrome leak through. This brief fixes those without touching inference/rendering logic.

**Ground rule for the whole pass:** every change below is CSS + Markdown/copy + minor structural `st.markdown(unsafe_allow_html=True)` wrapper changes in `app_streamlit.py`. **Do not touch** `scripts/render_annotated_video.py`, `src/realtime/`, `src/smoothing/`, checkpoint paths, or any inference/rendering call. This is a presentation-layer pass only.

---

## 1. Sidebar is illegible and still reads as "a Streamlit sidebar"

**Root cause:** `assets/style.css` styles `.stCheckbox label` for *weight* but never forces a foreground color, so labels inherit Streamlit's default light-gray theme text against the `--muted-surface` background — this is why "Enable Grad-CAM overlay," "Tier-2a smoothing," and "Performance HUD" are barely visible in the screenshot.

**Fix:**
- In `assets/style.css`, add explicit, high-specificity color rules for every sidebar text node, not just checkboxes:
  ```css
  [data-testid="stSidebar"] * {
      color: var(--ink-black) !important;
  }
  [data-testid="stSidebar"] .stCheckbox label p,
  [data-testid="stSidebar"] .stSlider label p,
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
      color: var(--ink-black) !important;
      opacity: 1 !important;
  }
  ```
  Verify this doesn't fight with disabled-state graying (the Grad-CAM-every-N slider *should* look muted when the checkbox above it is off — keep that, just make the *enabled* state fully legible).
- Remove the gear emoji from "⚙️ Options" — replace the sidebar header with plain Raleway uppercase: `CONTROL PANEL` or `SESSION CONTROLS`, no icon.
- Restructure the sidebar into three visually separated bordered blocks (Grad-CAM / Smoothing / Display), each with a thin top rule and a small monospace section tag, e.g.:
  ```
  [ 01 ] GRAD-CAM
  [ 02 ] SMOOTHING
  [ 03 ] DISPLAY
  ```
  instead of plain `st.subheader()` calls with no visual separation — right now the three groups run together with only whitespace between them.
- The "MODEL INFORMATION" and "CLASSES" expander buttons currently carry emoji icons (`ℹ️`, `🗂️`). Replace with bracket-tag prefixes matching the mono/technical language elsewhere: `[ MODEL INFO ]`, `[ CLASSES ]`.

**Acceptance check:** every sidebar label is pure black (or `--text-secondary` only when genuinely disabled), zero emoji anywhere in the sidebar, and the three control groups are visually distinct blocks, not a continuous scroll of checkboxes.

---

## 2. The hero title says nothing and is styled wrong

**Root cause:** `_render_header()` renders "FETSCAN" in full-caps Playfair Display at a huge size with only a technical subtitle beneath it (`REAL-TIME SONOGRAPHIC PLANE ANALYSIS · 8-CLASS DEEP LEARNING SYSTEM`). A first-time visitor sees a brand name and a jargon line — no explanation of what the tool does, what problem it solves, or what "8-class" means in plain terms.

**Fix — rewrite the masthead as three tiers, not two:**
1. **Wordmark** (keep Playfair, but stop all-caps — brand names in full caps read as shouting, not as a name): `FetScan`, sentence case as designed originally, moderately large (this is currently oversized relative to its information content — reduce ~20–25%).
2. **One-sentence positioning line** directly under the wordmark, in Raleway, larger than the current caption size (this replaces the buried "Upload a session scan →..." line entirely — see §3):
   > *An AI assistant that watches a fetal ultrasound scan in real time and tells the sonographer which standard anatomical plane is on screen — and how confident it is.*
3. **Technical strapline** below that, small monospace, exactly as-is: `REAL-TIME SONOGRAPHIC PLANE ANALYSIS · 8-CLASS DEEP LEARNING SYSTEM`.

Then, immediately below the masthead (before any marketing prose), add a **compact 2–3 sentence problem statement** — not buried in a two-column essay later in the page, but right up front, plain Raleway body text at readable size:
> During a fetal anatomy scan, a sonographer must correctly identify and capture ~7 standard reference planes (brain, abdomen, femur, thorax, cervix) to complete a valid exam. Missed or mislabeled planes are a leading source of scan-quality variability between operators. FetScan classifies the current frame in real time and holds a stable label even as the probe moves, so the operator gets continuous feedback on what plane they're looking at.

This does the job the current hidden caption line was trying to do, but as primary content instead of an ignorable footnote.

**Acceptance check:** a reader who has never seen this tool before understands, within 10 seconds of landing on the page and without scrolling, (a) what it's called, (b) what problem it solves, (c) roughly how.

---

## 3. The "Upload a session scan →..." line: kill it in its current form

**Root cause:** This line is doing real explanatory work (what happens when you upload) but is styled as a `st.caption()` — smallest, grayest, lowest-priority text on the page, directly under a hard rule that visually seals it off from the title above. It is, by design, unreadable-in-practice. This is the single biggest "nobody will read this" issue you flagged, and it's correct.

**Fix:** Split its two jobs and give each a proper home:
- **The plain-English "what happens" explanation** merges into the one-sentence positioning line in §2 — don't repeat it separately.
- **The "how it works" mechanics** (upload → inference → smoothing → annotated output) becomes a **4-step visual strip**, not a sentence. Use four compact bordered cells in a row (collapsing to a stacked list on narrow viewports per the accessibility requirement), each with a large monospace step number and a short label:
  ```
  01               02                03                  04
  UPLOAD    →      INFERENCE  →      SMOOTHING    →      ANNOTATED OUTPUT
  clip in           per-frame          EMA + majority       labeled, stable
  MP4/AVI           classification     vote filter          video out
  ```
  This is the "How It Works" module the original design brief explicitly asked for (`[layout]` section: "video input and processing area" should be prioritized #2) and it currently doesn't exist as a distinct element at all.
- **The technical credentials line** (`Powered by convnext_tiny... · Macro-F1 0.8927 · EMA + Tier-2A...`) stays small and monospace — that's correctly deprioritized info, keep it as a footnote-style line, but move it to sit near the Model Information panel/sidebar rather than immediately under the title, since it's not orienting content, it's a spec sheet detail.

**Acceptance check:** the four-step strip is visually one of the first things a user sees (top of page, above the marketing prose), and no single sentence on the page is trying to carry both "why this exists" and "how it technically works."

---

## 4. Information order is backwards for a functional tool

**Root cause:** Current top-to-bottom order is: Title → tiny hidden explanation → two paragraphs of marketing prose with large decorative images → (scroll) → actual upload/output UI → (scroll) → examples. The thing the tool *does* is the least visible part of the page. The original design brief's own priority list puts "video input and processing area" and "annotated output" at positions #2 and #3, ahead of general background — this was not followed.

**Fix — reorder the page top-to-bottom to:**
1. Masthead + positioning line + problem statement (§2)
2. How-it-works step strip (§3)
3. **The functional module: upload panel + analysis output, side by side** — moved up to immediately follow #2, not buried after the "Precision AI for Diagnostic Obstetrics" essay
4. Clinical/technical background ("Precision AI for Diagnostic Obstetrics" / "Temporal Stability & Interpretability") — this can stay largely as-is content-wise but now serves as supporting/secondary material below the fold, and should be visually treated as secondary (narrower column, more compact) rather than the first thing after the title
5. Model information / metrics (can also live partly in an expander, which it partially already does — keep that)
6. Example clips
7. Footer

This is purely a reordering of existing `st.` calls in `main()` in `app_streamlit.py` plus corresponding CSS class adjustments — no new functionality.

**Acceptance check:** the upload control is visible without scrolling past marketing copy on a standard 1440×900 viewport.

---

## 5. Two images look AI-generated and cheap, and one fabricates fake patient data

**Root cause:** The "clinical diagnostic workstation" photo (Fig. 01) is a generic stock-style image and reads fine. The second image (Fig. 02) is a comic-panel-style composite with fabricated overlay text — `PATIENT: J. DOE (GA: 20w 4d)`, `BPM: 146`, `ECHO: 3.4dB`, fake waveform charts — none of which correspond to anything the model actually outputs. This is the "generated in a very crude way" problem you flagged, and it's worse than just crude: it invents clinical-looking numbers that have no relationship to FetScan's real output, which is actively misleading in a medical-adjacent context even as a mockup.

**Fix:**
- **Remove the fabricated-data image entirely.** Do not replace it with another AI-generated "hologram" style image — this whole visual genre (sci-fi HUD overlays, fake vitals) fights the actual newsprint/clinical-journal aesthetic the design brief calls for, which explicitly says "Do NOT introduce... excessive decorative medical icons" and asks for restraint.
- Replace it with one of:
  - **A real screenshot from the actual pipeline** — an annotated output frame (label + confidence + Grad-CAM overlay) pulled from an actual rendered clip. This is real content, on-brand, and truthful.
  - **A simple line-diagram of the pipeline architecture** (capture → preprocess → backbone → classification head → temporal smoothing → display), drawn as clean technical line art in the same black/off-white/red palette — no photorealism, no fake data, matches "technical drafting paper" language from the design brief's texture section.
- Keep Fig. 01 (the workstation photo) if it fits the intake panel context, but caption it precisely, e.g. `FIG. 01 — ROUTINE SONOGRAPHIC EXAMINATION` rather than implying it's *this* system in clinical use (it isn't).

**Acceptance check:** no image on the page displays numbers, patient identifiers, or vitals that the actual model does not produce.

---

## 6. Two empty black-bordered boxes render with no content

**Root cause:** Visible in the second screenshot, directly above "Intake Panel" and "Analysis Output" — two hard-bordered rectangular divs render completely empty. This is very likely a stray `<div class="fpc-...">...</div>` pairing bug in `app_streamlit.py` (e.g., an opening div meant to wrap a caption/rule element where the intended inner content call was never placed inside it, or a leftover decorative element from CSS iteration that was supposed to be removed).

**Fix:**
- Locate the corresponding markup in `_render_upload_col()` / `_render_output_col()` (or wherever the `.fpc-intake-panel` / `.fpc-analysis-panel` wrappers open) and either:
  - (a) delete the empty div if it's dead code, or
  - (b) if it was intended as a section-label bar, fill it with the actual label content — e.g. a mono technical tag like `INPUT / 01` and `OUTPUT / 02` — so it does the job the design brief describes ("technical figure numbering... should be part of the visual language rather than hidden").
- Grep the file for any `st.markdown('<div class="fpc-...">', unsafe_allow_html=True)` calls that aren't immediately followed by real content before their matching close, since that's the most likely bug pattern.

**Acceptance check:** no bordered box on the page is visually empty; every border either contains content or is removed.

---

## 7. The `st.info()` box is raw, unstyled Streamlit blue

**Root cause:** `assets/style.css` targets `.stAlert` generally but the info-state background color override isn't taking effect — the "Upload a diagnostic scan and click ▶ Process Video to begin analysis." message renders in Streamlit's default blue notification chrome, which is the single element on the page that completely breaks the monochrome/red design system.

**Fix:** Add explicit overrides per alert *kind*, since Streamlit applies different `data-baseweb` / class combinations per severity:
```css
[data-testid="stAlert"] {
    background-color: var(--bg-paper) !important;
    border: var(--border-thin) !important;
    border-left: 6px solid var(--ink-black) !important;
    color: var(--ink-black) !important;
}
[data-testid="stAlert"] svg { display: none; } /* remove default colored icon, or recolor to ink-black */
```
Reserve `--accent-red` for the left border only on genuine error/warning states, keep info/success at black-left-border to stay monochrome per the design brief's color rules ("the application should remain visually convincing even if the accent color were removed completely").

Also reconsider whether this message needs to be an `st.info()` box at all — a plain bordered placeholder panel with mono caption text (`AWAITING INPUT`) inside the empty output module (see §4) may fit the aesthetic better than a notification-style alert for what is really just an empty-state placeholder, not a system notification.

**Acceptance check:** zero blue, green, or default-themed Streamlit alert colors appear anywhere on the page.

---

## 8. Remove every emoji from headings, labels, and buttons

**Root cause:** Emojis appear in: sidebar header (`⚙️ Options`), expander buttons (`ℹ️ MODEL INFORMATION`, `🗂️ CLASSES`), section header (`📥 Intake Panel`), examples header (`🎬 Example Clips`). This directly conflicts with the design brief's own instruction to avoid "excessive decorative medical icons" and undermines the "precision instrument, not a marketing website" tone.

**Fix:** Search `app_streamlit.py` for every emoji character in `st.header()`, `st.subheader()`, `st.markdown()`, `st.expander()`, and button labels, and remove them, replacing where a visual marker is genuinely useful with:
- A monospace bracket tag (`[ MODEL INFO ]`), or
- Nothing at all — plain uppercase Raleway headers carry sufficient weight in this system without an icon crutch.

**Acceptance check:** grep the file for common emoji Unicode ranges returns zero matches in any user-facing string.

---

## 9. Example clip captions don't match the rest of the typographic system

**Root cause:** The design brief specifically calls for figure-style captions (`FIG. 01 — TRANS-CEREBELLAR`), but the implemented captions are plain sentence-case filenames (`Brain Trans cerebellum clip01`) with no figure numbering or visual treatment distinct from body text.

**Fix:** In `_render_examples()`, reformat the caption string to the mono figure-tag style already used elsewhere on the page (`FIG 01. CLINICAL DIAGNOSTIC WORKSTATION INTEGRATION` is already correctly styled for the hero images — apply the identical CSS class to the example-clip captions):
```python
caption = f"FIG. {i+1:02d} — {clip.stem.replace('_', ' ').upper()}"
```
and give it the same `.fpc-figure-caption` class already defined in the CSS for the hero images, rather than a bare `st.caption()`.

**Acceptance check:** example-clip captions are visually and structurally identical in treatment to the Fig. 01 / Fig. 02 captions higher on the page.

---

## 10. Miscellaneous polish items

- **The "PROCESS VIDEO" button looks disabled/inactive** even in states where it should read as an inviting primary action. Confirm the enabled-state button has strong black-fill/white-text contrast per the design brief's button spec, and that Streamlit's native `:disabled` opacity isn't leaking through when the button *is* actually enabled (upload present).
- **The blank black top bar** (Streamlit's default header) currently contributes empty vertical space with zero content. Either (a) hide it via `header[data-testid="stHeader"] { display: none; }` and rely entirely on the custom masthead below, or (b) repurpose it as the system status ticker the original design brief suggested (`FRAME 00421 · 24 FPS INPUT · ... · STABLE · CONFIDENCE 94.2%`) if that's feasible to populate from real session state — do **not** leave it as dead space either way.
- **Copy length overall**: the two-column "Precision AI for Diagnostic Obstetrics" / "Temporal Stability & Interpretability" essay is dense prose that most users will skim past. Cut each paragraph roughly in half, lead with the single most important sentence, and consider converting the second half of each into 2–3 short bullet-style facts instead of full paragraphs — this fits the "high information density... editorial hierarchy" goal from the design brief better than continuous prose blocks.
- **Verify keyboard focus states** exist on all interactive elements (buttons, checkboxes, expanders) — the design brief's accessibility section requires visible focus states and none are evident in the CSS reviewed.
- **Verify responsive collapse**: confirm the two-column masthead/background sections and the three-clip example grid collapse to single-column on narrow viewports rather than compressing.

---

## Suggested execution order for the agent

1. Fix the two structural bugs first (empty boxes §6, blue alert §7) — these are pure CSS/markup bugs, fastest wins, zero content-writing needed.
2. Do the emoji removal pass (§8) — mechanical, low-risk, touches many small strings.
3. Rewrite the masthead + problem statement + how-it-works strip (§2, §3) — this is the highest-impact content change.
4. Reorder the page sections (§4).
5. Swap out the fabricated-data image (§5).
6. Sidebar contrast + restructuring (§1).
7. Example clip captions (§9) and remaining polish (§10).

After each step, re-render the app and re-screenshot to confirm against the acceptance checks listed above before moving to the next.
