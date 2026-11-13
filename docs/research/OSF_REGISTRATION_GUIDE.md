# How to lodge the R5 pre-registration on OSF (Gate G4)

**Who this is for:** the study PI, doing this once. It takes about 30–45 minutes, mostly copying and pasting.
**Why it matters:** the R5 training matrix must not start until the analysis plan is time-stamped publicly. Otherwise the claim "pre-registered" does not hold. `scripts/research/r5_run_matrix.py` refuses to run until step 6 is done.

The OSF help pages describe the process at <https://help.osf.io/article/158-create-a-preregistration>. OSF is changing its interface ("Project Transition"), so button names may differ slightly from what is written here. The steps themselves do not change.

---

## Before you start

- **Account.** You need a free OSF account (osf.io → *Sign Up*). Use the PI's institutional email.
- **Have these open:**
  - `docs/research/PREREGISTRATION.md`, the document you are registering.
  - `docs/research/GATE_G4.md`: the line `PREREGISTRATION commit: <hash>` gives the git commit to quote.
- **Optional:** a PDF of PREREGISTRATION.md, which reviewers find easier to read. Open the file in VS Code's Markdown preview and print to PDF.
- **Fees.** None were stated on the OSF help pages we read, and registering on OSF is normally free. Check if your institution says otherwise.

## Step 1: create a project to hold the files (recommended)

1. On osf.io, click **Create new project**.
2. Title it something like *Anatomy or Acquisition? (R5 anatomy-focused training: pre-registration)*. Keep it **private** for now.
3. In the project, open **Files** and upload `PREREGISTRATION.md` (and the PDF if you made one).

A registration started from a project archives that project's files with it, up to 5 GB. Files cannot be added or changed after you submit, so upload everything now.

> If you skip the project, start from **My OSF → My Registrations → Add a Registration**. Answer **No** to "Do you have content for registration in an existing OSF project?". You then attach the document inside the form. A project is still created for you after submission.

## Step 2: start the registration

1. In the project, open the **Registrations** tab and click **Add a Registration**.
2. Choose the template **OSF Preregistration**. OSF describes it as the "standard, comprehensive, and general purpose preregistration form".
3. Click **Create Draft**. OSF emails you a link to the draft, and drafts are listed under My Registrations → Drafts. You can stop and come back later.

## Step 3: fill in the form

The template asks the standard preregistration questions. You do not need to write anything new: every answer is already in PREREGISTRATION.md. For each field, paste the matching section below, followed by *"Full text: attached PREREGISTRATION.md, §n (git commit `<hash>`)."*

| The form asks about… | Paste from PREREGISTRATION.md |
|---|---|
| Title, authors, description | Title block and §1 Background |
| Hypotheses | §2: P4a is the primary hypothesis; P4b, P4c and the confirmatory P3b |
| Study type and design (no human participants, secondary analysis of public datasets) | §3 Data and §4 Arms, backbones, seeds and recipe |
| Existing data | §9 What was already known at registration (see the note below) |
| Data collection, sample size and its rationale | §3 Data: datasets, image counts, frozen splits and their SHA-256 hashes |
| Stopping rule | §7 Stopping and re-run rules |
| Variables / outcome measures | §5: the primary endpoint is pooled external full-argmax balanced accuracy |
| Statistical models, inference criteria, multiplicity | §5: paired cluster bootstrap, Holm–Bonferroni, success rule, non-inferiority margin −0.02 |
| Data exclusion, missing data | §3 exclusions (QC-flagged landmarks, failed sector segmentation) and §7 |
| Exploratory analyses | §6 Secondary and exploratory analyses |
| Other | §8 Deviations from the original plan, decided before registration |

**Note on "Existing data".** The images are public and were already analysed with the *legacy* models in R1–R4. The R5 models, and their predictions, do not exist yet. The honest answer is the option closest to *"registration prior to analysis of the data"*. In the explanation box, paste §9, which lists what we already knew. Do not tick an option claiming that nobody has seen the data. This choice is the PI's; this is our recommendation.

**Mention the commit.** Put the commit hash from `GATE_G4.md` in the description, or under *Other*, as follows: "The analysis code that will run is fixed at git commit `<hash>`. The primary-analysis script records its own SHA-256 in its output."

## Step 4: choose when it becomes public

After **Register**, then **Continue**, OSF asks about privacy:

- **Make registration public immediately.** This is the simplest option, and only public registrations get a DOI.
- **Enter registration into embargo.** The registration stays private until a date you choose, at most 4 years ahead, and then becomes public automatically. The time-stamp still counts from submission. Choose this if you do not want the plan visible before the paper is submitted.

Either option satisfies Gate G4.

## Step 5: submit and wait for approval

- After you submit, project admins are emailed. The registration shows **"Pending registration approval"**.
- It is approved automatically after 48 hours, or straight away if every admin approves. One admin rejecting it sends it back to a draft.
- **After submission nothing can be edited.** Changes go through a formal *Update*, which needs a written justification. Check the attached file and the hash before you submit.

## Step 6: record the link and unblock the matrix

1. When the registration exists, copy its address from the browser. It looks like `https://osf.io/abcde/`. Once public, the registration also has a DOI (`10.17605/OSF.IO/ABCDE`, shown under *Metadata*). Either works.
2. In `docs/research/GATE_G4.md`, replace the placeholder line `OSF registration: (pending …)` with exactly one line:

   ```
   OSF: https://osf.io/abcde/
   ```

3. Commit that change, or just tell Claude the link and it will add it and commit.
4. Check the gate (it should print a list of runs, not an error):

   ```
   D:\miniconda\envs\fetalplane\python.exe scripts/research/r5_run_matrix.py --dry-run
   ```

5. Tell Claude to continue. The matrix (24 runs plus 1 sensitivity run, about 15–20 GPU-hours) then starts. Evaluation, the pre-registered primary analysis and the final RESULTS.md follow.

## If something goes wrong

| Problem | What to do |
|---|---|
| "Gate G4 not passed: no 'OSF: https://...' line" | The line must start at the beginning of the line with `OSF: http` and contain no other text. |
| "PREREGISTRATION commit … is not an ancestor of HEAD" | Do not rewrite git history. Ask Claude to check. |
| You found a mistake after submitting | Do not delete or re-register. Use OSF's *Update* with a justification, and log it as a deviation in `RESEARCH_LOG.md`. |
| You would rather not use OSF | Say so explicitly. Claude will then record the decision in `GATE_G4.md` and `RESEARCH_LOG.md`, and RESULTS.md will state that R5 was not externally registered. Any alternative registry, such as AsPredicted, needs the same `OSF:`-style line. Ask Claude to adjust the gate. |
