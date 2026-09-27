# Independent review brief for Codex

You are being asked for a genuinely independent second opinion on a hackathon
submission that Claude (Anthropic) built and audited. **Do not defer to
anything below as settled fact** — every claim here should be treated as "a
prior AI's account of its own work," not ground truth. Verify by reading the
actual files and running the actual commands yourself. Where you disagree
with something stated here, say so explicitly and explain why — the entire
point of asking a second, differently-trained model is to catch what the
first one is systematically blind to (including confident-sounding wrong
claims), so agreement should be earned, not assumed. If you find nothing
wrong after actually checking, say that plainly too — "verified, no issue
found" is a useful and different claim from "not checked."

## Exact locations (verify these paths exist before trusting anything about them)

- **Local repo root**: `C:\Users\Sanjar\Desktop\Genz_eliminationtaskrepo`
- **GitHub (public)**: https://github.com/tordev1/Genz_eliminationtaskrepo
- **Live public demo**: https://genzwiut.com (deployed on the user's own
  domain via cPanel's Python App / Passenger WSGI, Python 3.9)
- **Official hackathon spec PDF** (ground truth for every rule — read this
  yourself, don't trust any paraphrase of it, including this document's):
  `C:\Users\Sanjar\Downloads\WIUT Hackathon _ CV Track Elimination Task.pdf`
- **Official starter kit** (the exact files the repo's `run_submission.py`
  and `evaluate.py` claim to be byte-identical copies of — diff them
  yourself): `C:\Users\Sanjar\Downloads\wiut_cv_scripts.zip`
- **Sample video links**: `C:\Users\Sanjar\Downloads\Videos.pdf` (4 Google
  Drive links; real footage from the actual camera)
- **Real sample videos, as they land**: `D:\GenzSamples\` (this is also
  reachable as `samples\` inside the repo via a Windows directory junction
  — `dir` both paths and confirm they show the same files before assuming
  either one is stale). At the time of writing, only one file
  (`C3905.MP4`, ~127.6s, 3840x2160, ~29.97fps) had finished downloading;
  the other three may or may not be complete by the time you read this —
  check `D:\GenzSamples\` directly rather than trusting this number.
- **In-repo handoff doc** (written by Claude for exactly this purpose —
  read it, but audit it, don't summarize it back as your review):
  `C:\Users\Sanjar\Desktop\Genz_eliminationtaskrepo\HANDOFF.md`
- **Technical report** (what worked / didn't / next steps, Claude's own
  account): `C:\Users\Sanjar\Desktop\Genz_eliminationtaskrepo\docs\REPORT.md`

## What this project is (one paragraph, verify against the PDF above)

A submission for a hackathon computer-vision track: given a fixed CCTV-style
road camera, detect 14 kinds of traffic events as time segments
(`solution.detect_events`, mandatory) and, as a bonus, output a causal
per-frame accident-risk score (`solution.RiskEstimator`, optional). Scored
by an organizer-supplied, must-be-unmodified `evaluate.py`/`run_submission.py`
pair against a hidden test set from the same camera. The submitted approach
is 100% classical computer vision — OpenCV background subtraction (MOG2) +
a hand-written greedy-IoU tracker + hand-written rules per class — no
trained model, no GPU requirement, explicitly one of the challenge's
allowed approaches ("a detector plus tracker with hand-written rules").

## What Claude says it already did (verify, don't assume)

1. Built the pipeline from the pasted spec before the real starter kit was
   available, then replaced the harness/evaluator with the real,
   organizer-supplied files once they arrived (claim: byte-identical —
   **diff them yourself**).
2. Ran an independent "Opus-tier" audit agent against an earlier state of
   the repo; claims 4 real bugs were found and fixed (weights-path
   resolution, an `accident`-class false-positive source from
   unknown-blob overlaps, overstated calibration docs, 3 perf issues) and
   that 2 other findings (a multiprocessing deadlock, an AP tie-handling
   bug) were against files since deleted and no longer apply. **This is
   exactly the kind of self-reported audit result that benefits most from
   a second, skeptical read** — re-check the specific commits
   (`git log`) rather than the prose summary.
3. Added a 31-case `pytest` suite and a `tools/preflight.py` local
   pre-submission gate; claims both are green.
4. Deployed the live demo (see URL above) and claims to have verified it
   end-to-end with real HTTP requests (homepage, static assets, a real
   video POST to `/api/detect_events`).
5. Verified (by grep, not exhaustively) that `solution.py` and everything
   under `src/` has zero network-related imports/calls, and that the
   official `run_submission.py`/`evaluate.py` only import stdlib + `cv2`.
6. **Has not yet**: annotated any real sample video into ground truth,
   tuned any threshold against real data, run the actual `detect_events()`
   pipeline against real footage and measured real wall-clock time (the
   one video downloaded so far, `C3905.MP4`, has not yet been run through
   the pipeline with timing measured — if you do this, that data point is
   new, not a re-check of an existing claim).
7. Six of the 14 event classes are admittedly unimplemented
   (`red_light`, `stop_line`, `illegal_u_turn`, `illegal_turn`,
   `solid_line_crossing`, `fire_smoke`) — Claude's stated reasoning is
   that guessing without calibration data would only add false positives.
   Independently judge whether that reasoning holds, and whether the
   6-class gap is documented honestly everywhere (README, website, report)
   rather than glossed over anywhere.

## Specifically worth your independent attention

These are either things the first audit didn't have the right files to
check, or judgment calls worth a second opinion on:

1. **Re-derive the Part A / Part B scoring algorithms directly from the
   spec PDF** and compare against `evaluate.py` (the file in the repo,
   which should be byte-identical to the one in `wiut_cv_scripts.zip` —
   confirm that first). Don't trust that "it's the official file" means
   "it's definitely correct" — the organizers could have shipped a bug
   too; your job is to check the actual scoring logic against the actual
   spec text, independent of who wrote which file.
2. **Timing at real scale.** Run
   `python solution.py`-style timing (see `solution.detect_events`) or use
   `run_submission.py` against whatever is in `D:\GenzSamples\` and
   measure real wall-clock time vs. the 3x-duration budget the spec
   requires. This has never actually been measured against real footage
   as of this writing — everything measured so far used short synthetic
   test clips (~12-20s). A 4K, ~2-minute real clip is the first real data
   point; if you get to it before Claude does, that's genuinely new
   information, not a re-check.
3. **The `accident`/`wrong_way`/`congestion` rules against a real, busy,
   multi-directional intersection.** `C3905.MP4` (once you look at it) is
   a fixed elevated view of a multi-lane intersection with crossing
   traffic in several legal directions, not a simple one-way road. Check
   `src/road_model.py` and `src/rules.py`'s `wrong_way` logic
   specifically: it computes one single "dominant flow direction" for the
   entire scene. At a real intersection with legitimate cross-traffic and
   turns, that assumption may be wrong and could cause false `wrong_way`
   positives on any vehicle going a legal-but-different direction than
   whatever the single global dominant cluster happens to be. Determine
   whether this is a real bug worth flagging before real annotation work
   builds on top of the current rule set.
4. **Determinism, causality, and the format validator** — re-verify
   independently rather than trusting the first audit's clean bill of
   health, since that audit ran against an earlier commit.
5. **The website and repo README/report for any claim that doesn't match
   what the code actually does** — this is the exact failure mode the
   first audit caught once already (see point 2 above); check whether it
   recurred anywhere after subsequent commits.

## How to verify things yourself, concretely

```bash
cd "C:\Users\Sanjar\Desktop\Genz_eliminationtaskrepo"
git log --oneline                          # full commit history, read the actual diffs, not just messages
git show <commit>                          # for any commit whose message you want to verify against its diff
pip install -r requirements.txt -r requirements-dev.txt
pytest tests/ -v                           # 31 cases, claimed green
python tools/preflight.py --videos samples --team Genz   # needs at least one .mp4 in samples/ (junction to D:\GenzSamples)
python evaluate.py --pred <a predictions.json> --validate-only
```

Diff the starter kit directly:
```bash
# unzip wiut_cv_scripts.zip somewhere, then:
diff "<extracted>/wiut_cv_scripts/run_submission.py" run_submission.py
diff "<extracted>/wiut_cv_scripts/evaluate.py" evaluate.py
```

## What to report back

A findings list, most-severe first, each with: file:line, what you checked,
what you found, and whether it's a real defect or a non-issue. Explicitly
include a section for things you checked and found **no** problem with —
that's useful signal too, not just a list of complaints. If you and the
prior Claude audit disagree about something, say so directly rather than
softening it — that disagreement is the actual value of getting a second
model's opinion.
