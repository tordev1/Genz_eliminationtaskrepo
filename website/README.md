# Genz -- team website

A static, no-build-step site covering the team, problem/approach, EDA,
results, a live local demo, an honest report, and links. Plain HTML/CSS/JS;
no npm install required.

## Viewing the static pages

Every section except the live demo works by just opening the file:

```
website/index.html
```

double-click it, or open it in a browser directly. No server needed for
Team / Problem & Approach / EDA / Results / Report / Links.

## Running the live demo

The live demo needs a tiny local server because it runs the repo's own
`solution.detect_events()` on an uploaded video and returns real results.

```bash
# 1. From the repository root:
pip install -r requirements.txt          # opencv-python, numpy (needed by solution.py)
pip install -r website/requirements.txt  # flask (only extra dependency this website adds)

# 2. Start the demo server:
python website/demo_server.py

# 3. Open in a browser:
http://localhost:5000
```

`demo_server.py` serves `index.html` itself (so you don't need to open the
file separately), plus one API route:

- `POST /api/detect_events` -- multipart form field `video`, an `.mp4` file.
  Runs `solution.detect_events()` from the repository root and returns
  `{"events": [[start_sec, end_sec, label], ...]}`.

Limits enforced by the server (and stated on the page): `.mp4` only, up to
200MB and up to 2 minutes of footage. Longer/larger uploads are rejected with
a clear error, not silently truncated.

The demo currently visualizes **Part A (event detection) only**. A Part B
risk-curve visualization in the demo is a nice-to-have, not yet built --
the page says "risk curve visualization: coming soon."

## Public hosting -- not done yet

**This site and demo are not publicly hosted anywhere.** Everything above
runs on `localhost` only. Before the submission deadline, a team member still
needs to manually deploy this with their own account on a platform such as
Vercel, Netlify, or Hugging Face Spaces (for the static pages) and, separately,
somewhere that can run a small Python/Flask process for the live-demo API
(e.g. a Hugging Face Space with a Python runtime, Render, Railway, or similar --
static hosts like plain Netlify/Vercel cannot run the Flask backend as-is).
Do not claim this is already deployed publicly until that step is actually
done.

## Files

- `index.html` -- the whole site (Team, Problem & Approach, EDA, Results,
  Live Demo, Report, Links), single page with anchor navigation.
- `assets/style.css` -- all styling, mobile-first, one shared palette.
- `assets/demo.js` -- live-demo client logic (upload, progress, timeline,
  table). Talks to `/api/detect_events`.
- `demo_server.py` -- the Flask app described above.
- `requirements.txt` -- this folder's only extra dependency (`flask`).

## Known placeholders (see the page itself for the visible/marked TODOs)

- Team member names, roles, contributions, and profile links (3 cards).
- EDA charts and results (no sample videos exist yet; the page says so
  honestly and shows placeholder skeleton charts rather than invented numbers).
- Repository URL and `predictions_samples.json` link (both `<!-- TODO -->`
  in `index.html`'s Links section).
- Public hosting URL (none exists yet -- see above).
