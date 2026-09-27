# Genz -- team website

A static, no-build-step site covering the team, problem/approach, EDA,
results, a live demo, an honest report, and links. Plain HTML/CSS/JS; no
npm install required.

**Live**: https://genzwiut.com -- the real site and demo, publicly hosted.

## Viewing the static pages

Every section except the live demo also works by just opening the file
directly, no server needed:

```
website/index.html
```

## Running the live demo yourself (locally)

```bash
# 1. From the repository root:
pip install -r requirements.txt          # opencv-python-headless, numpy (needed by solution.py)
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

## Public hosting -- how genzwiut.com actually runs it

Deployed via cPanel's **Setup Python App** feature (Phusion Passenger WSGI),
Python 3.9, on the team's own GoDaddy-hosted domain:

- `passenger_wsgi.py` (repo root when deployed there) exposes
  `website.demo_server.app` as `application`, the WSGI callable Passenger
  looks for -- it never triggers `demo_server.py`'s own
  `if __name__ == "__main__": app.run(...)` block, since Passenger imports
  the module rather than executing it as a script.
- Dependencies (`opencv-python-headless`, `numpy`, `flask`) install cleanly
  into cPanel's managed virtualenv via its "Run Pip Install" button --
  headless OpenCV needs no system-level libraries root access would be
  required for, which is exactly why it was the right choice here over
  plain `opencv-python`.
- Verified end to end against the live URL: homepage and static assets
  serve, and a real `.mp4` posted to `/api/detect_events` returns a valid
  `{"events": [...]}` response computed by the actual pipeline.

**Deployment package**: the exact file set cPanel's app root needs
(`passenger_wsgi.py`, `requirements.txt`, `solution.py`, `src/`, `website/`,
`weights/`) is assembled ad hoc for upload via cPanel's File Manager -- it's
not a tracked folder in this repo, since it's just a repackaging of files
that already live here for a specific host's directory layout.

### Alternative: Docker (Render, Hugging Face Spaces, etc.)

`Dockerfile` in this folder + `../render.yaml` at the repo root are a
ready, tested alternative deploy path (build context = repo root,
`CMD ["python", "website/demo_server.py"]`, reads `PORT` from the
environment) -- useful as a backup or for a teammate who'd rather not touch
cPanel. Render's free web services don't require payment-method
verification, unlike Hugging Face Spaces' Docker/Gradio SDKs on newer
accounts (Static-only Spaces are free but can't run a Python backend at
all). Not currently deployed anywhere; genzwiut.com is the live instance.

## Files

- `index.html` -- the whole site (Team, Problem & Approach, EDA, Results,
  Live Demo, Report, Links), single page with anchor navigation.
- `assets/style.css` -- all styling, mobile-first, one shared palette.
- `assets/demo.js` -- live-demo client logic (upload, progress, timeline,
  table). Talks to `/api/detect_events`.
- `demo_server.py` -- the Flask app described above.
- `requirements.txt` -- this folder's only extra dependency (`flask`).
- `Dockerfile` -- alternative Docker-based deploy path, see above.

## Known placeholders (see the page itself for the visible/marked TODOs)

- Team member names, roles, contributions, and profile links (3 cards).
- EDA charts and results (no sample videos exist yet; the page says so
  honestly and shows placeholder skeleton charts rather than invented numbers).
- `predictions_samples.json` link (`<!-- TODO -->` in `index.html`'s Links
  section) -- not generated yet, needs real sample videos first.
