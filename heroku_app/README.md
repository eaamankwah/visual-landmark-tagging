# Landmark Snap — Voila app + Heroku deployment

A standalone [Voila](https://voila.readthedocs.io/) app that serves the trained
landmark classifier as a simple web page: upload a photo, get the top-5
predicted landmarks with a styled probability bar chart.

This folder is self-contained — it does **not** depend on the `src/` package
from the training project. The exported TorchScript model
(`checkpoints/transfer_exported.pt`) already bundles the preprocessing
transforms, trained weights, and class names.

## Files

| File | Purpose |
|---|---|
| `app.ipynb` | The Voila app itself (styled UI, upload + URL-based classification) |
| `checkpoints/transfer_exported.pt` | **You must add this** — copy it from Part 2 (or Part 1) of the training project after running the export cell |
| `requirements.txt` | Python dependencies (uses PyTorch's CPU wheel index, since GPU wheels are far too large for Heroku's slug size limit) |
| `Procfile` | Tells Heroku how to start the app (`voila ...`) |
| `.python-version` | Pins the Python version Heroku's buildpack uses (Heroku deprecated `runtime.txt` in favor of this) |
| `.slugignore` | Keeps caches/checkpoints out of the deployed slug where not needed |
| `app.json` | Optional metadata, lets you deploy via a "Deploy to Heroku" button if you push this repo to GitHub |

## 1. Run it locally first

```bash
cd heroku_app
pip install -r requirements.txt
# Copy your trained, exported model here:
cp /path/to/your/proj/checkpoints/transfer_exported.pt checkpoints/
voila app.ipynb --show_tracebacks=True
```

This opens the app in your browser at `http://localhost:8866`. Confirm the
upload + classify flow works before deploying.

## 2. Deploy to Heroku

Make sure you have the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
installed and are logged in (`heroku login`).

```bash
cd heroku_app

# Initialize a git repo scoped to just this folder (if you haven't already)
git init
git add .
git commit -m "Landmark Snap Voila app"

# Create the Heroku app (pick your own name, or omit it to get a random one)
heroku create landmark-snap-yourname

# Push to Heroku -- this triggers the build
git push heroku main
# (if your default branch is "master" instead of "main", use:
#  git push heroku master)

# Make sure at least one web dyno is running
heroku ps:scale web=1

# Open it in your browser
heroku open
```

Heroku will detect `requirements.txt`, use the official Python buildpack, and
start the app using the command in `Procfile`.

## 3. Important things to know before/if you hit issues

- **Model file size**: `transfer_exported.pt` for a frozen ResNet18 backbone is
  typically ~45 MB. That's fine for Heroku's slug size limit (500 MB
  compressed), but if you experiment with a much larger backbone, keep an eye
  on total slug size (`heroku builds:info` after a deploy).
- **CPU-only PyTorch**: `requirements.txt` points at PyTorch's CPU wheel index
  (`https://download.pytorch.org/whl/cpu`). Heroku dynos don't have GPUs, and
  the default (CUDA-enabled) PyTorch wheels are several times larger and would
  likely blow past the slug size limit.
- **Memory**: standard Heroku dynos have 512 MB of RAM. Loading PyTorch,
  Voila/Jupyter, and a ResNet18 checkpoint comfortably fits in that budget for
  inference (no training happens here), but if you see `Error R14 (Memory
  quota exceeded)` in `heroku logs --tail`, consider upgrding to a dyno with
  more RAM (e.g. Standard-2X).
- **Cold start / boot timeout**: Heroku expects your web process to bind to
  `$PORT` within 60 seconds of boot. Importing torch + loading the model is
  usually fast enough, but on a cold, freshly-built dyno the very first
  request can be slow. If you see `Error R10 (Boot timeout)`, check
  `heroku logs --tail` for what's happening during startup.
- **Debugging deploys**: `heroku logs --tail` is your best friend for both
  build and runtime issues.
- **`.python-version`**: currently set to `3.12`. Heroku only supports the
  latest patch release of each major Python version and periodically drops
  old ones as they reach end-of-life — if a deploy fails with a message like
  `Requested runtime is not available for this stack`, check
  https://devcenter.heroku.com/articles/python-support for the currently
  supported versions and update this file accordingly.

## 4. Customizing further

- Colors/fonts/layout are all defined in the first cell of `app.ipynb` (a
  single `<style>` block), so you can restyle the whole app without touching
  any of the model logic.
- The prediction bars, banner text, and footer are all generated from small
  HTML-building functions in the second cell (`render_predictions_html`,
  `format_landmark_name`) — easy to tweak independently of the UI wiring.
