# Re-recording the demo GIF

The previous `assets/demo.cast` and `assets/demo.gif` were removed during
the v0.1.0 wording cleanup. Use the steps below to produce a fresh
recording with the updated wording.

## Prerequisites

```bash
# macOS
brew install asciinema agg

# Linux (Debian/Ubuntu)
pipx install asciinema   # or: pip install --user asciinema
cargo install --git https://github.com/asciinema/agg   # GIF renderer
```

- `asciinema` records the terminal session to a `.cast` file.
- `agg` (asciinema gif generator) renders the `.cast` to a `.gif`.
- Alternatively, any cast → gif renderer works (e.g. `asciicast2gif`).

## 1. Make sure the project deps are installed

```bash
pip install -r requirements.txt
```

## 2. Record the cast

From the repo root:

```bash
asciinema rec assets/demo.cast \
  --command "bash assets/record_demo.sh" \
  --overwrite
```

The script runs:

```
python -m demo.killer_demo --model all-MiniLM-L6-v2 --no-export
```

with brief pauses for readability. Total run time is roughly 20–30s
depending on machine speed.

## 3. Render the GIF

```bash
agg assets/demo.cast assets/demo.gif \
  --font-size 14 \
  --speed 1.0
```

Tweak `--font-size` and `--cols` / `--rows` for legibility. Aim for a
GIF under ~3 MB so GitHub renders it inline.

## 4. Verify the GIF contains no banned terms

Visually scan the generated GIF (or replay the cast with
`asciinema play assets/demo.cast`) and confirm the rendered output is
clean against the project's anonymity-safety word list.

Run the project's two release-checklist greps from the repo root
before committing the new assets — both must return zero matches:

```bash
# Patterns are written with single-char character classes so this
# doc file does not match itself. They are functionally identical
# to the unescaped patterns.
EXCL='--exclude-dir=.git --exclude-dir=__pycache__ --exclude-dir=.pytest_cache --exclude=RECORD_DEMO.md'

grep -RniE 'or[d]er[- ]?b[l]ind(ness)?' . $EXCL
grep -RniE '\b(pa[p]er|manus[c]ript|sub[m]ission|un[d]er review|pre[p]rint)\b' . $EXCL
```

The current `assets/demo_replay.py` and `demo/killer_demo.py` are
already cleaned, so a fresh recording from `main` should be safe by
construction.

## 5. Restore the README link

Once the new `assets/demo.gif` is in place and verified, replace the
placeholder block near the top of `README.md`:

```markdown
> 🎬 **Demo GIF pending re-record.** Regenerate locally with
> `bash assets/record_demo.sh` and drop the output at `assets/demo.gif`.
```

with:

```markdown
![Embedding Debugger demo](assets/demo.gif)
```

Then keep the existing italic caption underneath it.

## 6. Commit

```bash
git add assets/demo.cast assets/demo.gif README.md
git commit -m "assets: re-record demo with cleaned wording"
```
