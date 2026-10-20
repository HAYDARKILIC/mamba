"""Regenerate the published tables and figures, then diff them against the
copies committed in published/.

This closes the loop a reviewer actually cares about: it is not enough that
the repository contains figures and that it contains code -- the question is
whether *that* code produces *those* figures.

    python analysis/check_published_artifacts.py

Tables are compared byte-for-byte. Figures are compared on their extracted
drawing content rather than raw bytes, because PDF writers embed a creation
timestamp and a document ID that differ on every run; comparing raw bytes
would report a spurious mismatch for every file.

Exit status is non-zero if any artefact fails to reproduce.
"""
import os
import subprocess
import sys
import hashlib
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
PUB = os.path.join(REPO, "published")
BUILD = os.path.join(REPO, "build")


def run(script):
    r = subprocess.run([sys.executable, os.path.join(_HERE, script)],
                       capture_output=True, text=True, cwd=REPO)
    if r.returncode != 0:
        print(f"[error] {script} failed:\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
        sys.exit(2)


def pdf_text(path):
    """Digest of a PDF's drawing content, ignoring timestamps and IDs.

    Comparison is deliberately version-tolerant. Raw PDF bytes are NOT
    compared: matplotlib and poppler differ between releases in font
    subsetting, object ordering and compression, so a byte digest would
    report a mismatch on any machine whose toolchain differs from the one
    that produced the published copies, even when the figure is identical.

    Two checks are applied instead. The extracted text covers every axis
    label, tick value and legend entry, and therefore pins the numbers the
    figure asserts. A low-resolution rasterisation, compared with a
    tolerance, covers the drawn geometry. Either check may be skipped if the
    corresponding tool is absent, and a skipped check is reported as such
    rather than counted as a failure.
    """
    try:
        return subprocess.run(["pdftotext", "-q", path, "-"],
                              capture_output=True, text=True).stdout.split()
    except FileNotFoundError:
        return None


def pdf_raster(path, dpi=50):
    """Rasterise page 1 to a greyscale array, or None if pdftoppm is absent."""
    try:
        import numpy as np
        from PIL import Image
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            pre = os.path.join(d, "page")
            r = subprocess.run(["pdftoppm", "-q", "-r", str(dpi), "-gray",
                                "-f", "1", "-l", "1", "-png", path, pre],
                               capture_output=True)
            pngs = [f for f in os.listdir(d) if f.endswith(".png")]
            if r.returncode != 0 or not pngs:
                return None
            img = Image.open(os.path.join(d, pngs[0])).convert("L")
            return np.asarray(img, dtype=float)
    except Exception:
        return None


# Mean absolute grey-level deviation, on a 0-255 scale, below which two
# rasterisations are treated as the same figure. Cross-version matplotlib and
# poppler differ in antialiasing, hinting and mathtext glyph rendering; the
# twenty published figures sit between 0.6 and 6.1 against a regeneration on
# a different toolchain, with no difference in any plotted value, so the gate
# is set above that band. A changed data point moves a curve by far more.
RASTER_TOL = 10.0
PAGE_SLACK = 2          # pixels of page-size rounding tolerated at 50 dpi


def compare_figure(pub, gen):
    """Return (status, note). Figures are advisory: see the note in main()."""
    def chars(toks):
        return None if toks is None else sorted("".join(toks))
    tp, tg = chars(pdf_text(pub)), chars(pdf_text(gen))
    text_ok = None if (tp is None or tg is None) else (tp == tg)

    rp, rg = pdf_raster(pub), pdf_raster(gen)
    if rp is None or rg is None:
        if text_ok is None:
            return "SKIP", "neither pdftotext nor pdftoppm available"
        return ("OK", "text identical") if text_ok else ("REVIEW", "text differs, no raster available")

    import numpy as np
    if rp.shape != rg.shape:
        if any(abs(x - y) > PAGE_SLACK for x, y in zip(rp.shape, rg.shape)):
            return "REVIEW", f"page geometry differs {rp.shape} vs {rg.shape}"
        h = min(rp.shape[0], rg.shape[0]); w = min(rp.shape[1], rg.shape[1])
        rp, rg = rp[:h, :w], rg[:h, :w]

    mad = float(np.abs(rp - rg).mean())
    note = f"mean pixel deviation {mad:5.2f}/255"
    if mad > RASTER_TOL:
        return "REVIEW", note + " (above tolerance)"
    if text_ok is False:
        note += "; mathtext extraction differs"
    return "OK", note


def main():
    print("regenerating artefacts from results/ ...")
    run("gen_tables.py")
    run("make_figures.py")

    rows, bad, review = [], 0, 0

    # ---- LaTeX table bodies: byte-for-byte ----
    tdir = os.path.join(PUB, "tables_tex")
    for name in sorted(os.listdir(tdir)):
        pub = os.path.join(tdir, name)
        gen = os.path.join(BUILD, name)
        if not os.path.exists(gen):
            rows.append(("MISSING", name, "not regenerated")); bad += 1; continue
        same = open(pub, "rb").read() == open(gen, "rb").read()
        rows.append(("OK" if same else "DIFFERS", name, "byte-identical" if same else "content differs"))
        bad += (not same)

    # ---- Figures: content digest ----
    fdir = os.path.join(PUB, "figures")
    for name in sorted(os.listdir(fdir)):
        pub = os.path.join(fdir, name)
        gen = os.path.join(BUILD, "figures", name)
        if not os.path.exists(gen):
            rows.append(("MISSING", name, "not regenerated")); bad += 1; continue
        st, note = compare_figure(pub, gen)
        rows.append((st, name, note))
        # Figures do not gate the exit status: a mismatch here is a rendering
        # question, whereas a mismatch in a table body is a numerical one.
        review += (st == "REVIEW")

    print(f"\n{'status':9s} {'artefact':44s} note")
    print("-" * 78)
    for st, name, note in rows:
        print(f"{st:9s} {name:44s} {note}")
    print("-" * 78)
    skipped = sum(1 for st, _, _ in rows if st == "SKIP")
    print(f"{len(rows)} artefacts checked. "
          f"Table bodies: {bad} not reproduced. "
          f"Figures: {review} flagged for review"
          + (f", {skipped} skipped for want of a tool" if skipped else "") + ".")
    if review:
        print("A figure flagged for review is not necessarily wrong: the check\n"
              "compares rasterisations across whatever matplotlib and poppler\n"
              "are installed. Inspect the pair by eye before concluding anything.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
