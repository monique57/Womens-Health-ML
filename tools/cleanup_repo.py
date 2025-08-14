import argparse
import fnmatch
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# =========================
# Repo cleanup configuration
# =========================

# 1) Keep-list (explicit safeties)
KEEP_EXACT = {
    # top level
    "README.md",
    "requirements.txt",
    ".gitignore",
    # models we want to keep
    "models/best_model_namcs2019.joblib",
    "models/best_model_v2.joblib",
    "models/class_labels.json",
    # data we want to keep
    "data/namcs2019_clean.csv",
}

# 2) Keep by glob (patterns we keep if they match)
KEEP_GLOBS = [
    # code we rely on
    "src/app_v2.py",
    "src/app_namcs.py",
    "src/train_models_v2.py",
    "src/evaluate_v2.py",
    "src/train_namcs2019.py",
    "src/test_namcs2019.py",
    "src/stability_eval.py",
    "src/cross_validate.py",
    "src/shap_explain.py",
    # results (we keep everything under results by default)
    "results/**",
]

# 3) Delete/move candidates by glob (safe defaults)
#    You can add more patterns here if you find clutter.
CLEANUP_GLOBS = [
    # Python caches & editor junk
    "**/__pycache__/**",
    "**/.ipynb_checkpoints/**",
    ".vscode/**",
    ".DS_Store",
    "Thumbs.db",

    # archives / zips
    "**/*.zip",

    # raw NAMCS or other large raw files we no longer need
    "data/*.sav",
    "data/raw/**",

    # old/unused app & eval scripts (adjust if you still need them)
    "src/app.py",
    "src/app_old.py",
    "src/evaluate.py",
    "src/beeswarm_plot.py",

    # models we don't ship
    "models/*.h5",
    "models/*.zip",
    "models/*.pkl",
    "models/*.pt",

    # temp outputs
    "outputs/**",
    "results/tmp/**",
]

# 4) Directories we never traverse (safety)
SKIP_DIRS = {".git", ".venv", "venv", ".mypy_cache", ".ruff_cache"}

# =========================

ROOT = Path(__file__).resolve().parents[1]  # repo root (one level up from tools/)
ARCHIVE_ROOT = ROOT / "archive"
NOW_TAG = datetime.now().strftime("%Y%m%d-%H%M%S")

def is_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False

def match_any(path: Path, patterns) -> bool:
    rel = str(path.relative_to(ROOT)).replace("\\", "/")
    for pat in patterns:
        if fnmatch.fnmatch(rel, pat):
            return True
    return False

def should_keep(path: Path) -> bool:
    rel = str(path.relative_to(ROOT)).replace("\\", "/")
    if rel in KEEP_EXACT:
        return True
    if match_any(path, KEEP_GLOBS):
        return True
    # always keep python source unless explicitly targeted by CLEANUP_GLOBS
    # (we'll filter with CLEANUP_GLOBS first anyway)
    return False

def list_candidates():
    candidates = []
    for p in ROOT.rglob("*"):
        # skip directories we don't traverse
        if p.is_dir() and p.name in SKIP_DIRS:
            # do not traverse into skip dirs
            continue
        # If directory matches cleanup glob, we add the directory itself (it will remove tree)
        if p.is_dir() and match_any(p, CLEANUP_GLOBS) and not should_keep(p):
            candidates.append(p)
            continue
        # Files:
        if p.is_file():
            if match_any(p, CLEANUP_GLOBS) and not should_keep(p):
                candidates.append(p)
    return dedupe_paths(candidates)

def dedupe_paths(paths):
    # Remove children if a parent directory is already present to act on
    final = []
    skip_under = set()
    for p in sorted(paths, key=lambda x: (len(str(x)), str(x))):
        if any(is_inside(p, s) for s in skip_under):
            continue
        final.append(p)
        if p.is_dir():
            skip_under.add(p)
    return final

def human_preview(items):
    files = [p for p in items if p.is_file()]
    dirs  = [p for p in items if p.is_dir()]
    return len(files), len(dirs)

def confirm(prompt: str) -> bool:
    print(prompt + " [y/N] ", end="", flush=True)
    ans = sys.stdin.readline().strip().lower()
    return ans == "y"

def archive_path_for(p: Path) -> Path:
    rel = p.relative_to(ROOT)
    dest = ARCHIVE_ROOT / f"{NOW_TAG}" / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest

def do_archive(items):
    moved = []
    for p in items:
        dest = archive_path_for(p)
        try:
            if p.is_dir():
                shutil.move(str(p), str(dest))
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(dest))
            moved.append((p, dest))
        except Exception as e:
            print(f"  ⚠️  Could not move {p}: {e}")
    return moved

def do_delete(items):
    removed = []
    for p in items:
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=False)
            else:
                p.unlink()
            removed.append(p)
        except Exception as e:
            print(f"  ⚠️  Could not delete {p}: {e}")
    return removed

def main():
    parser = argparse.ArgumentParser(description="Project cleanup helper (safe, interactive).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Only list what would be removed/moved.")
    mode.add_argument("--archive", action="store_true", help="Move candidates to archive/<timestamp>/ ...")
    mode.add_argument("--delete", action="store_true", help="Permanently delete candidates.")
    args = parser.parse_args()

    print(f"📁 Repo root: {ROOT}")
    print("🔍 Scanning for cleanup candidates...")
    items = list_candidates()
    if not items:
        print("✅ No cleanup candidates found. You're all set!")
        return

    n_files, n_dirs = human_preview(items)
    print(f"Found {len(items)} items → {n_files} files, {n_dirs} directories.\n")
    for p in items[:30]:
        print("  -", p.relative_to(ROOT))
    if len(items) > 30:
        print(f"  ... and {len(items) - 30} more")

    if args.dry_run:
        print("\n💡 Dry run only. No changes made.")
        return

    action = "archive" if args.archive else "delete" if args.delete else None
    if not action:
        print("\nℹ️ No mode chosen. Use one of: --dry-run | --archive | --delete")
        sys.exit(1)

    if action == "archive":
        if not confirm(f"\nMove {len(items)} item(s) into {ARCHIVE_ROOT}/{NOW_TAG}/ ?"):
            print("❎ Aborted.")
            return
        moved = do_archive(items)
        print(f"✅ Archived {len(moved)} item(s) under {ARCHIVE_ROOT}/{NOW_TAG}/")
    else:
        if not confirm(f"\nPermanently DELETE {len(items)} item(s)? This cannot be undone."):
            print("❎ Aborted.")
            return
        removed = do_delete(items)
        print(f"✅ Deleted {len(removed)} item(s).")

if __name__ == "__main__":
    main()
