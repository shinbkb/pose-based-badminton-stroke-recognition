"""Export notebook workflows as guarded, executable Python scripts.

Notebook code remains the source of truth. Markdown becomes comments and cell
boundaries are preserved for editors that support # %% cells.
"""
import argparse
import ast
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = {
    "data-preprocessing": "scripts/preprocess_data.py",
    "yolov26pose": "scripts/extract_yolo_pose.py",
    "motionbert_3d_extraction": "scripts/extract_motionbert_3d.py",
    "stgcn_experiments": "experiments/train_stgcn.py",
    "stgcn_experiments_motionbert": "experiments/train_stgcn_motionbert.py",
    "stgcn_ablation": "experiments/run_stgcn_ablation.py",
}


def render(stem, destination):
    notebook = json.loads((ROOT / "notebooks" / f"{stem}.ipynb").read_text())
    header = f'''"""Exported from notebooks/{stem}.ipynb.

Regenerate with: python tools/export_notebooks.py
Run --help for usage. Importing this script does not start the workflow.
"""
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--non-interactive", action="store_true",
        help="Use the Agg plotting backend when running without a desktop.",
    )
    args = parser.parse_args()
    from IPython.display import display

    if args.non_interactive:
        os.environ["MPLBACKEND"] = "Agg"
    os.chdir(_ROOT)

'''
    chunks = []
    # The first cell only bootstraps Jupyter; the script has its own bootstrap.
    for index, cell in enumerate(notebook["cells"][1:], start=1):
        source = "".join(cell.get("source", [])).rstrip()
        if not source:
            continue
        if cell["cell_type"] == "markdown":
            chunks.append("# %% [markdown]\n" + "\n".join("# " + line for line in source.splitlines()))
        elif cell["cell_type"] == "code":
            lines = []
            for line in source.splitlines():
                if line.strip() == "!nvidia-smi":
                    line = 'import subprocess; subprocess.run(["nvidia-smi"], check=False)'
                elif line.lstrip().startswith("%") or (line.lstrip().startswith("!") and not line.lstrip().startswith("!=")):
                    raise ValueError(f"Unsupported notebook magic: {stem}, cell {index}: {line}")
                lines.append(line)
            source = "\n".join(lines)
            ast.parse(source, filename=f"{stem}:cell{index}")
            chunks.append(f"# %% Cell {index}\n{source}")
    result = header + textwrap.indent("\n\n".join(chunks), "    ") + "\n"
    ast.parse(result, filename=destination)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check exports without writing.")
    args = parser.parse_args()
    stale = []
    for stem, destination in WORKFLOWS.items():
        rendered = render(stem, destination)
        path = ROOT / destination
        if args.check:
            if not path.exists() or path.read_text() != rendered:
                stale.append(destination)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered)
            print(destination)
    if stale:
        parser.exit(1, "Outdated exports: " + ", ".join(stale) + "\n")


if __name__ == "__main__":
    main()
