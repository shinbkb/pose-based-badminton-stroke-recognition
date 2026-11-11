"""Smoke tests for the reorganized source tree; no video extraction or training."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from export_notebooks import WORKFLOWS, render


class ProjectStructureTests(unittest.TestCase):
    def test_exports_are_current_and_import_without_running(self):
        for stem, destination in WORKFLOWS.items():
            with self.subTest(workflow=stem):
                path = ROOT / destination
                self.assertEqual(path.read_text(), render(stem, destination))
                spec = importlib.util.spec_from_file_location(stem, path)
                spec.loader.exec_module(importlib.util.module_from_spec(spec))

    def test_help_outside_repository(self):
        for destination in WORKFLOWS.values():
            with self.subTest(workflow=destination):
                result = subprocess.run(
                    [sys.executable, str(ROOT / destination), "--help"],
                    cwd="/tmp", capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--non-interactive", result.stdout)

    def test_notebook_cells_compile(self):
        for path in (ROOT / "notebooks").glob("*.ipynb"):
            notebook = json.loads(path.read_text())
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    source = "".join(cell["source"]).replace("!nvidia-smi", "pass")
                    compile(source, f"{path.name}:cell{index}", "exec")

    def test_paths_and_legacy_dataset(self):
        from badminton_stroke.paths import PROJECT_ROOT, EXPERIMENTS_DIR
        from badminton_stroke.dataset import JOINT_DIR
        self.assertEqual(PROJECT_ROOT, ROOT)
        self.assertEqual(EXPERIMENTS_DIR, ROOT / "experiments" / "runs")
        self.assertTrue(Path(JOINT_DIR).is_absolute())

    def test_model_forward_with_both_skeletons(self):
        import torch
        from badminton_stroke.models.graphs import build_coco_adjacency, build_h36m_adjacency
        from badminton_stroke.models.stgcn import STGCNV2
        torch.set_num_threads(1)
        for channels, graph in [(5, build_coco_adjacency), (7, build_h36m_adjacency)]:
            with self.subTest(channels=channels):
                adjacency = graph()
                self.assertEqual(tuple(adjacency.shape), (3, 17, 17))
                model = STGCNV2(channels, 18, adjacency).eval()
                with torch.no_grad():
                    logits, attention = model(torch.randn(2, channels, 16, 17, 2), return_attention=True)
                self.assertEqual(tuple(logits.shape), (2, 18))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue(torch.allclose(attention.sum(dim=1), torch.ones(2)))


if __name__ == "__main__":
    unittest.main()
