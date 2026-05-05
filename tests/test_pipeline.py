import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iac_guard.collector.diff_collector import collect_from_unified_diff
from iac_guard.model import IaCMaliciousChangeModel
from iac_guard.pipeline import run_detection


def test_forward_smoke():
    text = """
diff --git a/x/main.tf b/x/main.tf
--- a/x/main.tf
+++ b/x/main.tf
@@ -1 +1,3 @@
+resource "aws_s3_bucket" "b" {}
"""
    d = collect_from_unified_diff(text, preferred_path="x/main.tf")
    assert d is not None
    m = IaCMaliciousChangeModel()
    with torch.no_grad():
        logits = m.forward_from_diff(d)
    assert logits.shape[-1] == 3


def test_detection_runs():
    text = """
diff --git a/k8s/a.yaml b/k8s/a.yaml
--- a/k8s/a.yaml
+++ b/k8s/a.yaml
@@ -1 +1,4 @@
 apiVersion: apps/v1
+kind: Deployment
"""
    d = collect_from_unified_diff(text, preferred_path="k8s/a.yaml")
    assert d is not None
    res = run_detection(IaCMaliciousChangeModel(), d)
    assert res.probs.numel() == 3
