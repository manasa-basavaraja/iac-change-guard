from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from iac_guard.collector.diff_collector import IaCDiff
from iac_guard.model import IaCMaliciousChangeModel


@dataclass
class DetectionResult:
    logits: torch.Tensor
    probs: torch.Tensor
    label: str
    risk_score: float
    malicious_probability: float


CLASS_NAMES = ("benign", "misconfiguration", "malicious")


def run_detection(
    model: IaCMaliciousChangeModel,
    diff: IaCDiff,
    *,
    device: torch.device | None = None,
    risk_threshold: float = 0.72,
) -> DetectionResult:
    model.eval()
    dev = device or torch.device("cpu")
    model.to(dev)
    with torch.no_grad():
        logits = model.forward_from_diff(diff).to(dev)
        probs = F.softmax(logits, dim=-1).squeeze(0)
        pred = int(torch.argmax(probs).item())
        mal_prob = float(probs[2].item()) if probs.numel() >= 3 else float(probs[-1].item())
        risk = float(mal_prob)
    return DetectionResult(
        logits=logits,
        probs=probs,
        label=CLASS_NAMES[pred] if pred < len(CLASS_NAMES) else "unknown",
        risk_score=risk,
        malicious_probability=mal_prob,
    )


def should_alert(result: DetectionResult, threshold: float = 0.72) -> bool:
    return result.malicious_probability >= threshold
