"""SEAF - Secure Explainable AI Framework.

Explanation-space anomaly detection for securing machine-learning decisions:
a model decision is explained with TreeSHAP, the attribution vector is scored
against a baseline of normal model reasoning, and confidence, anomaly and
stability are combined into a trust score that routes low-trust decisions to
human review.
"""
from .core import ACCEPT, REVIEW, SEAF, SEAFResult

__all__ = ["SEAF", "SEAFResult", "ACCEPT", "REVIEW"]
__version__ = "1.0.0"
