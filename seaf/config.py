"""Global constants and the registry of evaluation domains.

Every stochastic component in the library draws its seed from ``SEED`` so that
training, experiments and the dashboard are reproducible end to end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SEED: int = 42

# Paper hyper-parameters (Section 3.4 - 3.6)
N_ESTIMATORS: int = 300
MAX_DEPTH: int = 10
TEST_SIZE: float = 0.20
IFOREST_ESTIMATORS: int = 200
JITTER_SCALE: float = 0.02          # +-2 % non-adversarial jitter
JITTER_SAMPLES: int = 8             # k
ATTACK_BUDGET: float = 0.35         # +-35 % adversarial perturbation
ATTACK_CANDIDATES: int = 40
ATTACK_TAU: float = 0.06            # max |delta p| for prediction preservation
FPR_BUDGET: float = 0.10            # review-capacity budget for recalibration
NORM_LO, NORM_HI = 5.0, 95.0        # clean-baseline percentiles for normalisation
HEALTHCARE_N: int = 4269            # downsample target (matches finance)
N_ATTACKED_PAPER: int = 60          # attacked instances per domain in the paper

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"
ARTIFACT_VERSION = "1.0.0"


def runtime_dir() -> Path:
    """Writable directory for ephemeral state (DB, refits, uploads).

    ``$SEAF_RUNTIME_DIR`` if set, else ``<repo>/runtime`` when writable, else the
    system temp dir (read-only app directories on some hosts).
    """
    import os
    import tempfile

    env = os.environ.get("SEAF_RUNTIME_DIR")
    candidates = [Path(env)] if env else []
    candidates += [ROOT / "runtime", Path(tempfile.gettempdir()) / "seaf_runtime"]
    for c in candidates:
        try:
            c.mkdir(parents=True, exist_ok=True)
            probe = c / ".write_probe"
            probe.write_text("ok")
            probe.unlink()
            return c
        except OSError:
            continue
    return Path(tempfile.gettempdir())


@dataclass(frozen=True)
class DomainSpec:
    """Static description of one evaluation domain."""

    key: str
    title: str
    subtitle: str
    csv: str
    target: str
    positive_label: str
    class_names: tuple[str, str]        # (negative, positive) display names
    decision_verb: str                  # e.g. "rejection probability"
    icon: str                           # material icon name for the UI
    sep: str = ","
    id_columns: tuple[str, ...] = ()
    feature_labels: dict[str, str] = field(default_factory=dict)
    # Display labels for encoded / coded values (UI only, never changes data)
    value_labels: dict[str, dict] = field(default_factory=dict)
    # UI-only unit conversion: col -> (factor, unit); shown value = raw * factor
    display_units: dict[str, tuple[float, str]] = field(default_factory=dict)


DOMAINS: dict[str, DomainSpec] = {
    "finance": DomainSpec(
        key="finance",
        title="Finance",
        subtitle="Loan approval",
        csv="loan_approval_dataset.csv",
        target="loan_status",
        positive_label="Rejected",
        class_names=("Approved", "Rejected"),
        decision_verb="rejection probability",
        icon="account_balance",
        id_columns=("loan_id",),
        feature_labels={
            "no_of_dependents": "Dependents",
            "education": "Education",
            "self_employed": "Self-employed",
            "income_annum": "Annual income",
            "loan_amount": "Loan amount",
            "loan_term": "Loan term (years)",
            "cibil_score": "CIBIL credit score",
            "residential_assets_value": "Residential assets",
            "commercial_assets_value": "Commercial assets",
            "luxury_assets_value": "Luxury assets",
            "bank_asset_value": "Bank assets",
        },
    ),
    "healthcare": DomainSpec(
        key="healthcare",
        title="Healthcare",
        subtitle="Cardiovascular diagnosis",
        csv="cardio_train.csv",
        target="cardio",
        positive_label="1",
        class_names=("No CVD", "CVD"),
        decision_verb="disease probability",
        icon="cardiology",
        sep=";",
        id_columns=("id",),
        feature_labels={
            "age": "Age (days)",
            "gender": "Gender",
            "height": "Height (cm)",
            "weight": "Weight (kg)",
            "ap_hi": "Systolic BP",
            "ap_lo": "Diastolic BP",
            "cholesterol": "Cholesterol",
            "gluc": "Glucose",
            "smoke": "Smoker",
            "alco": "Alcohol intake",
            "active": "Physically active",
        },
        display_units={"age": (1 / 365.25, "years")},
        value_labels={
            "gender": {1: "Female", 2: "Male"},
            "cholesterol": {1: "Normal", 2: "Above normal", 3: "Well above normal"},
            "gluc": {1: "Normal", 2: "Above normal", 3: "Well above normal"},
            "smoke": {0: "No", 1: "Yes"},
            "alco": {0: "No", 1: "Yes"},
            "active": {0: "No", 1: "Yes"},
        },
    ),
    "recruitment": DomainSpec(
        key="recruitment",
        title="Recruitment",
        subtitle="Employee attrition",
        csv="WA_Fn-UseC_-HR-Employee-Attrition.csv",
        target="Attrition",
        positive_label="Yes",
        class_names=("Stays", "Leaves"),
        decision_verb="attrition probability",
        icon="badge",
        id_columns=("EmployeeNumber",),
    ),
}
