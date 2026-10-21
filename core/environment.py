"""Real, inspectable facts about the runtime environment and roadmap.

Used by the Overview page and by the unit tests. Every value is read
from the running interpreter or from installed package metadata —
nothing here is hard-coded marketing text.
"""

import platform
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

#: Every dependency the project relies on, in requirements.txt order.
TRACKED_PACKAGES: tuple[str, ...] = (
    "streamlit",
    "pandas",
    "numpy",
    "plotly",
    "scikit-learn",
    "google-genai",
    "python-dotenv",
    "pydantic",
    "sqlalchemy",
    "pytest",
)


@dataclass(frozen=True)
class PackageStatus:
    """Installation state of one tracked package."""

    name: str
    version: str | None
    installed: bool


@dataclass(frozen=True)
class Milestone:
    """One step of the project roadmap.

    Status is one of "done", "in_progress" or "planned".
    """

    id: str
    title: str
    status: str
    summary: str


#: Current state of the project. Update the status field as milestones
#: are verified; the Overview page and the tests read from this tuple.
MILESTONES: tuple[Milestone, ...] = (
    Milestone(
        id="M1",
        title="Project setup",
        status="done",
        summary="Virtual environment, dependencies, git hygiene and documentation.",
    ),
    Milestone(
        id="M2",
        title="Design system & app shell",
        status="done",
        summary="Navy theme, design tokens, navigation and reusable components.",
    ),
    Milestone(
        id="M3",
        title="Data upload & KPIs",
        status="done",
        summary="CSV/XLSX upload, dataset profiling and quality checks.",
    ),
    Milestone(
        id="M4",
        title="Interactive charts",
        status="done",
        summary="Plotly visuals themed with the design tokens.",
    ),
    Milestone(
        id="M5",
        title="Gemini AI assistant",
        status="done",
        summary="Grounded analysis chat: Gemini plans, pandas calculates, Gemini explains.",
    ),
    Milestone(
        id="M6",
        title="Models, storage & tests",
        status="done",
        summary="scikit-learn models (classification, regression, clustering), SQLAlchemy persistence and pytest coverage.",
    ),
    Milestone(
        id="M7",
        title="Model comparison & explainability",
        status="done",
        summary="Multi-model comparison on one shared split, honest best-model selection, feature attribution with permutation fallback.",
    ),
    Milestone(
        id="M8",
        title="Export, reports & reproducibility",
        status="done",
        summary="CSV exports for metrics, run metadata and predictions, a self-contained HTML analysis report and recorded run configuration.",
    ),
)


def python_version() -> str:
    """Version of the interpreter currently running the app."""
    return platform.python_version()


def package_statuses() -> list[PackageStatus]:
    """Look up the installed version of each tracked package."""
    statuses = []
    for name in TRACKED_PACKAGES:
        try:
            statuses.append(PackageStatus(name=name, version=version(name), installed=True))
        except PackageNotFoundError:
            statuses.append(PackageStatus(name=name, version=None, installed=False))
    return statuses
