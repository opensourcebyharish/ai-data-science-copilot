"""Tests for the real environment facts shown on the Overview page."""

from core.environment import MILESTONES, TRACKED_PACKAGES, package_statuses, python_version


def test_python_version_is_supported():
    major, minor, *_ = python_version().split(".")
    assert major == "3"
    assert minor in {"11", "12"}


def test_every_tracked_package_is_installed():
    statuses = package_statuses()
    assert [status.name for status in statuses] == list(TRACKED_PACKAGES)
    missing = [status.name for status in statuses if not status.installed]
    assert missing == []


def test_streamlit_version_is_reported():
    streamlit_status = next(s for s in package_statuses() if s.name == "streamlit")
    assert streamlit_status.installed
    assert streamlit_status.version is not None


def test_milestone_ids_are_unique():
    ids = [milestone.id for milestone in MILESTONES]
    assert len(set(ids)) == len(ids)


def test_milestone_statuses_are_valid_and_ordered():
    valid = {"done", "in_progress", "planned"}
    statuses = [milestone.status for milestone in MILESTONES]
    assert all(status in valid for status in statuses)
    # M1–M8 verified.
    assert statuses == [
        "done",
        "done",
        "done",
        "done",
        "done",
        "done",
        "done",
        "done",
    ]
