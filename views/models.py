"""Models page — scikit-learn training, comparison and run history.

The ML engine lives in core/models.py, the comparison engine in
core/comparison.py, attribution in core/explain.py and persistence in
core/storage.py. This module only turns those results into UI. The
dataset is read from session state (``loaded_dataset``) set by the
Data page.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.comparison import (
    METRIC_DEFINITIONS,
    METRIC_DIRECTIONS,
    METRIC_LABELS,
    ComparisonResult,
    compare_models,
)
from core.explain import explain, group_importances
from core.export import (
    ReportContext,
    build_comparison_report,
    build_single_report,
    comparison_csv,
    config_csv,
    config_from_comparison,
    config_from_result,
    config_json,
    predictions_csv,
    run_metadata_csv,
    safe_filename,
)
from core.loader import LoadedDataset
from core.models import (
    TASK_LABELS,
    ModelError,
    ModelResult,
    available_models,
    dataset_hash,
    suitable_target_columns,
    train_run,
)
from core.storage import delete_run, load_runs, save_run
from ui.components import (
    card_title,
    render_empty_state,
    render_kpis,
    render_page_header,
    section_card,
    section_heading,
)

#: Session-state key for the last trained result.
RESULT_KEY = "models_last_result"

#: Session-state key for the fitted run behind RESULT_KEY (Milestone 8):
#: the estimator plus the exact split, so predictions can be exported.
RUN_KEY = "models_last_run"

#: Session-state keys recording the configuration a single run was
#: trained with, so the export metadata describes what actually ran.
SINGLE_TEST_SIZE_KEY = "models_single_test_size"
SINGLE_SEED_KEY = "models_single_seed"
SINGLE_N_TRAIN_KEY = "models_single_n_train"
SINGLE_N_TEST_KEY = "models_single_n_test"
SINGLE_DROPPED_KEY = "models_single_dropped"

#: The seed every training run uses — recorded for reproducibility.
RANDOM_STATE = 42

#: Session-state key for the last comparison result and the form
#: settings it was produced with (a stale comparison is discarded when
#: the configuration changes, so old numbers are never shown as if they
#: belonged to the current setup).
COMPARISON_KEY = "models_comparison_result"
COMPARISON_SIGNATURE_KEY = "models_comparison_signature"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def dropped_rows_of(run, frame) -> int:
    """How many frame rows the run could not use (missing features/target).

    Derived from the matrices the estimator actually saw, so the exported
    metadata describes reality rather than an estimate.
    """
    if run.X_train is None:
        return 0
    scored = len(run.X_train) + (len(run.X_test) if run.X_test is not None else 0)
    return max(0, len(frame) - scored)


def _dataset() -> LoadedDataset | None:
    """Return the loaded dataset from session state, or None."""
    return st.session_state.get("loaded_dataset")


def _render_run_history() -> None:
    """Collapsible section showing past runs from the database."""
    runs = load_runs()
    with st.expander(f"Run history ({len(runs)} run{'s' if len(runs) != 1 else ''})", expanded=False):
        if not runs:
            st.caption("No runs saved yet. Train a model to record your first run.")
            return
        for run in runs:
            metrics_str = "  ·  ".join(
                f"{k}: {v:.4g}" for k, v in run["metrics"].items()
            )
            features_str = ", ".join(run["feature_columns"])
            target_str = f" → {run['target_column']}" if run["target_column"] else ""
            st.markdown(
                f"**#{run['id']}** &nbsp; `{run['model_name']}`"
                f" &nbsp;·&nbsp; {run['task_type'].title()}{target_str}"
            )
            st.caption(f"Features: {features_str}")
            st.caption(metrics_str)
            col_del, _ = st.columns([1, 6])
            with col_del:
                if st.button("Delete", key=f"del_run_{run['id']}", type="tertiary"):
                    delete_run(run["id"])
                    st.rerun()
            st.divider()

def _render_importances(result: ModelResult) -> None:
    """Bar chart of feature importances when available.

    One-hot pieces are folded back onto their original column names so
    the chart reads like the uploaded data, not like the encoding.
    """
    if not result.feature_importances:
        return
    import pandas as pd

    section_heading("Feature importances")
    grouped = group_importances(result.feature_importances, result.feature_columns)
    imp_df = (
        pd.DataFrame(
            grouped.items(),
            columns=["Feature", "Importance"],
        )
        .sort_values("Importance", ascending=False)
        .head(15)
    )
    st.bar_chart(imp_df.set_index("Feature"), color="#22d3ee")
    st.caption(
        "Importance describes association with this model's predictions, "
        "not causation."
    )


def _persist_comparison(comparison: ComparisonResult, frame) -> None:
    """Save every successful comparison run to the history.

    A failed write is reported once and never hides the comparison.
    """
    digest = dataset_hash(frame)
    failures = 0
    for outcome in comparison.outcomes:
        if not outcome.ok:
            continue
        try:
            save_run(outcome.result, dataset_hash=digest)
        except Exception:
            failures += 1
    if failures:
        st.warning(
            f"{failures} comparison run(s) trained successfully but could not "
            "be saved to the run history (database unavailable)."
        )


def _render_comparison_result(result: ComparisonResult) -> None:
    """Table, best-model badge, warnings, chart, definitions, attribution."""
    import pandas as pd

    section_heading("Comparison results")
    st.badge(
        f"{TASK_LABELS[result.task_type]} · {len(result.outcomes)} model(s) compared",
        color="green",
        help="Every model used the same split and preprocessing",
    )

    marks = {"higher": "↑", "lower": "↓"}
    rows = []
    for row in result.table_rows():
        entry = {"Model": f"{row['model']} ★" if row["best"] else row["model"]}
        for metric in result.metrics:
            value = row["metrics"].get(metric)
            label = f"{METRIC_LABELS[metric]} {marks[METRIC_DIRECTIONS[metric]]}"
            entry[label] = "—" if value is None else round(value, 4)
        entry["Status"] = "OK" if row["ok"] else (row["error"] or "failed")
        rows.append(entry)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if result.best_model:
        st.success(
            f"Best by {METRIC_LABELS[result.primary_metric]}: {result.best_model} — "
            "only valid, finite scores are ranked."
        )

    for warning in result.warnings:
        st.warning(warning)

    _render_comparison_chart(result)
    _render_metric_definitions(result)
    _render_comparison_explanation(result)


def _render_comparison_chart(result: ComparisonResult) -> None:
    """Bar chart of the task's primary metric across valid models."""
    import pandas as pd

    valid = [
        (row["model"], row["metrics"][result.primary_metric])
        for row in result.table_rows()
        if row["ok"] and row["metrics"].get(result.primary_metric) is not None
    ]
    if not valid:
        return
    direction = (
        "higher is better"
        if METRIC_DIRECTIONS[result.primary_metric] == "higher"
        else "lower is better"
    )
    metric = METRIC_LABELS[result.primary_metric]
    st.caption(f"{metric} across the compared models — {direction}.")
    chart = pd.DataFrame(
        {metric: [value for _, value in valid]},
        index=[name for name, _ in valid],
    )
    st.bar_chart(chart, color="#22d3ee")


def _render_metric_definitions(result: ComparisonResult) -> None:
    """Collapsible definitions for every metric in the table."""
    with st.expander("What these metrics mean"):
        for metric in result.metrics:
            arrow = (
                "↑ higher is better"
                if METRIC_DIRECTIONS[metric] == "higher"
                else "↓ lower is better"
            )
            st.markdown(
                f"**{METRIC_LABELS[metric]}** — {METRIC_DEFINITIONS[metric]} *({arrow})*"
            )


def _render_comparison_explanation(result: ComparisonResult) -> None:
    """Per-model attribution for the best (or first) successful model."""
    import pandas as pd

    successful = result.successful
    if not successful:
        return
    names = [o.model_name for o in successful]
    default_idx = names.index(result.best_model) if result.best_model in names else 0
    chosen = st.selectbox(
        "Explain model",
        options=names,
        index=default_idx,
        help="Feature attribution for this model's predictions — association, never causation.",
    )
    explanation = explain(result.runs[chosen])
    if not explanation.available or not explanation.importances:
        st.caption(explanation.note)
        return
    scores = (
        pd.DataFrame(
            sorted(explanation.importances.items(), key=lambda kv: kv[1], reverse=True)[:15],
            columns=["Feature", "Importance"],
        )
        .set_index("Feature")
    )
    st.bar_chart(scores, color="#22d3ee")
    st.caption(explanation.note)


# ---------------------------------------------------------------------------
# Exports (Milestone 8)
# ---------------------------------------------------------------------------


def _report_context(dataset: LoadedDataset, frame) -> ReportContext:
    """Aggregate dataset context for a report — never raw rows."""
    return ReportContext(
        dataset_filename=dataset.filename,
        dataset_rows=len(frame),
        dataset_columns=frame.shape[1],
        dataset_hash=dataset_hash(frame),
    )


def _download(stem: str, *, extension: str, data: str, label: str, key: str, mime: str) -> None:
    """One download button with a safe, sensible filename."""
    st.download_button(
        label,
        data=data,
        file_name=safe_filename(stem, extension=extension),
        mime=mime,
        key=key,
        icon=":material/download:",
        width="stretch",
    )


def _file_stem(dataset: LoadedDataset, suffix: str) -> str:
    """A readable file stem for a download: ``sales-Linear-Regression``."""
    base = Path(dataset.filename).stem or "dataset"
    return f"{base}-{suffix}"


def _render_single_exports(dataset: LoadedDataset, frame, result: ModelResult) -> None:
    """Download section for the last single-model training run."""
    section_heading("Export this run")
    with section_card():
        card_title(
            "Downloads",
            hint="Predictions cover the held-out rows only. Nothing sensitive is included.",
        )
        run = st.session_state.get(RUN_KEY)
        config = config_from_result(
            result,
            test_size=st.session_state.get(SINGLE_TEST_SIZE_KEY, 0.2),
            random_state=st.session_state.get(SINGLE_SEED_KEY, 42),
            n_clusters=(
                int(result.metrics["n_clusters"])
                if result.task_type == "clustering" and "n_clusters" in result.metrics
                else None
            ),
            n_train=st.session_state.get(SINGLE_N_TRAIN_KEY),
            n_test=st.session_state.get(SINGLE_N_TEST_KEY),
            dropped_rows=st.session_state.get(SINGLE_DROPPED_KEY, 0),
        )
        stem = _file_stem(dataset, result.model_name)

        columns = st.columns(2)
        with columns[0]:
            if run is not None:
                try:
                    _download(
                        f"{stem}-predictions",
                        extension="csv",
                        data=predictions_csv(run),
                        label="Predictions (CSV)",
                        key="dl_single_predictions",
                        mime="text/csv",
                    )
                except ModelError as exc:
                    st.caption(str(exc))
        with columns[1]:
            _download(
                f"{stem}-config",
                extension="csv",
                data=config_csv(config),
                label="Configuration (CSV)",
                key="dl_single_config_csv",
                mime="text/csv",
            )

        columns2 = st.columns(2)
        with columns2[0]:
            _download(
                f"{stem}-config",
                extension="json",
                data=config_json(config),
                label="Configuration (JSON)",
                key="dl_single_config_json",
                mime="application/json",
            )
        with columns2[1]:
            explanation = explain(run) if run is not None else None
            report = build_single_report(
                result=result,
                run=run,
                context=_report_context(dataset, frame),
                config=config,
                importances=(
                    explanation.importances
                    if explanation and explanation.available
                    else None
                ),
                importances_note=explanation.note if explanation else None,
            )
            _download(
                f"{stem}-report",
                extension="html",
                data=report,
                label="Analysis report (HTML)",
                key="dl_single_report",
                mime="text/html",
            )
        st.caption(
            "The report and prediction files are built from this run only — no raw "
            "dataset rows, API keys or model binaries are exported."
        )


def _render_comparison_exports(dataset: LoadedDataset, frame, result: ComparisonResult) -> None:
    """Download section for a completed comparison."""
    section_heading("Export comparison")
    with section_card():
        card_title(
            "Downloads",
            hint="Comparison metrics, per-model metadata and a full HTML report.",
        )
        stem = _file_stem(dataset, "comparison")
        columns = st.columns(2)
        with columns[0]:
            _download(
                f"{stem}-metrics",
                extension="csv",
                data=comparison_csv(result),
                label="Comparison metrics (CSV)",
                key="dl_comparison_metrics",
                mime="text/csv",
            )
        with columns[1]:
            _download(
                f"{stem}-runs",
                extension="csv",
                data=run_metadata_csv(result),
                label="Run metadata (CSV)",
                key="dl_comparison_metadata",
                mime="text/csv",
            )

        columns2 = st.columns(2)
        with columns2[0]:
            successful = result.successful
            chosen = (
                result.runs.get(result.best_model)
                if result.best_model in result.runs
                else (result.runs[successful[0].model_name] if successful else None)
            )
            config_text = (
                config_csv(config_from_comparison(result, chosen))
                if chosen is not None
                else ""
            )
            if config_text:
                _download(
                    f"{stem}-config",
                    extension="csv",
                    data=config_text,
                    label="Configuration (CSV)",
                    key="dl_comparison_config",
                    mime="text/csv",
                )
        with columns2[1]:
            report = build_comparison_report(
                result=result,
                context=_report_context(dataset, frame),
            )
            _download(
                f"{stem}-report",
                extension="html",
                data=report,
                label="Analysis report (HTML)",
                key="dl_comparison_report",
                mime="text/html",
            )
        st.caption(
            "Failed and undefined results are exported as empty cells and listed "
            "explicitly — never as zero."
        )


def _render_comparison_section(
    dataset: LoadedDataset,
    frame,
    task_type: str,
    models_for_task: list[str],
    feature_columns: list[str],
    target_column: str | None,
    test_size: float,
    n_clusters: int,
) -> None:
    """Controls, run and rendering for the model comparison."""
    section_heading("Compare models")
    with section_card():
        card_title(
            "Model comparison",
            hint=(
                "Every selected model is trained and scored on one shared split "
                "and preprocessing pass, so the numbers are directly comparable."
            ),
        )
        selected = st.multiselect(
            "Models to compare",
            options=models_for_task,
            default=models_for_task,
            help=(
                "Two or more models give a meaningful comparison; clustering can "
                "also compare its algorithms head to head."
            ),
        )

        compare_clicked = st.button(
            "Compare models",
            type="secondary",
            disabled=not feature_columns or not selected,
            icon=":material/scale:",
            width="stretch",
        )
        if not feature_columns:
            st.caption("Select at least one feature column to compare models.")

    # A comparison produced under different settings is discarded, so
    # stale numbers are never presented as if they matched the form.
    signature = (
        task_type,
        tuple(feature_columns),
        target_column,
        round(test_size, 4),
        n_clusters if task_type == "clustering" else None,
    )
    if st.session_state.get(COMPARISON_SIGNATURE_KEY) != signature:
        st.session_state.pop(COMPARISON_KEY, None)

    if compare_clicked and selected:
        with st.spinner(f"Comparing {len(selected)} model(s)…"):
            try:
                comparison = compare_models(
                    frame=frame,
                    task_type=task_type,
                    model_names=selected,
                    feature_columns=feature_columns,
                    target_column=target_column,
                    test_size=test_size,
                    n_clusters=n_clusters,
                )
                st.session_state[COMPARISON_KEY] = comparison
                st.session_state[COMPARISON_SIGNATURE_KEY] = signature
                _persist_comparison(comparison, frame)
            except ModelError as exc:
                st.session_state.pop(COMPARISON_KEY, None)
                st.session_state.pop(COMPARISON_SIGNATURE_KEY, None)
                st.error(str(exc))

    comparison: ComparisonResult | None = st.session_state.get(COMPARISON_KEY)
    if comparison is not None:
        _render_comparison_result(comparison)
        _render_comparison_exports(dataset, frame, comparison)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the full Models page."""
    render_page_header(
        "Models",
        subtitle="Train and compare scikit-learn models on your dataset, with explanations.",
        chip="M8",
        chip_tone="progress",
    )

    dataset = _dataset()
    if dataset is None:
        render_empty_state(
            "No dataset loaded",
            "Upload a file on the Data page first, then return here to train a model.",
            hint="Use the sidebar to navigate to the Data page.",
            icon="🗂️",
        )
        st.link_button("Go to Data page", url="/data", icon=":material/database:")
        _render_run_history()
        return

    frame = dataset.dataframe

    # --- Configuration form ---
    section_heading("Configure training")

    task_options = list(TASK_LABELS.keys())
    task_labels = [TASK_LABELS[t] for t in task_options]

    with section_card():
        card_title(
            "Model configuration",
            hint="Pick the task, the algorithm and the columns the model learns from.",
        )
        task_col, model_col = st.columns(2)
        with task_col:
            task_label = st.selectbox(
                "Task type",
                options=task_labels,
                help="Classification predicts a category, regression predicts a number, clustering groups rows.",
            )
        task_type = task_options[task_labels.index(task_label)]

        models_for_task = available_models(task_type)
        with model_col:
            model_name = st.selectbox(
                "Model",
                options=models_for_task,
                help="Choose the algorithm to train.",
            )

        # Target column (not shown for clustering)
        target_column: str | None = None
        if task_type != "clustering":
            candidate_targets = suitable_target_columns(frame, task_type)
            if not candidate_targets:
                st.warning(
                    f"No suitable target columns found for {task_label}. "
                    "For regression, upload a dataset with numeric columns. "
                    "For classification, a column must have 2–50 distinct values."
                )
                _render_run_history()
                return
            target_column = st.selectbox(
                "Target column",
                options=candidate_targets,
                help="The column the model will learn to predict.",
            )

        # Feature columns
        default_features = [c for c in frame.columns if c != target_column]
        feature_columns = st.multiselect(
            "Feature columns",
            options=[c for c in frame.columns if c != target_column],
            default=default_features[:10],  # cap default selection at 10
            help="The columns the model uses as inputs.",
        )

        # Train/test split
        test_pct = st.slider(
            "Test split",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            format="%d%%",
            help="Percentage of rows held back for evaluation.",
        )

        # Clustering-specific: number of clusters
        n_clusters = 3
        if task_type == "clustering":
            n_clusters = st.slider(
                "Number of clusters (K)",
                min_value=2,
                max_value=min(15, max(2, len(frame) // 5)),
                value=3,
                help="How many groups K-Means will create.",
            )

        train_disabled = not feature_columns
        if train_disabled:
            st.caption("Select at least one feature column to train.")

        train_clicked = st.button(
            "Train model",
            type="primary",
            disabled=train_disabled,
            icon=":material/model_training:",
            width="stretch",
        )

    if train_clicked and feature_columns:
        with st.spinner("Training…"):
            try:
                run = train_run(
                    frame=frame,
                    task_type=task_type,
                    model_name=model_name,
                    feature_columns=feature_columns,
                    target_column=target_column,
                    test_size=test_pct / 100,
                    n_clusters=n_clusters,
                    random_state=RANDOM_STATE,
                )
                result = run.result
                st.session_state[RESULT_KEY] = result
                st.session_state[RUN_KEY] = run
                # Record what actually ran, for the export metadata.
                st.session_state[SINGLE_TEST_SIZE_KEY] = test_pct / 100
                st.session_state[SINGLE_SEED_KEY] = RANDOM_STATE
                st.session_state[SINGLE_N_TRAIN_KEY] = (
                    len(run.X_train) if run.X_train is not None else None
                )
                st.session_state[SINGLE_N_TEST_KEY] = (
                    len(run.X_test) if run.X_test is not None else None
                )
                st.session_state[SINGLE_DROPPED_KEY] = dropped_rows_of(run, frame)
                # Persist to run history; a failed write must never hide
                # the trained result from the user.
                try:
                    config = config_from_result(
                        result,
                        test_size=test_pct / 100,
                        random_state=RANDOM_STATE,
                        n_clusters=(
                            int(result.metrics["n_clusters"])
                            if result.task_type == "clustering"
                            and "n_clusters" in result.metrics
                            else None
                        ),
                        n_train=st.session_state[SINGLE_N_TRAIN_KEY],
                        n_test=st.session_state[SINGLE_N_TEST_KEY],
                        dropped_rows=st.session_state[SINGLE_DROPPED_KEY],
                    )
                    save_run(result, dataset_hash=dataset_hash(frame), config=config.to_dict())
                except Exception:
                    st.warning(
                        "The model trained, but this run could not be saved "
                        "to the run history (database unavailable)."
                    )
            except ModelError as exc:
                st.error(str(exc))

    # --- Results ---
    result: ModelResult | None = st.session_state.get(RESULT_KEY)
    if result is not None:
        st.divider()
        section_heading("Results")

        target_label = f" → {result.target_column}" if result.target_column else ""
        st.badge(
            f"{result.model_name} · {TASK_LABELS[result.task_type]}{target_label}",
            color="green",
            help="Last trained model",
        )

        render_kpis(result.metrics_as_kpis())

        for caveat in result.caveats:
            st.warning(caveat)

        _render_importances(result)
        _render_single_exports(dataset, frame, result)

    st.divider()
    _render_comparison_section(
        dataset=dataset,
        frame=frame,
        task_type=task_type,
        models_for_task=models_for_task,
        feature_columns=feature_columns,
        target_column=target_column,
        test_size=test_pct / 100,
        n_clusters=n_clusters,
    )

    st.divider()
    _render_run_history()


render()
