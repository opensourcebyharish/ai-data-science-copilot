# AI Data Science Copilot

A portfolio project: a Streamlit app that lets you upload datasets, inspect
them, explore them with interactive Plotly charts, and ask questions about
them in plain language using Google Gemini.

> **Current status: Milestone 8 — export, reports & reproducibility, on a
> premium analytics UI.**
> The Data page loads CSV and Excel files, profiles them and reports
> quality issues without ever modifying the data; the Charts page plots
> that same in-memory dataset; the AI Assistant answers questions about
> it with a grounded plan → calculate → explain pipeline; and the
> Models page trains scikit-learn models, compares several of them on
> one shared split, explains which columns each model relied on, and
> exports every result as CSV, a self-contained HTML report and a
> reproducibility record.
>
> **Deployment status:** preparing for a first public release — the app
> works on local data end to end, the Gemini key is optional and is only
> read from the environment, a `.env` file or Streamlit secrets, and
> every trained run can be taken off the screen as a file.

## Tech stack

| Area | Tools |
| --- | --- |
| Language | Python 3.11 / 3.12 (3.12 recommended) |
| Web app | Streamlit |
| Data | pandas, NumPy, openpyxl (Excel) |
| Charts | Plotly |
| AI | Google Gemini (`google-genai` SDK), python-dotenv |
| Machine learning | scikit-learn |
| Validation / storage | Pydantic, SQLAlchemy, SQLite |
| Testing | pytest |

## Project structure

```
ai-data-science-copilot/
├── app.py                 # Entry point: theme + navigation wiring
├── views/                 # One module per page
│   ├── overview.py        # Working home: welcome, dataset status, quick actions
│   ├── data.py            # Working: upload, profile, quality (Milestone 3)
│   ├── charts.py          # Working: Plotly charts (Milestone 4)
│   ├── ai.py              # Working: Gemini chat + code examples (Milestone 5)
│   └── models.py          # Working: training, comparison, export UI, run history (M6/M7/M8)
├── core/                  # Framework-free logic (unit-testable)
│   ├── loader.py          # CSV/XLSX validation and parsing
│   ├── profiling.py       # Metrics, column profiles, quality report
│   ├── charts.py          # Plotly figure builders, themed from the tokens
│   ├── ai_schemas.py      # Pydantic models: plan, answer, code example
│   ├── gemini.py          # Gemini client, config resolution, error taxonomy
│   ├── analysis.py        # Deterministic pandas executor for approved ops
│   ├── assistant.py       # Orchestration: plan → calculate → explain
│   ├── codegen.py         # Display-only code examples (never executed)
│   ├── models.py          # scikit-learn engine: train + reusable fit primitives (M6/M7/M8)
│   ├── comparison.py      # Shared-split multi-model comparison (Milestone 7)
│   ├── explain.py         # Feature attribution: importances + permutation (M7)
│   ├── export.py          # CSV exports, HTML report, run configuration (M8)
│   ├── storage.py         # SQLite run history via SQLAlchemy (Milestone 6/8)
│   └── environment.py     # Real environment/roadmap facts
├── ui/                    # Design system
│   ├── styles.css         # Tokens + premium component/chrome styles
│   ├── theme.py           # Loads and injects the CSS
│   └── components.py      # Header, cards, KPIs, status strip, empty state
├── tests/                 # pytest suite (353 tests)
├── .streamlit/
│   └── config.toml        # Dark theme + 50 MB upload limit
├── .env.example           # Documented Gemini settings (copy to .env)
├── requirements.txt
├── .gitignore             # Ignores .env, secrets and ml_runs.db
└── README.md
```

`views/` is used instead of `pages/` because Streamlit auto-discovers a
`pages/` folder as legacy multipage navigation.

## Setup (PowerShell)

```powershell
cd C:\Users\hp\ai-data-science-copilot

# 1. Get Python 3.12 (skip if already installed)
uv python install 3.12

# 2. Create the virtual environment
uv venv --python 3.12

# 3. Install dependencies
uv pip install -r requirements.txt

# 4. Run the app
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Open **http://localhost:8501**. If activation is blocked by the execution
policy, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
first, or skip activation:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Data page (Milestone 3)

- Upload **CSV** (comma, semicolon, tab or pipe separated) or **Excel
  `.xlsx`**, up to **50 MB** (enforced by the app and by Streamlit).
- Invalid extensions, empty files, binary/corrupt files, oversized uploads
  and malformed rows show a clear message instead of crashing.
- Duplicate or blank column headers are renamed safely in memory
  (`price`, `price__2`, `column_1`, …) and the originals are reported.
- KPI cards: rows, columns, missing cells, duplicate rows, memory usage.
- Tabs: **Preview** (selectable row count), **Columns** (type, non-null,
  missing %, unique), **Statistics** (numeric describe + categorical
  summaries), **Quality** (severity-coded findings).
- Quality findings are read-only: missing values, duplicate rows, constant
  columns, empty and mostly-missing columns, numeric-looking text columns.
- The loaded dataset is kept in session memory for the rest of the visit,
  so the Charts page can plot it without a second upload.
- Uploaded content is never executed as code; parsing happens in memory and
  nothing is written to disk.

### Known limitations

- Only the **first worksheet** of an Excel workbook is read.
- The **first non-blank row** is treated as the header.
- The whole file is held in memory (hence the 50 MB cap); there is no
  sampling or streaming.
- Ragged CSV rows are rejected rather than silently repaired.
- Datetime columns appear in the column profile but are not summarised in
  the Statistics tab yet.
- Profiling results are cached per upload; nothing is persisted between
  sessions.

## Charts page (Milestone 4)

- Reads the dataset loaded on the **Data page** (session memory — upload
  once, chart everywhere). With no dataset, the page explains what to do
  and links back to the Data page.
- **Distribution** — histogram of any numeric column with a bin slider,
  mean and median markers, and count/mean/median/std/min/max KPI cards.
- **Relationship** — scatter of any two numeric columns, optionally
  coloured by a group column (at most 12 groups) or by another number
  (continuous cyan-to-violet scale).
- **Correlation** — heatmap of all numeric columns that vary, Pearson or
  Spearman, from −1 to +1. Constant and all-empty columns are excluded.
- **Time series** — a numeric column bucketed by hour, day, week or month
  (Automatic picks from the date span) with mean/sum/min/max/count.
- Colours and surfaces come straight from the design tokens, so charts
  match the rest of the theme (`#22d3ee`, `#a78bfa`, `#0b1220`,
  `#111c33`, `#223252`, …).
- Figures are built from copies of the data — plotting never modifies the
  loaded frame.

### Known limitations

- Column kinds come from pandas dtypes: numeric-looking text is not
  treated as a number, and a text column counts as dates only when at
  least 90% of its values parse as dates (plain years like `2024` are
  deliberately skipped).
- Scatter plots render every row; very large files can be slow to draw.
- Time-series buckets are calendar-based. Periods without rows appear as
  gaps for mean/min/max and as zero for sum/count.
- Correlation is pairwise and limited to Pearson and Spearman.
- Charts live in session memory only — restarting the app means
  re-uploading the file.

## AI Assistant page (Milestone 5)

Ask questions in plain language about the dataset loaded on the Data
page. The pipeline is **plan → calculate → explain**:

1. **Plan** — Gemini receives only dataset *metadata* (column names,
   dtypes, missing counts, summary statistics — never raw rows) and
   returns a structured `AnalysisPlan` validated with Pydantic against a
   fixed set of operations: `overview`, `column_stats`, `group_aggregate`,
   `top_n`, `correlation`, `timeseries`, `data_quality`, `cannot_answer`.
2. **Calculate** — the app validates the plan's columns, operations,
   aggregations and limits against the loaded dataset, then executes it
   **deterministically with pandas** in Python. Model-generated code is
   never executed.
3. **Explain** — Gemini receives the verified, JSON-safe result of
   that one calculation — aggregates, group labels, value counts and,
   for a top-N request, the ranked rows — always hard-capped by
   `compact_json()` at 15 entries per list and 4,000 characters, and
   writes a short explanation. It cannot invent numbers: only the
   calculated facts are sent to it.

Other page features:

- Native `st.chat_message` / `st.chat_input` chat with session-persistent
  history, dataset-aware suggested questions, loading indicators and a
  "Clear conversation" button.
- Calculated results are clearly badged *Calculated by the app* and shown
  as tables/KPI cards **before** the AI's explanation; caveats (small
  samples, missing data) are listed under the answer.
- Unanswerable questions (prediction, causation, external data) get a
  limitation message that explains *why*, from the model's own reason.
- **Code examples**: ask for Python (pandas) or SQL on any task; the
  example is generated from your real column names, validated, shown
  syntax-highlighted with a copy/download button, and labelled
  *display only, never executed*. SQL examples carry a note that no
  database is connected. Snippets containing system-touching constructs
  (`import os`, `eval(`, non-SELECT SQL, …) are discarded outright.
- API status card at the top: the model name and where the key was read
  from — the key itself is never displayed, logged, or sent anywhere but
  Google. Errors (missing key, invalid key, rate limits, timeouts, bad
  model name, network failures) surface as friendly messages with the
  raw detail scrubbed of the key.

### Privacy guarantees

| Rule | How it is enforced |
| --- | --- |
| Minimal data leaves the app | Plan calls send metadata only — no cell values; explain calls send the verified result of the one calculation that ran, hard-capped at 15 entries and 4,000 characters by `compact_json()`, plus the last few chat messages (`core/assistant.py` builds every prompt). |
| No model-generated code runs | Execution is `pandas` only, dispatched from a fixed operation whitelist (`core/analysis.py`). The code-example path is display-only. |
| No filesystem / env / OS access from model output | Model output is parsed as data (Pydantic), never `eval`/`exec`'d; generated code with blocked constructs is discarded before display. |
| Keys stay out of source, UI and logs | Read from env var / `.env` / Streamlit secrets; scrubbed from every error message; `.env` and `secrets.toml` are git-ignored. |
| Nothing persisted | The conversation lives in `st.session_state` for the browser session only; API calls use `store=False`. |

### Configuring the Gemini API key

1. Create a key in [Google AI Studio](https://aistudio.google.com/apikey).
2. Copy `.env.example` to `.env` in the project root and fill it in:

   ```toml
   GEMINI_API_KEY=your-key-here
   # optional — overrides the default model (gemini-3.8-flash)
   GEMINI_MODEL=
   ```

   Alternatively add the same keys to `.streamlit/secrets.toml`.
3. Restart the Streamlit server. The status card on the AI Assistant
   page turns green. Without a key the rest of the app works normally;
   only the assistant's input and code generation are disabled.

### Manual smoke test (live API)

The automated suite mocks Gemini and never calls the network. To verify
the live integration yourself:

1. Configure the key as above and start the app.
2. On the **Data page**, upload `sample_sales.csv` (Date, Product,
   Region, Sales, Profit).
3. Open the **AI Assistant** page — the status card should show the
   model and key source.
4. Click the suggested question *"Which region has the highest total
   sales?"* — you should see a *Calculated by the app* table of regional
   totals followed by a short Gemini explanation.
5. Ask a follow-up: *"And which product?"* — the assistant should use
   the conversation context.
6. Ask *"Predict next quarter's sales"* — you should get a limitation
   message, not a fabricated forecast.
7. In **Code examples**, enter `average profit by region` and generate —
   confirm the SQL/Python snippet appears with the *never executed*
   badge and a download button.

### Known limitations

- Answers depend on the model's plan; a confusing question may produce a
  plan that fails validation, which is reported honestly rather than
  retried silently.
- Only the approved operations exist — anything needing modelling,
  forecasting or external data is refused with an explanation.
- The planner sees only summary statistics; the explainer sees the
  verified result of the calculation that ran — including ranked rows
  for a top-N request (capped at 15 entries / 4,000 characters), so it
  can quote rows an analysis returned but never invent others.
- Live behaviour (latency, answer quality, rate limits) depends on the
  Gemini API; the automated tests use a mock client and make no network
  calls.

## Design system

A premium dark analytics workspace — deep navy surfaces with a cyan
primary accent and a violet secondary — built entirely from CSS, with no
extra frontend dependencies and no change to native widget behaviour.

- **`.streamlit/config.toml`** — native theme: dark navy background
  (`#0B1220`), cyan primary (`#22D3EE`), muted-blue surfaces.
- **`ui/styles.css`** — the single source of truth, organised in layers:
  1. **Tokens** — colour, semantic, typography, spacing (`--gap-1…5`),
     radii (`--radius-xs…xl`, `--radius-pill`), layered shadows
     (`--shadow-1…3`), gradients (`--g-brand`, `--g-surface`,
     `--g-ambient`) and motion (`--ease`, `--ease-out`).
  2. **Base** — ambient background, typography and thin scrollbars.
  3. **Chrome** — sidebar, header and navigation states (the active page
     gets a cyan accent bar via `a[aria-current="page"]`).
  4. **Components** — buttons (brand-gradient primary), inputs, tabs,
     expanders, native bordered containers, dataframes, alerts (via
     `:has()` on Streamlit's own alert-type test ids), chat bubbles,
     badges, code, tooltips and dividers.
  5. **Page pieces** — headers with a brand hairline, chips, KPI cards,
     empty states, the compact key/value strip and Plotly surfaces.
  6. **States** — a restrained skeleton shimmer for real loading.
- **`ui/components.py`** — reusable building blocks only: page header,
  section heading, `card_title`, `section_card` (a native
  `st.container(border=True)`), `status_strip`, KPI row, empty state and
  the planned-page banner. Views compose these instead of duplicating CSS.
- **Targeting** — only stable Streamlit test ids and our own class names
  are styled, so working controls are never replaced by decorative
  mockups and broad selectors cannot hide real widgets.

Accessibility and motion: visible `:focus-visible` rings, sufficient
contrast, native keyboard support, a `prefers-reduced-motion` override
that disables transitions *and* hover lift, and a `max-width: 900px`
breakpoint for narrower screens.

## Pages

| Page | Status |
| --- | --- |
| Overview | Working — welcome block, dataset status, quick actions, recent runs, KPIs and roadmap |
| Data | Working — upload card, summary metrics, key/value strip, profile and quality tabs |
| Charts | Working — distribution, relationship, correlation, time series with per-type guidance |
| AI Assistant | Working — grounded Gemini chat, suggested prompts, display-only code examples |
| Models | Working — training cards, metrics, shared-split comparison, explanations, downloads, run history |

## Models page (Milestone 6)

Pick a task and a model, choose the feature columns, and train — all
locally with scikit-learn, on the dataset loaded on the Data page:

- **Tasks**: classification (Logistic Regression, Random Forest,
  Decision Tree), regression (Linear, Random Forest, Decision Tree) and
  clustering (K-Means, MiniBatch K-Means, Agglomerative Clustering).
  The target list is filtered per task: regression needs numeric
  columns; classification needs 2–50 distinct values.
- **Determinism**: every estimator runs with a fixed `random_state`, so
  the same dataset and settings produce the same metrics.
- **Honest metrics**: accuracy / F1 / precision / recall for
  classification, R² / MAE / RMSE / MSE for regression, silhouette and
  inertia for clustering — with automatic caveats for small test sets,
  single-class test splits and rows dropped for missing values
  (features *or* target).
- **Feature importances** are shown when the model exposes them (tree
  ensembles) or derived from absolute coefficients (linear models),
  with one-hot pieces folded back onto their original column names.
- **Run history**: every training run is recorded in a local SQLite
  database (`ml_runs.db`, created on first use and git-ignored; move it
  with the `ML_RUNS_DB_PATH` environment variable). Only metadata is
  stored — task, model, column names, metrics and a dataset-shape hash.
  No rows, cell values or model binaries are persisted, and runs can be
  deleted from the page.

## Model comparison & explanations (Milestone 7)

Below the single-model flow, the same configuration drives a head-to-
head comparison:

- **Multiple models per task** — pick any selection of the task's
  models (classification and regression: three each; clustering: the
  three clustering algorithms) and compare them in one click.
- **One shared split, one shared preprocessing pass** — every
  supervised model is fitted and scored on exactly the same train and
  test rows, so the numbers are directly comparable. The one-hot
  vocabulary is fitted on the training rows only: a category that
  appears only in the held-out rows never enters the feature matrix.
- **Comparison table** — accuracy / precision / recall / F1 for
  classification, MAE / RMSE / R² for regression, silhouette for
  clustering, plus a per-model status column and the task's primary
  metric as a bar chart. A ★ marks the best model.
- **Best only when valid** — the ★ requires a finite primary metric
  from a successful run. A single class in the test set suppresses the
  highlight entirely (the metrics would be misleading); an undefined
  silhouette shows a dash and is never ranked — comparison results
  never turn "undefined" into a fake 0.0.
- **Per-model failure isolation** — a model that cannot train (for
  example one incompatible with the data) is reported with its own
  error while the remaining models still run and rank.
- **Explanations** — pick any successful model to see its feature
  attribution: impurity importances for tree ensembles, absolute
  coefficients for linear models, and permutation importance on the
  held-out split as a deterministic fallback for models that expose
  neither. Scores always describe **association with the model's
  predictions, not causation**.
- **Metric definitions** — an expander explains every metric in the
  table and whether higher or lower is better; evaluation warnings
  (small test sets, dropped rows, degenerate splits) are shown next to
  the results.
- Successful comparison runs are saved to the same run history as
  single training runs — metadata only, as before.

### Known limitations

- The comparison ranks models on a single held-out split; it is a
  quick, honest comparison, not cross-validation, and small test sets
  can still produce noisy metrics (a warning says so when it applies).
- Clustering is compared by silhouette only (there is no ground-truth
  label to score against), and no feature-importance claim is made for
  clustering runs — grouping rows is not prediction.
- Explanations describe how sensitive the fitted model is to each
  column; they are not causal statements about the data.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Coverage: loader validation and column-name safety (`tests/test_loader.py`),
profiling math and non-mutation guarantees (`tests/test_profiling.py`),
chart building, error handling and theming (`tests/test_charts.py`),
design tokens and the restyled Streamlit chrome (`tests/test_theme.py`),
components including the new cards and status strip
(`tests/test_components.py`), environment facts (`tests/test_core.py`),
AI schemas, Gemini error translation, the analysis executor, orchestration,
privacy guarantees and code validation (`tests/test_ai.py`), model
training, validation errors and SQLite persistence (`tests/test_models.py`),
the shared-split comparison engine — metric directions, failure
isolation, determinism, missing values, invalid datasets and the
no-leakage encoding (`tests/test_comparison.py`) — and feature
attribution (`tests/test_explain.py`), the export layer — CSV contents,
empty results, failed models, special characters, serialisation,
privacy safeguards and reproducibility metadata (`tests/test_export.py`)
— plus full-page rendering via `AppTest` (`tests/test_shell.py`,
`tests/test_ai_shell.py`, `tests/test_models_shell.py`).
All fixtures are generated in memory and Gemini is mocked — no API key,
no network access, no private files. The shell tests point the run
history at a throwaway database (`ML_RUNS_DB_PATH`) so they neither
depend on nor write to the real local history.

## API keys

The AI Assistant reads `GEMINI_API_KEY` (required) and `GEMINI_MODEL`
(optional, defaults to `gemini-3.8-flash`) from, in order: the process
environment, a `.env` file in the project root, or
`.streamlit/secrets.toml`. All three locations are git-ignored; the key
is never hardcoded, displayed, logged, or included in error messages.
`.env.example` documents both settings.

## Export, reports & reproducibility (Milestone 8)

Every trained run and every comparison can now be taken off the screen
as files. All of it lives in `core/export.py` (framework-free) and is
surfaced on the Models page as download buttons inside the existing
design system.

### What you can download

| File | Contents |
| --- | --- |
| **Comparison metrics (CSV)** | One row per model: every metric, a status column (`OK` / `failed`), and a column marking the best model. Metric headers state the direction (`RMSE (lower is better)`). |
| **Run metadata (CSV)** | Per-model task, target, feature columns, split sizes, dropped-row count and error — failed models included, so nothing is hidden. |
| **Predictions (CSV)** | The **held-out rows only**: a row identifier, the actual value and the prediction. Clustering exports the assigned cluster label instead. |
| **Configuration (CSV / JSON)** | The reproducibility record: task, model, target, selected features, test split, random seed, cluster count, train/test/dropped row counts and the preprocessing decisions. |
| **Analysis report (HTML)** | A self-contained report: dataset context, model configuration, valid metrics, warnings, the full comparison and inline SVG charts. |

Filenames are slugged automatically (`sales.csv-comparison-metrics.csv`),
so special characters in a dataset or model name can never produce a
broken download.

### The report

The HTML report is one file with an inline stylesheet — no external
assets, no network access, no JavaScript. It opens in any browser and
prints to PDF. It contains:

- **Dataset context** — file name, row and column counts and the dataset
  hash (aggregate context only, never raw rows).
- **Model configuration & reproducibility** — every setting listed above,
  including the preprocessing decisions that were actually applied.
- **Evaluation metrics** — only *valid* metrics, each labelled with its
  direction, plus the task's primary metric.
- **Warnings** — evaluation warnings (small test sets, dropped rows,
  degenerate splits) listed verbatim.
- **Model comparison** — a table of the successful models and a separate
  table of the failed ones with their errors, plus a bar chart of the
  primary metric.
- **Feature importance** — when attribution is available, a bar chart and
  the note that importance describes **association with the model's
  predictions, not causation**.

### Honesty guarantees

- A failed model is exported with its error and **empty** metric cells —
  never a fabricated `0`.
- An undefined metric (an invalid silhouette, for example) is an empty
  cell and renders as a dash. When the evaluation is not valid, no best
  model is declared at all.
- Warnings appear exactly once, in their own section.
- The report always states that the metrics were computed by the app,
  not by a language model.
- Feature importance is never presented as causal evidence.

### Privacy guarantees

- No raw dataset rows are exported. The only row-level output is the
  prediction file you explicitly asked for, and it contains the held-out
  row identifier, the actual value and the prediction — **never the
  feature values**.
- No API keys, secrets or model binaries are ever written. The exporter
  records *configuration settings* only.
- The stored run configuration is metadata only: task, model, target,
  features, seed, split sizes and preprocessing notes.

### Reproducibility record

`core/storage.py` gained a `config` column that stores the run's
reproducibility settings next to its metrics. Existing databases are
upgraded in place (`ALTER TABLE ... ADD COLUMN`), so no run history is
lost. A run saved before Milestone 8 simply reports an empty
configuration.

The recorded seed is the seed the app actually used (`42` today), and
the split sizes are the exact ones the estimator saw — so a run can be
repeated from the recorded configuration.

### Known limitations

- The report is HTML, not PDF. Printing to PDF uses the browser's own
  print dialog; there is no server-side PDF renderer (that would need an
  extra dependency).
- The predictions file contains the held-out rows only. Re-scoring the
  training rows is deliberately not offered, because that is not the
  data the metrics were computed on.
- The recorded configuration describes the *settings*; it does not embed
  the dataset itself, so reproducing a run still requires the same source
  file (the dataset hash makes a mismatch detectable).
- Cluster labels are exported for the rows the clustering run was fitted
  on; there is no held-out split for an unsupervised task.

## Milestones

1. Project setup — done.
2. Design system & application shell — done.
3. Dataset upload & inspection — done.
4. Interactive Plotly charts — done.
5. Gemini AI assistant — done.
6. scikit-learn models, persistence and expanded tests — done.
7. Model comparison & explainability — done.
8. **Export, reports & reproducibility — done.**
