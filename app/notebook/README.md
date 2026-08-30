# Wadati Quality-Control Notebook — environment & launch

## 1. Generate `environment.yml` and the notebook

From the repository root:

```bash
python -m app.notebook.build_notebook
```

This writes into `notebooks/`:

- `notebooks/environment.yml` — reproducible conda-forge environment (Python 3.11)
- `notebooks/wadati_qc.ipynb` — the runnable Wadati workflow notebook
- `notebooks/wadati_ingest.py` — pure-Python CSV / QuakeML loaders and validation
- `notebooks/wadati_samples.py` — deterministic sample CSV tables and QuakeML builder
- `notebooks/wadati_discovery.py` — real ObsPy FDSN discovery of LO / OSPL stations and
  Hispaniola-region earthquakes
- `notebooks/wadati_subset.py` — the exhaustive Wadati subset search and its CSV exports
- `notebooks/wadati_subset_examples.py` — the five worked subset-search examples
- `notebooks/wadati_dashboard.py` — the embedded Panel dashboard: session logic, manual
  re-picking, optional SeisBench assistance and the second pass

The `.py` modules are copies of `app/notebook/ingest.py`,
`app/notebook/samples.py`, `app/notebook/discovery.py`,
`app/notebook/subset_search.py`, `app/notebook/subset_examples.py` and
`app/notebook/dashboard.py`;
the notebook imports them from its own directory, so
the notebook exercises exactly the shipped loader code. Running the notebook
also creates `notebooks/data/` with the sample pick tables, the generated
`sample_catalog.xml`, and the normalized `canonical_picks.csv`.

## Canonical pick schema

Both input families normalize onto one row per event/station P–S pair:
`event_id`, `station_id`, optional `network` / `channel`, `source`,
`origin_time`, `p_time`, `s_time`, `p_travel_time`, `s_minus_p`.

Accepted CSV layouts:

- long — `event_id, station_id, phase, time` (+ `origin_time`) or
  `event_id, station_id, phase, travel_time`
- wide — `event_id, station_id, p_time, s_time` (+ `origin_time`) or
  `event_id, station_id, p_travel_time, s_minus_p`

QuakeML: anything `read_events(path, format="quakeml")` accepts. Picks are
paired per event and station via `pick.waveform_id`, the phase comes from
`pick.phase_hint` (falling back to the origin arrival phase), and P travel
times are measured against the preferred origin, then the first available
origin carrying a time.

Validation codes: `malformed_time`, `unknown_phase`, `duplicate_phase`,
`missing_p`, `missing_s`, `s_not_after_p`, `missing_origin_time`,
`origin_time_conflict`, `insufficient_pairs`. `print(SCHEMA_DOC)` in the
notebook is the authoritative reference.

## Ingestion result API

`load_csv_picks`, `load_quakeml_picks` and `load_picks` all return an
`IngestResult`. The methods that actually exist on it are:
`pairs`, `usable_pairs`, `event_ids`, `event_pairs(event_id)`,
`event_summary()`, `station_summary()`, `as_rows()`, `to_dataframe()`,
`issues` / `errors` / `warnings`, `ok` and `report()`. Use
`write_canonical_csv(result, path)` to persist the canonical table.
`print(SCHEMA_DOC)` documents the same list in the notebook.

## Real data discovery — LO / OSPL and Hispaniola

**Hispaniola** is the Caribbean island shared by the **Dominican Republic** and
**Haiti**. Network **LO** is the *Observatorio Sismológico Politécnico Loyola*
(**OSPL**) network of the **Dominican Republic**.

`wadati_discovery.py` provides:

- `fetch_lo_stations(starttime=…, endtime=…, bbox=HISPANIOLA_BBOX)` — station
  metadata for network LO from **EARTHSCOPE**, the only FDSN node that serves LO.
  Tidy rows: `station_uid` (`LO.LOBH`), `network`, `station_id`, `site`,
  `latitude`, `longitude`, `elevation_m`, `start_date`, `end_date`, `channels`.
- `fetch_hispaniola_events(starttime=…, endtime=…, minmagnitude=3.0, bbox=…)` —
  regional earthquakes from **USGS**. Tidy rows: `event_uid`, `event_id`,
  `origin_time`, `latitude`, `longitude`, `depth_km`, `magnitude`,
  `magnitude_type`, `event_type`, `region`.
- `HISPANIOLA_BBOX = (16.5, 20.5, -75.5, -67.0)` — the documented default region:
  16.5–20.5 N, 75.5–67.0 W, broad enough for the whole island plus offshore
  seismicity (Septentrional and Enriquillo fault zones, Muertos trough, western
  Puerto Rico trench approaches). `print(BBOX_DOC)` in the notebook.
- `validate_window`, `validate_bbox`, `default_window(days)` and
  `DiscoveryResult` with `rows`, `options()`, `ids()`, `select(uids)`,
  `to_dataframe()`, `messages` / `errors` / `warnings`, `ok` and `report()`.

No query runs at import time — the notebook drives every query from a Panel
button, and the selections come from `MultiChoice` / `MultiSelect` widgets
populated with the real inventory and catalog. No-data (HTTP 204), malformed
date or magnitude ranges, provider failures (HTTP 400, service outage) and
offline/DNS errors are all reported as actionable messages on the returned
result (`no_data`, `malformed_range`, `bad_request`, `provider_unreachable`,
`provider_failure`, `offline`) instead of raising.

FDSN discovery needs internet access; without it, work from the sample CSV and
QuakeML files, which are fully offline.

## Exhaustive Wadati subset search

`wadati_subset.py` (`app/notebook/subset_search.py`) implements step 3:

1. Fit `s_minus_p` against `p_travel_time` over **all** valid station pairs with
   `scipy.stats.linregress`.
2. If Pearson `|r| >= min_correlation` (default **0.9**), the retained station
   count reaches `min_stations`, and the implied `Vp/Vs` is inside the optional
   `[vp_vs_min, vp_vs_max]` bounds — accept every pick, remove nothing.
3. Otherwise enumerate subsets by removing one point, then two, then three …
   larger subsets are always tested before smaller ones, and the search stops at
   the **first** removal depth that produces qualifying candidates.
4. Choose deterministically: highest `|r|`, then lowest RMSE, then the stable
   station ordering of the removed stations.
5. Reject the event when no qualifying subset remains at or above
   `min_stations`, which is clamped up to the absolute floor of **3**.

Status / reason codes: `accepted_all_picks`, `accepted_after_removal`,
`rejected_correlation` (inconsistent picks), `rejected_vp_vs` (consistent picks,
anomalous velocity ratio — deliberately distinct from a correlation failure),
`rejected_insufficient_stations`, `rejected_search_truncated`.

API:

- `SearchConfig(min_correlation=0.9, min_stations=4, vp_vs_min=1.50,
  vp_vs_max=2.10, max_removals=None, max_combinations=200_000)`, with
  `normalized()` and `describe()`; `print(CONFIG_DOC)` in the notebook.
- `search_event(event_id, points, config)` → `EventQC` with `status`, `reason`,
  `reason_detail`, `retained` / `removed` / `outlier_stations`, `fit` and
  `initial_fit` (`slope`, `intercept`, `pearson_r`, `r_squared`, `p_value`,
  `slope_stderr`, `intercept_stderr`, `rmse`, `mae`, `max_abs_residual`,
  `residual_std`, `vp_vs`, `vp_vs_stderr`), `search_depth`,
  `max_depth_searched`, `combinations_evaluated`, `as_row()`, `pick_rows()`,
  `plot_frame()`, `plot_fit()` and `report()`.
- `points_from_pairs(pairs)` converts canonical `IngestResult` pairs to
  `WadatiPoint`s; `run_subset_search(ingest_result, config)` → `DatasetQC` with
  `accepted`, `rejected`, `reason_counts()`, `summary()`, `event_rows()`,
  `pick_rows()`, `event_dataframe()`, `pick_dataframe()` and `report()`.
- Exports: `write_event_qc_csv(dataset, path)`,
  `write_pick_qc_csv(dataset, path)`, `event_qc_csv_text(dataset)` and
  `pick_qc_csv_text(dataset)` — the last two back the notebook's
  `FileDownload` widgets.

Worked examples (`python -m app.notebook.subset_examples`): `all_pass`,
`one_outlier`, `two_outliers`, `anomalous_vp_vs`, `irrecoverable`.
`check_examples()` asserts the expected status, reason, search depth and removed
stations for each.

## Embedded Panel dashboard (step 4 — the workflow is complete)

`wadati_dashboard.py` (`app/notebook/dashboard.py`) has two separate layers.

**Logic — `DashboardSession(ingest_result, config)`** (pure Python, testable):

- `original` — the immutable first-pass `DatasetQC`; `current(event_id)` returns the
  revised `EventQC` once the event has been rerun, otherwise the original.
- `event_ids()`, `filtered_event_ids(f)` with `f` in `all` / `accepted` / `rejected` /
  `edited` / `revised`, and `neighbour(event_id, ±1, filter)` for ◀ / ▶ navigation.
- `picks(event_id)`, `pick(event_id, station_id)`, `points(event_id)`,
  `station_ids(event_id)` and `pick_table(event_id)` — the editable table rows carry
  `state` (`retained` / `outlier`), `revised`, `residual`, `apparent_vp_vs` and
  `provenance`.
- Editing: `stage_edit(event_id, station_id, p_travel_time=…, s_minus_p=…, p_time=…,
  s_time=…, provenance=…, note=…)` validates first (P travel time and S−P must be
  finite and strictly positive, absolute ISO times need the event origin time, and an
  edit that changes nothing is refused), then `apply_pending()`, `discard_pending()`,
  `undo_last(event_id)`, `reset_event(event_id)` and `reset_all()`. Every applied edit
  is a `PickEdit` provenance record — `edit_log_rows()` is the audit table.
- Second pass: `rerun_event(event_id)`, `rerun_all()`, `revised_dataset()`.
- Comparison and exports: `comparison_rows(event_id)`, `dataset_comparison()`,
  `export_csv(which, kind)` for `original`/`revised` × `events`/`picks`, and
  `write_exports(directory)`.
- `self_check()` asserts all of the above fully offline and loads no model weights;
  run it with `python -m app.notebook.dashboard`.

**Panel — `build_dashboard(session, export_dir=…)`**: the dense asymmetric workspace
centred on the real Wadati plot (`wadati_figure`) — warm mineral paper, charcoal and
slate type, geological rule lines, deep **teal** retained/accepted, **seismic red**
outlier/rejected, **amber** revised picks. Event navigation and status filtering, the
linked editable pick table, staged/applied/undo/reset controls, the second-pass
buttons, the side-by-side original-versus-revised metric and dataset tables, and the
four `FileDownload` QC exports.

**Optional waveform-assisted SeisBench re-picking** (right-hand column, all lazy):

- `SUPPORTED_MODELS = ("PhaseNet", "EQTransformer", "GPD")` — real SeisBench classes.
- `available_weights(model)` calls the real `list_pretrained()`; `load_picker(model,
  weights)` loads on the **CPU** in `eval()` mode and caches it, only when requested.
- `read_mseed(bytes | path)` reads uploaded or local MiniSEED with ObsPy;
  `repick_stream(stream, model, weights, min_confidence=…)` runs the official
  `model.classify(stream)` API and returns `CandidatePick`s with their real confidence.
- A candidate reaches a pick only through `stage_candidate_pair(...)` after an explicit
  selection — picks are never overwritten silently, and there are no mock predictions.
- Truthful error codes: `seisbench_missing`, `torch_missing`, `unsupported_model`,
  `unknown_weights`, `weights_unavailable`, `offline`, `inference_failed`,
  `read_failed`. The first weight load downloads to `~/.seisbench/models`; offline with
  an empty cache is reported, not faked.

The dashboard also embeds the real FDSN selection: LO / OSPL stations from EARTHSCOPE
and Hispaniola-region earthquakes from USGS, using the same `fetch_lo_stations` /
`fetch_hispaniola_events` helpers and never querying until a button is pressed.

## 2. Create the environment (conda-forge highest priority)

```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
conda env create -f notebooks/environment.yml
conda activate wadati-qc
```

`mamba env create -f notebooks/environment.yml` is a faster drop-in.

## 3. Launch JupyterLab

```bash
jupyter lab notebooks/wadati_qc.ipynb
```

Panel is initialised inside the notebook with `pn.extension()`. `pyviz_comms`
must be present in the *same* environment that runs JupyterLab (it is pinned in
`environment.yml`), so no manual lab extension install is needed on JupyterLab 4.

## Notes

- PyTorch is pinned to the CPU build; SeisBench models run on CPU only.
- ObsPy QuakeML files must be read with an explicit format:
  `read_events(path, format="quakeml")`.
- The first SeisBench weight load downloads to `~/.seisbench/models/`.
