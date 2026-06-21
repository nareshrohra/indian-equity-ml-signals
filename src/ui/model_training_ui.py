import os
import json
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import ipywidgets as widgets
from IPython.display import display, clear_output

from modules.data_helper import load_data
from modules.model_helper import prepare_features, create_target, filter_training_data
from modules.models import ClassifierModelsRegistry

_FEATURE_CONFIGS_PATH = 'config/feature_configs.json'
_FILTER_CONFIGS_PATH = 'config/filter_configs.json'
_SAVE_BASE = os.path.join('..', 'trained_models', 'bullish', 'classifiers')
_plt_lock = threading.Lock()


def _sanitize_experiment(name: str) -> str:
    name = name.strip().strip('/\\')
    if not name:
        return ''
    if '..' in name or any(c in name for c in '<>:"|?*'):
        raise ValueError(f'Invalid experiment name: "{name}"')
    return name


def _load_configs() -> dict[str, list[str]]:
    with open(_FEATURE_CONFIGS_PATH) as f:
        data = json.load(f)
    feature_groups = data.get('features', {})
    resolved = {}
    for config_name, groups in data.get('configs', {}).items():
        cols: list[str] = []
        seen: set[str] = set()
        for item in groups:
            entries = feature_groups.get(item, [item])
            for col in entries:
                if col not in seen:
                    seen.add(col)
                    cols.append(col)
        resolved[config_name] = cols
    return resolved


def _load_filter_configs() -> dict[str, str]:
    if not os.path.exists(_FILTER_CONFIGS_PATH):
        return {}
    with open(_FILTER_CONFIGS_PATH) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# JOB ROW
# ══════════════════════════════════════════════════════════════════════════════

class _JobRow:
    def __init__(self, model_key: str, config_key: str):
        self.model_key = model_key
        self.config_key = config_key

        self.cb = widgets.Checkbox(
            value=True, indent=False,
            layout=widgets.Layout(width='44px'),
        )
        self.status = widgets.HTML(
            value='<span style="color:#aaa">● Pending</span>',
            layout=widgets.Layout(width='220px'),
        )
        self.out = widgets.Output(layout=widgets.Layout(
            max_height='300px', overflow_y='auto',
        ))
        self._acc = widgets.Accordion(children=[self.out])
        self._acc.set_title(0, f'{model_key}  ×  {config_key}')
        self._acc.selected_index = None  # collapsed by default

        header = widgets.HBox([
            self.cb,
            widgets.Label(model_key, layout=widgets.Layout(width='100px')),
            widgets.Label(config_key, layout=widgets.Layout(width='160px')),
            self.status,
        ])
        self.widget = widgets.VBox([header, self._acc])

    def set_status(self, html: str):
        self.status.value = html

    def expand(self):
        self._acc.selected_index = 0


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING WORKER
# ══════════════════════════════════════════════════════════════════════════════

def _run_job(job: _JobRow, feature_list: list, training_data, test_data,
             cutoff: str, end_date, experiment: str, filter_names: list[str]):
    import time
    from sklearn.metrics import (
        classification_report as cr_fn,
        roc_auc_score,
        confusion_matrix as cm_fn,
    )

    model_name = f'{job.model_key}_{job.config_key}'
    save_dir = os.path.join(_SAVE_BASE, experiment, model_name) if experiment \
        else os.path.join(_SAVE_BASE, model_name)

    job.set_status('<span style="color:#1a73e8">⟳ Training…</span>')
    job.expand()

    with job.out:
        try:
            if os.path.exists(save_dir):
                job.set_status('<span style="color:#e8a000">⚠ Already exists</span>')
                print(f'Skipped — "{model_name}" already exists.')
                return

            available = [f for f in feature_list if f in training_data.columns]
            missing = set(feature_list) - set(available)
            if missing:
                print(f'⚠ {len(missing)} feature(s) absent from loaded data (ignored): {sorted(missing)}')
            if not available:
                job.set_status('<span style="color:#c00">✗ No features</span>')
                print('No valid features found in loaded data.')
                return

            print(
                f'Training {job.model_key} on {len(training_data):,} rows '
                f'with {len(available)} features  (config: {job.config_key})…'
            )
            model_class = ClassifierModelsRegistry.get(job.model_key)
            model = model_class(feature_columns=available)
            train_start = time.perf_counter()
            model.train(training_data)
            training_time_secs = time.perf_counter() - train_start
            print(f'Training took {training_time_secs:.2f}s.')

            # ── Evaluate (bypass show_prediction_report to keep thread-safe) ──
            print(f'\nEvaluating on {len(training_data):,} training rows…')
            train_prep = prepare_features(training_data, available)
            train_prep = create_target(train_prep)
            y_preds_train, y_probs_train = model.predict(train_prep)
            y_true_train = train_prep['Target']

            print(f'Evaluating on {len(test_data):,} held-out rows…')
            test_prep = prepare_features(test_data, available)
            test_prep = create_target(test_prep)
            y_preds, y_probs = model.predict(test_prep)
            y_true = test_prep['Target']

            def _stats(y_t, y_p, y_pr):
                return {
                    'roc_auc': round(float(roc_auc_score(y_t, y_pr)), 4),
                    'classification_report': cr_fn(y_t, y_p, output_dict=True),
                    'confusion_matrix': cm_fn(y_t, y_p).tolist(),
                }

            # ── Metadata ──────────────────────────────────────────────────────
            importance = model.get_feature_importance()
            features_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)

            metadata = {
                'name': model_name,
                'guid': str(uuid.uuid4()),
                'saved_at': str(date.today()),
                'train_start': str(training_data['Date'].min()),
                'train_cutoff': cutoff,
                'data_end': str(end_date),
                'n_train_rows': len(training_data),
                'n_test_rows': len(test_data),
                'training_filters': filter_names,
                'training_time_secs': round(training_time_secs, 2),
                'features': [
                    {'name': f, 'importance': round(float(imp), 6)}
                    for f, imp in features_sorted
                ],
                'stats': {
                    'train': _stats(y_true_train, y_preds_train, y_probs_train),
                    'test': _stats(y_true, y_preds, y_probs),
                },
            }

            model.save_to_dir(save_dir, metadata)

            # Serialise matplotlib calls to avoid cross-thread figure state issues
            with _plt_lock:
                model.save_eval_plots(save_dir, y_true, y_preds, y_probs)

            roc_train = metadata['stats']['train']['roc_auc']
            roc_test = metadata['stats']['test']['roc_auc']
            job.set_status(
                f'<span style="color:#0a0">✓ Done  ROC train {roc_train:.3f} / test {roc_test:.3f}  '
                f'({training_time_secs:.1f}s)</span>'
            )
            print(
                f'\n✓ Saved → {save_dir}  '
                f'(ROC AUC — train: {roc_train:.4f}, test: {roc_test:.4f}; '
                f'training time: {training_time_secs:.2f}s)'
            )

        except Exception as exc:
            import traceback
            job.set_status('<span style="color:#c00">✗ Error</span>')
            print(f'Error: {exc}\n{traceback.format_exc()}')


# ══════════════════════════════════════════════════════════════════════════════
# RENDER
# ══════════════════════════════════════════════════════════════════════════════

def render():
    try:
        feature_configs = _load_configs()
    except FileNotFoundError:
        print(f'Feature configs not found at: {_FEATURE_CONFIGS_PATH}')
        return

    filter_configs = _load_filter_configs()

    model_keys = list(ClassifierModelsRegistry.list().keys())
    config_keys = list(feature_configs.keys())

    # ── Date pickers & load ────────────────────────────────────────────────────
    w_end_date = widgets.DatePicker(
        description='Data end date',
        value=date(2026, 3, 31),
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='260px'),
    )
    w_cutoff = widgets.DatePicker(
        description='Training cutoff',
        value=date(2024, 3, 31),
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='260px'),
    )
    btn_load = widgets.Button(
        description='Load Training Data',
        button_style='info',
        icon='download',
        layout=widgets.Layout(width='200px'),
    )
    out_load = widgets.Output()

    w_experiment = widgets.Text(
        description='Experiment name',
        placeholder='optional, e.g. high_turnover_universe',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='320px'),
    )

    w_filters = widgets.SelectMultiple(
        options=list(filter_configs.keys()),
        description='Training filters',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='320px', height='80px'),
    )

    # ── Job matrix ────────────────────────────────────────────────────────────
    jobs = [
        _JobRow(mk, ck)
        for mk in model_keys
        for ck in config_keys
    ]

    btn_select_all = widgets.Button(
        description='Select all',
        layout=widgets.Layout(width='100px'),
    )
    btn_deselect_all = widgets.Button(
        description='Deselect all',
        layout=widgets.Layout(width='100px'),
    )

    # ── Training controls ─────────────────────────────────────────────────────
    w_workers = widgets.BoundedIntText(
        value=2, min=1, max=max(1, len(jobs)),
        description='Max parallel',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='190px'),
    )
    btn_start = widgets.Button(
        description='Start Training',
        button_style='success',
        icon='rocket',
        disabled=True,
        layout=widgets.Layout(width='180px'),
    )
    out_status = widgets.Output()

    state: dict = {}

    # ── Handlers ──────────────────────────────────────────────────────────────
    def on_load(b):
        with out_load:
            clear_output()
            print('Loading training data…')
        end = w_end_date.value
        start = end - timedelta(days=365 * 10)
        try:
            df = load_data('learning', start, end)
            cutoff = str(w_cutoff.value)
            state['training_data'] = df[df['Date'] <= cutoff].copy()
            state['test_data']     = df[df['Date'] > cutoff].copy()
            state['cutoff']    = cutoff
            state['end_date']  = end
            btn_start.disabled = False
            n_train = len(state['training_data'])
            n_test  = len(state['test_data'])
            with out_load:
                clear_output()
                print(
                    f'✓ Loaded {len(df):,} rows  →  '
                    f'{n_train:,} train / {n_test:,} test  (cutoff {cutoff})'
                )
        except Exception as exc:
            with out_load:
                clear_output()
                print(f'Error: {exc}')

    def on_select_all(b):
        for j in jobs:
            j.cb.value = True

    def on_deselect_all(b):
        for j in jobs:
            j.cb.value = False

    def on_start(b):
        selected = [j for j in jobs if j.cb.value]
        if not selected:
            with out_status:
                clear_output()
                print('No jobs selected.')
            return

        try:
            experiment = _sanitize_experiment(w_experiment.value)
        except ValueError as exc:
            with out_status:
                clear_output()
                print(f'Error: {exc}')
            return

        filter_names = list(w_filters.value)
        try:
            train_data = filter_training_data(
                state['training_data'],
                [filter_configs[name] for name in filter_names],
            )
        except Exception as exc:
            with out_status:
                clear_output()
                print(f'Error applying training filter(s): {exc}')
            return

        if filter_names:
            with out_status:
                clear_output()
                print(
                    f'Applied filter(s) {filter_names}: '
                    f'{len(state["training_data"]):,} → {len(train_data):,} training rows'
                )

        btn_start.disabled = True
        btn_load.disabled  = True
        for j in selected:
            j.set_status('<span style="color:#aaa">⌛ Queued</span>')

        def _run_all():
            try:
                with ThreadPoolExecutor(max_workers=w_workers.value) as pool:
                    futures = {
                        pool.submit(
                            _run_job, j,
                            feature_configs[j.config_key],
                            train_data,
                            state['test_data'],
                            state['cutoff'],
                            state['end_date'],
                            experiment,
                            filter_names,
                        ): j
                        for j in selected
                    }
                    for _ in as_completed(futures):
                        pass

                done = sum(1 for j in selected if '✓' in j.status.value)
                errs = sum(1 for j in selected if '✗' in j.status.value)
                with out_status:
                    clear_output()
                    print(f'All done.  ✓ {done} saved  ✗ {errs} errors  ({len(selected)} jobs total)')
            except Exception as exc:
                import traceback
                with out_status:
                    clear_output()
                    print(f'Error: {exc}\n{traceback.format_exc()}')
            finally:
                btn_start.disabled = False
                btn_load.disabled  = False

        threading.Thread(target=_run_all, daemon=True).start()

    btn_load.on_click(on_load)
    btn_select_all.on_click(on_select_all)
    btn_deselect_all.on_click(on_deselect_all)
    btn_start.on_click(on_start)

    # ── Layout ────────────────────────────────────────────────────────────────
    col_header = widgets.HBox([
        widgets.HTML('', layout=widgets.Layout(width='44px')),
        widgets.HTML('<b>Model</b>',          layout=widgets.Layout(width='100px')),
        widgets.HTML('<b>Feature config</b>', layout=widgets.Layout(width='160px')),
        widgets.HTML('<b>Status</b>'),
    ])

    jobs_box = widgets.VBox(
        [col_header] + [j.widget for j in jobs],
        layout=widgets.Layout(
            border='1px solid #ddd',
            padding='8px',
        ),
    )

    ui = widgets.VBox([
        widgets.HTML('<h3 style="margin:8px 0">Batch Model Training</h3>'),
        widgets.HTML('<hr style="margin:4px 0">'),
        widgets.HBox([w_end_date, w_cutoff], layout=widgets.Layout(gap='16px')),
        widgets.HBox([btn_load], layout=widgets.Layout(margin='6px 0 2px')),
        out_load,
        widgets.HTML('<hr style="margin:8px 0">'),
        widgets.HTML('<b>Training jobs</b>'),
        w_experiment,
        w_filters,
        widgets.HBox(
            [btn_select_all, btn_deselect_all, w_workers, btn_start],
            layout=widgets.Layout(gap='8px', align_items='center', margin='4px 0 6px'),
        ),
        jobs_box,
        out_status,
    ], layout=widgets.Layout(padding='12px', width='760px'))

    display(ui)
