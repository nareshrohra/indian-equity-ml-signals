import json
import os

import ipywidgets as widgets
from IPython.display import display, clear_output, Image as IPImage
from datetime import date

from modules.data_helper import load_data
from modules.models import ClassifierModel

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    'live_data': None,
}

_MODELS_BASE = os.path.join('..', 'trained_models', 'bullish', 'classifiers')
_ROOT_LABEL = '(root)'


def _is_model_dir(path: str) -> bool:
    return os.path.exists(os.path.join(path, 'metadata.json'))


def _list_experiments() -> list[str]:
    """Experiment folders under _MODELS_BASE that hold trained models, plus
    '(root)' if _MODELS_BASE itself has model folders saved directly inside it."""
    options = []
    if not os.path.isdir(_MODELS_BASE):
        return options
    if any(_is_model_dir(os.path.join(_MODELS_BASE, d)) for d in os.listdir(_MODELS_BASE)
           if os.path.isdir(os.path.join(_MODELS_BASE, d))):
        options.append(_ROOT_LABEL)
    for d in sorted(os.listdir(_MODELS_BASE)):
        if d == '_old':
            continue
        full = os.path.join(_MODELS_BASE, d)
        if not os.path.isdir(full) or _is_model_dir(full):
            continue
        if any(_is_model_dir(os.path.join(full, sub)) for sub in os.listdir(full)
               if os.path.isdir(os.path.join(full, sub))):
            options.append(d)
    return options


def _resolve_base(experiment: str) -> str:
    if not experiment or experiment == _ROOT_LABEL:
        return _MODELS_BASE
    return os.path.join(_MODELS_BASE, experiment)


# ══════════════════════════════════════════════════════════════════════════════
# RUN PREDICTIONS TAB
# ══════════════════════════════════════════════════════════════════════════════

w_experiment_selector = widgets.Dropdown(
    options=[],
    value=None,
    description='Experiment',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='320px')
)

w_model_selector = widgets.Dropdown(
    options=[],
    value=None,
    description='Model',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='320px')
)

btn_refresh_models = widgets.Button(
    description='Refresh',
    button_style='',
    icon='refresh',
    layout=widgets.Layout(width='100px')
)

w_live_start = widgets.DatePicker(
    description='Live start date',
    value=date(2025, 4, 1),
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='260px')
)

btn_run_preds = widgets.Button(
    description='Run Predictions',
    button_style='primary',
    icon='play',
    layout=widgets.Layout(width='180px')
)

out_run = widgets.Output()

out_model_details = widgets.Output(
    layout=widgets.Layout(
        overflow_y='auto', overflow_x='hidden',
        max_height='560px', width='100%',
    )
)


def _refresh_experiment_list():
    options = _list_experiments()
    current = w_experiment_selector.value
    w_experiment_selector.options = options
    w_experiment_selector.value = current if current in options else (options[0] if options else None)


def _refresh_model_list():
    base = _resolve_base(w_experiment_selector.value)
    options = []
    if os.path.isdir(base):
        folders = sorted(
            [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))],
            key=lambda d: os.path.getmtime(os.path.join(base, d)),
            reverse=True,
        )
        for folder in folders:
            meta_path = os.path.join(base, folder, 'metadata.json')
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    options.append((meta['name'], folder))
                except Exception:
                    options.append((folder, folder))
    current_val = w_model_selector.value
    w_model_selector.options = options
    all_values = [v for _, v in options]
    if current_val not in all_values:
        w_model_selector.value = options[0][1] if options else None


def show_model_details(folder):
    with out_model_details:
        clear_output()
        if not folder:
            print('No model selected.')
            return

        base = _resolve_base(w_experiment_selector.value)
        meta_path = os.path.join(base, folder, 'metadata.json')
        if not os.path.exists(meta_path):
            print('metadata.json not found.')
            return
        with open(meta_path) as f:
            meta = json.load(f)
        cm = meta['stats']['confusion_matrix']

        print(f"{'═' * 38}")
        print(f"  {meta['name']}")
        print(f"{'═' * 38}")
        print(f"Saved : {meta['saved_at']}")
        print(f"Train : {meta['train_start']}  →  {meta['train_cutoff']}")
        print(f"End   : {meta['data_end']}")
        print(f"Rows  : {meta['n_train_rows']:,} train / {meta['n_test_rows']:,} test")
        print(f"\n── Evaluation ─────────────────────")
        print(f"ROC AUC : {meta['stats']['roc_auc']}")
        print(f"\nClassification Report:")
        print(meta['stats']['classification_report'])
        print(f"Confusion Matrix:")
        print(f"  TN={cm[0][0]:,}  FP={cm[0][1]:,}")
        print(f"  FN={cm[1][0]:,}  TP={cm[1][1]:,}")
        print(f"\n── Top Features ───────────────────")
        feats = meta['features']
        max_imp = feats[0]['importance'] if feats else 1
        for feat in feats[:15]:
            bar = '█' * int(feat['importance'] / max_imp * 12)
            print(f"  {feat['name']:<24} {feat['importance']:.4f}  {bar}")
        for fname in ['roc_pr_curves.png', 'confusion_matrix.png']:
            img_path = os.path.join(base, folder, fname)
            if os.path.exists(img_path):
                display(IPImage(img_path, width=360))


def on_model_selector_change(change):
    show_model_details(change['new'])

w_model_selector.observe(on_model_selector_change, names='value')


def on_experiment_selector_change(change):
    _refresh_model_list()

w_experiment_selector.observe(on_experiment_selector_change, names='value')


def on_refresh_models(b):
    _refresh_experiment_list()
    _refresh_model_list()

btn_refresh_models.on_click(on_refresh_models)


def on_run_predictions(b):
    with out_run:
        clear_output()
        folder = w_model_selector.value
        if not folder:
            print('No model selected. Use the Refresh button to load available models.')
            return
        try:
            save_dir = os.path.join(_resolve_base(w_experiment_selector.value), folder)
            model = ClassifierModel.load_from_dir(save_dir)
            print(f'Loaded model from "{folder}".')
        except Exception as exc:
            print(f'Error loading model: {exc}')
            return
        print('Loading live data…')
        try:
            live_data = load_data(
                'live',
                w_live_start.value,
                date.today()
            )
            print(f'Loaded {len(live_data):,} records. Running predictions…')
            live_preds, live_probs = model.predict(live_data)
            live_data = live_data.copy()
            live_data['predictions'] = live_preds
            live_data['probability'] = live_probs
            state['live_data'] = live_data
            bullish_count = int((live_data['predictions'] == 1).sum())
            print(f'✓ Done. {bullish_count:,} bullish signals found across all dates.')
            print('Switch to the View Predictions tab to filter by date.')
        except Exception as exc:
            print(f'Error: {exc}')


btn_run_preds.on_click(on_run_predictions)

run_controls = widgets.VBox([
    widgets.HTML('<h3 style="margin:8px 0">Run predictions on live data</h3>'),
    widgets.HTML('<b>Select experiment, then model</b>'),
    widgets.HBox([w_experiment_selector, btn_refresh_models],
                 layout=widgets.Layout(gap='8px', align_items='center', margin='0 0 8px 0')),
    w_model_selector,
    w_live_start,
    widgets.HBox([btn_run_preds], layout=widgets.Layout(margin='8px 0')),
    out_run,
], layout=widgets.Layout(width='450px', padding='12px'))

run_details_panel = widgets.VBox([
    widgets.HTML('<b style="font-size:13px; margin:4px 0">Model Details</b>'),
    out_model_details,
], layout=widgets.Layout(
    width='390px',
    padding='8px 10px',
    border='1px solid #d0d0d0',
    margin='8px 0 0 8px',
    overflow='hidden',
))

run_tab = widgets.HBox(
    [run_controls, run_details_panel],
    layout=widgets.Layout(width='860px')
)


# ══════════════════════════════════════════════════════════════════════════════
# VIEW PREDICTIONS TAB
# ══════════════════════════════════════════════════════════════════════════════

w_target_date = widgets.DatePicker(
    description='Target date',
    value=date.today(),
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='260px')
)

w_min_prob = widgets.FloatSlider(
    value=0.85,
    min=0.3,
    max=0.95,
    step=0.05,
    description='Min probability',
    style={'description_width': 'initial'},
    readout_format='.2f',
    layout=widgets.Layout(width='420px')
)

btn_view = widgets.Button(
    description='View Signals',
    button_style='warning',
    icon='search',
    layout=widgets.Layout(width='160px')
)

out_view = widgets.Output()


def on_view(b):
    with out_view:
        clear_output()
        live_data = state.get('live_data')
        if live_data is None:
            print('No predictions available — go to the Run Predictions tab first.')
            return
        target = str(w_target_date.value)
        min_prob = w_min_prob.value
        result = live_data[
            (live_data['predictions'] == 1) &
            (live_data['Date'] == target) &
            (live_data['probability'] >= min_prob)
        ][['Identifier', 'Date', 'Close', 'probability']].sort_values(
            'probability', ascending=False
        ).reset_index(drop=True)
        print(f'Bullish signals on {target} (prob ≥ {min_prob:.2f}): {len(result)} stocks\n')
        display(result)


btn_view.on_click(on_view)

view_tab = widgets.VBox([
    widgets.HTML('<h3 style="margin:8px 0">View predictions for a target date</h3>'),
    widgets.HBox([w_target_date, w_min_prob], layout=widgets.Layout(gap='16px', align_items='center')),
    widgets.HBox([btn_view], layout=widgets.Layout(margin='8px 0')),
    out_view
], layout=widgets.Layout(padding='12px'))


# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE TABS
# ══════════════════════════════════════════════════════════════════════════════

tabs = widgets.Tab(children=[run_tab, view_tab])
tabs.set_title(0, '▶ Run Predictions')
tabs.set_title(1, '📊 View Predictions')
tabs.layout = widgets.Layout(width='860px')

_refresh_experiment_list()
_refresh_model_list()


def render():
    display(tabs)
