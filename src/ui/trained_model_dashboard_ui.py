import os
import json

import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output, FileLink
from IPython.display import Image as IPImage

_BASE = os.path.join('..', 'trained_models', 'bullish', 'classifiers')
_ROOT_LABEL = '(root)'


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _is_model_dir(path: str) -> bool:
    return os.path.exists(os.path.join(path, 'metadata.json'))


def _list_experiments() -> list[str]:
    """Experiment folders under _BASE that hold trained models, plus '(root)'
    if _BASE itself has any model folders saved directly inside it."""
    options = []
    if not os.path.isdir(_BASE):
        return options
    if any(_is_model_dir(os.path.join(_BASE, d)) for d in os.listdir(_BASE)
           if os.path.isdir(os.path.join(_BASE, d))):
        options.append(_ROOT_LABEL)
    for d in sorted(os.listdir(_BASE)):
        if d == '_old':
            continue
        full = os.path.join(_BASE, d)
        if not os.path.isdir(full) or _is_model_dir(full):
            continue
        if any(_is_model_dir(os.path.join(full, sub)) for sub in os.listdir(full)
               if os.path.isdir(os.path.join(full, sub))):
            options.append(d)
    return options


def _resolve_base(experiment: str) -> str:
    if not experiment or experiment == _ROOT_LABEL:
        return _BASE
    return os.path.join(_BASE, experiment)


def _load_all_models(base: str) -> dict:
    models = {}
    if not os.path.isdir(base):
        return models
    for name in sorted(os.listdir(base)):
        meta_path = os.path.join(base, name, 'metadata.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    models[name] = json.load(f)
            except Exception:
                pass
    return models


def _parse_cr(report) -> dict:
    """Normalise a sklearn classification_report (dict or legacy string).
    Returns {'0': {precision, recall, f1}, '1': {...}, 'accuracy': float}
    """
    if isinstance(report, dict):
        result = {}
        for label, vals in report.items():
            if label == 'accuracy':
                result['accuracy'] = float(vals)
            elif isinstance(vals, dict) and label in ('0', '1'):
                result[label] = {
                    'precision': float(vals['precision']),
                    'recall':    float(vals['recall']),
                    'f1':        float(vals['f1-score']),
                }
        return result

    result = {}
    for line in report.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        label = parts[0]
        if label in ('0', '1') and len(parts) >= 5:
            result[label] = {
                'precision': float(parts[1]),
                'recall':    float(parts[2]),
                'f1':        float(parts[3]),
            }
        elif label == 'accuracy' and len(parts) >= 3:
            result['accuracy'] = float(parts[-2])
    return result


def _format_cr(report) -> str:
    """Render a classification_report (dict or legacy string) as readable text."""
    if isinstance(report, str):
        return report

    lines = [f"{'':>14}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}", '']
    for label in ('0', '1'):
        vals = report.get(label)
        if not vals:
            continue
        lines.append(
            f"{label:>14}{vals['precision']:>10.2f}{vals['recall']:>10.2f}"
            f"{vals['f1-score']:>10.2f}{vals['support']:>10.0f}"
        )
    lines.append('')
    lines.append(f"{'accuracy':>14}{'':>20}{report.get('accuracy', float('nan')):>10.2f}")
    for label in ('macro avg', 'weighted avg'):
        vals = report.get(label)
        if not vals:
            continue
        lines.append(
            f"{label:>14}{vals['precision']:>10.2f}{vals['recall']:>10.2f}"
            f"{vals['f1-score']:>10.2f}{vals['support']:>10.0f}"
        )
    return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MODEL VIEW
# ══════════════════════════════════════════════════════════════════════════════

def _show_detail(base: str, name: str, meta: dict, out: widgets.Output):
    folder  = os.path.join(base, name)
    stats   = meta['stats']
    train_s = stats['train']
    test_s  = stats['test']
    cm      = test_s['confusion_matrix']
    feats   = meta['features']
    max_imp = feats[0]['importance'] if feats else 1.0

    with out:
        clear_output(wait=True)
        print(f"{'═' * 54}")
        print(f"  {meta['name']}")
        print(f"{'═' * 54}")
        print(f"Saved      : {meta['saved_at']}")
        print(f"Train      : {meta['train_start']}  →  {meta['train_cutoff']}")
        print(f"Data end   : {meta['data_end']}")
        print(f"Rows       : {meta['n_train_rows']:,} train  /  {meta['n_test_rows']:,} test")
        print(f"\nROC AUC    : train {train_s['roc_auc']}   /   test {test_s['roc_auc']}")
        print(f"\nClassification Report (train):\n{_format_cr(train_s['classification_report'])}")
        print(f"Classification Report (test):\n{_format_cr(test_s['classification_report'])}")
        print(f"Confusion Matrix (test):")
        print(f"  TN = {cm[0][0]:>9,}    FP = {cm[0][1]:>9,}")
        print(f"  FN = {cm[1][0]:>9,}    TP = {cm[1][1]:>9,}")

        roc_path = os.path.join(folder, 'roc_pr_curves.png')
        if os.path.exists(roc_path):
            display(IPImage(roc_path, width=560))

        cm_path = os.path.join(folder, 'confusion_matrix.png')
        if os.path.exists(cm_path):
            display(IPImage(cm_path, width=280))

        print(f"\n── Feature Importance ({len(feats)} features) " + "─" * 20)
        for feat in feats:
            bar = '█' * int(feat['importance'] / max_imp * 18) if max_imp > 0 else ''
            print(f"  {feat['name']:<36} {feat['importance']:.6f}  {bar}")


def _build_model_view(base: str, models: dict) -> widgets.Widget:
    names = list(models.keys())
    if not names:
        out = widgets.Output()
        with out:
            print('No models found.')
        return out

    w_select = widgets.Select(
        options=names,
        value=names[0],
        rows=min(20, len(names)),
        layout=widgets.Layout(width='240px', height='650px'),
    )
    out_detail = widgets.Output(layout=widgets.Layout(
        overflow_y='auto', max_height='650px', flex='1',
    ))

    def on_select(change):
        _show_detail(base, change['new'], models[change['new']], out_detail)

    w_select.observe(on_select, names='value')
    _show_detail(base, names[0], models[names[0]], out_detail)

    return widgets.HBox(
        [
            widgets.VBox(
                [widgets.HTML('<b style="font-size:13px">Models</b>'), w_select],
                layout=widgets.Layout(padding='8px'),
            ),
            out_detail,
        ],
        layout=widgets.Layout(width='100%'),
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODELS COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def _build_models_table(base: str, models: dict) -> widgets.Widget:
    rows = []
    for name, meta in models.items():
        train_s = meta['stats']['train']
        test_s  = meta['stats']['test']
        cr_train = _parse_cr(train_s['classification_report'])
        cr_test  = _parse_cr(test_s['classification_report'])
        cm = test_s['confusion_matrix']
        rows.append({
            'Model':          name,
            'Saved':          meta['saved_at'],
            'Cutoff':         meta['train_cutoff'],
            'Train rows':     meta['n_train_rows'],
            'Test rows':      meta['n_test_rows'],
            'ROC AUC (train)': train_s['roc_auc'],
            'ROC AUC (test)':  test_s['roc_auc'],
            'F1 (1, train)':  cr_train.get('1', {}).get('f1', float('nan')),
            'F1 (1, test)':   cr_test.get('1', {}).get('f1',  float('nan')),
            'Prec (0)':   cr_test.get('0', {}).get('precision', float('nan')),
            'Rec (0)':    cr_test.get('0', {}).get('recall',    float('nan')),
            'F1 (0)':     cr_test.get('0', {}).get('f1',        float('nan')),
            'Prec (1)':   cr_test.get('1', {}).get('precision', float('nan')),
            'Rec (1)':    cr_test.get('1', {}).get('recall',    float('nan')),
            'Accuracy':   cr_test.get('accuracy', float('nan')),
            'TN': cm[0][0], 'FP': cm[0][1],
            'FN': cm[1][0], 'TP': cm[1][1],
        })

    df = pd.DataFrame(rows).set_index('Model')

    float_cols = ['ROC AUC (train)', 'ROC AUC (test)', 'F1 (1, train)', 'F1 (1, test)',
                  'Prec (0)', 'Rec (0)', 'F1 (0)', 'Prec (1)', 'Rec (1)', 'Accuracy']
    int_cols   = ['Train rows', 'Test rows', 'TN', 'FP', 'FN', 'TP']

    fmt = {c: '{:.4f}' for c in float_cols}
    fmt.update({c: '{:,}' for c in int_cols})

    _th = [('background-color', '#f0f2f5'), ('font-weight', 'bold'),
           ('text-align', 'center'), ('white-space', 'nowrap'), ('padding', '6px 10px')]
    _td = [('text-align', 'center'), ('white-space', 'nowrap'), ('padding', '5px 10px')]

    out_table = widgets.Output(layout=widgets.Layout(overflow_x='auto'))
    with out_table:
        styled = (
            df.style
            .format(fmt, na_rep='—')
            .background_gradient(subset=['ROC AUC (train)', 'ROC AUC (test)', 'F1 (1, train)', 'F1 (1, test)', 'Prec (1)', 'Rec (1)'], cmap='RdYlGn')
            .background_gradient(subset=['F1 (0)'],             cmap='Blues')
            .set_table_styles([
                {'selector': 'th', 'props': _th},
                {'selector': 'td', 'props': _td},
            ])
        )
        display(styled)

    btn_dl = widgets.Button(
        description='Download CSV', icon='download',
        layout=widgets.Layout(width='160px'),
    )
    out_dl = widgets.Output()

    def _on_dl_models(b):
        with out_dl:
            clear_output()
            path = os.path.join(base, 'models_comparison.csv')
            df.to_csv(path)
            display(FileLink(path, result_html_prefix='Saved: '))

    btn_dl.on_click(_on_dl_models)

    # Images accordion (collapsed by default)
    img_cells = []
    for name in models:
        img_path = os.path.join(base, name, 'roc_pr_curves.png')
        out_img = widgets.Output()
        with out_img:
            if os.path.exists(img_path):
                display(IPImage(img_path, width=370))
            else:
                print('(image not found)')
        img_cells.append(widgets.VBox(
            [widgets.HTML(f'<b style="font-size:11px">{name}</b>'), out_img],
            layout=widgets.Layout(margin='4px'),
        ))

    acc = widgets.Accordion(children=[
        widgets.HBox(img_cells, layout=widgets.Layout(flex_flow='row wrap', gap='8px'))
    ])
    acc.set_title(0, 'ROC & PR Curves (all models)')
    acc.selected_index = None

    return widgets.VBox([widgets.HBox([btn_dl]), out_dl, out_table, acc], layout=widgets.Layout(padding='8px'))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_importances(features: list) -> dict:
    raw = {f['name']: f['importance'] for f in features}
    total = sum(raw.values())
    return {k: v / total * 100 for k, v in raw.items()} if total > 0 else raw


def _build_features_table(base: str, models: dict) -> widgets.Widget:
    data = {
        name: _normalize_importances(meta['features'])
        for name, meta in models.items()
    }
    df = pd.DataFrame(data)               # index=features, columns=models
    df = df.reindex(sorted(models.keys()), axis=1)
    df = df.loc[df.mean(axis=1).sort_values(ascending=False).index]

    _th = [('background-color', '#f0f2f5'), ('font-weight', 'bold'),
           ('text-align', 'center'), ('white-space', 'nowrap'), ('padding', '5px 10px')]
    _td = [('text-align', 'right'), ('white-space', 'nowrap'),
           ('padding', '4px 10px'), ('font-family', 'monospace')]

    out = widgets.Output(layout=widgets.Layout(
        overflow_x='auto', overflow_y='auto', max_height='700px',
    ))
    with out:
        styled = (
            df.style
            .background_gradient(cmap='YlOrRd', axis=None)
            .format('{:.4f}', na_rep='—')
            .set_sticky(axis='index')
            .set_table_styles([
                {'selector': 'th', 'props': _th},
                {'selector': 'td', 'props': _td},
            ])
        )
        display(styled)

    btn_dl = widgets.Button(
        description='Download CSV', icon='download',
        layout=widgets.Layout(width='160px'),
    )
    out_dl = widgets.Output()

    def _on_dl_features(b):
        with out_dl:
            clear_output()
            path = os.path.join(base, 'features_comparison.csv')
            df.to_csv(path)
            display(FileLink(path, result_html_prefix='Saved: '))

    btn_dl.on_click(_on_dl_features)

    return widgets.VBox([widgets.HBox([btn_dl]), out_dl, out], layout=widgets.Layout(padding='8px'))


# ══════════════════════════════════════════════════════════════════════════════
# RENDER
# ══════════════════════════════════════════════════════════════════════════════

def render():
    w_experiment = widgets.Dropdown(
        options=[], value=None,
        description='Experiment',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='260px'),
    )
    btn_refresh   = widgets.Button(
        description='Refresh', icon='refresh',
        layout=widgets.Layout(width='110px'),
    )
    out_container = widgets.Output()

    def _refresh_experiments():
        options = _list_experiments()
        current = w_experiment.value
        w_experiment.options = options
        w_experiment.value = current if current in options else (options[0] if options else None)

    def _render_into():
        base = _resolve_base(w_experiment.value)
        models = _load_all_models(base)
        if not models:
            with out_container:
                clear_output(wait=True)
                print(f'No models found in: {base}')
            return

        tab1 = _build_model_view(base, models)
        tab2 = _build_models_table(base, models)
        tab3 = _build_features_table(base, models)

        tabs = widgets.Tab(children=[tab1, tab2, tab3])
        tabs.set_title(0, 'Model View')
        tabs.set_title(1, 'Models')
        tabs.set_title(2, 'Features')
        tabs.layout = widgets.Layout(width='980px')

        with out_container:
            clear_output(wait=True)
            display(tabs)

    def _on_refresh(b):
        _refresh_experiments()
        _render_into()

    btn_refresh.on_click(_on_refresh)
    w_experiment.observe(lambda change: _render_into(), names='value')

    display(widgets.VBox([
        widgets.HBox(
            [
                widgets.HTML('<h3 style="margin:6px 0">Trained Model Dashboard</h3>'),
                widgets.HTML('<span style="width:16px;display:inline-block"></span>'),
                w_experiment,
                btn_refresh,
            ],
            layout=widgets.Layout(align_items='center'),
        ),
        out_container,
    ]))

    _refresh_experiments()
    _render_into()
