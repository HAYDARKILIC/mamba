"""Generate the LaTeX table bodies from the released result frames."""
import os
# ---------------------------------------------------------------------
# Repo-goreli yollar. Hicbir mutlak yol YOKTUR; betikler repo klasorunun
# herhangi bir yerinden calistirilabilir.
# ---------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
RESULTS = os.path.join(REPO, "results")
OUT = os.path.join(REPO, "build")
os.makedirs(OUT, exist_ok=True)
import pandas as pd, numpy as np, io

d = pd.read_csv(os.path.join(RESULTS,'master_results.csv'))
defn = pd.read_csv(os.path.join(RESULTS,'defense_results.csv'))
abl = pd.read_csv(os.path.join(RESULTS,'ablation_results.csv'))
# The pipeline now emits the significance analysis as two frames: T2 carries
# the pre-specified comparator (the Transformer) and T2b the post-hoc
# strongest competitor.  The primary table below is recomputed from
# master_results.csv rather than read from either, so only the supplementary
# post-hoc table consumes a frame here.
t2b = pd.read_csv(os.path.join(RESULTS,'tables',
                               'T2b_istatistiksel_anlamlilik_posthoc.csv'))
t3 = pd.read_csv(os.path.join(RESULTS,'tables','T3_saldiri_turu_kirilganligi.csv'))
t4 = pd.read_csv(os.path.join(RESULTS,'tables','T4_cikarim_verimliligi.csv'))

DS = ['NSL-KDD', 'NF-UNSW-NB15', 'CICIDS2018']
DSN = {'NSL-KDD': 'NSL-KDD', 'NF-UNSW-NB15': 'NF-UNSW-NB15', 'CICIDS2018': 'CSE-CIC-IDS2018'}
MO = ['Mamba', 'Transformer', 'CNN1D', 'LSTM', 'RandomForest']
# Display names: the recurrent baseline is bidirectional (Section 3.4),
# so it is reported as BiLSTM throughout the manuscript.
DISP = {'LSTM': 'BiLSTM', 'RandomForest': 'Random Forest'}
def disp(m):
    return DISP.get(m, m)
out = {}


def ms(m, s, nd=4):
    if pd.isna(m):
        return '--'
    if pd.isna(s):
        return f'{m:.{nd}f}'
    return f'{m:.{nd}f}\\,$\\pm$\\,{s:.{nd}f}'


def bold_max(vals, fmt='{:.4f}'):
    arr = np.array([v if v is not None else -np.inf for v in vals], float)
    i = int(np.nanargmax(arr))
    return i


# ---------------- Table 1: clean performance ----------------
rows = []
cl = d[d.attack == 'clean']
for ds in DS:
    for j, mo in enumerate(MO):
        g = cl[(cl.dataset == ds) & (cl.model == mo)]
        lead = f'\\multirow{{5}}{{*}}{{{DSN[ds]}}}' if j == 0 else ''
        rows.append(f"{lead} & {disp(mo)} & {ms(g.accuracy.mean(), g.accuracy.std())} & "
                    f"{ms(g.precision_macro.mean(), g.precision_macro.std())} & "
                    f"{ms(g.recall_macro.mean(), g.recall_macro.std())} & "
                    f"{ms(g.f1_macro.mean(), g.f1_macro.std())} & "
                    f"{ms(g.auc.mean(), g.auc.std())} \\\\")
    rows.append('\\midrule' if ds != DS[-1] else '')
out['T1'] = '\n'.join(r for r in rows if r)

# ---------------- Table 2: white-box robust accuracy ----------------
EPS = [0.01, 0.03, 0.05, 0.1]
rows = []
for ds in DS:
    mm = [m for m in MO if m != 'RandomForest']
    for j, mo in enumerate(mm):
        lead = f'\\multirow{{4}}{{*}}{{{DSN[ds]}}}' if j == 0 else ''
        cells = []
        for atk in ['FGSM', 'PGD']:
            for e in EPS:
                g = d[(d.dataset == ds) & (d.model == mo) & (d.attack == atk) & (d.eps == e)]
                cells.append(f'{g.accuracy.mean():.3f}')
        rows.append(f"{lead} & {disp(mo)} & " + ' & '.join(cells) + ' \\\\')
    rows.append('\\midrule' if ds != DS[-1] else '')
out['T2'] = '\n'.join(r for r in rows if r)

# std companion (for the caption / appendix)
rows = []
for ds in DS:
    for mo in [m for m in MO if m != 'RandomForest']:
        cells = []
        for atk in ['FGSM', 'PGD']:
            for e in EPS:
                g = d[(d.dataset == ds) & (d.model == mo) & (d.attack == atk) & (d.eps == e)]
                cells.append(f'{g.accuracy.std():.3f}')
        rows.append(f"{DSN[ds]} & {disp(mo)} & " + ' & '.join(cells) + ' \\\\')
out['T2std'] = '\n'.join(rows)

# ---------------- Table 3: black-box transfer ----------------
clm = cl.groupby(['dataset', 'model'])['accuracy'].mean()
rows = []
for ds in DS:
    mm = [m for m in MO if m != 'RandomForest']
    for j, mo in enumerate(mm):
        lead = f'\\multirow{{4}}{{*}}{{{DSN[ds]}}}' if j == 0 else ''
        cells = []
        for e in [0.03, 0.1]:
            g = d[(d.dataset == ds) & (d.model == mo) & (d.attack == 'BlackBoxTransfer') & (d.eps == e)]
            a = g.accuracy.mean()
            asr = 1 - a / clm.loc[(ds, mo)]
            cells += [f'{a:.3f}\\,$\\pm$\\,{g.accuracy.std():.3f}', f'{asr:.3f}']
        rows.append(f"{lead} & {disp(mo)} & " + ' & '.join(cells) + ' \\\\')
    rows.append('\\midrule' if ds != DS[-1] else '')
out['T3'] = '\n'.join(r for r in rows if r)

# ---- Table 4 (primary): pre-specified Transformer comparator ----
from scipy import stats as _st
_rows = []
for ds in DS:
    for atk in ['clean', 'FGSM', 'PGD', 'BlackBoxTransfer']:
        sub = d[(d.dataset == ds) & (d.attack == atk)]
        mm_ = sub[sub.model == 'Mamba'].groupby('seed')['accuracy'].mean()
        tt_ = sub[sub.model == 'Transformer'].groupby('seed')['accuracy'].mean()
        k = sorted(set(mm_.index) & set(tt_.index))
        mm_, tt_ = mm_.loc[k], tt_.loc[k]
        dif = mm_ - tt_
        ts, pv = _st.ttest_rel(mm_, tt_)
        _, wp = _st.wilcoxon(mm_, tt_)
        _rows.append(dict(dataset=ds, attack=atk, mamba=mm_.mean(),
                          transformer=tt_.mean(), t=ts, dz=dif.mean() / dif.std(),
                          p=pv, wp=wp))
_pre = pd.DataFrame(_rows)
# Holm-Bonferroni over the family of twelve
_order = _pre.p.sort_values().index
_run, _corr, _m = 0.0, {}, len(_pre)
for _rank, _i in enumerate(_order):
    _adj = (_m - _rank) * _pre.p[_i]
    _run = max(_run, _adj)
    _corr[_i] = min(1.0, _run)
_pre['p_holm'] = pd.Series(_corr)
rows = []
for ds in DS:
    sub = _pre[_pre.dataset == ds]
    for j, (_, r) in enumerate(sub.iterrows()):
        lead = f'\\multirow{{4}}{{*}}{{{DSN[ds]}}}' if j == 0 else ''
        atk = ('Transfer' if r['attack'] == 'BlackBoxTransfer'
               else 'Clean' if r['attack'] == 'clean' else r['attack'])
        star = '$\\ast$' if r['p_holm'] < 0.05 else ''
        rows.append(f"{lead} & {atk} & {r['mamba']:.4f} & {r['transformer']:.4f} & "
                    f"{r['t']:.2f} & {r['dz']:.2f} & {r['p']:.4f} & {r['wp']:.4f} & "
                    f"{r['p_holm']:.4f}{star} \\\\")
    rows.append('\\midrule' if ds != DS[-1] else '')
out['T4'] = '\n'.join(r for r in rows if r)

# ---- Table 4b (secondary, supplementary): post-hoc strongest competitor ----
rows = []
for ds in DS:
    sub = t2b[t2b.dataset == ds]
    for j, (_, r) in enumerate(sub.iterrows()):
        lead = f'\\multirow{{4}}{{*}}{{{DSN[ds]}}}' if j == 0 else ''
        star = '$\\ast$' if r['p_value_holm'] < 0.05 else ''
        atk = ('Transfer' if r['attack'] == 'BlackBoxTransfer'
               else 'Clean' if r['attack'] == 'clean' else r['attack'])
        rows.append(f"{lead} & {atk} & {disp(r['en_iyi_karsilastirma_modeli'])} & "
                    f"{r['mamba_ortalama']:.4f} & {r['rakip_ortalama']:.4f} & "
                    f"{r['t_stat']:.2f} & {r['p_value']:.4f} & "
                    f"{r['p_value_holm']:.4f}{star} \\\\")
    rows.append('\\midrule' if ds != DS[-1] else '')
out['T4b'] = '\n'.join(r for r in rows if r)

# ---------------- Table 5: category evasion matrix ----------------
mm = ['Mamba', 'Transformer', 'CNN1D', 'LSTM']
rows = []
for ds in DS:
    sub = t3[t3.dataset == ds]
    cats = sub.groupby('attack_category')['n_samples'].first().sort_values(ascending=False)
    for j, cat in enumerate(cats.index):
        lead = f'\\multirow{{{len(cats)}}}{{*}}{{{DSN[ds]}}}' if j == 0 else ''
        cells = []
        for mo in mm:
            v = sub[(sub.attack_category == cat) & (sub.model == mo)]['evasion_success_rate']
            cells.append(f'{v.iloc[0]:.3f}' if len(v) else '--')
        rows.append(f"{lead} & {cat} & {int(cats[cat])} & " + ' & '.join(cells) + ' \\\\')
    rows.append('\\midrule' if ds != DS[-1] else '')
out['T5'] = '\n'.join(r for r in rows if r)

# ---------------- Table 6: efficiency ----------------
rows = []
for _, r in t4.sort_values('latency_ms_bs1').iterrows():
    npar = '--' if r['n_parameters'] < 0 else f"{int(r['n_parameters']):,}".replace(',', '\\,')
    rows.append(f"{disp(r['model'])} & {npar} & {r['latency_ms_bs1']:.4f} & "
                f"{r['throughput_sps_bs1']:.1f} & {r['latency_ms_bs512']:.4f} & "
                f"{r['throughput_sps_bs512']:.1f} \\\\")
out['T6'] = '\n'.join(rows)

# ---------------- Table 7: defense ----------------
wb = defn[defn.attack.isin(['FGSM', 'PGD'])]
rows = []
for ds in DS:
    for j, mo in enumerate(['Mamba', 'CNN1D']):
        lead = f'\\multirow{{2}}{{*}}{{{DSN[ds]}}}' if j == 0 else ''
        g = wb[(wb.dataset == ds) & (wb.model == mo)]
        n = g[g.defense == 'none'].accuracy.mean()
        a = g[g.defense == 'adv_train'].accuracy.mean()
        c = defn[(defn.dataset == ds) & (defn.model == mo) & (defn.attack == 'clean')]
        cn = c[c.defense == 'none'].accuracy.mean()
        ca = c[c.defense == 'adv_train'].accuracy.mean()
        rows.append(f"{lead} & {disp(mo)} & {cn:.4f} & {ca:.4f} & {ca-cn:+.4f} & "
                    f"{n:.4f} & {a:.4f} & {a-n:+.4f} \\\\")
    rows.append('\\midrule' if ds != DS[-1] else '')
out['T7'] = '\n'.join(r for r in rows if r)

# ---------------- Table 8: ablation ----------------
ag = abl.groupby(['dataset', 'model', 'd_model'])['accuracy'].agg(['mean', 'std'])
np_ = abl.groupby(['model', 'd_model'])['n_parameters'].first()
rows = []
mm = ['Mamba', 'Transformer', 'CNN1D', 'LSTM']
for ds in DS:
    for j, mo in enumerate(mm):
        lead = f'\\multirow{{4}}{{*}}{{{DSN[ds]}}}' if j == 0 else ''
        c = []
        for dm in [32, 64]:
            m, s = ag.loc[(ds, mo, dm)]
            c.append(f'{m:.4f}\\,$\\pm$\\,{s:.4f}')
        rows.append(f"{lead} & {disp(mo)} & " + ' & '.join(c) + ' \\\\')
    rows.append('\\midrule' if ds != DS[-1] else '')
out['T8'] = '\n'.join(r for r in rows if r)

for k, v in out.items():
    # A file that is \input inside a tabular must NOT end with a row
    # terminator: the end-of-file boundary would then open a new row and
    # \bottomrule would be flagged as a misplaced \noalign. The parent
    # document supplies the final \\ instead.
    body = v.rstrip()
    if body.endswith('\\\\'):
        body = body[:-2].rstrip()
    with open(os.path.join(OUT, f'tab_{k}.tex'), 'w') as f:
        f.write(body + '%\n')
    print('===', k, '===')
    print(v)
    print()
