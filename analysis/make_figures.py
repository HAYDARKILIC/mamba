"""Regenerate every published figure from the released result frames."""
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

OUT = os.path.join(OUT, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': ':',
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
})

master = pd.read_csv(os.path.join(RESULTS,'master_results.csv'))
defense = pd.read_csv(os.path.join(RESULTS,'defense_results.csv'))
ablation = pd.read_csv(os.path.join(RESULTS,'ablation_results.csv'))
t3 = pd.read_csv(os.path.join(RESULTS,'tables','T3_saldiri_turu_kirilganligi.csv'))
t4 = pd.read_csv(os.path.join(RESULTS,'tables','T4_cikarim_verimliligi.csv'))

DS = ['NSL-KDD', 'NF-UNSW-NB15', 'CICIDS2018']
TITLE = {'NSL-KDD': 'NSL-KDD', 'NF-UNSW-NB15': 'NF-UNSW-NB15',
         'CICIDS2018': 'CSE-CIC-IDS2018'}
MODELS = ['Mamba', 'Transformer', 'CNN1D', 'LSTM', 'RandomForest']
# Display names must match the manuscript: the recurrent baseline is
# bidirectional and is reported as BiLSTM throughout.
DISP = {'LSTM': 'BiLSTM', 'RandomForest': 'Random Forest'}
def disp(m):
    return DISP.get(m, m)
COL = {'Mamba': '#0072B2', 'Transformer': '#D55E00', 'CNN1D': '#009E73',
       'LSTM': '#CC79A7', 'RandomForest': '#7F7F7F'}
MRK = {'Mamba': 'o', 'Transformer': 's', 'CNN1D': '^', 'LSTM': 'D',
       'RandomForest': 'v'}
FS = (3.6, 2.7)


def save(fig, name):
    fig.savefig(f'{OUT}/{name}.pdf')
    plt.close(fig)
    print('  ->', name)


# ---------- 1. clean accuracy ----------
for ds in DS:
    sub = master[(master.dataset == ds) & (master.attack == 'clean')]
    g = sub.groupby('model')['accuracy'].agg(['mean', 'std']).reindex(MODELS)
    fig, ax = plt.subplots(figsize=FS)
    ax.bar(range(len(g)), g['mean'], yerr=g['std'], capsize=3,
           color=[COL[m] for m in g.index], edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(['Mamba', 'Transf.', 'CNN1D', 'BiLSTM', 'RF'], rotation=20)
    ax.set_ylabel('Clean test accuracy')
    ax.set_title(TITLE[ds])
    lo = max(0.0, float(g['mean'].min()) - 0.12)
    ax.set_ylim(lo, min(1.02, float(g['mean'].max()) + 0.06))
    save(fig, f'clean_acc_{ds}')

# ---------- 2. robustness curves ----------
for atk in ['FGSM', 'PGD']:
    for ds in DS:
        fig, ax = plt.subplots(figsize=FS)
        for m in [x for x in MODELS if x != 'RandomForest']:
            s = master[(master.dataset == ds) & (master.model == m) &
                       (master.attack == atk)]
            g = s.groupby('eps')['accuracy'].agg(['mean', 'std'])
            c = master[(master.dataset == ds) & (master.model == m) &
                       (master.attack == 'clean')]['accuracy'].mean()
            x = [0.0] + list(g.index)
            y = [c] + list(g['mean'])
            e = [0.0] + list(g['std'].fillna(0))
            ax.errorbar(x, y, yerr=e, marker=MRK[m], color=COL[m], label=disp(m),
                        capsize=2, markersize=4, linewidth=1.2)
        ax.set_xlabel(r'Perturbation budget $\varepsilon$ ($\ell_\infty$)')
        ax.set_ylabel('Robust accuracy')
        ax.set_title(f'{TITLE[ds]} -- {atk}')
        ax.set_ylim(-0.02, 1.05)
        ax.legend(frameon=False, loc='lower left', ncol=2)
        save(fig, f'robust_{atk.lower()}_{ds}')

# ---------- 3. category vulnerability ----------
for ds, mo in [('NSL-KDD', 'Mamba'), ('NSL-KDD', 'LSTM'),
               ('NF-UNSW-NB15', 'CNN1D')]:
    s = t3[(t3.dataset == ds) & (t3.model == mo)].sort_values(
        'evasion_success_rate', ascending=False)
    fig, ax = plt.subplots(figsize=FS)
    y = np.arange(len(s))
    ax.barh(y - 0.2, s['clean_detection_rate'], height=0.4,
            color='#4C72B0', edgecolor='black', linewidth=0.4, label='Clean')
    ax.barh(y + 0.2, s['adv_detection_rate'], height=0.4,
            color='#C44E52', edgecolor='black', linewidth=0.4,
            label=r'FGSM $\varepsilon=0.1$')
    ax.set_yticks(y)
    ax.set_yticklabels(s['attack_category'])
    ax.invert_yaxis()
    ax.set_xlabel('Detection rate')
    ax.set_xlim(0, 1.05)
    ax.set_title(f'{disp(mo)} -- {TITLE[ds]}')
    ax.legend(frameon=False, loc='lower right')
    save(fig, f'catvuln_{ds}_{mo}')

# ---------- 4. efficiency vs robustness ----------
rob = master[master.attack != 'clean'].groupby('model')['accuracy'].mean()
for bs, col in [(1, 'latency_ms_bs1'), (512, 'latency_ms_bs512')]:
    fig, ax = plt.subplots(figsize=FS)
    for _, r in t4.iterrows():
        m = r['model']
        if m not in rob.index:
            continue
        ax.scatter(r[col], rob[m], s=55, color=COL[m], marker=MRK[m],
                   edgecolor='black', linewidth=0.5, zorder=3)
        ax.annotate(disp(m), (r[col], rob[m]), textcoords='offset points',
                    xytext=(6, 4), fontsize=7)
    ax.set_xscale('log')
    ax.set_xlabel(f'Inference latency (ms/sample, $b$={bs})')
    ax.set_ylabel('Mean robust accuracy')
    ax.set_title(f'Latency vs.\\ robustness ($b$={bs})'.replace('\\', ''))
    save(fig, f'latency_bs{bs}')

fig, ax = plt.subplots(figsize=FS)
for _, r in t4.iterrows():
    m = r['model']
    if m not in rob.index or r['n_parameters'] < 0:
        continue
    ax.scatter(r['n_parameters'], rob[m], s=55, color=COL[m], marker=MRK[m],
               edgecolor='black', linewidth=0.5, zorder=3)
    ax.annotate(disp(m), (r['n_parameters'], rob[m]), textcoords='offset points',
                xytext=(6, 4), fontsize=7)
ax.set_xscale('log')
ax.set_xlabel('Number of parameters')
ax.set_ylabel('Mean robust accuracy')
ax.set_title('Capacity vs. robustness')
save(fig, 'params_vs_robust')

# ---------- 5. defence effectiveness ----------
for ds, mo in [('NSL-KDD', 'Mamba'), ('CICIDS2018', 'CNN1D')]:
    fig, ax = plt.subplots(figsize=(3.9, 2.8))
    for defn, ls, lab in [('none', '--', 'Undefended'),
                          ('adv_train', '-', 'Adv.\\ trained')]:
        for atk, mk in [('FGSM', 'o'), ('PGD', 's')]:
            s = defense[(defense.dataset == ds) & (defense.model == mo) &
                        (defense.defense == defn) & (defense.attack == atk)]
            if s.empty:
                continue
            g = s.groupby('eps')['accuracy'].mean()
            ax.plot(g.index, g.values, ls, marker=mk, markersize=4,
                    linewidth=1.2,
                    color='#D55E00' if atk == 'FGSM' else '#0072B2',
                    label=f'{atk}, {lab}'.replace('\\', ''))
    ax.set_xlabel(r'Perturbation budget $\varepsilon$ ($\ell_\infty$)')
    ax.set_ylabel('Robust accuracy')
    ax.set_title(f'{disp(mo)} -- {TITLE[ds]}')
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False, fontsize=7, loc='lower left', ncol=2)
    save(fig, f'defense_{ds}_{mo}')

# ---------- 6. capacity ablation ----------
for ds in DS:
    s = ablation[ablation.dataset == ds]
    g = s.groupby(['model', 'd_model'])['accuracy'].agg(['mean', 'std'])
    fig, ax = plt.subplots(figsize=FS)
    for m in ['Mamba', 'Transformer', 'CNN1D', 'LSTM']:
        if m not in g.index.get_level_values(0):
            continue
        sub = g.loc[m]
        ax.errorbar(sub.index, sub['mean'], yerr=sub['std'].fillna(0),
                    marker=MRK[m], color=COL[m], label=disp(m), capsize=3,
                    markersize=5, linewidth=1.2)
    ax.set_xticks([32, 64])
    ax.set_xlabel(r'Embedding width $d_{\mathrm{model}}$')
    ax.set_ylabel('Clean test accuracy')
    ax.set_title(TITLE[ds])
    ax.legend(frameon=False, ncol=2, fontsize=7)
    save(fig, f'ablation_{ds}')

print('done')
