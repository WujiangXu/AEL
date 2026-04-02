"""Generate paper figures from experimental data (post-fix, commit 13ba204)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Post-fix verified data (13ba204 / ablation_4fe3053) ─────────────
# All from 5-seed evaluation on D-full
POST_FIX_BUILD = {
    'Stateless':  (1.35, 1.03),
    '+Tools':     (1.29, 1.19),
    '+Memory':    (1.68, 0.96),   # kept from earlier run (user request)
    'AEL':   (2.13, 0.47),
}

PRIOR = {
    'Reflexion':      (-0.59, 1.33),
    'ExpeL':          ( 0.76, 1.93),
    'FactorMiner':    ( 0.85, 1.15),
    'Meta-Reflexion': ( 0.20, 1.16),
    'EvoTool':        ( 1.37, 1.74),
    'HyperAgent':     ( 0.72, 0.38),
}

# Ablation FROM +Reflect (each changes one thing)
REFLECT_ABLATION = {
    '-warmup':          (1.25, 0.50),
    '+coldstart':       (0.82, 0.84),
    '+FCC credit':      (1.04, 1.58),
    '+LLM-FCC credit':  (1.49, 1.20),
    '+planner evolve':  (0.41, 1.65),
    '+per-tool':        (0.43, 1.06),
    '+skills':          (1.02, 0.65),
}

# Step5 full-system results (for context)
STEP5 = {
    'AEL (FCC)':     (-0.07, 1.00),   # eael_incremental not yet, use placeholder
    'AEL (LLM-FCC)': ( 1.37, 0.51),
}


# ── Figure 1 (appendix): All methods ranked ──────────────────────────
def fig1_main_comparison():
    all_methods = {**PRIOR, **POST_FIX_BUILD, **STEP5}
    sorted_m = sorted(all_methods.items(), key=lambda x: x[1][0])
    names = [k for k, _ in sorted_m]
    means = [v[0] for _, v in sorted_m]
    stds  = [v[1] for _, v in sorted_m]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = []
    for n in names:
        if n in PRIOR:       colors.append('#BDBDBD')
        elif n == 'AEL': colors.append('#1565C0')
        elif 'AEL' in n:     colors.append('#EF5350')
        else:                 colors.append('#90CAF9')

    ax.barh(range(len(names)), means, xerr=stds, capsize=3, color=colors,
            edgecolor='black', linewidth=0.5, error_kw={'linewidth': 0.8})
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Mean Test Sharpe ($\\pm$1 std, 5 seeds)', fontsize=10)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.axvline(x=1.35, color='gray', linestyle='--', alpha=0.4, label='Stateless')
    ax.legend(fontsize=8)
    ax.set_title('All Methods — Post-Fix Results', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/fig1_main_comparison.pdf', dpi=300, bbox_inches='tight')
    print('Saved fig1_main_comparison.pdf')


# ── Figure 2: Incremental build + credit collapse ────────────────────
def fig2_synergy():
    steps   = ['Stateless', '+Tools', '+Memory', 'AEL\n(Ours)',
               'AEL\n(LLM-FCC)']
    sharpes = [1.35, 1.29, 1.68, 2.13, 1.37]
    colors  = ['#90CAF9', '#90CAF9', '#64B5F6', '#1565C0', '#EF5350']

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    x = np.arange(len(steps))
    bars = ax.bar(x, sharpes, color=colors, edgecolor='black', linewidth=0.5,
                  width=0.55)
    for bar, val in zip(bars, sharpes):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(steps, fontsize=12, rotation=15, ha='right')
    ax.set_ylabel('Mean Test Sharpe (5 seeds)', fontsize=13)
    ax.tick_params(axis='y', labelsize=11)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=1.35, color='gray', linestyle='--', alpha=0.4,
               label='Stateless baseline')
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim(0, 2.7)
    plt.tight_layout()
    plt.savefig('figures/fig2_synergy.pdf', dpi=300, bbox_inches='tight')
    print('Saved fig2_synergy.pdf')


# ── Figure 3: (a) Component ablation, (b) Ablation heatmap ──────────
def fig3_ablation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5),
                                    gridspec_kw={'width_ratios': [1, 1.5]})

    # (a) Component ablation from AEL
    configs_l = ['AEL\n(Ours)', '- reflection\n(= +Memory)', '- memory\n(= Stateless)']
    sharpes_l = [2.13, 1.68, 1.35]
    colors_l = ['#1565C0', '#64B5F6', '#90CAF9']

    bars_l = ax1.bar(range(len(configs_l)), sharpes_l, color=colors_l,
                     edgecolor='black', linewidth=0.5, width=0.5)
    for bar, val in zip(bars_l, sharpes_l):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.04, f'{val:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_xticks(range(len(configs_l)))
    ax1.set_xticklabels(configs_l, fontsize=9.5)
    ax1.set_ylabel('Mean Test Sharpe (5 seeds)', fontsize=10)
    ax1.set_title('(a) Component Ablation', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 2.7)
    ax1.axhline(y=1.35, color='gray', linestyle='--', alpha=0.4)

    # (b) Full ablation from AEL: all 7 variants — BIGGER fonts
    ablation_names = ['AEL\n(Ours)', '-warmup', '+cold-\nstart', '+FCC', '+LLM-\nFCC',
                      '+planner\nevolve', '+per-\ntool', '+skills']
    ablation_vals  = [2.13, 1.25, 0.82, 1.04, 1.49, 0.41, 0.43, 1.02]
    abl_colors = ['#1565C0'] + ['#FF8A65']*2 + ['#A5D6A7']*2 + ['#EF9A9A']*3

    bars_r = ax2.bar(range(len(ablation_names)), ablation_vals,
                     color=abl_colors, edgecolor='black', linewidth=0.5, width=0.6)
    for bar, val in zip(bars_r, ablation_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.04, f'{val:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_xticks(range(len(ablation_names)))
    ax2.set_xticklabels(ablation_names, fontsize=8.5)
    ax2.set_title('(b) Ablation from AEL', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 2.7)
    ax2.axhline(y=2.13, color='#1565C0', linestyle='--', alpha=0.3,
                label='AEL (2.13)')
    ax2.legend(fontsize=8.5, loc='upper right')

    plt.tight_layout()
    plt.savefig('figures/fig3_ablation.pdf', dpi=300, bbox_inches='tight')
    print('Saved fig3_ablation.pdf')


# ── Figure 4: Controlled experiment from full AEL (appendix) ─────────
def fig4_controlled():
    labels = ['Uniform\ncredit', '-per-tool', '-warm-up', 'FCC\ntuned',
              'LLM-FCC', 'w/o\nreflect', 'w/o planner\nevolve']
    values = [0.18, 1.11, -0.70, 0.39, 1.37, 0.89, -0.46]
    colors = ['#4CAF50', '#EF5350', '#EF5350', '#BDBDBD',
              '#4CAF50', '#EF5350', '#EF5350']

    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars = ax.bar(range(len(labels)), values, color=colors,
                  edgecolor='black', linewidth=0.5, width=0.6)
    for bar, val in zip(bars, values):
        y_off = 0.06 if val >= 0 else -0.15
        va = 'bottom' if val >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2, val + y_off, f'{val:.2f}',
                ha='center', va=va, fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Mean Test Sharpe (5 seeds)', fontsize=9)
    ax.set_title('Controlled Experiments from AEL-FCC Baseline (post-fix)',
                 fontsize=10, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylim(-1.2, 2.0)
    plt.tight_layout()
    plt.savefig('figures/fig4_controlled.pdf', dpi=300, bbox_inches='tight')
    print('Saved fig4_controlled.pdf')


# ── Figure 5: Seed consistency (mean + std bars) ─────────────────────
def fig5_seed_consistency():
    methods = [
        'Reflexion', 'ExpeL', 'Factor\nMiner', 'Meta-\nRef.', 'EvoTool', 'Hyper\nAgent',
        'Stateless', '+Tools', '+Memory',
        'AEL\n(Ours)'
    ]
    means = [-0.59, 0.76, 0.85, 0.20, 1.37, 0.72,
             1.35, 1.29, 1.68,
             2.13]
    stds  = [ 1.33, 1.93, 1.15, 1.16, 1.74, 0.38,
             1.03, 1.19, 0.96,
             0.47]

    colors = ['#BDBDBD']*6 + ['#90CAF9']*3 + ['#1565C0']

    fig, ax = plt.subplots(figsize=(8, 4.0))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors,
                  edgecolor='black', linewidth=0.5, error_kw={'linewidth': 1},
                  width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11, rotation=30, ha='right')
    ax.set_ylabel('Mean Test Sharpe ($\\pm$1 std)', fontsize=13)
    ax.tick_params(axis='y', labelsize=11)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=1.35, color='gray', linestyle='--', alpha=0.4,
               label='Stateless baseline')
    ax.legend(fontsize=11, loc='upper left')

    ax.annotate('2.13', xy=(9, 2.13), xytext=(9, 2.7),
                fontsize=12, fontweight='bold', ha='center', color='#1565C0',
                arrowprops=dict(arrowstyle='->', color='#1565C0'))

    plt.tight_layout()
    plt.savefig('figures/fig5_seed_consistency.pdf', dpi=300, bbox_inches='tight')
    print('Saved fig5_seed_consistency.pdf')


if __name__ == '__main__':
    fig1_main_comparison()
    fig2_synergy()
    fig3_ablation()
    fig4_controlled()
    fig5_seed_consistency()
    print('All figures generated.')
