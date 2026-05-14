import numpy as np                                                                                      
import os
                                                                                                        
atlas_dir = '/mnt/alphaidz/brain-score-language/atlases'                                              
atlases = {
    'LanA': np.load(f'{atlas_dir}/lh_rh_lana_atlas_fsavg5_top_10pct_mask.npy').astype(bool),
    'A1': np.load(f'{atlas_dir}/glasser_a1_fsavg5_mask_bool_fixed.npy').astype(bool),
    'A2': np.load(f'{atlas_dir}/glasser_a2_fsavg5_mask.npy').astype(bool),
    'A1 Extended': np.load(f'{atlas_dir}/fsavg5_auditory_mask_conservative.npy').astype(bool),
    'V1': np.load(f'{atlas_dir}/glasser_v1_fsavg5_mask_bool_fixed.npy').astype(bool),
    'V2': np.load(f'{atlas_dir}/glasser_v2_fsavg5_mask.npy').astype(bool),
    'V4': np.load(f'{atlas_dir}/glasser_v4_fsavg5_mask.npy').astype(bool),
    'IT': np.load(f'{atlas_dir}/glasser_it_fsavg5_mask.npy').astype(bool),
    'MD': np.load(f'{atlas_dir}/yeo_md_fsavg5_mask.npy').astype(bool),
    'DMN': np.load(f'{atlas_dir}/yeo_dmn_fsavg5_mask.npy').astype(bool),
    'ToM': np.load(f'{atlas_dir}/saxe_tom_fsavg5_mask.npy').astype(bool),
    'DLPFC': np.load(f'{atlas_dir}/glasser_dlpfc_fsavg5_mask.npy').astype(bool),
}

score_dir = '/mnt/alphaidz/brain-score-language/brain_olmo_stages_correct_28_word/ordered'
score_files = sorted([f for f in os.listdir(score_dir) if f.endswith('.npy')])

# Print header
print(f"{'Stage':<35} {'LanA':>8} {'A1':>8} {'A2':>8} {'A1 Extended':>8} {'V1':>8} {'V2':>8} {'V4':>8} {'IT':>8} {'MD':>8} {'DMN':>8} {'ToM':>8} {'DLPFC':>8} {'Whole':>8}")
print('-' * 100)

rows = []
for fname in score_files:
    scores = np.load(f'{score_dir}/{fname}')
    stage = fname.replace('_layer28_voxel_scores_reordered.npy', '')
    lana = scores[atlases['LanA']].mean()
    a1 = scores[atlases['A1']].mean()
    a2 = scores[atlases['A2']].mean()
    a1_extended = scores[atlases['A1 Extended']].mean()
    v1 = scores[atlases['V1']].mean()
    v2 = scores[atlases['V2']].mean()
    v4 = scores[atlases['V4']].mean()
    it = scores[atlases['IT']].mean()
    md = scores[atlases['MD']].mean()
    dmn = scores[atlases['DMN']].mean()
    tom = scores[atlases['ToM']].mean()
    dlpfc = scores[atlases['DLPFC']].mean()
    whole = scores.mean()
    print(f"{stage:<35} {lana:>8.4f} {a1:>8.4f} {a2:>8.4f} {a1_extended:>8.4f} {v1:>8.4f} {v2:>8.4f} {v4:>8.4f} {it:>8.4f} {md:>8.4f} {dmn:>8.4f} {tom:>8.4f} {dlpfc:>8.4f} {whole:>8.4f}")
    rows.append((stage, {'LanA': lana, 'A1': a1, 'A2': a2, 'A1 Extended': a1_extended, 'DLPFC': dlpfc, 'V1': v1, 'V2': v2, 'V4': v4, 'IT': it, 'MD': md, 'DMN': dmn, 'ToM': tom, 'Whole': whole}))

cols = ['LanA', 'A1', 'A2', 'A1 Extended', 'DLPFC', 'V1', 'V2', 'V4', 'IT', 'MD', 'DMN', 'ToM', 'Whole']

# --- Canonical stage ordering ---
stage_order = [
    'stage1-step0', 'stage1-step10000', 'stage1-step50000',
    'stage1-step100000', 'stage1-step250000', 'stage1-step500000',
    'stage1-step750000', 'stage1-step1000000', 'stage1-step1413814',
    'stage2-step47684', 'base', 'sft', 'dpo', 'rlvr',
]
stage_labels = [
    'Untrained', 'Pre: 41.9B', 'Pre: 209.7B', 'Pre: 419.4B',
    'Pre: 1.05T', 'Pre: 2.10T', 'Pre: 3.15T', 'Pre: 4.19T',
    'Pre: 5.93T', 'Mid: 6.03T', 'Long: 6.08T',
    'SFT', 'DPO', 'RLVR',
]
score_lookup = {s: v for s, v in rows}

# Reorder rows by stage_order
rows_ordered = [(s, score_lookup[s]) for s in stage_order if s in score_lookup]

# --- LaTeX generation ---
posttrain_names = {'base': 'Base', 'sft': 'SFT', 'dpo': 'DPO', 'rlvr': 'RLVR'}

pretrain_rows = [(s, v) for s, v in rows_ordered if s.startswith('stage1-step')]
stage2_rows = [(s, v) for s, v in rows_ordered if s.startswith('stage2-step')]
posttrain_rows = [(s, v) for s, v in rows_ordered if s in posttrain_names]

# Find column maxima across ALL rows for bolding
col_max = {}
for c in cols:
    col_max[c] = max(v[c] for _, v in rows)

def fmt_val(val, col):
    s = f'{val:.3f}'
    if abs(val - col_max[col]) < 1e-6:
        return f'\\textbf{{{s}}}'
    return s

def fmt_stage(stage):
    # Pretty-print stage names
    if stage in posttrain_names:
        return posttrain_names[stage]
    if stage.startswith('stage1-step'):
        step = stage.replace('stage1-step', '')
        # Format step number nicely
        n = int(step)
        if n == 0:
            return 'step 0'
        elif n >= 1_000_000:
            val = n / 1_000_000
            return f'step {val:.1f}M'.replace('.0M', 'M')
        elif n >= 1000:
            return f'step {n//1000}k'
        return f'step {n}'
    if stage.startswith('stage2'):
        return 'Stage 2'
    return stage

def latex_row(stage, vals):
    cells = ' & '.join(fmt_val(vals[c], c) for c in cols)
    return f'  {fmt_stage(stage):<14} & {cells} \\\\'

n_cols = len(cols) + 1  # +1 for Stage column

print('\n\n% === GENERATED LATEX ===')
print(r'\begin{frame}{Per-Region Encoding Accuracy}')
print(r'  \centering')
print(r'  \resizebox{\textwidth}{!}{%')
print(f'  \\begin{{tabular}}{{l {"c" * len(cols)}}}')
print(r'  \toprule')
print(f'  Stage & {" & ".join(cols)} \\\\')
print(r'  \midrule')

if pretrain_rows:
    print(f'  \\multicolumn{{{n_cols}}}{{l}}{{\\textit{{Pre-training checkpoints}}}} \\\\')
    for stage, vals in pretrain_rows:
        print(latex_row(stage, vals))

if stage2_rows:
    print(r'  \midrule')
    for stage, vals in stage2_rows:
        print(latex_row(stage, vals))

if posttrain_rows:
    print(r'  \midrule')
    print(f'  \\multicolumn{{{n_cols}}}{{l}}{{\\textit{{Post-training stages}}}} \\\\')
    for stage, vals in posttrain_rows:
        print(latex_row(stage, vals))

print(r'  \bottomrule')
print(r'  \end{tabular}%')
print(r'  }')
print(r'\end{frame}')

# --- TikZ graph generation ---
# Only include stages that have score data, sequential x-coordinates
available = [(x, s) for x, (i, s) in enumerate(
    ((i, s) for i, s in enumerate(stage_order) if s in score_lookup), start=0)]

def tikz_coords(region):
    parts = []
    for x, stage in available:
        y = score_lookup[stage][region]
        parts.append(f'({x+1},{y:.4f})')
    return ' '.join(parts)

# Graph 1: Language-Relevant (+ auditory)
graph1_regions = [
    ('A2',           'red!60',         'square',     '', ''),
    ('A1',           'red',            'square*',    '', ''),
    ('A1 Extended',  'red!30',         'diamond',    '', ''),
    ('ToM',          'purple',         'triangle*',  '', ''),
    ('LanA',         'blue',           '*',          '', ''),
    ('DMN',          'orange',         'diamond*',   '', ''),
    ('MD',           'teal',           'pentagon*',  '', ''),
]

# Graph 2: Visual & Prefrontal
graph2_regions = [
    ('DLPFC', 'brown',         'star',   '', ''),
    ('V4',    'green!40!black', 'x',     '', ''),
    ('V2',    'green!70!black', 'o',     '', ''),
    ('V1',    'green!50!black', 'o',     'dashed', ''),
    ('IT',    'gray',           '+',     '', ''),
]

def print_graph(title, regions, ymax):
    n = len(available)
    xticks = ','.join(str(x+1) for x, _ in available)
    # Map each available stage back to its label via stage_order index
    stage_to_label = dict(zip(stage_order, stage_labels))
    xlabels = ',\n          '.join(f'{{{stage_to_label[s]}}}' for _, s in available)

    print(f'\n  \\begin{{frame}}{{{title}}}')
    print(r'  \centering')
    print(r'  \begin{tikzpicture}')
    print(r'  \begin{axis}[')
    print(f'      width=14cm,')
    print(f'      height=7cm,')
    print(f'      xtick={{{xticks}}},')
    print(f'      xticklabels={{')
    print(f'          {xlabels}')
    print(f'      }},')
    print(r'      x tick label style={rotate=45, anchor=east, font=\tiny},')
    print(f'      ymin=0, ymax={ymax},')
    print(r'      legend style={')
    print(r'          at={(1.02,1)},')
    print(r'          anchor=north west,')
    print(r'          font=\tiny,')
    print(r'          cells={anchor=west},')
    print(r'      },')
    print(r'      grid=major,')
    print(r'      grid style={dashed, gray!30},')
    print(r'      every axis plot/.append style={thick, mark size=1.5pt},')
    print(r'  ]')

    for region, color, mark, style, _ in regions:
        style_str = f', {style}' if style else ''
        print(f'\n  \\addplot[color={color}, mark={mark}{style_str}] coordinates {{')
        print(f'      {tikz_coords(region)}')
        print(r'  };')
        print(f'  \\addlegendentry{{{region}}}')

    print(r'')
    print(r'  \end{axis}')
    print(r'  \end{tikzpicture}')
    print(r'  \end{frame}')

print('\n\n% === GENERATED TIKZ GRAPHS ===')
print_graph('Brain Region Scores Across OLMo-3 Training -- Language-Relevant', graph1_regions, 0.45)
print_graph('Brain Region Scores Across OLMo-3 Training -- Visual \\& Prefrontal', graph2_regions, 0.35)
