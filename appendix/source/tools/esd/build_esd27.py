"""ESD (27-feature) heatmap raster + TikZ overlay + schema/statistics table.
Input: the 300x27 scene-phase feature matrix whose SHA-256 equals the released
split provenance (input_sha256), joined with the released per-cell scores."""
import pandas as pd, numpy as np, json, sys
from PIL import Image
S=sys.argv[1]; OUT=sys.argv[2]
m=pd.read_csv(f'{S}/esd27/esd_300x27_release.csv')
d=json.load(open(f'{S}/dl/splits/scene_splits.json')); w=d['provenance']['weights']; F=list(w)
fam=[('Annotation effort',['iter_mean','iter_max']),('Scene complexity',['obj_mean','obj_std','obj_consist']),
 ('Occlusion',['occ_mean','occ_p90','occ_heavy']),('Depth quality',['depth_invalid','depth_invalid_mask','depth_std','depth_std_mask']),
 ('Photometric conflict',['specular','dark']),('Temporal stability',['area_cv','area_drop','vis_instability']),
 ('Camera motion',['trans_mean','trans_p90','rot_mean','rot_p90','jerk']),
 ('Fisheye/stereo',['fisheye_dark','fisheye_bright','fisheye_sharpness','fisheye_corr','fisheye_texture'])]
order=[f for _,g in fam for f in g]; assert sorted(order)==sorted(F) and len(order)==27
# ---- heatmap raster: percentile rank per feature (as entering the score), cells ordered by the supplementary ESD score
m=m.sort_values('rpx_ds',kind='mergesort').reset_index(drop=True)
P=np.stack([m[f].rank(method='average').values/len(m) for f in order])   # 27 x 300
lo=np.array([247,249,251]); hi=np.array([30,58,95])
rgb=(lo[None,None,:]*(1-P[...,None])+hi[None,None,:]*P[...,None]).astype(np.uint8)
Image.fromarray(rgb).resize((300*8,27*40),Image.Resampling.NEAREST).save(f'{OUT}/assets/images/appendix/esd_heatmap_27x300.png')
cuts=[int((m.difficulty=='easy').sum()), int((m.difficulty!='hard').sum())]
assert list(m.difficulty.iloc[:cuts[0]].unique())==['easy'] and list(m.difficulty.iloc[cuts[1]:].unique())==['hard']
# score strip (vector polyline, 30 samples averaged)
sc=m.rpx_ds.values; smin,smax=0.27,0.74
pts=' '.join(f'({(i+.5)/300:.4f},{(v-smin)/(smax-smin):.4f})' for i,v in enumerate(sc) if i%3==1)
# ---- TikZ overlay
T=[r'''% ESD descriptor heatmap (supplementary 27-descriptor characterization). Matrix raster: one block per
% scene-phase cell (columns, ordered by the supplementary ESD score) x descriptor (rows);
% value = within-feature percentile rank, as entering the score.
\begin{tikzpicture}[x=1cm,y=1cm,lab/.style={inner sep=0pt,font=\fontsize{7.5}{9}\selectfont}]
\def\mx{5.3}\def\mw{12.4}\def\rh{0.29}
\pgfmathsetmacro\mh{27*\rh}
\node[anchor=north west,inner sep=0pt] at (\mx,0) {\includegraphics[width=\mw cm,height=\mh cm]{assets/images/appendix/esd_heatmap_27x300.png}};
\draw[rpxAccent,line width=.4pt] (\mx,0) rectangle ++(\mw,-\mh);''']
r=0
for fi,(fn,g) in enumerate(fam):
    y0=-r*0.29
    T.append(rf'\node[lab,anchor=north west,text=rpxInk] at (0,{y0-0.045:.3f}) {{{fn}}};')
    for k,f in enumerate(g):
        yy=-(r+k+.5)*0.29
        T.append(rf'\node[lab,anchor=east,font=\fontsize{{7.5}}{{9}}\selectfont\ttfamily,text=rpxSlate] at (\mx-.08,{yy:.3f}) {{{f.replace("_",r"\_")}}};')
    r+=len(g)
    if fi<len(fam)-1:
        T.append(rf'\draw[white,line width=1.6pt] (\mx,{-r*0.29:.3f}) -- ++(\mw,0);')
        T.append(rf'\draw[rpxHair,line width=.3pt] (0,{-r*0.29:.3f}) -- (\mx-.05,{-r*0.29:.3f});')
for c in cuts:
    T.append(rf'\draw[rpxInk,line width=.6pt,dash pattern=on 2.2pt off 1.4pt] (\mx+{c/300:.4f}*\mw,0.95) -- (\mx+{c/300:.4f}*\mw,-\mh);')
for (a,b),col,lab in zip([(0,cuts[0]),(cuts[0],cuts[1]),(cuts[1],300)],['tierE','tierM','tierH'],['Easy (100)','Medium (100)','Hard (100)']):
    T.append(rf'\node[lab,anchor=base,text={col}] at (\mx+{(a+b)/600:.4f}*\mw,1.02) {{{lab}}};')
T.append(r'''% ESD characterization score of each column
\begin{scope}[shift={(\mx,0.12)},x=\mw cm,y=0.68cm]
  \draw[rpxHair,line width=.3pt] (0,0) rectangle (1,1);
  \draw[rpxAccent,line width=.6pt] plot coordinates {'''+pts+r'''};
\end{scope}
\node[lab,anchor=east,text=rpxSlate] at (\mx-.08,.46) {ESD score};
\node[lab,anchor=north west,text=rpxSlate] at (\mx,-\mh-.12) {300 scene--phase cells, ordered by ESD characterization score $\rightarrow$};
% colour key
\shade[left color={rgb,255:red,247;green,249;blue,251},right color={rgb,255:red,30;green,58;blue,95}] (\mx+\mw-4.05,-\mh-.2) rectangle ++(1.9,-.22);
\draw[rpxAccent,line width=.3pt] (\mx+\mw-4.05,-\mh-.2) rectangle ++(1.9,-.22);
\node[lab,anchor=east,text=rpxSlate] at (\mx+\mw-4.13,-\mh-.31) {0};
\node[lab,anchor=west,text=rpxSlate] at (\mx+\mw-2.07,-\mh-.31) {1\enspace percentile rank};
\end{tikzpicture}''')
open(f'{OUT}/figures/fig_esd_heatmap.tex','w').write('\n'.join(T)+'\n')
# ---- schema + weights + statistics table (one display unit per row)
def row_fmt(vals):
    mx=max(abs(v) for v in vals)
    k=0 if mx>=0.1 else int(-np.floor(np.log10(mx)))       # shown = raw x 10^k
    sv=[v*10**k for v in vals]; smx=mx*10**k
    dec=0 if smx>=100 else (1 if smx>=10 else (2 if smx>=1 else 3))
    unit='' if k==0 else rf'$10^{{-{k}}}$'
    def f(x):
        if x==0: return '0'
        if abs(x)<0.5*10**-dec: return r'${<}$'+f'{10**-dec:.{dec}f}'
        return f'{x:.{dec}f}'
    return unit,[f(x) for x in sv]
L=[]
for fi,(fn,g) in enumerate(fam):
    for k,f in enumerate(g):
        v=m[f]; unit,q=row_fmt([v.mean(),v.std(),v.median(),v.quantile(.1),v.quantile(.9),v.min(),v.max()])
        L.append((fn if k==0 else '')+r' & \texttt{'+f.replace('_',r'\_')+'} & '+f'{w[f]:.3f} & {unit} & '+' & '.join(q)+r' \\')
    if fi<len(fam)-1: L.append(r'\addlinespace[2.5pt]')
tab=r"""% Generated by tools/esd/build_esd27.py: 27 supplementary difficulty descriptors, weights, and statistics.
\begin{table*}[t]
\centering
\caption{Supplementary difficulty descriptors: schema, weights in the ESD characterization score (Eq.~\eqref{eq:appendix-esd-score}), and raw distribution over the 300 scene--phase cells. Statistics are raw descriptor values before percentile ranking, shown in the unit listed (e.g.\ $10^{-3}$: value $\times 10^{-3}$).}
\label{tab:appendix-esd-features}
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\renewcommand{\arraystretch}{1.02}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcc rrrrrrr@{}}
\toprule
Group & Descriptor & Weight & Unit & Mean & SD & Median & P10 & P90 & Min & Max \\
\midrule
"""+'\n'.join(L)+r"""
\midrule
\multicolumn{4}{@{}l}{27 descriptors $=$ 2 effort $+$ 25 non-effort; $\sum_i w_i=1$} & \multicolumn{7}{r@{}}{$n=300$ scene--phase cells} \\
\bottomrule
\end{tabular*}
\end{table*}
"""
open(f'{OUT}/generated/esd_feature_table.tex','w').write(tab)
print('cuts',cuts,'weights sum',sum(w.values()))
