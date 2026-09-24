"""Build generated/sos_catalogue.tex. Usage: build_catalogue.py <last ID of each column block except the last, e.g. 2,19,36,53>"""
import json, sys
import os
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
ents=json.load(open(os.path.join(HERE,'catalogue_canonical.json')))
splits=[int(x) for x in sys.argv[1].split(',')]   # last ID of every column block except the final one
bounds=[0]+splits+[70]
head=r'\toprule\textbf{ID} & \textbf{Object and canonical response}\\\midrule'
col=r'@{}>{\bfseries}r@{\hspace{5pt}}>{\raggedright\arraybackslash}p{\dimexpr\columnwidth-2.3em\relax}@{}'
L=[r'''% Table XIII -- canonical SOS questionnaire catalogue (all 70 objects), set as
% four column blocks balanced over the last two pages; each continuation block
% repeats the table number and the column heads.
\begingroup
\fontsize{\catfs}{\catlead}\selectfont
\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1}''']
for k in range(len(bounds)-1):
    rows=[]
    for i in range(bounds[k],bounds[k+1]):
        n,name,resp=ents[i]
        sep=r'\\\catsep' if i<bounds[k+1]-1 else r'\\'
        rows.append(rf'{n} & \textbf{{{name}}}\newline {resp}{sep}')
    L.append(r'\noindent\begin{minipage}[t]{\columnwidth}\centering')
    if k==0:
        L.append(r'{\normalsize\captionof{table}{Canonical SOS questionnaire catalogue. Each entry lists the object name and one lightly normalized canonical category, colour, material, and function response.}\label{tab:sos-canonical}}')
    else:
        L.append(r'{\footnotesize TABLE~\thetable{} (continued)}\par\vspace{4pt}')
    L.append(r'\begin{tabular}{'+col+'}\n'+head)
    L+=rows
    L.append(r'\bottomrule'+'\n'+r'\end{tabular}'+'\n'+r'\end{minipage}')
    if k<len(bounds)-2: L.append(r'\newpage')
L.append(r'\endgroup')
open(os.path.join(ROOT,'generated','sos_catalogue.tex'),'w').write('\n'.join(L)+'\n')
