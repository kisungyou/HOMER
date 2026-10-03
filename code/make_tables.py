import argparse
import json
import pandas as pd
from settings import RESULTS as R, TABLES, CONFIG, MODE
# Display labels are independent of the historical result-file method keys.
METHOD_LABELS={'PH(1)':'HOMER(1)', 'PH(16)':'HOMER(16)', 'Adaptive(2)':'Adaptive HOMER'}
def table(name,caption,label,columns,header,rows,size='\\small'):
 text='\\begin{table}[ht]\n\\centering\n'+size+'\n\\setlength{\\tabcolsep}{3pt}\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\begin{tabular}{'+columns+'}\n\\toprule\n'+header+' \\\\\n\\midrule\n'+'\n'.join(' & '.join(map(str,row))+' \\\\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
 (TABLES/name).write_text(text)
def cv(row):return f"{row.coverage:.3f} [{row.mc_low:.3f},{row.mc_high:.3f}]"
def scalar():
 s=pd.read_csv(R/'scalar_summary.csv');rows=[]
 for schedule in ['slow','boundary','many']:
  for a in [4,5,6]:
   q=s[(s.schedule==schedule)&(s.a==a)].set_index('metric');base=q.iloc[0]
   rows.append([schedule,int(base.k),int(base.m),*[cv(q.loc[j]) for j in ['plugin_mean_covered','plugin_target_covered','oracle_mean_covered','oracle_target_covered']],f'{base.mean_plugin_length:.4f}',f'{base.mean_oracle_length:.4f}'])
 table('scalar_primary_table.tex','HOMER(1) scalar coverage with pointwise 95\\% Wilson intervals. Each cell uses REPLICATIONS replications. E and P denote estimated and population sandwich variance. The last columns give mean interval lengths.'.replace('REPLICATIONS',f'{int(s.reps.iloc[0]):,}'),'tab:scalar-primary','lrrccccrr','Schedule & $k$ & $m$ & E, mean & E, target & P, mean & P, target & E length & P length',rows,'\\scriptsize')


def functional():
 f=pd.read_csv(R/'functional_inference_summary.csv');rows=[]
 for spectrum in ['concentrated','diffuse']:
  for distribution in ['gaussian','exponential']:
   for k in [16,32,64]:
    q=f[(f.spectrum==spectrum)&(f.distribution==distribution)&(f.k==k)].set_index('contrast');a=q.loc['interval_average'];b=q.loc['evening_minus_morning'];c=q.loc['smoothed_evaluation']
    rows.append([spectrum,'Gaussian' if distribution=='gaussian' else 'Exponential',k,int(a.m),cv(a),cv(b),cv(c),f'{a.mean_length:.4f}',f'{b.mean_length:.4f}',f'{c.mean_length:.4f}'])
 table('functional_coverage_table.tex','HOMER(1) functional mean coverage and pointwise 95\\% Wilson intervals, with REPLICATIONS replications per cell. IA is the interval average, EM is evening minus morning, and SE is smoothed evaluation. Lengths are averaged across replications.'.replace('REPLICATIONS',f'{int(f.reps.iloc[0]):,}'),'tab:functional-coverage','llrrcccrrr','Spectrum & Law & $k$ & $m$ & IA coverage & EM coverage & SE coverage & IA length & EM length & SE length',rows,'\\scriptsize')
 p=pd.read_csv(R/'functional_point_summary.csv');rows=[]
 for spectrum in ['concentrated','diffuse']:
  for distribution in ['gaussian','student_t4']:
   for method in ['Mean','MOM','PH(1)','PH(16)','Adaptive(2)']:
    q=p[(p.spectrum==spectrum)&(p.distribution==distribution)&(p.method==method)];clean=q[q.mechanism=='clean'].iloc[0];block=q[(q.mechanism=='whole_block')&(q.magnitude==100)].iloc[0];raw=q[(q.mechanism=='raw_dispersed')&(q.magnitude==100)].iloc[0]
    rows.append([spectrum,'Gaussian' if distribution=='gaussian' else '$t_4$',METHOD_LABELS.get(method,method),f'{clean.mse*1e4:.3f}',f'{block.q95:.3f} [{block.q95_mc_low:.3f},{block.q95_mc_high:.3f}]',f'{raw.q95:.3f} [{raw.q95_mc_low:.3f},{raw.q95_mc_high:.3f}]'])
 table('point_summary_table.tex','Clean squared error and stressed 95th-percentile norm error. HOMER uses pseudo-Huber loss; parentheses give fixed standardized thresholds, and adaptive HOMER uses multiplier two. Every cell has REPLICATIONS replications. Stresses use magnitude 100 and bootstrap intervals use RESAMPLES resamples. Heavy-tailed squared errors are reported without normal Monte Carlo intervals.'.replace('REPLICATIONS',f'{int(p.reps.iloc[0]):,}').replace('RESAMPLES',f"{json.loads((R/'functional_manifest.json').read_text())['quantile_bootstrap_replicates']:,}"),'tab:point-summary','lllrrr','Spectrum & Law & Method & \\shortstack{Clean MSE\\\\$\\times10^4$} & Whole-block quantile & Dispersed-raw quantile',rows,'\\small')


def wearable():
 w=pd.read_csv(R/'wearable_results.csv');rows=[]
 activities={1:'Walking',2:'Upstairs',3:'Downstairs',4:'Sitting',5:'Standing',6:'Laying'}
 for activity in range(1,7):
  row=[activities[activity]]
  for mechanism in ['one_block','dispersed_subjects']:
   q=w[(w.activity==activity)&(w.mechanism==mechanism)&(w.amplitude_multiplier==10)].set_index('method')
   row += [f'{q.loc[method,"shift_relative_clean_mean"]:.3f}' for method in ['Mean','MOM','Adaptive(2)']]
  rows.append(row)
 table('wearable_table.tex','Descriptive energy shifts from each method\'s own clean estimate, divided by the activity\'s clean empirical subject mean. Adaptive HOMER uses pseudo-Huber loss and multiplier two. Perturbed energy is multiplied by 100. One-block corruption affects six participants. Dispersed corruption affects three participants in three blocks.','tab:wearable','lrrrrrr','Activity & \\multicolumn{3}{c}{One block} & \\multicolumn{3}{c}{Dispersed participants} \\\\ \n & Mean & MOM & Adaptive HOMER & Mean & MOM & Adaptive HOMER',rows,'\\small')



def covariance():
 # Matrix-valued nonlinear subset-estimator illustration.
 cov=pd.read_csv(R/'covariance_summary.csv');rows=[]
 for delta in [0,8,32]:
  q=cov[cov.delta==delta].set_index('method')
  rows.append([str(delta)]+[f'{q.loc[method,"mean_frobenius_error"]:.3f} [{q.loc[method,"mean_error_mc_low"]:.3f},{q.loc[method,"mean_error_mc_high"]:.3f}]' for method in ['Mean','GMed','Adaptive PH']])
 table('covariance_table.tex','Mean Frobenius covariance error with pointwise 95\\% bootstrap Monte Carlo intervals from REPLICATIONS paired replications. Inflation $\\delta$ acts in four of 32 blocks. The mean averages the same block covariances. Adaptive HOMER uses pseudo-Huber loss and multiplier two.'.replace('REPLICATIONS',f'{int(cov.replications.iloc[0]):,}'),'tab:covariance','lccc','Inflation $\\delta$ & Mean & Geometric median & Adaptive HOMER',rows,'\\small')


def main():
 parser=argparse.ArgumentParser(description="Generate standalone LaTeX tables from the selected results")
 parser.add_argument('--study',nargs='+',choices=['all','scalar','functional','wearable','covariance'],default=['all'])
 args=parser.parse_args()
 studies=['scalar','functional','wearable','covariance'] if 'all' in args.study else args.study
 for study in dict.fromkeys(studies):
  globals()[study]()
 print(f'Tables saved under {TABLES}; mode={MODE}')

if __name__=='__main__':main()
