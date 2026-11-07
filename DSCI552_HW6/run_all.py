"""Run the complete APS Homework 6 pipeline, including plots and all model fits.
Run from the submission folder: python run_all.py
"""
from pathlib import Path
import subprocess,sys,os
ROOT=Path(__file__).resolve().parent
ENV=os.environ.copy();ENV['OPENBLAS_NUM_THREADS']='1';ENV['OMP_NUM_THREADS']='1';ENV['MKL_NUM_THREADS']='1'

def go(*args):
    print('Running:',*args,flush=True)
    subprocess.run([sys.executable,*args],cwd=ROOT,check=True,env=ENV)

if __name__=='__main__':
    go('run_original_baselines.py')
    for smote in ('False','True'):
      for a in ('0.0001','0.0003','0.001'):
        go('cv_one_config.py',a,smote)
    go('finalize_model_trees.py')
    go('recolor_model_tree_figures.py')
    go('make_report.py')
    print('All model results and the PDF report have been rebuilt.',flush=True)
