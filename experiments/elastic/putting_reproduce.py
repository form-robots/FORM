"""Re-run the frozen putting protocol, including both searches and all checks.

The source bundle supplies material estimates and robot assets. No archived
particle trajectories or reported stopping positions are used as inputs.
"""
from pathlib import Path
import argparse
import os
import subprocess
import sys
from experiments.elastic.putting import ROOT,digest
from experiments.elastic.putting_study import prepare,read

def reproduce(source,out,devices):
    hashes=read(source/'reproduction_source_hashes.json')
    different=[name for name,h in hashes.items() if digest(ROOT/name)!=h]
    if different:raise RuntimeError(f'Source differs from the archived experiment: {different}. Use the archived source revision before reproducing.')
    prepare(out,source);logs=out/'logs';logs.mkdir()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    def pair(commands,label):
        running=[]
        for k,args in zip('AB',commands):
            log=(logs/f'{label}_{k}.log').open('w')
            proc=subprocess.Popen([sys.executable,'-m','experiments.elastic.putting_study',*args,'--out',str(out)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            running.append((proc,log))
        for proc,log in running:
            code=proc.wait();log.close()
            if code:raise RuntimeError(f'{label} failed; see {logs}')
    pair([['plan','--material',m,'--device',dev] for m,dev in zip('AB',devices)],'planning')
    for swapped in [False,True]:
        pair([['execute','--material',m,'--plan',('B' if m=='A' else 'A') if swapped else m,'--device',dev]
              for m,dev in zip('AB',devices)],'swapped' if swapped else 'matched')
    for prefix,extra in [('fine',['--grid','320']),('half_contact',['--coupling','.0000625'])]:
        for swapped in [False,True]:
            pair([['check','--material',m,'--plan',('B' if m=='A' else 'A') if swapped else m,'--device',dev,'--prefix',prefix,*extra]
                  for m,dev in zip('AB',devices)],f'{prefix}_{swapped}')
    with (logs/'independent_repeat_A.log').open('w') as log:
        subprocess.run([sys.executable,'-m','experiments.elastic.putting_study','check','--out',str(out),
            '--material','A','--plan','A','--grid','320','--prefix','repeat_fine','--device',devices[0]],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,'-m','experiments.elastic.putting_report','all','--out',str(out)],cwd=ROOT,env=env,check=True)
    from experiments.elastic.putting_audit import summarize
    summarize(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--devices',nargs=2,default=['cuda:0','cuda:1'])
    a=ap.parse_args();reproduce(a.source.resolve(),a.out.resolve(),a.devices)
