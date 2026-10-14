# -*- coding: utf-8 -*-
import subprocess, os, glob
import pandas as pd, numpy as np

def get(hdfs, tag):
    subprocess.run(['docker','exec','namenode','rm','-rf','/tmp/e_'+tag], check=False)
    subprocess.run(['docker','exec','namenode','hdfs','dfs','-get',hdfs,'/tmp/e_'+tag], check=True)
    loc = '/tmp/loc_'+tag
    os.system('rm -rf ' + loc)
    subprocess.run(['docker','cp','namenode:/tmp/e_'+tag, loc], check=True)
    fs = sorted(glob.glob(loc+'/*.csv')) + sorted(glob.glob(loc+'/**/*.csv', recursive=True))
    fs = sorted(set(fs))
    print('  %s: %d csv' % (tag, len(fs)))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)

ev = get('/opt/work/data-out/audit_src','ev'); ev.columns=[c.strip() for c in ev.columns]
vid = get('/opt/work/data-out/audit_vid','vid'); vid.columns=[c.strip() for c in vid.columns]
print('events', ev.shape, list(ev.columns), '| videos', vid.shape, list(vid.columns))
ev = ev[ev.bvid.notna()].copy(); vid = vid[vid.bvid.notna()].copy()
ev['n_methods']=ev.n_methods.astype(int)
np.random.seed(42)
ALLOC = {2: 30, 3: 35, 4: 25}
picked = []
for k, n in ALLOC.items():
    pool = sorted(ev[ev.n_methods == k].bvid.unique())
    t = min(n, len(pool)); picked += list(np.random.choice(pool, t, replace=False))
    print('  n_methods=%d 池 %d 抽 %d' % (k, len(pool), t))
sel = ev[ev.bvid.isin(picked)].merge(vid, on='bvid', how='left', suffixes=('','_v'))
sel = sel.sort_values(['n_methods','bvid','event_id']).reset_index(drop=True)
sel['video_url'] = 'https://www.bilibili.com/video/' + sel.bvid + '?t=' + sel.start_sec.astype(int).astype(str)
sel['verdict'] = ''
sel['dur_s'] = sel['duration'] if 'duration' in sel.columns else ''
out = sel[['bvid','tname','event_id','start_sec','end_sec','score_fused','n_methods','video_url','verdict']]
print('审核清单:', out.shape[0], '行 /', out.bvid.nunique(), '视频')
print('  按 n_methods:', out.groupby('n_methods').size().to_dict())
out.to_csv('/home/molly/projects/danmaku-v2/data-out/candidate_audit.csv', index=False, encoding='utf-8-sig')
print('已写 data-out/candidate_audit.csv')