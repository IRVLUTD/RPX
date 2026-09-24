"""Fetch the middle frame of every MOS capture (3 exo phases + ego) for the
100-scene catalogue, using HTTP range reads into the released RGB tar shards."""
import requests, tarfile, io, os, sys, json, time, concurrent.futures as cf
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_url, get_token
OUT='/home/naren/.cache/rpx_catalogue_frames'; os.makedirs(OUT,exist_ok=True)
f=pd.read_parquet('/tmp/RPX_two_column_final_20260923_v2/appendix_data/hf_frames_v1.parquet')
f=f[f.scene_type.isin(['multi_object','ego'])]
H={'Authorization':f'Bearer {get_token()}'}
def session():
    s=requests.Session(); s.headers.update(H); return s
def rng(s,url,a,b):
    for k in range(5):
        try:
            r=s.get(url,headers={'Range':f'bytes={a}-{b}'},timeout=90)
            if r.status_code==206: return r.content
            if r.status_code==200: return r.content[a:b+1]
        except Exception as e: time.sleep(2*(k+1))
    raise RuntimeError(f'range failed {url[:60]} {a}')
def headers_in(buf,base):
    out=[]
    for o in range(0,len(buf)-511,512):
        blk=buf[o:o+512]
        if blk[257:262]!=b'ustar': continue
        try: ti=tarfile.TarInfo.frombuf(blk,'utf-8','surrogateescape')
        except Exception: continue
        out.append((os.path.basename(ti.name),base+o+512,ti.size))
    return out
def job(key):
    try: return _job(key)
    except Exception as e: return key,'FAILED'
def _job(key):
    scene,view,target,n,frac=key
    dst=f'{OUT}/{scene}_{view}.webp'
    if os.path.exists(dst): return key,'cached'
    s=session(); path=f'scenes/{scene}/{view}/rgb.tar'
    for k in range(6):
        try:
            r=s.head(hf_hub_url('IRVLUTD/RPX',path,repo_type='dataset'),allow_redirects=True,timeout=60); break
        except Exception: time.sleep(5*(k+1))
    else: raise RuntimeError('head failed')
    url=r.url; total=int(r.headers['Content-Length'])
    win=6_000_000 if view=='ego' else 1_500_000
    est=int(total*frac)
    for attempt in range(8):
        a=max(0,(est-win//2)//512*512); b=min(total-1,a+win-1)
        buf=rng(s,url,a,b); hs=headers_in(buf,a)
        names=[h[0] for h in hs]
        if target in names:
            _,doff,size=hs[names.index(target)]
            data=buf[doff-a:doff-a+size] if doff-a+size<=len(buf) else rng(s,url,doff,doff+size-1)
            open(dst,'wb').write(data); return key,'ok'
        if not hs: est+=win//2; continue
        seen=sorted(names)
        est = est - win//2 if target<seen[0] else est + win//2
    return key,'FAILED'
jobs=[]
for (scene,st,ph),g in f.groupby(['scene_id','scene_type','phase']):
    view='ego' if st=='ego' else str(ph)
    names=sorted(g.frame_filename); n=len(names); t=names[n//2]
    jobs.append((scene,view,t,n,(n//2+0.5)/n))
print(len(jobs),'jobs'); sys.stdout.flush()
res={}
with cf.ThreadPoolExecutor(8) as ex:
    for k,st in ex.map(job,jobs):
        res[f'{k[0]}_{k[1]}']={'frame':k[2],'n':k[3],'status':st}
        if st!='cached' and len(res)%20==0: print(len(res),st,flush=True)
json.dump(res,open(f'{OUT}/selection.json','w'),indent=1)
print('failed',[k for k,v in res.items() if v['status']=='FAILED'])
