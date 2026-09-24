import requests, tarfile, os, time
from huggingface_hub import hf_hub_url, get_token
S=requests.Session(); S.headers.update({'Authorization':f'Bearer {get_token()}'})
def resolve(path):
    for k in range(5):
        try:
            r=S.head(hf_hub_url('IRVLUTD/RPX',path,repo_type='dataset'),allow_redirects=True,timeout=60); return r.url,int(r.headers['Content-Length'])
        except Exception: time.sleep(3)
def rng(url,a,b):
    for k in range(5):
        try:
            r=S.get(url,headers={'Range':f'bytes={a}-{b}'},timeout=90)
            if r.status_code==206: return r.content
        except Exception: time.sleep(3)
    raise RuntimeError('range')
def headers(buf,base):
    out=[]
    for o in range(0,len(buf)-511,512):
        blk=buf[o:o+512]
        if blk[257:262]!=b'ustar': continue
        try: ti=tarfile.TarInfo.frombuf(blk,'utf-8','surrogateescape')
        except Exception: continue
        out.append((ti.name,base+o+512,ti.size))
    return out
def first(path,n=4096):
    url,tot=resolve(path); return headers(rng(url,0,n-1),0), tot
def fetch(path,want,frac,win=3_000_000):
    """want: predicate on member name; frac: estimated relative offset."""
    url,tot=resolve(path); est=int(tot*frac)
    for _ in range(12):
        a=max(0,(est-win//2)//512*512); b=min(tot-1,a+win-1); buf=rng(url,a,b); hs=headers(buf,a)
        hit=[h for h in hs if want(h[0])]
        if hit:
            name,off,size=hit[0]; data=buf[off-a:off-a+size] if off-a+size<=len(buf) else rng(url,off,off+size-1); return name,data
        if not hs: est+=win//2; continue
        names=sorted(h[0] for h in hs); est+= win//2 if all(n<'~' for n in names) and want(names[-1]+'~')==False and names[-1]<'zzz' else 0
        # ordering heuristic handled by caller-specific key
        raise RuntimeError(f'not in window: {names[:2]}..{names[-2:]}')
