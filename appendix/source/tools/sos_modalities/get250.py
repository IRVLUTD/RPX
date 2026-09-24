import sys; sys.path.insert(0,'.')
from rangefetch import resolve, rng, headers
def get(path,target,lo_frac,hi_frac,win=2_000_000):
    url,tot=resolve(path); lo,hi=int(tot*lo_frac),int(tot*hi_frac)
    for _ in range(30):
        mid=(lo+hi)//2; a=max(0,(mid-win//2)//512*512); b=min(tot-1,a+win-1); buf=rng(url,a,b); hs=headers(buf,a)
        names=[h[0] for h in hs]
        if target in names:
            n,off,size=hs[names.index(target)]; return buf[off-a:off-a+size] if off-a+size<=len(buf) else rng(url,off,off+size-1)
        if not names: lo=mid; continue
        if target<names[0]: hi=mid
        else: lo=mid
    raise RuntimeError(target)
o='power_drill'
for path,t,lo,hi,out in [(f'objects/{o}/0/rgb.tar','rgb/00250.webp',0,1,'rgb.webp'),(f'objects/{o}/0/depth.tar','depth/00250.png',0,1,'depth.png'),
                         (f'objects/{o}/0/fisheye.tar','fisheye/left/00250.webp',0,.5,'fl.webp'),(f'objects/{o}/0/fisheye.tar','fisheye/right/00250.webp',.5,1,'fr.webp')]:
    open(out,'wb').write(get(path,t,lo,hi)); print('ok',out)
