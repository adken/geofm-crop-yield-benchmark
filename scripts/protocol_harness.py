import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error
from benchmark_embeddings.s2_indices import load_s2_index_features

ALPHAS=[0.01,0.1,1,10,100]
cen=pd.read_csv("/tmp/out/county_centroids.csv",dtype={"county_id":str}).set_index("county_id")
man=pd.read_csv("data/group_kfold_county_T7.csv",dtype={"county_id":str})
lab=pd.read_csv("data/labels/county_yield.csv")
lab["county_id"]=lab["county"].astype(str).str.zfill(5)
lab=lab[["county_id","year","yield"]].dropna()

def feats_alphaearth():
    d=pd.read_parquet("/tmp/out/embeddings/alphaearth.parquet")
    g=d.groupby(["county_id","year"])["embedding"]
    rows=[]
    for (c,y),s in g:
        e=np.stack(s.values)
        rows.append((str(c).zfill(5),int(y),np.concatenate([e.mean(0),e.std(0)])))
    return pd.DataFrame(rows,columns=["county_id","year","f"])

def feats_s2():
    s2=load_s2_index_features("data/s2_daymet_merged.xlsx",
        fips_map="data/geometry/county_fips_map.csv")
    rows=[(r.county_id,int(r.year),np.asarray(r.features,dtype=float))
          for r in s2.itertuples()]
    return pd.DataFrame(rows,columns=["county_id","year","f"])

def build(fn):
    F=fn().merge(lab,on=["county_id","year"])
    keys=set(man.fips_year)
    F["k"]=F.county_id+"-"+F.year.astype(str)
    F=F[F.k.isin(keys)].copy()
    # Canonical row order so index-based splits are comparable ACROSS frames.
    F=F.sort_values(["county_id","year"]).reset_index(drop=True)
    return F

def fit_eval(F,tr,va,te):
    Xtr=np.stack(F.f[tr]); Xva=np.stack(F.f[va]); Xte=np.stack(F.f[te])
    ytr=F["yield"].values[tr]; yva=F["yield"].values[va]; yte=F["yield"].values[te]
    best=None
    for a in ALPHAS:
        sc=StandardScaler().fit(Xtr)
        m=Ridge(alpha=a).fit(sc.transform(Xtr),ytr)
        rm=np.sqrt(mean_squared_error(yva,m.predict(sc.transform(Xva))))
        if best is None or rm<best[0]: best=(rm,a)
    a=best[1]
    Xfull=np.vstack([Xtr,Xva]); yfull=np.concatenate([ytr,yva])
    sc=StandardScaler().fit(Xfull)
    m=Ridge(alpha=a).fit(sc.transform(Xfull),yfull)
    p=m.predict(sc.transform(Xte))
    return r2_score(yte,p), float(np.sqrt(mean_squared_error(yte,p))), p, yte, F.county_id.values[te]

def hav(la1,lo1,la2,lo2):
    R=6371.0; p1,p2=np.radians(la1),np.radians(la2)
    dp=p2-p1; dl=np.radians(lo2-lo1)
    a=np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(a))
