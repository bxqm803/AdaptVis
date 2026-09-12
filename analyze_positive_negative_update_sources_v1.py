#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Post-process analyze_real_update_module_head_sources_v1.py outputs.
No model forward, no generation.

For each realized update:
    B_REAL = B_ATTN + B_MLP
positive/helpful if B_REAL>threshold, negative/harmful if B_REAL<-threshold.

For sign s=sign(B_REAL):
    module same-sign mass = max(s * B_module, 0)
    module opposing mass  = max(-s * B_module, 0)

Heads are a decomposition INSIDE Attention, not a third peer module:
    B_ATTN ~= sum_h B_HEAD[h]
"""

from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd

EPS = 1e-12
KEYS = ["sid", "update_layer", "real_position"]


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--threshold", type=float, default=1e-8)
    p.add_argument("--top-heads", type=int, default=30)
    p.add_argument("--topk-per-update", default="1,3,5")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def boolify(x):
    if isinstance(x, (bool, np.bool_)): return bool(x)
    if pd.isna(x): return False
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y", "t"}


def ensure_dir(p: Path, overwrite: bool):
    if overwrite and p.exists(): shutil.rmtree(p)
    if p.exists() and any(p.iterdir()):
        raise RuntimeError(f"Non-empty output directory: {p}")
    p.mkdir(parents=True, exist_ok=True)


def safe_mean(x):
    a = np.asarray(list(x), float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else np.nan


def safe_median(x):
    a = np.asarray(list(x), float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else np.nan


def load_tables(inp: Path):
    mp = inp / "per_module_decision_contribution.csv"
    hp = inp / "per_head_decision_contribution.csv"
    if not mp.exists(): raise FileNotFoundError(mp)
    if not hp.exists(): raise FileNotFoundError(hp)
    m, h = pd.read_csv(mp), pd.read_csv(hp)
    req_m = {"sid","gt","baseline_correct","real_position","update_layer","B_real","B_attn","B_mlp"}
    req_h = {"sid","gt","baseline_correct","real_position","update_layer","head","head_name","head_decision_score"}
    if req_m-set(m.columns): raise RuntimeError(f"module CSV missing {sorted(req_m-set(m.columns))}")
    if req_h-set(h.columns): raise RuntimeError(f"head CSV missing {sorted(req_h-set(h.columns))}")
    for df in (m,h):
        df["sid"] = pd.to_numeric(df["sid"]).astype(int)
        df["update_layer"] = pd.to_numeric(df["update_layer"]).astype(int)
        df["real_position"] = pd.to_numeric(df["real_position"]).astype(int)
        df["baseline_correct"] = df["baseline_correct"].map(boolify)
    for c in ("B_real","B_attn","B_mlp"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    h["head"] = pd.to_numeric(h["head"]).astype(int)
    h["head_decision_score"] = pd.to_numeric(h["head_decision_score"], errors="coerce")
    for c in ("is_direction_head","is_centroid_head","is_spatial_head"):
        h[c] = h[c].map(boolify) if c in h.columns else False
    return m,h,mp,hp


def classify_updates(m, thr):
    x = m.copy()
    br = x.B_real.to_numpy(float)
    s = np.where(br>thr,1,np.where(br<-thr,-1,0))
    x["update_sign"] = s
    x["update_polarity"] = np.where(s>0,"positive",np.where(s<0,"negative","neutral"))
    ba, bm = x.B_attn.to_numpy(float), x.B_mlp.to_numpy(float)
    a_same, m_same = np.maximum(s*ba,0), np.maximum(s*bm,0)
    a_opp, m_opp = np.maximum(-s*ba,0), np.maximum(-s*bm,0)
    den = a_same + m_same
    x["attn_same_sign_mass"] = a_same
    x["mlp_same_sign_mass"] = m_same
    x["attn_opposing_mass"] = a_opp
    x["mlp_opposing_mass"] = m_opp
    x["attn_same_sign_share"] = np.where(den>EPS,a_same/den,np.nan)
    x["mlp_same_sign_share"] = np.where(den>EPS,m_same/den,np.nan)
    dom = np.full(len(x),"none",object)
    dom[(den>EPS)&(a_same>m_same+EPS)] = "attention"
    dom[(den>EPS)&(m_same>a_same+EPS)] = "mlp"
    dom[(den>EPS)&(np.abs(a_same-m_same)<=EPS)] = "tie"
    x["dominant_same_sign_module"] = dom
    pattern=[]
    for sr,aa,mm in zip(s,ba,bm):
        sa = 1 if aa>thr else -1 if aa<-thr else 0
        sm = 1 if mm>thr else -1 if mm<-thr else 0
        if sr==0: p="neutral"
        elif sa==sr and sm==sr: p="both_same_sign"
        elif sa==sr and sm==-sr: p="attention_drives_mlp_opposes"
        elif sm==sr and sa==-sr: p="mlp_drives_attention_opposes"
        elif sa==sr: p="attention_only_same_sign"
        elif sm==sr: p="mlp_only_same_sign"
        else: p="other"
        pattern.append(p)
    x["module_sign_pattern"] = pattern
    return x


def summarize_modules(df, group_cols):
    rows=[]
    z=df[df.update_sign!=0]
    for keys,g in z.groupby(group_cols, dropna=False):
        if not isinstance(keys,tuple): keys=(keys,)
        base=dict(zip(group_cols,keys))
        for pol in ("positive","negative"):
            q=g[g.update_polarity==pol]
            if not len(q): continue
            vc=q.dominant_same_sign_module.value_counts()
            pv=q.module_sign_pattern.value_counts()
            row={**base,"update_polarity":pol,"N_updates":len(q),
                 "mean_abs_B_real":safe_mean(abs(q.B_real)),
                 "mean_B_attn":safe_mean(q.B_attn),"mean_B_mlp":safe_mean(q.B_mlp),
                 "mean_attn_same_sign_share":safe_mean(q.attn_same_sign_share),
                 "mean_mlp_same_sign_share":safe_mean(q.mlp_same_sign_share),
                 "median_attn_same_sign_share":safe_median(q.attn_same_sign_share),
                 "median_mlp_same_sign_share":safe_median(q.mlp_same_sign_share),
                 "dominant_attention_fraction":vc.get("attention",0)/len(q),
                 "dominant_mlp_fraction":vc.get("mlp",0)/len(q)}
            for p in ("both_same_sign","attention_drives_mlp_opposes","mlp_drives_attention_opposes","attention_only_same_sign","mlp_only_same_sign","other"):
                row[f"fraction_{p}"]=pv.get(p,0)/len(q)
            rows.append(row)
    return pd.DataFrame(rows)


def attach_heads(h, updates):
    meta=updates[KEYS+["update_sign","update_polarity","B_real","B_attn","B_mlp"]].drop_duplicates(KEYS)
    z=h.merge(meta,on=KEYS,how="inner",validate="many_to_one")
    s=z.update_sign.to_numpy(float); bh=z.head_decision_score.to_numpy(float)
    z["head_same_sign_mass"]=np.maximum(s*bh,0)
    z["head_opposing_mass"]=np.maximum(-s*bh,0)
    z["head_same_sign"]=(s*bh)>0
    return z


def summarize_heads(z):
    rows=[]
    z=z[z.update_sign!=0]
    for pol in ("positive","negative"):
        g=z[z.update_polarity==pol]
        total_same=max(float(g.head_same_sign_mass.sum()),EPS)
        for name,q in g.groupby("head_name"):
            f=q.iloc[0]; same=float(q.head_same_sign_mass.sum())
            rows.append({"update_polarity":pol,"head_name":name,"update_layer":int(f.update_layer),"head":int(f.head),
                         "is_direction_head":bool(f.is_direction_head),"is_centroid_head":bool(f.is_centroid_head),
                         "N_messages":len(q),"N_samples":q.sid.nunique(),
                         "mean_B_head":safe_mean(q.head_decision_score),
                         "mean_abs_B_head":safe_mean(abs(q.head_decision_score)),
                         "same_sign_fraction":float(q.head_same_sign.mean()),
                         "total_same_sign_mass":same,
                         "share_attention_same_sign_mass":same/total_same})
    out=pd.DataFrame(rows)
    if len(out): out["same_sign_mass_rank"]=out.groupby("update_polarity").total_same_sign_mass.rank(method="min",ascending=False).astype(int)
    return out


def topk_enrichment(z, ks):
    rows=[]; z=z[z.update_sign!=0]
    for pol in ("positive","negative"):
        g=z[z.update_polarity==pol]
        for col,label in (("is_direction_head","direction"),("is_centroid_head","centroid"),("is_spatial_head","spatial_union")):
            base=float(g[col].mean())
            for k in ks:
                parts=[]
                for _,q in g.groupby(KEYS,sort=False):
                    parts.append(q.sort_values("head_same_sign_mass",ascending=False).head(k))
                sel=pd.concat(parts,ignore_index=True) if parts else g.iloc[:0]
                rate=float(sel[col].mean()) if len(sel) else np.nan
                rows.append({"update_polarity":pol,"head_type":label,"top_k_per_update":k,
                             "base_rate":base,"topk_rate":rate,
                             "enrichment":rate/base if base>EPS else np.nan,"N_top_rows":len(sel)})
    return pd.DataFrame(rows)


def main():
    a=parse_args(); inp=Path(a.input_dir); out=Path(a.output_dir); ensure_dir(out,a.overwrite)
    m,h,mp,hp=load_tables(inp)
    u=classify_updates(m,a.threshold)
    u.to_csv(out/"per_update_source_classification.csv",index=False)

    allu=u.copy(); allu["cohort"]="all"
    corr=u.copy(); corr["cohort"]=np.where(corr.baseline_correct,"baseline_correct","baseline_wrong")
    stack=pd.concat([allu,corr],ignore_index=True)
    overall=summarize_modules(stack,["cohort"])
    byl=summarize_modules(stack,["cohort","update_layer"])
    byr=summarize_modules(stack,["cohort","gt"])
    overall.to_csv(out/"module_source_summary.csv",index=False)
    byl.to_csv(out/"module_source_by_layer.csv",index=False)
    byr.to_csv(out/"module_source_by_relation.csv",index=False)

    hz=attach_heads(h,u)
    hs=summarize_heads(hz)
    hs.to_csv(out/"head_source_summary.csv",index=False)
    pos=hs[hs.update_polarity=="positive"].sort_values("total_same_sign_mass",ascending=False).head(a.top_heads)
    neg=hs[hs.update_polarity=="negative"].sort_values("total_same_sign_mass",ascending=False).head(a.top_heads)
    pos.to_csv(out/"top_heads_positive_updates.csv",index=False)
    neg.to_csv(out/"top_heads_negative_updates.csv",index=False)
    enrich=topk_enrichment(hz,[int(x) for x in a.topk_per_update.split(",") if x.strip()])
    enrich.to_csv(out/"head_topk_enrichment.csv",index=False)

    cols=["cohort","update_polarity","N_updates","mean_abs_B_real","mean_B_attn","mean_B_mlp",
          "mean_attn_same_sign_share","mean_mlp_same_sign_share","dominant_attention_fraction","dominant_mlp_fraction",
          "fraction_both_same_sign","fraction_attention_drives_mlp_opposes","fraction_mlp_drives_attention_opposes"]
    report=["="*170,"POSITIVE / NEGATIVE UPDATE SOURCE DECOMPOSITION","="*170,
            f"N updates={len(m)} | N head messages={len(h)}","","MODULE SUMMARY","-"*170,
            overall[cols].to_string(index=False,float_format=lambda x:f"{x:.5f}"),"",
            "TOP HEADS FOR POSITIVE UPDATES","-"*170,
            pos.head(15).to_string(index=False,float_format=lambda x:f"{x:.6f}") if len(pos) else "EMPTY","",
            "TOP HEADS FOR NEGATIVE UPDATES","-"*170,
            neg.head(15).to_string(index=False,float_format=lambda x:f"{x:.6f}") if len(neg) else "EMPTY","",
            "INTERPRETATION:",
            "  1) Compare dominant_attention_fraction vs dominant_mlp_fraction separately for positive/negative updates.",
            "  2) attention_drives_mlp_opposes: Attention determines the net sign while MLP resists it.",
            "  3) mlp_drives_attention_opposes: MLP determines the net sign while Attention resists it.",
            "  4) Heads are nested inside Attention; inspect top_heads_* only after checking Attention's module-level role.",
            "  5) This is exact additive attribution of the realized update, not independent causal mediation."]
    text="\n".join(report)+"\n"; print(text); (out/"analysis_summary.txt").write_text(text,encoding="utf-8")
    (out/"metadata.json").write_text(json.dumps({"script":"analyze_positive_negative_update_sources_v1.py","input_dir":str(inp),"threshold":a.threshold,"model_forward":False,"generation":False,"intervention":False},indent=2),encoding="utf-8")

if __name__=="__main__": main()
