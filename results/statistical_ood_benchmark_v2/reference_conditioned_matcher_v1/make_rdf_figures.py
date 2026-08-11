#!/usr/bin/env python3
"""RDF / S(k) fingerprint figures + inter-family confusion heatmap.

Reference curves come straight from the cached anchors (no recompute). Two candidates
(Steven1, one amorphous) are re-described to overlay match vs mismatch.
"""
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
import build_and_run_matcher as M
import match_multi_domain as MD

SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; GRID="#e7e6e3"
CAT={"CsCl":"#2a78d6","Th3P4":"#008300","Cu3Au":"#e87ba4","Ni3Sn":"#eda100",
     "zangenite":"#1baf7a","Cr3Si_A15":"#eb6834","rocksalt_B1":"#4a3aa7"}
mpl.rcParams.update({"figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,
    "font.size":11,"axes.edgecolor":INK2,"axes.linewidth":0.8,"text.color":INK,
    "axes.labelcolor":INK,"xtick.color":INK2,"ytick.color":INK2,
    "axes.spines.top":False,"axes.spines.right":False})

anchors,tau=pickle.load(open(HERE/"library_cache.pkl","rb"))
scales=M.anchor_channel_scales(anchors)
fams=list(M.FAMILIES)


def fig_reference_rdfs():
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for f in fams:
        a=anchors[f]
        axes[0].plot(a["r"],a["gr"]["total"],color=CAT[f],lw=1.8,label=f)
        axes[1].plot(a["k"],a["sk_shape"],color=CAT[f],lw=1.8,label=f)
    axes[0].set_xlabel("r / d₀  (neighbour distance)"); axes[0].set_ylabel("g(r)  total RDF")
    axes[0].set_title("Real-space fingerprint: total RDF of each reference crystal",fontsize=11.5,loc="left")
    axes[0].axhline(1,color=INK2,lw=0.7,ls=":")
    axes[1].set_xlabel("k · d₀  (reciprocal)"); axes[1].set_ylabel("S(k) shape")
    axes[1].set_title("Reciprocal-space fingerprint: structure factor",fontsize=11.5,loc="left")
    for ax in axes: ax.grid(color=GRID,lw=0.8); ax.legend(fontsize=8.5,frameon=False,ncol=2)
    fig.tight_layout(); fig.savefig(FIG/"5_reference_fingerprints.png",dpi=200); plt.close(fig)


def fig_match_vs_mismatch():
    import gsd.hoomd
    # candidates
    with gsd.hoomd.open(str(M.UP/"data/test_set/steven_example_1.gsd"),"r") as t:
        p,ti,b=MD.canonical(t[-1])
    nn=MD.robust_first_shell_distance(p,b); dom=M.gas_safe_domain(p,b,nn)
    s1=MD.describe_domain(p,ti,b,dom)
    amo="/projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/preprocessing_pipeline/nicoles_dataset/phase_diagram/debye5.25_volumefraction.015943113378_seed2/trajectory.gsd"
    with gsd.hoomd.open(amo,"r") as t:
        p,ti,b=MD.canonical(t[-1])
    nn=MD.robust_first_shell_distance(p,b); dom=M.gas_safe_domain(p,b,nn)
    am=MD.describe_domain(p,ti,b,dom)

    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    # match: Steven1 vs Ni3Sn
    axes[0].plot(anchors["Ni3Sn"]["r"],anchors["Ni3Sn"]["gr"]["total"],color=CAT["Ni3Sn"],lw=2.2,label="Ni3Sn reference")
    axes[0].plot(s1["r"],s1["gr"]["total"],color=INK,lw=1.6,ls="--",label="Steven 1 candidate")
    axes[0].set_title("MATCH: Steven 1 lands on Ni3Sn (peaks line up)",fontsize=11.5,loc="left",color="#0ca30c")
    # mismatch: amorphous vs CsCl
    axes[1].plot(anchors["CsCl"]["r"],anchors["CsCl"]["gr"]["total"],color=CAT["CsCl"],lw=2.2,label="CsCl reference")
    axes[1].plot(am["r"],am["gr"]["total"],color=INK,lw=1.6,ls="--",label="amorphous candidate")
    axes[1].set_title("NO MATCH: amorphous is washed-out (no sharp peaks)",fontsize=11.5,loc="left",color="#d03b3b")
    for ax in axes:
        ax.set_xlabel("r / d₀"); ax.set_ylabel("g(r) total RDF"); ax.axhline(1,color=INK2,lw=0.7,ls=":")
        ax.grid(color=GRID,lw=0.8); ax.legend(fontsize=9.5,frameon=False)
    fig.tight_layout(); fig.savefig(FIG/"6_match_vs_mismatch_rdf.png",dpi=200); plt.close(fig)


def fig_confusion():
    D=pd.read_csv(HERE/"inter_family_distance_matrix.csv",index_col=0)
    fig,ax=plt.subplots(figsize=(7.2,6.2))
    m=D.values.copy(); np.fill_diagonal(m,np.nan)
    im=ax.imshow(m,cmap="cividis_r",vmin=0.5,vmax=1.4)
    ax.set_xticks(range(7)); ax.set_xticklabels(D.columns,rotation=40,ha="right",fontsize=9)
    ax.set_yticks(range(7)); ax.set_yticklabels(D.index,fontsize=9)
    for i in range(7):
        for j in range(7):
            if i==j: ax.text(j,i,"—",ha="center",va="center",color=INK2,fontsize=9)
            else:
                v=D.values[i,j]; ax.text(j,i,f"{v:.2f}",ha="center",va="center",
                    color="white" if v<0.95 else INK,fontsize=8.5)
    cb=fig.colorbar(im,fraction=0.046,pad=0.04); cb.set_label("anchor-scaled distance (lower = more confusable)")
    ax.set_title("Structural confusion map: how far each reference is from every other\n"
                 "(Th3P4↔zangenite and the 1:3 cluster are the confusable pairs)",fontsize=11,loc="left")
    fig.tight_layout(); fig.savefig(FIG/"7_confusion_matrix.png",dpi=200); plt.close(fig)


if __name__=="__main__":
    fig_reference_rdfs(); fig_confusion()
    try: fig_match_vs_mismatch()
    except Exception as e: print("match/mismatch fig failed:",e)
    print("wrote:",*[p.name for p in sorted(FIG.glob('[567]_*.png'))])
