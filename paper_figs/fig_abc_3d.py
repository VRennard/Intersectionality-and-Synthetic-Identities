"""3D coefficient space for triples: humans rise to (1,1,1); the model lies
flat in the budget plane alpha+beta+gamma=1 (the simplex triangle)."""
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.stats import gaussian_kde
import style

z = np.load(os.path.join(style.HERE, "abc_points.npz"))
RNG = np.random.default_rng(1)
LIM = (-0.5, 2.0)

def prep(pts, n=14000):
    m = np.all((pts > LIM[0]) & (pts < LIM[1]), axis=1)
    p = pts[m]
    idx = RNG.choice(len(p), min(n, len(p)), replace=False)
    p = p[idx]
    kde = gaussian_kde(p[RNG.choice(len(p), min(6000, len(p)), replace=False)].T,
                       bw_method=0.25)
    dens = kde(p.T)
    order = np.argsort(dens)          # draw dense points last
    return p[order], dens[order]

def panel(ax, pts, dens, title, med):
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=dens, cmap="Blues", s=2.2,
               alpha=0.5, linewidths=0, rasterized=True)
    # budget plane (simplex triangle)
    tri = np.array([[1,0,0],[0,1,0],[0,0,1]])
    ax.add_collection3d(Poly3DCollection([tri], color="#c23b22", alpha=0.18,
                                         edgecolor="#c23b22", lw=1.2))
    for v in tri:
        ax.scatter(*v, color="#c23b22", s=28, edgecolor="white", lw=0.6, zorder=5)
    ax.scatter(1,1,1, marker="*", s=210, color="#e58606", edgecolor="white",
               lw=0.8, zorder=6)
    ax.text(1.02,1.02,1.14, "perfect\ncomposition", color="#b26a04", fontsize=7.5)
    ax.text(1.25,-0.25,0.0, "budget plane\n$\\alpha{+}\\beta{+}\\gamma{=}1$",
            color="#c23b22", fontsize=7.5)
    ax.scatter(*med, marker="P", s=90, color="white", edgecolor="black",
               lw=1.0, zorder=7)
    ax.set_xlim(LIM); ax.set_ylim(LIM); ax.set_zlim(LIM)
    ax.set_xlabel(r"$\alpha$", labelpad=-4); ax.set_ylabel(r"$\beta$", labelpad=-4)
    ax.set_zlabel(r"$\gamma$", labelpad=-4)
    ax.set_title(title, fontsize=9, pad=0)
    ax.view_init(elev=16, azim=42)
    ax.tick_params(pad=-2, labelsize=6.5)
    try: ax.set_box_aspect((1,1,1))
    except Exception: pass

fig = plt.figure(figsize=(7.2, 3.7))
for i, (key, title) in enumerate((("hp","real subgroups (steering space)"),
                                   ("mp","LLM pair bias (gpt-4o-mini), 3 features"))):
    pts = z[key]
    med = np.median(pts, axis=0)
    p, d = prep(pts)
    ax = fig.add_subplot(1, 2, i+1, projection="3d")
    panel(ax, p, d, f"{title}\nmedian ({med[0]:.2f}, {med[1]:.2f}, {med[2]:.2f}), total {med.sum():.2f}", med)
fig.subplots_adjust(left=0.0, right=0.98, bottom=0.02, top=0.88, wspace=0.02)
style.save(fig, "fig_abc_3d")
