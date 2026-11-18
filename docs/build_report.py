"""Rebuild the report from saved notebook outputs, without retraining models.

Run from any directory: uv run --locked python docs/build_report.py
The notebook outputs are the experiment snapshot; optional run caches are not
needed. All scientific plots are exported as vector SVG, then Typst makes PDF.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import typst

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
ASSETS = DOCS / "assets"
MIME = "application/vnd.plotly.v1+json"
COLORS = {"CE": "#2563eb", "RobustCE(0.05)": "#0891b2",
          "RobustCE(0.1)": "#ea580c", "RobustCE(0.2)": "#7c3aed",
          "Smoothing(0.2)": "#16803b"}
STYLES = {"CE": "-", "RobustCE(0.05)": (0, (6, 2)),
          "RobustCE(0.1)": "--", "RobustCE(0.2)": "-.",
          "Smoothing(0.2)": ":"}
NAMES = {"CE": "CE", "RobustCE(0.05)": r"Robust ($\delta=0.05$)",
         "RobustCE(0.1)": r"Robust ($\delta=0.1$)",
         "RobustCE(0.2)": r"Robust ($\delta=0.2$)",
         "Smoothing(0.2)": r"Smoothing ($\varepsilon=0.2$)"}
RG = LinearSegmentedColormap.from_list("red_white_green", ["#991b1b", "white", "#166534"])
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "cm",
                     "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": .65, "lines.linewidth": 1.5,
                     "svg.fonttype": "path", "savefig.facecolor": "white"})


def notebook(prefix):
    path = next((ROOT / "notebooks").glob(prefix + "*.ipynb"))
    return path, json.loads(path.read_text(encoding="utf-8"))


def plots(nb, cell_id):
    cell = next(c for c in nb["cells"] if c.get("id") == cell_id)
    result = [o["data"][MIME] for o in cell.get("outputs", []) if MIME in o.get("data", {})]
    if not result:
        raise ValueError(f"Missing saved Plotly output in cell {cell_id}; execute the notebook first.")
    return result


def all_plots(nb):
    return [o["data"][MIME] for c in nb["cells"] for o in c.get("outputs", [])
            if MIME in o.get("data", {})]


def save(fig, name):
    fig.savefig(ASSETS / f"{name}.svg", bbox_inches="tight", pad_inches=.04)
    plt.close(fig)


def grid(ax):
    ax.grid(alpha=.18, linewidth=.6)
    ax.set_axisbelow(True)


def legend(fig, handles=None, labels=None, columns=3):
    if handles is None:
        handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.015),
               ncol=columns, frameon=False, handlelength=2.4, columnspacing=1.3)


def draw_trace(ax, t, error=True):
    kw = dict(color=COLORS[t["name"]], linestyle=STYLES[t["name"]],
              label=NAMES[t["name"]])
    if error and "error_y" in t:
        ax.errorbar(t["x"], t["y"], yerr=t["error_y"]["array"],
                    marker="o", markersize=3.5, capsize=2, elinewidth=.8, **kw)
    else:
        ax.plot(t["x"], t["y"], **kw)


def theory_figures():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), layout="constrained")
    for ax, m, delta in zip(axes, [.45, 0], [.3, .2]):
        center = (1-m, m)
        ax.add_patch(Circle(center, delta * np.sqrt(2), facecolor="#dbeafe",
                            edgecolor="#93b4d7", linewidth=1))
        ax.plot([0, 1], [1, 0], color=".25", lw=1.2)
        q0, q1 = max(0, m-delta), min(1, m+delta)
        ax.plot([1-q0, 1-q1], [q0, q1], lw=4, color="#7c3aed", solid_capstyle="round")
        ax.scatter(*center, s=25, color="#ea580c", zorder=4)
        ax.text(center[0]+.035, center[1]+.055, r"$\mu$", fontsize=12)
        ax.scatter([0, 1], [1, 0], s=15, color=".2", zorder=4)
        ax.set(xlim=(-.12, 1.32), ylim=(-.3, 1.1), xlabel=r"$c_0$", ylabel=r"$c_1$",
               title=fr"$m={m:g},\ \delta={delta:g}$; $q\in[{q0:g},{q1:g}]$")
        ax.set_aspect("equal"); ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    save(fig, "simplex_interval")

    # Actual direct numerical maximization, independently of the analytic rule.
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    records = []
    for y in [0, 1]:
        for z in [-8., -2., 0., 2., 8.]:
            for delta in [0., .05, .2, .5, 1., 1.5]:
                q = torch.linspace(max(0, y-delta), min(1, y+delta), 10001, dtype=torch.float64)
                numeric = F.binary_cross_entropy_with_logits(torch.full_like(q, z), q, reduction="none").max().item()
                analytic = np.logaddexp(0, z) - y*z + min(1, delta)*max(0, (2*y-1)*z)
                records.append(dict(y=y, z=z, delta=delta, analytic=analytic, numeric=numeric,
                                    error=abs(analytic-numeric)))
    assert max(r["error"] for r in records) < 1e-10
    for r in records:
        nominal = np.logaddexp(0, r["z"]) - r["y"] * r["z"]
        assert r["analytic"] >= nominal - 1e-12
        if r["delta"] == 0:
            assert abs(r["analytic"] - nominal) < 1e-12
    for y in [0, 1]:
        for z in [-8., -2., 0., 2., 8.]:
            values = [r["analytic"] for r in records if r["y"] == y and r["z"] == z]
            assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))
    for p, expected in [(.1, .5448054311250702), (.4, .5919186453876236), (.6, .916290731874155)]:
        z = np.log(p / (1-p))
        assert abs(np.logaddexp(0, z) + .2 * max(0, -z) - expected) < 1e-10
    fig, axes = plt.subplots(2, 3, figsize=(7, 3.4), sharex=True, sharey=True)
    for row, y in enumerate([0, 1]):
        for col, z in enumerate([-2., 0., 2.]):
            ax = axes[row, col]
            q0, q1 = max(0, y-.2), min(1, y+.2)
            q = np.linspace(0, 1, 101); ce = np.logaddexp(0, z)-q*z
            qgrid = torch.linspace(q0, q1, 10001, dtype=torch.float64)
            values = F.binary_cross_entropy_with_logits(torch.full_like(qgrid, z), qgrid, reduction="none")
            k = int(values.argmax())
            analytic = np.logaddexp(0, z)-y*z+.2*max(0, (2*y-1)*z)
            ax.axvspan(q0, q1, color="#dbeafe", zorder=0)
            ax.plot(q, ce, color=".75", label="CE on the simplex")
            ax.plot([q0, q1], [np.logaddexp(0,z)-q0*z, np.logaddexp(0,z)-q1*z],
                    color="#2563eb", lw=2.5, label="Feasible interval")
            ax.axhline(analytic, color="#16803b", ls="--", lw=1, label="Analytic maximum")
            ax.scatter(qgrid[k], values[k], color="#ea580c", s=24, zorder=4, label="Numerical maximum")
            ax.set(title=fr"$y={y},\ z={z:g}$", ylim=(0, 2.4), xlim=(-.04, 1.04))
            if z == 0: ax.text(.5, 1.2, "Every feasible target\nmaximizes the loss", ha="center", fontsize=8)
            if row == 1: ax.set_xlabel(r"Target $q$")
            if col == 0: ax.set_ylabel(r"$L_{\rm CE}(z,q)$")
    fig.tight_layout(rect=(0, .15, 1, 1)); legend(fig, columns=4)
    save(fig, "numerical_maximization")

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.65))
    m = np.linspace(-4, 6, 800); ce = np.logaddexp(0, -m)
    for method, delta in [("CE", 0), ("RobustCE(0.1)", .1), ("Smoothing(0.2)", .1)]:
        loss = ce if method == "CE" else ce+delta*(m if method.startswith("Smoothing") else np.maximum(0, m))
        axes[0].plot(m, loss, color=COLORS[method], ls=STYLES[method], label=NAMES[method])
    axes[0].axvline(0, color=".6", lw=.7); axes[0].set(xlabel=r"Signed margin $s=(2y-1)z$", ylabel="Loss", ylim=(0, 4.5))
    axes[1].plot(m, .1*np.maximum(0, -m), color="#7c3aed")
    axes[1].axvspan(-4, 0, color="#f3e8ff", zorder=0)
    axes[1].set(xlabel=r"Signed margin $s$", ylabel=r"$L_{\rm Robust}-L_{\rm Smooth}$",
                title=r"$\varepsilon=2\delta=0.2$", ylim=(-.01, .43))
    axes[1].text(1.7, .17, "Identical on\npositive margins", ha="center", fontsize=10)
    for ax in axes: grid(ax)
    fig.tight_layout(rect=(0, .14, 1, 1)); legend(fig, columns=3)
    save(fig, "margin_losses")
    return records


def main_figures(nb):
    tuning = plots(nb, "tuning-plot")[0]
    metrics = plots(nb, "heldout-plot")[0]
    learning = plots(nb, "learning-plot")[0]
    confidence = plots(nb, "confidence-comparison")[0]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))
    for i, ax in enumerate(axes):
        for t in tuning["data"][i*5:(i+1)*5]:
            draw_trace(ax, t)
            selected = t["marker"]["symbol"].index("star")
            ax.scatter(t["x"][selected], t["y"][selected], marker="*", s=70,
                       color=COLORS[t["name"]], edgecolor="white", lw=.5, zorder=4)
        ax.set(xscale="log", xlabel="Learning rate", ylabel="Clean validation CE",
               title=["Gaussian / logistic", "Moons / tanh network"][i])
        ax.set_xticks([.001, .01, .05], ["0.001", "0.01", "0.05"]); grid(ax)
    fig.tight_layout(rect=(0,.18,1,1)); legend(fig); save(fig,"lr_selection")

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.05))
    for d, ax in enumerate(axes):
        points = []
        for t in tuning["data"][d*5:d*5+4]:
            k = t["marker"]["symbol"].index("star")
            points.append((t["y"][k], t["error_y"]["array"][k]))
        ax.errorbar([0, .05, .1, .2], [p[0] for p in points],
                    yerr=[p[1] for p in points], color="#ea580c", marker="o", ms=3.5, capsize=2)
        ax.set(xlabel=r"Robust training radius $\delta$ (0 = CE)", ylabel="Clean validation CE",
               title=["Gaussian", "Moons"][d], xticks=[0, .05, .1, .2])
        grid(ax)
    fig.tight_layout(); save(fig, "radius_validation")

    # Use the saved exact means and sample standard deviations, not pixel reads.
    fig, axes = plt.subplots(2, 2, figsize=(7, 3.8))
    for d in range(2):
        for j, metric in enumerate([0, 2]):
            ax = axes[d,j]
            for t in metrics["data"][d*15+metric*5:d*15+metric*5+5]: draw_trace(ax,t)
            ax.set_title(["Gaussian", "Moons"][d]+": "+["accuracy (%)", r"robust CE ($\delta_{eval}=0.1$)"][j])
            ax.set_xticks([0,.1], ["0", "0.1"]); ax.set_xlim(-.008,.108); grid(ax)
            if d == 1: ax.set_xlabel(r"Training-label flip probability $\eta$")
    fig.tight_layout(rect=(0,.14,1,1)); legend(fig); save(fig,"auxiliary_metrics")
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.75))
    for d, ax in enumerate(axes):
        for t in metrics["data"][d*15+5:d*15+10]: draw_trace(ax,t)
        ax.set(title=["Gaussian / logistic", "Moons / tanh network"][d], ylabel="Clean test CE (nats)",
               xlabel=r"Training-label flip probability $\eta$")
        ax.set_xticks([0,.1]); ax.set_xlim(-.008,.108); grid(ax)
    fig.tight_layout(rect=(0,.18,1,1)); legend(fig); save(fig,"clean_ce")
    fig, axes = plt.subplots(2, 2, figsize=(7, 3.6))
    for i, ax in enumerate(axes.flat):
        for t in learning["data"][i*5:i*5+5]: draw_trace(ax,t,error=False)
        ax.set(title=[r"Gaussian, $\eta=0$",r"Gaussian, $\eta=0.1$",
                      r"Moons, $\eta=0$",r"Moons, $\eta=0.1$"][i], ylabel="Clean validation CE")
        ax.set_ylim(.13 if i<2 else .22,.65); grid(ax)
        if i>=2: ax.set_xlabel("Epoch")
    fig.tight_layout(rect=(0,.14,1,1)); legend(fig); save(fig,"learning_curves")
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.35))
    for d, ax in enumerate(axes):
        for t in confidence["data"][d*5:d*5+5]: draw_trace(ax,t)
        ax.set(title=["Gaussian", "Moons"][d], ylabel="Mean confidence (%)", xlabel=r"Training noise $\eta$")
        ax.set_xticks([0,.1]); ax.set_ylim(60,96); grid(ax)
    fig.tight_layout(rect=(0,.2,1,1)); legend(fig); save(fig,"confidence")

    for noise, source in enumerate(plots(nb,"boundary-plot")):
        fig, axes = plt.subplots(2, 3, figsize=(7,4.0))
        for d in range(2):
            for col, method_index in enumerate([0,2,4]):
                ax=axes[d,col]; start=(d*5+method_index)*4
                t=source["data"][start]; z=np.asarray(t["z"])
                im=ax.pcolormesh(t["x"],t["y"],z,cmap="RdBu_r",vmin=0,vmax=1,shading="auto",rasterized=True)
                ax.contour(t["x"],t["y"],z,levels=[.5],colors="black",linewidths=.9)
                for k,marker in [(2,"o"),(3,"x")]:
                    points=source["data"][start+k]
                    ax.scatter(points["x"],points["y"],s=6,c=["#dbeafe","#fee2e2"][k-2],
                               marker=marker,linewidths=.45,edgecolors=".2" if marker=="o" else None,alpha=.8)
                ax.set_title(["CE",r"Robust $\delta=0.1$",r"Smoothing $\varepsilon=0.2$"][col])
                ax.set_aspect("equal"); ax.set_xticks([-2,0,2]); ax.set_yticks([-2,0,2])
                if col == 0: ax.set_ylabel(["Gaussian\n", "Moons\n"][d]+r"Standardized $x_2$")
                if d==1: ax.set_xlabel(r"Standardized $x_1$")
        fig.tight_layout(rect=(0,0,.92,1))
        cax=fig.add_axes([.94,.21,.015,.61]); fig.colorbar(im,cax=cax,label=r"$p_\theta(Y=1\mid x)$")
        save(fig,"boundaries_"+["clean","noisy"][noise])

    rows=[]
    for d in range(2):
        for method_index, method in enumerate(COLORS):
            for i,eta in enumerate([0,.1]):
                row=dict(dataset=["gaussian","moons"][d], method=method, eta=eta)
                for key,idx in [("accuracy_percent",0),("clean_test_ce",1),("robust_test_ce",2)]:
                    t=metrics["data"][d*15+idx*5+method_index]
                    row[key]=t["y"][i]; row[key+"_sd"]=t["error_y"]["array"][i]
                t=confidence["data"][d*5+method_index]
                row["confidence_percent"]=t["y"][i]; row["confidence_percent_sd"]=t["error_y"]["array"][i]
                rows.append(row)
    return rows


def heatmap(ax, t, title, xlabel, limit=None, annotate=True):
    z=np.array(t["z"],dtype=float)
    limit=limit or max(np.abs(z).max(),1e-6)
    image=ax.imshow(z,cmap=RG,norm=Normalize(-limit,limit),aspect="auto")
    ax.set(xticks=range(len(t["x"])),xticklabels=t["x"],yticks=range(len(t["y"])),
           yticklabels=t["y"],ylabel=r"Actual training noise $\eta$",xlabel=xlabel,title=title)
    ax.tick_params(axis="x",rotation=40,labelsize=8)
    ax.set_xticks(np.arange(-.5,z.shape[1],1),minor=True);ax.set_yticks(np.arange(-.5,z.shape[0],1),minor=True)
    ax.grid(which="minor",color="#dddddd",linewidth=.6);ax.tick_params(which="minor",bottom=False,left=False)
    for s in ax.spines.values(): s.set_visible(False)
    if annotate:
        for (i,j),value in np.ndenumerate(z):
            if abs(value)<1e-6: label="0" if value==0 else f"{value:.0e}"
            elif abs(value)<.001: label=f"{value:.0e}"
            else: label=f"{value:+.3f}"
            color="white" if abs(value)/limit>.65 else "#222222"
            ax.text(j,i,label,ha="center",va="center",fontsize=7.2,color=color)
    return image


def sweep_figures(sources):
    robust,smooth,matched,noise=sources
    common=max(abs(np.asarray(t["z"],dtype=float)).max() for t in [robust,smooth])
    for name,t,title,xlabel in [
        ("robust_grid",robust,"CE baseline − RobustCE",r"Training radius $\delta$"),
        ("smoothing_grid",smooth,"CE baseline − Smoothing",r"Smoothing strength $\varepsilon$"),
        ("matched_grid",matched,"Smoothing − RobustCE",r"Robust radius $\delta$; smoothing uses $\varepsilon=2\delta$"),
        ("noise_aware_grid",noise,"CE baseline − NoiseAware",r"Assumed flip probability $\alpha=\eta_{expected}$")]:
        fig,ax=plt.subplots(figsize=(7,2.85))
        im=heatmap(ax,t,title,xlabel,limit=common if name in ["robust_grid","smoothing_grid"] else None)
        if name=="noise_aware_grid":
            for i,row in enumerate(t["customdata"]):
                for j,c in enumerate(row):
                    if c[12]: ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,edgecolor="#172554",lw=1.5))
        fig.colorbar(im,ax=ax,pad=.025,fraction=.03,label="Difference in clean test CE (nats)")
        fig.tight_layout();save(fig,name)
    diagonal=[c for row in noise["customdata"] for c in row if c[12]]
    fig,axes=plt.subplots(1,2,figsize=(7,2.5))
    etas=[c[0] for c in diagonal]
    axes[0].plot(etas,[c[2] for c in diagonal],"o-",color=COLORS["CE"],label="CE baseline",ms=4)
    axes[0].plot(etas,[c[3] for c in diagonal],"o-",color="#16803b",label=r"NoiseAware ($\alpha=\eta$)",ms=4)
    axes[0].set(ylabel="Mean clean test CE (nats)",xlabel=r"Actual training noise $\eta$")
    axes[0].legend(frameon=False)
    axes[1].errorbar(etas,[c[4] for c in diagonal],yerr=[c[6] for c in diagonal],color="#16803b",marker="o",ms=4,capsize=2)
    axes[1].axhline(0,color=".5",lw=.7)
    axes[1].set(ylabel="Paired CE − NoiseAware (nats)",xlabel=r"Actual training noise $\eta$")
    for ax in axes: grid(ax)
    fig.tight_layout();save(fig,"noise_aware_diagonal")
    return diagonal


def noise_diagram():
    fig,ax=plt.subplots(figsize=(7,1.5));ax.set(xlim=(0,10),ylim=(0,2));ax.axis("off")
    labels=[r"Features $x$", r"Clean $Y$"+"\n"+r"$p_\theta=\sigma(z)$",
            r"Flip channel"+"\n"+r"$T_\alpha$",r"Observed $\widetilde{Y}$"+"\n"+r"$q_\theta=\alpha+(1-2\alpha)p_\theta$"]
    positions=[(.1,1.35),(2.1,1.65),(4.55,1.55),(6.85,3.0)]
    for (x,w),label in zip(positions,labels):
        ax.add_patch(FancyBboxPatch((x,.55),w,1,boxstyle="round,pad=0.1",facecolor="#f1f5f9",edgecolor="#64748b",lw=.8))
        ax.text(x+w/2,1.05,label,ha="center",va="center",fontsize=10)
    for a,b in [(1.55,1.98),(3.87,4.43),(6.23,6.73)]:
        ax.add_patch(FancyArrowPatch((a,1.05),(b,1.05),arrowstyle="->",mutation_scale=11,color="#334155"))
    ax.text(3, .1,"Clean test: score this probability",ha="center",fontsize=8,color="#166534")
    ax.text(8.3,.1,"Training: likelihood of this observation",ha="center",fontsize=8,color="#92400e")
    save(fig,"noise_channel")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-only",action="store_true")
    args=parser.parse_args()
    ASSETS.mkdir(exist_ok=True)
    notebooks=[notebook(f"{i:02d}") for i in range(1,7)]
    checks=theory_figures()
    main_rows=main_figures(notebooks[1][1])
    sources=[all_plots(nb)[0]["data"][0] for _,nb in notebooks[2:]]
    diagonal=sweep_figures(sources)
    noise_diagram()
    # Check sign conventions and matching CE baselines across the three sweeps.
    for r,s,n in zip(sources[0]["customdata"],sources[1]["customdata"],sources[3]["customdata"]):
        assert abs(r[0][2]-s[0][2])<1e-12 and abs(r[0][2]-n[0][2])<1e-12
        assert all(abs(c[2]-c[3]-c[4])<1e-7 for c in r+s+n)
    for row in sources[2]["customdata"]:
        assert all(abs(c[3]-c[4]-c[5])<1e-7 for c in row)
    snapshot={"source": "Saved numeric Plotly outputs embedded in the notebooks; no retraining",
              "notebook_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p,_ in notebooks},
              "main_results":main_rows,"numerical_loss_check":checks,
              "sweeps":{name:{"x":t["x"],"y":t["y"],"customdata":t["customdata"],"plotted_difference":t["z"]}
                        for name,t in zip(["robust","smoothing","smoothing_minus_robust","ce_minus_noise_aware"],sources)},
              "noise_aware_diagonal":diagonal}
    (ASSETS/"report_data.json").write_text(json.dumps(snapshot,indent=2,ensure_ascii=True)+"\n",encoding="utf-8")
    print(f"Created {len(list(ASSETS.glob('*.svg')))} vector figures. Loss check: {len(checks)} cases, max error {max(r['error'] for r in checks):.3e}.")
    if not args.figures_only:
        typst.compile(str(DOCS/"report.typ"),output=str(DOCS/"report.pdf"),root=str(ROOT))
        import pymupdf
        with pymupdf.open(DOCS/"report.pdf") as pdf:
            print(f"Compiled {DOCS/'report.pdf'} ({len(pdf)} pages).")


if __name__ == "__main__":
    main()
