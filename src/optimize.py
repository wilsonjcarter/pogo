#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import re
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
from contextlib import redirect_stdout, redirect_stderr
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker

import MDAnalysis as mda
from MDAnalysis.analysis import pca
from numba import njit, prange
from scipy import stats


# -----------------------------
# Subprocess helper
# -----------------------------
def run_cmd(cmd: str, *, input_text: Optional[str] = None, cwd: Optional[Path] = None, quiet: bool = False) -> int:
    if not quiet:
        print(f"\n>>> {cmd}")
    kwargs = {
        "shell": True,
        "input": (input_text.encode() if input_text else None),
        "cwd": str(cwd) if cwd else None,
        "check": False,
    }
    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    res = subprocess.run(cmd, **kwargs)
    return res.returncode


# -----------------------------
# Math kernels
# -----------------------------
@njit(fastmath=True)
def LJ(sigma: float, epsilon: float, r: float) -> float:
    return 4.0 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)


@njit(parallel=True)
def get_go_pot(go_bonds: np.ndarray, trajectory: np.ndarray) -> np.ndarray:
    # go_bonds[:,2]=sigma(Å),[:,3]=epsilon; trajectory in nm
    kT = 2.49
    nb = go_bonds.shape[0]
    nf = trajectory.shape[0]
    out = np.zeros((nb, nf))
    for frame in prange(nf):
        for i in range(nb):
            i1 = int(go_bonds[i, 0]) - 1
            i2 = int(go_bonds[i, 1]) - 1
            R = np.linalg.norm(trajectory[frame, i1] - trajectory[frame, i2])
            out[i, frame] = LJ(go_bonds[i, 2], go_bonds[i, 3], R) / kT
    return out


@njit(parallel=True)
def get_covar(go_potential_cg: np.ndarray, nres: int, aver: np.ndarray, nframes: int) -> np.ndarray:
    covar = np.zeros((nres, nres))
    for i in range(nres):
        for j in prange(i, nres):
            s = 0.0
            for frame in range(nframes):
                s += (go_potential_cg[i, frame] - aver[i]) * (go_potential_cg[j, frame] - aver[j])
            cov = s / nframes
            covar[i, j] = cov
            covar[j, i] = cov
    return covar


@njit(fastmath=True, cache=True)
def get_score(eigvec_T, B, eigval, prob_diff_ref, lambd, threshold):
    xi = eigvec_T @ lambd
    C = 0.5 * np.dot(eigval * eigval, xi * xi)  # eqn 6 term
    prob_diff = B @ xi - C
    n = min(threshold, prob_diff.shape[0], prob_diff_ref.shape[0])
    score = 0.0
    for i in range(n):
        d = prob_diff_ref[i] - prob_diff[i]
        score += abs(d)
    return (score / max(1, n)) * 100.0


# -----------------------------
# Trajectory / RTP utilities
# -----------------------------
def get_traj(univ: mda.Universe, sel: mda.core.groups.AtomGroup) -> np.ndarray:
    nframes = len(univ.trajectory)
    nsel = sel.n_atoms
    traj = np.zeros((nframes, nsel, 3), dtype=float)
    for i, _ in enumerate(univ.trajectory):
        traj[i] = sel.positions / 10.0  # Å→nm
    return traj


def read_go_bonds(path: Path) -> np.ndarray:
    lines: List[List[str]] = []
    with path.open() as fh:
        for line in fh:
            if line.strip() and not line.startswith("["):
                fields = line.split()
                if len(fields) >= 5:
                    lines.append(fields)
    arr = np.zeros((len(lines), 4), dtype=float)
    for k, f in enumerate(lines):
        arr[k, 0] = int(f[0].split("_")[-1])      # i
        arr[k, 1] = int(f[1].split("_")[-1])      # j
        arr[k, 2] = float(f[3])                   # sigma
        arr[k, 3] = float(f[4])                   # epsilon
    return arr


def write_go_nbparams(path: Path, go_bonds: np.ndarray) -> None:
    with path.open("w") as f:
        f.write("[ nonbond_params ]\n")
        for i, j, sigma, eps in go_bonds:
            #if eps > 0.1:
            f.write(f"molecule_0_{int(i)} molecule_0_{int(j)} 1 {sigma:.8f} {eps:.8f}\n")


# -----------------------------
# Optimization wrapper (FuzzyPSO)
# -----------------------------
def optimize_lambda(search_space: List[Tuple[float, float]],
                    args_pack: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int],
                    log_path: Optional[Path] = None) -> np.ndarray:
    from fstpso import FuzzyPSO

    eigvec, B, eigval, prob_diff_ref, threshold = args_pack
    iter_counter = {"i": 0}

    def objective(lambd_list: List[float]) -> float:
        lambd = np.array(lambd_list, dtype=float)
        eigvec_T = np.ascontiguousarray(eigvec.T)
        B_c      = np.ascontiguousarray(B)
        lambd_c  = np.ascontiguousarray(lambd)
        eigval_c = np.ascontiguousarray(eigval)

        score = get_score(eigvec_T, B_c, eigval_c, prob_diff_ref, lambd_c, threshold)
        iter_counter["i"] += 1
        if log_path is not None and (iter_counter["i"] % 100 == 0):
            with open(str(log_path), "a") as fh:
                fh.write(f"{score:.6f}\n")
        return score

    FP = FuzzyPSO()
    with open(os.devnull, "w") as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
        FP.set_search_space(search_space)
        FP.set_fitness(lambda x: objective(x))

    initial = [[0.0] * len(search_space)]
    bestpos, _ = FP.solve_with_fstpso(max_iter=500, initial_guess_list=initial)
    return np.array(bestpos.X, dtype=float)


# -----------------------------
# Plotting helpers
# -----------------------------
def _style_axis(ax, i, j, boundary, units="Å"):
    ax.set_xlabel(f"PC{i+1} [{units}]")
    ax.set_ylabel(f"PC{j+1} [{units}]")
    ax.set_xlim(-boundary, boundary)
    ax.set_ylim(-boundary, boundary)

    minor = ticker.AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor)
    ax.yaxis.set_minor_locator(minor)

    ax.tick_params(axis='x', which='major', bottom=True, width=1., length=4, direction='out', zorder=1000)
    ax.tick_params(axis='x', which='minor', bottom=True, width=1., length=3, direction='out', zorder=1000)
    ax.tick_params(axis='y', which='major', left=True,  width=1., length=4, direction='out', zorder=1000)
    ax.tick_params(axis='y', which='minor', left=True,  width=1., length=3, direction='out', zorder=1000)

    for side in ['top','right','left','bottom']:
        ax.spines[side].set_linewidth(1.)
        ax.spines[side].set_zorder(100)

    # three nicely spaced major ticks within ~90% of boundary
    ax.set_xticks(np.linspace(-int(boundary*0.9), int(boundary*0.9), 3))
    ax.set_yticks(np.linspace(-int(boundary*0.9), int(boundary*0.9), 3))

    # square data aspect so ticks line up
    ax.set_aspect(1 / ax.get_data_ratio())

def _fe_from_hist2d(x, y, bins, eps=1e-12):
    """Normalize PDF to [0,1] then return -log(pdf) as 'free energy'."""
    H, *_ = np.histogram2d(x, y, bins=bins, density=True)
    H = H / (H.max() + eps)
    return -np.log(H.T + eps)  # transpose so axes align with extent


def _overlay_fe(ax, fe, extent, levels, cmap_fill, cmap_line, alpha_fill=0.4, alpha_line=0.8, lw=2):
    ax.contourf(fe, levels=levels, cmap=cmap_fill, alpha=alpha_fill, extent=extent)
    ax.contour (fe, levels=levels, cmap=cmap_line, alpha=alpha_line, extent=extent, linewidths=lw)


def _save(fig, out_dir: Path, name: str, dpi=600):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_fe(pc_proj_cg: np.ndarray,
            pc_proj_aa_ref: np.ndarray,
            boundary: float,
            dimensions: int,
            out_dir: Path,
            opt_step: int) -> None:

    # choose up to first two adjacent PC pairs, like original
    pairs = [(i, i+1) for i in range(min(2, max(1, dimensions-1)))]

    bins    = np.linspace(-boundary, boundary, 31)
    extent  = [bins[0], bins[-1], bins[0], bins[-1]]
    levels  = [0, 1, 2, 3, 4, 5]

    # precompute FE maps
    AA_fe = []
    CG_fe = []
    for i, j in pairs:
        AA_fe.append(_fe_from_hist2d(pc_proj_aa_ref[:, i], pc_proj_aa_ref[:, j], bins))
        CG_fe.append(_fe_from_hist2d(pc_proj_cg[:, i],     pc_proj_cg[:, j],     bins))

    # figure
    fig, axs = plt.subplots(1, len(pairs), figsize=(5, 2), sharex=True, sharey=True)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    fig.suptitle(f"Iteration {opt_step}")

    for ax, (i, j), fe_AA, fe_CG in zip(axs, pairs, AA_fe, CG_fe):
        _overlay_fe(ax, fe_AA, extent, levels, cmap_fill="Oranges_r", cmap_line="Oranges")
        _overlay_fe(ax, fe_CG, extent, levels, cmap_fill="Blues_r",   cmap_line="Blues")
        _style_axis(ax, i, j, boundary)

    # legend (AA/CG color cue)
    axs.flat[0].legend(
        [Line2D([0], [0], color='tab:orange', lw=4),
         Line2D([0], [0], color='tab:blue',   lw=4)],
        ["AA", "CG"], frameon=False
    )

    _save(fig, out_dir, f"fe_{opt_step}.png")


def plot_shift(pc_proj_cg: np.ndarray,
               pc_proj_aa_ref: np.ndarray,
               prob_diff: np.ndarray,
               boundary: float,
               out_dir: Path,
               opt_step: int) -> None:

    n_cg = pc_proj_cg.shape[0]
    positive_cg = pc_proj_cg[prob_diff[:n_cg] > 0]
    negative_cg = pc_proj_cg[prob_diff[:n_cg] < 0]

    colors = {"AA": "tab:blue", "CG-": "tab:orange", "CG+": "tab:green"}

    # same adjacent pairs convention, up to two panels
    dims = pc_proj_cg.shape[1]
    pairs = [(i, i+1) for i in range(min(2, max(1, dims-1)))]

    fig, axs = plt.subplots(1, len(pairs), figsize=(5, 2), sharex=True, sharey=True)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    fig.suptitle(f"Iteration {opt_step}")

    for ax, (i, j) in zip(axs, pairs):
        ax.scatter(pc_proj_aa_ref[:, i], pc_proj_aa_ref[:, j], s=5, color=colors["AA"],  label="AA",  zorder=10)
        ax.scatter(negative_cg[:, i],     negative_cg[:, j],   s=5, color=colors["CG-"], label="CG –", zorder=15)
        ax.scatter(positive_cg[:, i],     positive_cg[:, j],   s=5, color=colors["CG+"], label="CG +", zorder=20)
        _style_axis(ax, i, j, boundary)

    axs.flat[0].legend(markerscale=2, handletextpad=0.3, columnspacing=0.6, ncols=2, frameon=False)

    _save(fig, out_dir, f"shift_cg_{opt_step}.png")



# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    env = os.environ
    p = argparse.ArgumentParser(
        description=(
            "PC-based FE comparison and force-field update loop.\n"
            "Takes multi-replica CG outputs (sim_*/cg_pbc.xtc), aligns to reference, "
            "computes FE surfaces, optimizes lambdas, and writes updated Go parameters."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    cwd = Path.cwd()
    p.add_argument("--work_dir", default=str(cwd))
    p.add_argument("--top", default=str(cwd / "topology"))

    p.add_argument("--ref-pdb", default="reference/reference.pdb")
    p.add_argument("--ref-traj", default="reference/reference.xtc")
    p.add_argument("--ref-ndx", default="reference/reference.ndx")

    p.add_argument("--dimensions", type=int, default=3)
    p.add_argument("--nreplicas", type=int, default=int(env.get("NNODS", "10")))
    p.add_argument("--optstep", type=int, default=int(env.get("NOPTSTEP", "-1")))
    p.add_argument("--trj-groups", default="10 1")

    p.add_argument("--out-cg-traj", default="cg_traj")
    p.add_argument("--out-fig-fe", default=str(Path("figures") / "FE"))
    p.add_argument("--out-fig-shift", default=str(Path("figures") / "shifts"))
    p.add_argument("--out-opt-graphs", default="opt_graphs")
    p.add_argument("--out-ff-param", default="ff_param")
    return p.parse_args()


# -----------------------------
# Misc
# -----------------------------
def infer_opt_step(ff_dir: Path, override_step: int) -> int:
    if override_step is not None and override_step >= 0:
        return int(override_step)
    steps = []
    for p in ff_dir.glob("go_nbparams_*.itp"):
        try:
            steps.append(int(p.stem.split("_")[-1]))
        except ValueError:
            pass
    return (max(steps) + 1) if steps else 0


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    if sys.version_info < (3, 7):
        print("[error] Python >= 3.7 required")
        sys.exit(1)

    args = parse_args()

    work_dir = Path(args.work_dir).resolve()
    top_dir  = Path(args.top).resolve()

    ff_dir        = (work_dir / args.out_ff_param).resolve()
    cg_traj_dir   = (work_dir / args.out_cg_traj).resolve()
    fig_fe_dir    = (work_dir / args.out_fig_fe).resolve()
    fig_shift_dir = (work_dir / args.out_fig_shift).resolve()
    opt_graph_dir = (work_dir / args.out_opt_graphs).resolve()

    for d in (cg_traj_dir, fig_fe_dir, fig_shift_dir, opt_graph_dir, ff_dir):
        d.mkdir(parents=True, exist_ok=True)

    TRJCAT  = f"gmx trjcat"
    TRJCONV = f"gmx trjconv"

    nreplicas = int(args.nreplicas)
    sim_dirs = [work_dir / f"sim_{i}" for i in range(nreplicas)]

    dimensions = int(args.dimensions)
    ref_pdb   = (work_dir / args.ref_pdb).resolve()
    ref_traj  = (work_dir / args.ref_traj).resolve()
    ref_ndx   = (work_dir / args.ref_ndx).resolve()

    if not ref_pdb.exists():
        print(f"[error] missing reference PDB: {ref_pdb}"); sys.exit(2)
    if not ref_traj.exists():
        print(f"[error] missing AA reference XTC: {ref_traj}"); sys.exit(2)
    if not ref_ndx.exists():
        print(f"[error] missing index file: {ref_ndx}"); sys.exit(2)

    print(f"WORKDIR:     {work_dir}")
    print(f"TOP_DIR:     {top_dir}")
    print(f"FF_DIR:      {ff_dir}")
    print(f"CG_TRAJ_DIR: {cg_traj_dir}")
    print(f"FE_DIR:      {fig_fe_dir}")
    print(f"SHIFT_DIR:   {fig_shift_dir}")
    print(f"OPT_DIR:     {opt_graph_dir}")

    opt_step = infer_opt_step(ff_dir, args.optstep)
    print(f"************* OPTSTEP: {opt_step} *************")

    # Concatenate CG replicas
    print("\nConcatenating trajectories...")
    xtc_list = []
    for i in range(nreplicas):
        p = work_dir / f"sim_{i}" / "cg_pbc.xtc"
        if p.exists():
            xtc_list.append(str(p))
        else:
            print(f"[warn] missing {p}; skipping in concatenation")
    if not xtc_list:
        print("[error] no cg_pbc.xtc files found in sim_*"); sys.exit(2)

    run_cmd(f"{TRJCAT} -f {' '.join(xtc_list)} -o concatenated.xtc -cat yes", quiet=True)

    # Fit to reference and write processed CG trajectory
    cg_out = cg_traj_dir / f"run_{opt_step}.xtc"
    run_cmd(f"{TRJCONV} -s {ref_pdb} -f concatenated.xtc -o {cg_out} -fit rot+trans -n {ref_ndx}",
            input_text=args.trj_groups, quiet=True)
    print("...done")

    # Load universes
    print("\nLoading trajectories...")
    u_cg      = mda.Universe(str(ref_pdb), str(cg_out))
    bb_cg     = u_cg.select_atoms("name BB")
    u_aa_ref  = mda.Universe(str(ref_pdb), str(ref_traj))
    bb_aa_ref = u_aa_ref.select_atoms("name BB")

    # PCA
    pca_AA         = pca.PCA(u_aa_ref, select="name BB", n_components=dimensions).run()
    pc_proj_cg     = pca_AA.transform(bb_cg, n_components=dimensions)
    pc_proj_aa_ref = pca_AA.transform(bb_aa_ref, n_components=dimensions)

    var = np.var(pc_proj_aa_ref[:,0])
    pc_boundary = int(5 * np.sqrt(var))

    # FE plots
    plot_fe(pc_proj_cg, pc_proj_aa_ref, pc_boundary, dimensions, fig_fe_dir, opt_step)
    print("FE plots saved.")

    # KDE densities
    CG = np.vstack([pc_proj_cg[:, i] for i in range(dimensions)])
    AA = np.vstack([pc_proj_aa_ref[:, i] for i in range(dimensions)])

    kernel_CG = stats.gaussian_kde(CG)
    kernel_AA = stats.gaussian_kde(AA)
    p_CGtoCG = kernel_CG(CG)
    p_AAtoCG = kernel_CG(AA)
    p_AAtoAA = kernel_AA(AA)
    p_CGtoAA = kernel_AA(CG)

    probAA = np.hstack((p_CGtoAA, p_AAtoAA))
    probCG = np.hstack((p_CGtoCG, p_AAtoCG))
    prob_diff_ref = np.log((probAA + 1e-12) / (probCG + 1e-12))

    # GO params + search space

    itp_files = sorted(ff_dir.glob("go_nbparams_*.itp"),
                       key=lambda f: int(re.search(r'\d+', f.stem).group()),
                       reverse=True)
    if itp_files:
        go_params_template = itp_files[0]
        print(f"[info] Using most recent ITP: {go_params_template}")
    else:
        go_params_template = (top_dir / "top" / "go_nbparams.itp")
        print(f"[warn] No go_nbparams_*.itp found in {ff_dir}. Will try {go_params_template}.")

    if not go_params_template.exists():
        print(f"[error] cannot find template GO params: {go_params_template}")
        sys.exit(2)

    go_bonds = read_go_bonds(go_params_template)

    search_space: List[Tuple[float, float]] = []
    for _, _, _, eps in go_bonds:
        lower_bound = (33.472 - eps) / eps if eps < 33.472 else 0.0
        search_space.append((-lower_bound, 1.0))

    # Trajectories in nm
    traj_cg = get_traj(u_cg, bb_cg)
    traj_aa = get_traj(u_aa_ref, bb_aa_ref)

    # early exit if severely under-sampled; will cause errors downstream
    D = go_bonds.shape[0]              # number of GO terms
    N = traj_cg.shape[0]               # CG samples
    MIN_RATIO = 7

    if N < MIN_RATIO * D * 0.5:
        print(f"[abort] Not enough sampling: # of frames={N} < 0.5x{MIN_RATIO}×(# of Gō-bonds)={MIN_RATIO*D}")
        sys.exit(80)
    elif N < MIN_RATIO * D:
        print(f"[warn] Might not have enough sampling: # of frames={N} < {MIN_RATIO}×(# of Gō-bonds)={MIN_RATIO*D}")

    go_cg  = get_go_pot(go_bonds, traj_cg)
    go_aa  = get_go_pot(go_bonds, traj_aa)
    go_all = np.hstack((go_cg, go_aa))

    aver_cg = go_cg.mean(axis=1)
    covar   = get_covar(go_cg, go_bonds.shape[0], aver_cg, traj_cg.shape[0])
    eigval, eigvec = np.linalg.eig(covar)

    def _ensure_real64(x, name, tol=1e-10):
        if np.iscomplexobj(x):
            if np.max(np.abs(np.imag(x))) <= tol:
                x = np.real(x)
            else:
                print(f"[abort] {name} has significant imaginary parts; likely under-sampled.")
                sys.exit(81)
        return np.ascontiguousarray(x, dtype=np.float64)

    eigval = _ensure_real64(eigval, "eigval")
    eigvec = _ensure_real64(eigvec, "eigvec")

    # Build B
    B = np.zeros((go_all.shape[1], go_bonds.shape[0]))
    for i in range(B.shape[0]):
        B[i] = eigvec.T @ (go_all[:, i] - aver_cg)

    threshold = traj_cg.shape[0]

    # Fresh log + optimize
    log_path = opt_graph_dir / f"opt_graph_{opt_step}.txt"
    with open(log_path, "w"):
        pass
    print("\nInitializing PSO...")
    lambd = optimize_lambda(search_space, (eigvec, B, eigval, prob_diff_ref, threshold), log_path=log_path)

    # Classify and plot shifts
    xi = eigvec.T @ lambd
    prob_diff = np.array([(B[i] * xi - 0.5 * (eigval ** 2) * (xi ** 2)).sum() for i in range(B.shape[0])])
    plot_shift(pc_proj_cg, pc_proj_aa_ref, prob_diff, pc_boundary, fig_shift_dir, opt_step)
    print("Shift plots saved.")

    # Write updated files
    print("\nWriting updated files...")
    go_bonds[:, 3] = go_bonds[:, 3] - lambd * go_bonds[:, 3]

    bb_ids = bb_aa_ref.atoms.ids.astype(int)
    for d in sim_dirs:
        out_nb = d / "top" / "go_nbparams.itp"
        print(f"writing: {out_nb}")
        try:
            write_go_nbparams(out_nb, go_bonds)
        except:
            print(f"[warn] could not write gobonds to: {out_nb}")

    out_ff = ff_dir / f"go_nbparams_{opt_step}.itp"
    print(f"writing: {out_ff}")
    write_go_nbparams(out_ff, go_bonds)

    try:
        (work_dir / "concatenated.xtc").unlink()
    except FileNotFoundError:
        pass

    print("...done")


if __name__ == "__main__":
    if sys.version_info < (3, 8):
        print("[error] Python >= 3.8 required")
        sys.exit(1)
    main()
