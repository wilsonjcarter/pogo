#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional


# -----------------------------
# helpers
# -----------------------------
def run_cmd(cmd: str, *, input_text: Optional[str] = None, cwd: Optional[Path] = None) -> int:
    """Run a shell command, stream stdout/stderr, return exit code."""
    print(f"\n>>> {cmd}")
    try:
        res = subprocess.run(
            cmd,
            shell=True,
            input=(input_text.encode() if input_text is not None else None),
            cwd=str(cwd) if cwd else None,
            check=False,
        )
        return res.returncode
    except FileNotFoundError as e:
        print(f"[run_cmd] executable not found: {e}")
        return 127


def rm_files(paths: Iterable[Path]) -> None:
    for p in paths:
        try:
            p.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception as e:
            print(f"[warn] could not delete {p}: {e}")


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy(src, dst)
    else:
        print(f"[warn] missing reference file: {src}")


# -----------------------------
# preparation
# -----------------------------
def prepare_dirs(sim_dirs: List[Path], init_dir: Path, top_dir: Path) -> None:
    """Create/clean sim directories and stage required input files."""
    extensions = ("log", "cpt", "tpr", "edr", "xtc", "trr", "gro")

    frames = list(init_dir.glob("frame*.gro"))
    if not frames:
        print(f"[error] no matching seed frames found in {init_dir}")
        sys.exit(2)

    for d in sim_dirs:
        d.mkdir(parents=True, exist_ok=True)

        # clean old outputs
        rm_files(Path(p) for ext in extensions for p in glob.glob(str(d / f"*.{ext}")))

        # random AA frame -> start.gro
        shutil.copy(random.choice(frames), d / "start.gro")

        # copy required topology files
        for fname in ("topol.top", "molecule_0.itp", "posre_BB.itp"):
            copy_if_exists(top_dir / fname, d / fname)

        # copy top/ tree if absent
        top_src, top_dst = top_dir / "top", d / "top"
        if not top_dst.exists():
            if top_src.exists():
                shutil.copytree(top_src, top_dst)
            else:
                print(f"[warn] missing reference directory: {top_src}")


# -----------------------------
# gromacs steps
# -----------------------------
def grompp(
    sim_dirs: List[Path],
    grompp_cmd: str,
    mdp_dir: Path,
    stage: str,
    input_gro: str,
    output_tpr: str,
    mdp_file: str,
    ref_gro: Optional[str] = None,
) -> None:
    """Run grompp in each directory."""
    for d in sim_dirs:
        ref_opt = f"-r {d / ref_gro}" if ref_gro else ""
        cmd = (
            f"{grompp_cmd} -c {d / input_gro} -f {mdp_dir / mdp_file} "
            f"-p {d / 'topol.top'} -o {d / output_tpr} {ref_opt} -maxwarn 3"
        )
        code = run_cmd(cmd)
        if code != 0:
            print(f"[error] grompp failed in {d} for stage {stage} (code {code})")
            sys.exit(code)


def mdrun_multidir(
    sim_dirs: List[Path],
    launcher: str,
    mdrun_cmd: str,
    nreplicas: int,
    mdargs: str,
    stage: str,
) -> None:
    """
    Run mdrun. If MPI build (we detect by 'mpi' in mdrun_cmd) *and* nreplicas > 1,
    use -multidir. Otherwise run one directory at a time.
    """
    is_mpi = "mpi" in mdrun_cmd
    if is_mpi and nreplicas > 1:
        dirs_str = " ".join(str(d) for d in sim_dirs)
        if launcher.strip():
            cmd = f"{launcher} -np {nreplicas} {mdrun_cmd} -multidir {dirs_str} -deffnm {stage} {mdargs}".strip()
        else:
            cmd = f"{mdrun_cmd} -multidir {dirs_str} -deffnm {stage} {mdargs}".strip()
        code = run_cmd(cmd)
        if code != 0:
            print(f"[error] mdrun failed for stage {stage} (code {code})")
            sys.exit(code)
    else:
        for d in sim_dirs:
            mdrun_single(d, launcher, mdrun_cmd, mdargs, stage)


def mdrun_single(sim_dir: Path, launcher: str, mdrun_cmd: str, mdargs: str, stage: str) -> None:
    """Run a single GROMACS mdrun job in one directory."""
    if launcher.strip():
        cmd = f"{launcher} {mdrun_cmd} -deffnm {sim_dir / stage} {mdargs}".strip()
    else:
        cmd = f"{mdrun_cmd} -deffnm {sim_dir / stage} {mdargs}".strip()

    print(f"[run] {cmd}")
    code = run_cmd(cmd)
    if code != 0:
        print(f"[error] mdrun failed for stage {stage} in {sim_dir} (code {code})")
        sys.exit(code)


def trj_post(
    sim_dirs: List[Path],
    trj_groups: str,
    stage_tpr: str,
    stage_xtc: str,
    out_xtc: str,
    ndx: Path,
    begin_ps: int,
) -> None:
    """Center + make whole using trjconv. Feeds index selections via stdin."""
    for d in sim_dirs:
        cmd = (
            "gmx trjconv "
            f"-f {d / stage_xtc} -o {d / out_xtc} -pbc mol -center -b {begin_ps} "
            f"-n {ndx} -s {d / stage_tpr}"
        )
        code = run_cmd(cmd, input_text=trj_groups)
        if code != 0:
            print(f"[warn] trjconv failed for {d} (code {code})")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    env = os.environ
    p = argparse.ArgumentParser(
        description=(
            "Run a multi-replica GROMACS pipeline (min -> eq0..eq3 -> md) with inputs from 'reference/'.\n"
            "Seed structures are drawn from 'initial/frame<N>.gro'."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Paths
    cwd = Path.cwd()
    p.add_argument("--workdir", default=str(cwd), help="Working directory (root for runs).")
    p.add_argument("--init", default=str(cwd / "initial"), help="Directory containing AA frames 'frame<N>.gro'.")
    p.add_argument("--mdp", default=str(cwd / "mdp"), help="Directory containing mdp files.")
    p.add_argument("--top", default=str(cwd / "topology"), help="Directory containing topology files and top/ tree.")
    p.add_argument("--index", default=str(cwd / "reference/reference.ndx"), help="Directory with reference files (reference.*).")
    p.add_argument("--nreplicas", type=int, default=int(env.get("NREPLICAS", "10")),
                   help="Number of replicas (sim_0..sim_{N-1}).")

    # Trajectory post-processing
    p.add_argument("--trj-begin-ps", type=int, default=50000, help="Start time (ps) for trjconv -b.")
    p.add_argument("--seed", type=int, default=None, help="Random seed for selecting AA frames.")
    return p.parse_args()


# -----------------------------
# main pipeline
# -----------------------------
def main() -> None:
    if sys.version_info < (3, 8):
        print("[error] Python >= 3.8 required")
        sys.exit(1)

    args = parse_args()

    # Initialize RNG if requested
    if args.seed is not None:
        random.seed(args.seed)

    work_dir = Path(args.workdir).resolve()
    top_dir = Path(args.top).resolve()
    init_dir = Path(args.init).resolve()
    mdp_dir = Path(args.mdp).resolve()
    index = Path(args.index).resolve()

    # Tool selection
    if args.nreplicas > 1:
        mdrun_default = "gmx_mpi mdrun"
        launcher_default = "mpirun"
    else:
        mdrun_default = "gmx mdrun"
        launcher_default = ""  # run directly

    MDRUN = os.getenv("MDRUN", mdrun_default)
    GROMPP = os.getenv("GROMPP", "gmx grompp")
    MDARGS = os.getenv("MDARGS", "")  # e.g., "-ntomp 1"
    LAUNCHER = os.getenv("LAUNCHER", launcher_default)

    # Build sim directories
    sim_dirs = [work_dir / f"sim_{i}" for i in range(args.nreplicas)]

    print(f"WORKDIR:   {work_dir}")
    print(f"TOP_DIR:   {top_dir}")
    print(f"INIT_DIR:  {init_dir}")
    print(f"MDP_DIR :  {mdp_dir}")
    print(f"NREPLICAS: {args.nreplicas}")
    print(f"GROMPP:    {GROMPP}")
    print(f"MDRUN:     {MDRUN}")
    print(f"LAUNCHER:  {LAUNCHER}")
    print(f"MDARGS:    {MDARGS}")
    print(f"REF_INDEX: {index}")
    print(f"TRJ_BEGIN: {args.trj_begin_ps} ps")

    # Prep
    prepare_dirs(sim_dirs, init_dir, top_dir)

    # Minimization
    grompp(sim_dirs, GROMPP, mdp_dir, stage="min", input_gro="start.gro", output_tpr="min.tpr", mdp_file="min.mdp", ref_gro="start.gro")
    mdrun_multidir(sim_dirs, LAUNCHER, MDRUN, args.nreplicas, MDARGS, "min")

    # Equilibration
    grompp(sim_dirs, GROMPP, mdp_dir, "eq0", "min.gro", "eq0.tpr", "eq0.mdp", "min.gro")
    mdrun_multidir(sim_dirs, LAUNCHER, MDRUN, args.nreplicas, MDARGS, "eq0")

    grompp(sim_dirs, GROMPP, mdp_dir, "eq1", "eq0.gro", "eq1.tpr", "eq1.mdp", "eq0.gro")
    mdrun_multidir(sim_dirs, LAUNCHER, MDRUN, args.nreplicas, MDARGS, "eq1")

    grompp(sim_dirs, GROMPP, mdp_dir, "eq2", "eq1.gro", "eq2.tpr", "eq2.mdp", "eq1.gro")
    mdrun_multidir(sim_dirs, LAUNCHER, MDRUN, args.nreplicas, MDARGS, "eq2")

    grompp(sim_dirs, GROMPP, mdp_dir, "eq3", "eq2.gro", "eq3.tpr", "eq3.mdp", "eq2.gro")
    mdrun_multidir(sim_dirs, LAUNCHER, MDRUN, args.nreplicas, MDARGS, "eq3")

    # Production
    grompp(sim_dirs, GROMPP, mdp_dir, "md", "eq3.gro", "md.tpr", "md.mdp")
    mdrun_multidir(sim_dirs, LAUNCHER, MDRUN, args.nreplicas, MDARGS, "md")

    # Post-processing
    trj_post(
        sim_dirs,
        trj_groups="1 1",
        stage_tpr="md.tpr",
        stage_xtc="md.xtc",
        out_xtc="cg_pbc.xtc",
        ndx=index,
        begin_ps=args.trj_begin_ps,
    )


if __name__ == "__main__":
    main()
