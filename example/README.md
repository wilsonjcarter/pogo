# PoGō Pipeline

Optimization alternates between two main stages:

1. **`generate.py`** — seeds, builds, and runs a GROMACS simulation pipeline  
```scss
python generate.py
--workdir        Root directory for runs (default: .)
--init           Input AA frames (default: ./initial)
--mdp            MDP parameter files (default: ./mdp)
--top            Topology directory (default: ./topology)
--mdrun          MD run command (default: gmx_mpi mdrun)
--mdargs         Extra args to mdrun (default: "")
--launcher       MPI launcher (default: mpirun)
--nreplicas      Number of replicas (default: 10)
--trj-begin-ps   Start time for trajectory (default: 10000 ps)
```

2. **`optimize.py`** — analyzes generated ensemble and finds an optimal perturbation to the Gō-potential to match the target reference.  

```scss
python optimize.py
--workdir         Root directory (default: .)
--ref-pdb         Reference PDB (default: ./reference/reference.pdb)
--ref-traj        Reference trajectory (default: ./reference/reference.xtc)
--ref-ndx         Index file (default: ./reference/reference.ndx)
--dimensions      PCA dimensionality (default: 3)
--nreplicas       Number of replicas (default: 10)
--optstep         Optimization step (default: -1)
--trj-groups      trjconv groups (default: "10 1")
```

---
## Step-by-step
Download the `/example` directory. Inside, you will find:

- `/initial`: contains a set of initial configurations (this system corresponds to the `topol.top` file)
- `/topology`: contains a `topol.top` file and `martinize2`-generated `.itp` files
- `/mdp`: contains `.mdp` files in a standard equilibration hierarchy
- `reference.[pdb/ndx/xtc]`: a forward-mapped atomistic trajectory, structure file, and GROMACS-style index file

Download the `/src` directory. From `/example` directory you can run:

```bash
python ../src/generate.py --init ./initial/ --mdp ./mdp/ --top ./topology/ --nreplicas 2
````

This will generate two directories and run simulations in each (note that if `gmx_mpi` is not installed, only one replicate should be used).

Next, run:

```bash
python ../src/optimize.py --ref-pdb ./reference/reference.pdb --ref-traj ./reference/reference.xtc --ref-ndx ./reference/reference.ndx --trj-groups "10 1" --dimensions 3 --nreplicas 2
```

This will concatenate the trajectories from the previous step and align them to the `BB` beads in `reference.pdb` (index 10 in the `reference.ndx` file). The first three principal components will be determined from `reference.xtc`, and both the `reference.xtc` and `concatenated.xtc` will be projected. The probability overlap between the two distributions will then be computed, the corresponding perturbation applied, and the `go_nbparams.itp` file in each run directory updated.

Figures of the distributions, simulated trajectories, Gō-bond topologies, and PSO optimization traces are saved in separate folders.


This cycle can be iterated straightforwardly:

```bash
for i in {1..20}; do
  echo "=== CYCLE $i ==="
  python ../src/generate.py --init ./initial/ --mdp ./mdp/ --top ./topology/ --nreplicas 2
  python ../src/optimize.py --ref-pdb ./reference/reference.pdb --ref-traj ./reference/reference.xtc --ref-ndx ./reference/reference.ndx --trj-groups "10 1" --dimensions 3 --nreplicas 2
done
```
---

## Comment on sampling/`md.mdp` time
Generating the ensemble at each step is the largest computational cost. Ideally, you would sample as much as in the reference ensemble; however, this may not be computationally feasible. In that case, we suggest increasing the production-run length as the optimization proceeds: early on, a couple hundred nanoseconds are typically sufficient to identify the direction of improvement, and later stages can be lengthened to better sample the shape of the target landscape.


<!--
## `generate.py` — Command-Line Flags
| **Flag**         | **Default**                   | **Description**                                                                           |
| ---------------- | ----------------------------- | ----------------------------------------------------------------------------------------- |
| `--init`         | `./initial`                   | Directory containing AA frames (`frame<N>.gro`).                                          |
| `--mdp`          | `./mdp`                       | Directory containing GROMACS `.mdp` parameter files (`min.mdp`, `eq0.mdp`, etc.).         |
| `--top`          | `./topology`                  | Directory containing topology files (`topol*`, `top/` tree).                              |
| `--ref-ndx`      | `./reference/reference.ndx`   | Reference index used to extract BB beads after simulations                                 |
| `--mdrun`        | `gmx_mpi mdrun`               | Command used to run molecular dynamics (MPI-enabled if applicable).                       |
| `--launcher`     | `mpirun`                      | MPI launcher command (e.g., `mpirun`, `srun`, or empty).                                  |
| `--nreplicas`    | `10`                          | Number of replicas (`sim_0..sim_{N-1}`).                                                  |
| `--trj-groups`   | `"1 1"`                       | Space-separated `trjconv` index selections via stdin (e.g., `"1 1"` = Protein/Protein).   |
| `--trj-begin`    | `50000`                       | Start time (ps) for `trjconv -b` trajectory trimming.                                     |


## `optimize.py` — Command-Line Flags
| **Flag**         | **Default**                 | **Description**                                                                           |
| ---------------- | --------------------------- | ----------------------------------------------------------------------------------------- |
| `--init`         | `./initial`                 | Directory with initial AA frames (`frame<N>.gro`, if used elsewhere).                     |
| `--mdp`          | `./mdp`                     | Directory containing `.mdp` files (if referenced).                                        |
| `--top`          | `./topology`                | Directory with topology files or include tree (if referenced).                            |
| `--ref-pdb`      | `./reference/reference.pdb` | Reference PDB used to read topology for CG/AA universes.                                  |
| `--ref-xtc-aa`   | `./reference/reference.xtc` | Aligned AA trajectory file (relative to `--ref`).                                         |
| `--ref-ndx`      | `./reference/reference.ndx` | Index file for group selections (relative to `--ref`).                                    |
| `--pc-boundary`  | `35`                        | Plot boundary for principal component (PC) axes.                                          |
| `--nreplicas`    | `10`                        | Number of replicas to collect (`sim_0..sim_{N-1}`).                                       |
| `--optstep`      | `-1`                        | Override optimization step index; use `-1` to infer automatically.                        |
| `--trj-groups`   | `"10 1"`                    | Space-separated selections passed to `trjconv` (e.g., `"10 1"`).                          |
| `--out-cg-traj`  | `"cg_traj"`                 | Output directory for processed CG trajectories.                                           |
| `--out-fe`       | `"fe_maps"`                 | Output directory for free energy maps.                                                    |
| `--out-params`   | `"optimized_params"`        | Directory where updated force-field parameters are written.                               | -->
