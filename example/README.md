# PoGō Pipeline

Optimization alternates between two main stages:

1. **`generate.py`** — seeds, builds, and runs a GROMACS simulation pipeline  
```scss
python generate.py
--workdir        Root directory for runs (default: .)
--init           Input AA frames (default: ./initial)
--mdp            MDP parameter files (default: ./mdp)
--top            Topology directory (default: ./topology)
--ff             Folder for updated go_nbparams.itp (default: ./ff_param)
--nreplicas      Number of replicas (default: 10)
--trj-total-ps   Target ensemble size  (default: 1000000). Will calculate per replica length requirement.
```

`generate.py` will attempt to read `MDRUN`, `GROMPP`, `MDARGS`, and `LAUNCHER` from the environment and assign standard defaults (i.e., `gmx`) if none are found. The `--trj-total-ps` flag specifies the total size of the generated ensemble in picoseconds and automatically determines how long each replica simulation should run, assuming a `--trj-equil-ps=50000` ps discard for equilibration.

2. **`optimize.py`** — analyzes generated ensemble and finds an optimal perturbation to the Gō-potential to match the target reference.  

```scss
python optimize.py
--workdir        Root directory (default: .)
--top            Topology directory (default: ./topology)
--ref-pdb        Reference PDB (default: ./reference/reference.pdb)
--ref-traj       Reference trajectory (default: ./reference/reference.xtc)
--ref-ndx        Index file (default: ./reference/reference.ndx)
--dimensions     PCA dimensionality (default: 3)
--nreplicas      Number of replicas (default: 10)
--trj-groups     trjconv groups (default: "10 1")
```

---
## Step-by-step
Download the `/example` directory. Inside, you will find:

- `/initial`: contains a set of initial configurations (this system corresponds to the `topol.top` file)
- `/topology`: contains a `topol.top` file and `martinize2`-generated `.itp` files
- `/mdp`: contains `.mdp` files in a standard equilibration hierarchy
- `reference.[pdb/ndx/xtc]`: a forward-mapped atomistic trajectory, structure file, and GROMACS-style index file


---
Download the `/src` directory. From the `/example` directory you can run:

```bash
python ../src/generate.py --init ./initial/ --mdp ./mdp/ --top ./topology/ --nreplicas 1 --trj-total-ps 100000 --trj-equil-ps 10000
```

For illustration, we run a short 100 ns simulation, discarding 10 ns as equilibration. For a proper optimization cycle, we recommend using an ensemble of 1 microsecond. Note that if `gmx_mpi` is not installed and `-multidir` cannot be used, `--nreplicas 1` should be chosen.

After the ensemble is generated, we run:

```bash
python ../src/optimize.py --ref-pdb ./reference/reference.pdb --ref-traj ./reference/reference.xtc --ref-ndx ./reference/reference.ndx --trj-groups "10 1" --dimensions 3 --nreplicas 1
```

This will concatenate the trajectories from the previous step and align them to the `BB` beads in `reference.pdb` (index 10 in the `reference.ndx` file). The first three principal components will be determined from `reference.xtc`, and both the `reference.xtc` and `concatenated.xtc` will be projected. The probability overlap between the two distributions will then be computed, the corresponding perturbation applied, and the `go_nbparams.itp` file in each run directory updated.

Figures of the distributions, simulated trajectories, Gō-bond topologies, and PSO optimization traces are saved in separate folders.

This cycle can be iterated straightforwardly:

```bash
for i in {1..20}; do
  echo "===== CYCLE ${cycle}/20 ====="
  python ../src/generate.py --init ./initial/ --mdp ./mdp/ --top ./topology/ --nreplicas 1 --trj-length 11000 --trj-equil-ps 1000
  python ../src/optimize.py --ref-pdb ./reference/reference.pdb --ref-traj ./reference/reference.xtc --ref-ndx ./reference/reference.ndx --trj-groups "10 1" --dimensions 3 --nreplicas 1
done
```

---

## Single node `-multidir`

To improve aggregate simulation performance on nodes with many CPUs or multiple GPUs, the `-multidir` option can be used. It is sufficient to simply specify `--nreplicas 2` to enable this functionality; each replicate will now run half of the total 100 ns.

```bash
for i in {1..20}; do
  echo "===== CYCLE ${cycle}/20 ====="
  python ../src/generate.py --init ./initial/ --mdp ./mdp/ --top ./topology/ --nreplicas 2 --trj-length 100000 --trj-equil-ps 10000
  python ../src/optimize.py --ref-pdb ./reference/reference.pdb --ref-traj ./reference/reference.xtc --ref-ndx ./reference/reference.ndx --trj-groups "10 1" --dimensions 3 --nreplicas 2
done
```
For use on a HPC system we have provided an example SLURM job script: `./example/slurm/slurm_single.dat`.
---

## Multi node

To further improve performance, one can parallelize the workflow. Below is an excerpt from an example SLURM script: `./example/slurm/slurm_multi.dat`:

```bash
for cycle in $(seq 1 "$NUM_CYCLES"); do
    export CYCLE="$cycle"                      # <-- make it visible to srun shells
    export RUN_ID="$(date +'%Y%m%d-%H%M%S').$(tr -dc 'a-zA-Z0-9' </dev/urandom | fold -w 8 | head -n 1)"

    echo "===== CYCLE ${cycle}/${NUM_CYCLES} ====="

    echo "[Cycle $cycle] Cleaning old temp files..."
    srun --export=ALL -N "${NODES}" -n "${NODES}" bash -lc '
        for d in ${TMPDIR:-/tmp}/'"${RUN_ID}"'.*; do
        [[ -d "$d" ]] || continue
        find "$d" -type f -name "#*#"  -delete
        find "$d" -type f -name "core*" -delete
        done
    '

    echo "[Cycle $cycle] Launching generate.py on all nodes..."
    srun --export=ALL -N "${NODES}" -n "${NODES}" --chdir="${TMPDIR:-/tmp}" bash -lc '
        set -euo pipefail
        SIM_TMP_DIR="${TMPDIR:-/tmp}/${RUN_ID}.${SLURM_PROCID}"
        SIM_PERM_DIR="${PROJ_ROOT}/sim_${SLURM_PROCID}"
        mkdir -p "$SIM_TMP_DIR" "$SIM_PERM_DIR"
        cd "$SIM_TMP_DIR"

        python "/home/cwilson/git/pogo/src/generate.py" \
        --init   "'"${INIT_DIR}"'" \
        --mdp    "'"${MDP_DIR}"'" \
        --top    "'"${TOP_DIR}"'" \
        --index  "'"${REF_NDX}"'" \
        --ff     "'"${FF_DIR}"'" \
        --nreplicas "1" \
        --trj-total-ps 100000 \
        --trj-equil-ps 10000 \
        > "'"${OUT_DIR}"'/generate_${RUN_ID}_cycle${CYCLE}_rep${SLURM_PROCID}.out"

        XTC_SRC="$(find "$SIM_TMP_DIR" -maxdepth 3 -type f -name "cg_pbc.xtc" | head -n 1 || true)"
        if [[ -n "${XTC_SRC}" ]]; then
        cp -f "$XTC_SRC" "$SIM_PERM_DIR/cg_pbc.xtc"
        fi
    '

    echo "[Cycle $cycle] Running optimize.py on a single node..."
    srun --export=ALL -N 1 -n 1 bash -lc '
        set -euo pipefail
        python "/home/cwilson/git/pogo/src/optimize.py" \
        --ref-pdb   "'"${PROJ_ROOT}"'/reference/reference.pdb" \
        --ref-traj  "'"${PROJ_ROOT}"'/reference/reference.xtc" \
        --ref-ndx   "'"${PROJ_ROOT}"'/reference/reference.ndx" \
        --trj-groups "10 1" \
        --dimensions 3 \
        --nreplicas "'"${NREPLICAS}"'" \
        > "'"${OUT_DIR}"'/optimize_${RUN_ID}_cycle${CYCLE}_n${NREPLICAS}.out"
    '

    echo "[Cycle $cycle] Completed."
done
```

Here we distribute simulation jobs across 10 nodes, each generating a 100 ns ensemble (i.e., 150 ns total – 50 ns equilibration), for a collective 1 microsecond. Trajectories are copied back to the main working directory, and a single node performs the optimization step before 10 new simulations are spawned using the updated parameters.

---

## Comment on sampling/`md.mdp` time
Generating the ensemble at each step is the largest computational cost. Ideally, you would sample as much as in the reference ensemble; however, this may not be computationally feasible. In that case, we suggest increasing the production-run length as the optimization proceeds: early on, a couple hundred nanoseconds are typically sufficient to identify the direction of improvement, and later stages can be lengthened to better sample the shape of the target landscape.
