# Crystal stability protocol

Build a perfect colloidal crystal from a CIF, then relax it with Langevin dynamics
and see whether it survives. Wraps PACSim's `LatticeBuilder` + `pacsim-run`.

To test a different crystal, copy `systems/cscl.yaml`, change `cif` and `types`,
and resubmit. Nothing else needs to be touched.

## Layout

    structures/       CIFs that design.py and build.py read by default.
                      Every file here is byte-identical to one in
                      ../cif_library/prototypes/, but two differ in NAME
                      (Th3P4.cif = Th3P4__8edd1f72.cif, zangenite.cif =
                      zangenite__98bb84a4.cif). Kept as the build path's own
                      input directory; the library is the deduplicated one.
    systems/          One YAML per crystal. THIS IS THE ONLY FILE YOU EDIT.
    build.py          systems/<x>.yaml -> runs/<x>/{configuration.yaml, run.yaml, <x>.cif}
    env.sh            conda activate + the OPENMM_PLUGIN_DIR fix (see below)
    submit_gpu.sbatch Build + relax on a GPU. Use this one.
    submit_cpu.sbatch Same, on CPU. ~100x slower; only for tiny test systems.
    runs/             One directory per run. Outputs land here.

`build.py` exists so the two YAMLs PACSim consumes can never drift apart:
`pacsim-create` reads `configuration.yaml`, which points at `run.yaml` for the
potential parameters it needs to energy-optimize the lattice spacing. Change the
Debye length in one place and both stay consistent.

## Run it

    sbatch submit_gpu.sbatch systems/cscl.yaml

Everything after the sbatch script is forwarded to `build.py`, so a melting scan
is three submissions:

    sbatch submit_gpu.sbatch systems/cscl.yaml --debye 5.0 --tag stable
    sbatch submit_gpu.sbatch systems/cscl.yaml --debye 4.1 --tag melt_some
    sbatch submit_gpu.sbatch systems/cscl.yaml --debye 3.5 --tag melt_all

`--tag` keeps the outputs in separate `runs/` directories, so the runs cannot
overwrite each other. (The cookbook's own Stability/ inputs all write to the same
`trajectory.gsd` and clobber each other when run in parallel; this avoids that.)

Interactively, without SLURM:

    source env.sh
    python build.py systems/cscl.yaml --repeats 3 --steps 2000 --tag smoke
    cd runs/cscl_smoke
    pacsim-create configuration.yaml first_frame.gsd
    pacsim-run run.yaml

## Outputs (in runs/<name>/)

    first_frame.gsd   the perfect crystal, 1 frame. Look at this before running.
    trajectory.gsd    the relaxation, n_frames + 1 frames
    final_frame.gsd   last frame
    state_data.csv    time, energies, temperature. Potential energy tells you if it held.
    ramp.csv          integrator temperature vs step
    checkpoint.chk    resume with: pacsim-run run.yaml -c checkpoint.chk

## Timings (6750 particles, 15^3 CsCl)

    pacsim-create   ~40 s   (the 50-point energy scan dominates)
    pacsim-run      ~11 min on one GPU at 1e6 steps  (~38 h on 8 CPU threads)

## The OpenMM plugin gotcha

The openmm in the pacsim env has its plugin path baked in as
`/home/sz5808/.conda/envs/pacsim/lib`, but the env lives on /scratch and that home
path does not exist. So openmm loads no plugins, only the `Reference` platform
registers, and `pacsim-run` dies with:

    OpenMMException: There is no registered Platform called "CPU"

`env.sh` works around it by exporting `OPENMM_PLUGIN_DIR`. The permanent fix is:

    ln -s /scratch/sz5808/.conda/envs/pacsim /home/sz5808/.conda/envs/pacsim

## What the numbers should look like (CsCl)

The CIF's lattice constant (4.14 A, atomic) is discarded; only the geometry is
kept. LatticeBuilder rescales it to colloidal sizes:

    effective radius  = radius + brush_length + radii_padding
                      = 85 + 10 + 20 = 115 nm (Cs),  105 + 10 + 20 = 135 nm (Cl)
    geometric scale   -> Cs-Cl = 250 nm  (a = 288.7 nm)  [no-overlap guess]
    energy-optimized  -> Cs-Cl = 211.7 nm (a = 244.5 nm) [0.847x, min of V_E + V_P]
                         surface gap = 21.7 nm ~ brush contact (2L = 20 nm)

So `radii_padding` sets the *starting guess* for the spacing; `optimize_energy`
then finds the actual potential minimum. If pacsim-create warns that the energy
minimum sits at the boundary of the scan range, widen `energy_scale_range`.
