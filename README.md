# TrivialCat v4

This repository can be used to:

- generate standardized VASP input sets from Materials Project,
- run the main *TrivialCat* flat-band analysis on VASP outputs,
- and compute **quantum geometric tensors (QGT)** from Wannier90 tight-binding Hamiltonians for validating *TrivialCat* results

This README focuses on the scripts:

- `main_create_input_for_MP_bands.py`
- `main_trivialcat.py`
- `main_QGT_from_wann_1_slice.py`
- `main_nonAbel_QGT_from_wann_2Dslices.py`
- `main_plot_bands_from_VASP.py`

---

## 1. Environment & Dependencies

### Python & packages

Install dependencies with:

```bash
pip install -r pip_requirements.txt
````

Key Python packages used (non-exhaustive):

* `numpy`, `scipy`, `pandas`
* `matplotlib`
* `pymatgen`, `mp_api`, `emmet-core`
* `spglib`
* plus standard scientific Python stack

### External tools & data

Some scripts assume the presence of:

* **VASP**: for SCF / bandstructure runs (producing `WAVECAR`, `OUTCAR`, `vasprun.xml`, etc.).
* **Wannier90**: to generate `wannier90_hr.dat` for QGT scripts.
* The `external/` directory:

  * `external/Compound_flatness_score_segments.csv`
  * `external/Flat_band_sublattices.csv`
  * `external/list_of_materials_with_BS_DOS_readable`
  * `external/model.keras`, `external/WaveTrans.exe`
* A **Materials Project API key** for scripts that talk to MP (`main_create_input_for_MP_bands.py`, `main_plot_bands_from_VASP.py`).

> **Note:** The MP API key is currently hard-coded inside some scripts. In practice, you should replace that string with your own key (or modify the scripts to read `os.environ["MP_API_KEY"]`).

---

## 2. Typical Workflow

A rough end-to-end workflow looks like:

1. **Generate VASP input sets for flat-band candidates**
   Use `main_create_input_for_MP_bands.py` to create `bands/` and `scf/` subfolders for a range of MP IDs.

2. **Run VASP**
   Run SCF (and optionally bandstructure / Wannier) VASP jobs in those generated folders, so that `WAVECAR`, `OUTCAR`, etc. exist.

3. **Run the TrivialCat analysis**
   Use `main_trivialcat.py` on each material to analyze flatness, charge-density retention and destructive interference, using the precomputed flatness and sublattice information in `external/*.csv`.

4. **Optional: compute QGT from Wannier90 TB models**
   Use:

   * `main_QGT_from_wann_1_slice.py` for **Abelian** QGT on a single 2D slice, and
   * `main_nonAbel_QGT_from_wann_2Dslices.py` for **non-Abelian** QGT and 3D-averaged geometric superfluid weight over an energy window.

5. **Optional: simple band plotting from VASP**
   Use (after editing) `main_plot_bands_from_VASP.py` to make basic bandstructure plots.

---

## 3. `main_create_input_for_MP_bands.py`

### Purpose

Generate **VASP input directories** for a set of Materials Project IDs that are flagged as flat-band candidates in:

* `external/Compound_flatness_score_segments.csv`

The script:

* filters to entries labeled `Flat/not-flat == True`,
* restricts to IDs that also appear in `external/list_of_materials_with_BS_DOS_readable`,
* for each selected materials project ID:

  * fetches the bandstructure and associated task from Materials Project,
  * extracts the structure and INCAR from the original calculation,
  * writes:

    * a **bandstructure non-SCF** input (`bands/`),
    * an **SCF** input (`scf/`),
    * optionally an **SCF+Wannier** input (with `LSCDM`, `LWANNIER90`, etc. turned on),
    * and optionally a reference bandstructure plot from MP.

By default, directories are written under a hard-coded path:

```python
read_path  = '/home/anupam/TrivialCat_v4/external/'
write_path = '/home/anupam/TrivialCat_v4/Calc_wavecar_new/'
```

You will likely want to **edit** these paths to match your environment.

### Inputs & prerequisites

* `external/Compound_flatness_score_segments.csv`
* `external/list_of_materials_with_BS_DOS_readable`
* a valid Materials Project API key (edit `MP_API_KEY` in the script or change it to use an environment variable).

### Command-line usage

The script expects:

```bash
python main_create_input_for_MP_bands.py START_ID END_ID [wavefn_save] [bands_input] [scf_input] [scf_wannier_input] [ref_BS_plot]
```

* `START_ID` and `END_ID` (required, `int`):
  Numeric MP IDs (e.g., `20536` for `mp-20536`).
  Only materials with `START_ID <= id <= END_ID` and present in the flat-band list and the readable list are processed.

* `wavefn_save` (optional, default: `True`):
  If provided, it is converted with `bool(sys.argv[3])`, so you should pass something like `1`/`0` or `True`/`False` but be aware that in Python `bool("False")` is `True`. The intent is a toggle to keep `LWAVE` and `LCHARG`.

* `bands_input` (optional, default: `True`):
  Whether to write `bands/` non-SCF bandstructure inputs.

* `scf_input` (optional, default: `True`):
  Whether to write `scf/` static SCF inputs (`MPStaticSet` with `ALGO="Normal", LWAVE=True, LCHARG=True, ICHARG=2`, etc.).

* `scf_wannier_input` (optional, default: `False`):
  Whether to additionally write SCF inputs tuned for **Wannier90** (same `scf/` folder, but with `LSCDM=True`, `LWANNIER90=True`, `NUM_WANN`, `NBANDS` logic, etc.).

* `ref_BS_plot` (optional, default: `False`):
  If `True`, computes a reference bandstructure using Materials Project data and writes a plot.

Example:

```bash
python main_create_input_for_MP_bands.py 20000 20100 1 1 1 0 0
```

This will:

* scan flat-band entries whose MP IDs fall between `mp-20000` and `mp-20100`,
* generate `bands/` and `scf/` inputs for each matched ID under `Calc_wavecar_new/mp-XXXXX/`,
* skip the Wannier-specific SCF and MP reference plot.

---

## 4. `main_trivialcat.py`

### Purpose

This is the **main TrivialCat analysis script**. Given a VASP calculation directory for a particular MP material, it:

1. Reads the relevant **flatness and sublattice information** from:

   * `external/Compound_flatness_score_segments.csv`
   * `external/Flat_band_sublattices.csv`
2. Uses that to define an energy window `[Emin, Emax]` around the Fermi level and to pick a **sublattice element** (or full lattice).
3. Reads **wavefunctions** from the VASP `WAVECAR` (via `get_wavecar_band_data` from `src`), plus the FFT grid etc.
4. Constructs a **structure graph** whose nodes are atomic sites and edges represent hopping paths up to a cutoff distance.
5. Uses **multiprocessing** to evaluate the charge density and wavefunction along each graph edge in real space (`CHG_along_path.charge_density_along_a_path`).
6. Builds an “extended graph” with the additional data (density along edges, signal-to-noise ratios, etc.).
7. Produces:

   * plots of the extended graph (various options: density, |ψ|², etc.),
   * plots of charge density along edges (truncated near core radii),
   * summary metrics of **charge density retention** and **charge drop** along paths,
   * an analysis of destructive interference at Γ if the retention exceeds a threshold.

The actual plotting and output file naming are implemented in functions imported from `src` (e.g. `Make_graph`, `create_extended_graph`, `plot_extended_graph`, `plot_chg_along_graph_edges`, `identify_charge_retention`, `analyse_wavefunctions_for_destructive_interference`).

### Inputs & prerequisites

* A calculation directory for one material, e.g.:

  ```text
  Calc_wavecar_new/mp-20536/
  ├── bands/      # generated by main_create_input_for_MP_bands.py
  └── scf/
      ├── WAVECAR
      ├── OUTCAR
      └── (other VASP outputs)
  ```

* `external/Compound_flatness_score_segments.csv`

* `external/Flat_band_sublattices.csv`

* `external/model.keras` and `external/WaveTrans.exe` (used internally by some `src` utilities).

* Necessary Python dependencies (including those in `src/`).

### Command-line usage

From the repository root (`Trivialcat_v4/`):

```bash
python main_trivialcat.py PATH_TO_MP_DIR
```

where `PATH_TO_MP_DIR` is a folder whose **basename** is an MP ID, e.g.:

```bash
python main_trivialcat.py Calc_wavecar_new/mp-20536
```

Internally, the script does:

```python
exe_model_path = os.path.abspath('external/')
filepath       = os.path.abspath(sys.argv[1] + '/scf/')
mp_id          = sys.argv[1].split('/')[-1]   # e.g. 'mp-20536'
os.chdir(filepath)
```

so:

* it assumes the WAVECAR / OUTCAR live in the `scf/` subfolder of the path you pass in,
* it uses the MP identifier from the last segment of that path to look up entries in the CSV files.

### Important parameters (inside the script)

You can tune:

* `flatness_threshold`
* `prediction_bandwidth` (energy window in eV used in ML flatness prediction)
* `n_samp_pts` (number of sample points along each path)
* `hopping_cutoff` (Å, maximum bond length used in the graph construction)
* `core_rad` (approximate core radius where density is unreliable)
* `charge_density_retention_threshold`
* `destructive_interference_threshold`
* `num_process_use` (number of CPUs used in the `multiprocessing.Pool`)

### Example batch usage

A sample SLURM script is provided in `sub_main_trivialcat.sh`, which does roughly:

```bash
for path in Calc_wavecar_new/mp-*; do
    python main_trivialcat.py "$path"
done
```

This loops over all `mp-*` directories under `Calc_wavecar_new/` and runs the TrivialCat analysis on each.

---

## 5. `main_QGT_from_wann_1_slice.py`

### Purpose

Compute the **(Abelian) quantum geometric tensor** for **individual bands** in a Wannier90 tight-binding Hamiltonian `wannier90_hr.dat` on a **2D k-mesh at fixed kz**.

It reads `wannier90_hr.dat`, constructs the Bloch Hamiltonian `H(k)`, and then:

* diagonalizes `H(k)` on a `nkx × nky` grid at fixed `kz`,
* computes velocity matrices and uses a gauge-invariant, interband (Kubo-like) formula to obtain:

  * quantum metric components `g_xx`, `g_yy`, `g_xy`,
  * Berry curvature `Ω_xy`,
* optionally plots 2D color maps for a selected band.

### Command-line usage

```bash
python main_QGT_from_wann_1_slice.py \
  --hr path/to/wannier90_hr.dat \
  --nk 101 101 \
  --kz 0.0 \
  --bands all \
  --out qgt_maps.npz \
  [--plot] \
  [--plot-band 0] \
  [--save-prefix my_prefix]
```

Arguments:

* `--hr` (**required**): path to `wannier90_hr.dat`.
* `--nk NKX NKY` (default: `101 101`): k-grid sizes along kx and ky.
* `--kz` (default: `0.0`): fixed kz in fractional coordinates.
* `--bands` (default: `"all"`):

  * `"all"` → include all Wannier bands,
  * or a comma-separated list of **0-based** band indices, e.g. `"0,1,2,3"`.
* `--out` (default: `"qgt_maps.npz"`): output file name.
* `--plot` (flag): if set, produces static 2D plots.
* `--plot-band` (default: `0`): index **within the chosen list** of bands for plotting (not absolute band index).
* `--save-prefix`: if provided, plots are saved as `"{prefix}_{key}_band{n}.png"` for each quantity (`Omega_xy`, `gxx`, `gyy`, `gxy`).

### Output

The compressed `.npz` file contains keys:

* `kxs`, `kys` (1D arrays of kx, ky grid points),
* `bands` (1D array of band indices),
* `gxx`, `gyy`, `gxy`, `Omega_xy` (shape: `n_bands × nkx × nky`).

If `--plot` is used, the script produces PNGs showing:

* Berry curvature `Ω_xy`
* Quantum metric components `g_xx`, `g_yy`, `g_xy`

for the chosen band.

---

## 6. `main_nonAbel_QGT_from_wann_2Dslices.py`

### Purpose

Compute the **non-Abelian** quantum geometric tensor for a **composite band manifold** defined by an **energy window** `[Emin, Emax]` over the full 3D Brillouin zone.

The script:

1. Reads `wannier90_hr.dat`.

2. Builds a 3D k-grid with sizes `(nkx, nky, nkz)` where `kx, ky, kz ∈ [0, 1)` (fractional).

3. For each `k`:

   * Diagonalizes `H(k)` to get eigenvalues/eigenvectors.
   * Constructs the **projector** `P(k)` onto all states with energies in `[Emin, Emax]`.

4. On each `(kx, ky)` plane at discrete `kz`:

   * computes non-Abelian QGT components:

     * `g_xx(k)`, `g_yy(k)`, `g_xy(k)`, `Ω_xy(k)`,
   * and tracks the subspace dimension `N_sub(k)` (number of bands in the window).

5. Averages `g_muν(k)` over the **full 3D grid** to get:

   * `<g_xx>_3D`, `<g_yy>_3D`, `<g_xy>_3D`.

6. Computes the **geometric superfluid weight** (flat, T=0, uniform gap approximation):

   ```text
   D^geom_{μν} ≈ 2 Δ ⟨ g_{μν}(k) ⟩_3D
   ```

7. Optionally stores and plots a single 2D slice (at a chosen kz index).

To keep memory usage manageable, it only keeps **one slice** fully in memory at a time.

### Command-line usage

```bash
python main_nonAbel_QGT_from_wann_2Dslices.py \
  --hr path/to/wannier90_hr.dat \
  --nk 31 31 \
  --nkz 16 \
  --Emin -0.1 \
  --Emax  0.1 \
  --Delta 0.01 \
  --out nonabelian_qgt_3d.npz \
  [--plot] \
  [--plot-kz-idx 0] \
  [--save-prefix CoSn]
```

Arguments:

* `--hr` (**required**): path to `wannier90_hr.dat`.
  The script derives the **input/output directory** as:

  ```python
  input_path = hr_path.rsplit('/', 1)[0]
  ```

  and writes outputs there.

* `--nk NKX NKY` (default: `31 31`): k-grid sizes along kx, ky.

* `--nkz` (default: `1`): number of kz slices in `[0,1)` (uniform grid).

* `--Emin` (**required**): lower edge of energy window.

* `--Emax` (**required**): upper edge of energy window.

* `--Delta` (**required**): pairing gap `Δ` (same energy units as TB), used in geometric superfluid weight.

* `--out` (default: `"2D_slices_nonabelian_qgt.npz"`): output `.npz` filename, written under `input_path`.

* `--plot` (flag): if set, plots QGT fields for one stored slice.

* `--plot-kz-idx` (default: `0`): index of kz slice to store/plot (`0 .. nkz-1`).

* `--save-prefix` (default: `'_'`): prefix appended to slice plots.

### Output

The script writes a compressed `.npz` file at:

```text
input_path / --out
```

with keys including:

* **3D-averaged quantities:**

  * `avg_gxx_3d`, `avg_gyy_3d`, `avg_gxy_3d`
  * `Dxx_geom`, `Dyy_geom`, `Dxy_geom` (geometric superfluid weight tensor components)
  * `Nsub_min`, `Nsub_max` (min/max dimension of the composite subspace over the BZ)

* **Optional 2D slice fields** (if `--plot-kz-idx` is within range):

  * `kxs`, `kys`
  * `kz_slice`
  * `gxx_slice`, `gyy_slice`, `gxy_slice`
  * `Omega_xy_slice`
  * `Nsub_slice`

It also writes a human-readable log:

```text
input_path/2D_slices_Superfluid_weight.txt
```

containing:

* confirmation of the output `.npz` file,
* keys in the file,
* `Nsub_min`, `Nsub_max`,
* `<gxx>_3D`, `<gyy>_3D`, `<gxy>_3D`,
* `Δ`, `Dxx_geom`, `Dyy_geom`, `Dxy_geom`.

If `--plot` is used, a helper function `plot_qgt_2d_slice(...)` is called to write PNGs for the chosen kz slice (with filenames based on `input_path + '/2D_slices_' + args.save_prefix`).

---

## 7. `main_plot_bands_from_VASP.py`

### Purpose

A **small helper/prototype script** to:

* load VASP outputs (e.g. `vasprun.xml`, charge densities, DOS),
* optionally fetch additional information from Materials Project for a given `mp_id`,
* produce a simple bandstructure plot saved as:

```text
Path + 'bands/bands.png'
```

### Notes & usage

This script is **not** set up as a general command-line tool:

* `mp_id` is currently hard-coded in the script:

  ```python
  mp_id = '20536'
  ```
* `Path` is also hard-coded to a particular path, e.g.:

  ```python
  Path = '/mnt/.../TrivialCat_validation_codes/Calc_wavecar/' + 'mp-' + mp_id + '/'
  ```

To use it in your own environment you should:

1. Edit `mp_id` at the top of the script.

2. Change `Path` to point at the directory containing your VASP outputs, e.g.:

   ```python
   Path = '/path/to/Calc_wavecar/' + 'mp-' + mp_id + '/'
   ```

3. Make sure the VASP outputs you need exist (e.g. `Path + 'scf/vasprun.xml'`).

4. Run:

   ```bash
   python main_plot_bands_from_VASP.py
   ```

The script uses `pymatgen`’s VASP I/O utilities, DOS plotter, etc., but large chunks are commented out or tailored to specific local paths. Treat this file as an example/template for bandstructure plotting rather than a fully polished CLI.

---

## 8. Miscellaneous

* `clear_vasp.sh` – a small shell utility to remove common VASP output files from the current directory (e.g., AEC*, ELFCAR, CHG, CHGCAR, WAVECAR, etc.).
* `sub_main_trivialcat.sh` – example SLURM job script that activates a `conda` environment (`trivialcat`) and runs `main_trivialcat.py` for all `Calc_wavecar_new/mp-*` directories.

---

If you adapt paths, thresholds, or MP API settings, it’s a good idea to keep this README in sync so future users know how each `main_*.py` entry point is intended to be used.

```
```

