# Write a short, self-contained Python script that computes the quantum geometric tensor
# (metric g and Berry curvature Omega) from a Wannier90 tight-binding Hamiltonian (wannier90_hr.dat).
#
# It expects k-points in *fractional* reciprocal coordinates (units of b1,b2,b3 / 2π),
# so the phase is exp(i * 2π * k · R). Results are given for the chosen two directions
# (mu, nu) — by default kx, ky — on a 2D k-mesh at fixed kz.
#
# Save this file as qgt_from_wannier90.py and run:
#   python qgt_from_wannier90.py --hr wannier90_hr.dat --nk 101 101 --kz 0.0 --bands all
#
# Notes
# -----
# 1) This uses the interband (Kubo-like) formula with velocity matrices from ∂H/∂k,
#    which is gauge-invariant and robust on dense meshes.
# 2) If your flat band touches others, consider computing a *composite* QGT by summing
#    projectors over a band subspace — this simple script evaluates single-band QGT.
# 3) The QGT components here are with respect to fractional k-coordinates. To express
#    them in Cartesian coordinates, apply the reciprocal-metric transformation using
#    your lattice vectors.


import argparse
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

TWOPI = 2.0 * np.pi


@dataclass
class HRData:
    num_wann: int
    nrpts: int
    R_list: np.ndarray          # (nrpts, 3) integers
    degen: np.ndarray           # (nrpts,) integers
    H_of_R: np.ndarray          # (nrpts, num_wann, num_wann) complex


def read_wannier90_hr(path: str) -> HRData:
    """Parse wannier90_hr.dat (minimal, robust)."""
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    it = 1
    num_wann = int(lines[it]); it += 1
    nrpts = int(lines[it]); it += 1

    # Degeneracy list may span multiple lines, 15 integers per line.
    degen_vals = []
    while len(degen_vals) < nrpts:
        degen_vals.extend([int(x) for x in lines[it].split()])
        it += 1
    degen = np.array(degen_vals[:nrpts], dtype=int)

    # The remaining lines are hopping entries:
    # R1 R2 R3 i j Re Im, repeated num_wann^2 * nrpts times
    n_expected = nrpts * num_wann * num_wann
    R_to_idx: Dict[Tuple[int, int, int], int] = {}
    R_list = []
    H_of_R = np.zeros((nrpts, num_wann, num_wann), dtype=np.complex128)

    r_count = 0  # count distinct R encountered in order
    filled = 0

    for L in lines[it:]:
        parts = L.split()
        if len(parts) != 7:
            # Some Wannier versions split big numbers; join next line if needed
            # but for simplicity assume well-formed 7-column lines.
            continue
        R1, R2, R3 = (int(parts[0]), int(parts[1]), int(parts[2]))
        i, j = int(parts[3]) - 1, int(parts[4]) - 1
        val = float(parts[5]) + 1j * float(parts[6])
        key = (R1, R2, R3)

        if key not in R_to_idx:
            if r_count >= nrpts:
                raise RuntimeError("Found more R-points than nrpts indicates.")
            R_to_idx[key] = r_count
            R_list.append(key)
            r_count += 1

        idx = R_to_idx[key]
        H_of_R[idx, i, j] = val
        filled += 1

    if filled != n_expected:
        raise RuntimeError(
            f"Incomplete HR parsing: expected {n_expected} entries, got {filled}."
        )

    return HRData(
        num_wann=num_wann,
        nrpts=nrpts,
        R_list=np.array(R_list, dtype=int),
        degen=degen,
        H_of_R=H_of_R
    )


def fourier_interpolate(HR: HRData, kvec: np.ndarray):
    """Return H(k) and velocity matrices dH/dk_mu for mu=0,1,2 at fractional kvec=(k1,k2,k3)."""
    # Phase factor: exp(i * 2π * k · R)
    phases = np.exp(1j * TWOPI * (HR.R_list @ kvec))
    w = (phases / HR.degen).astype(np.complex128)[:, None, None]  # (nrpts,1,1)

    Hk = np.sum(w * HR.H_of_R, axis=0)  # (num_wann, num_wann)

    # Velocities: ∂H/∂k_mu = i * 2π * Σ_R R_mu * H(R)/degen * e^{i 2π k·R}
    dH = []
    for mu in range(3):
        fac = 1j * TWOPI * HR.R_list[:, mu]
        dH_mu = np.sum((fac * w.squeeze())[:, None, None] * HR.H_of_R, axis=0)
        dH.append(dH_mu)
    return Hk, dH  # dH is list of 3 arrays (num_wann,num_wann)


def qgt_at_k(Hk: np.ndarray, dH: Tuple[np.ndarray, np.ndarray, np.ndarray]):
    """Compute QGT for all bands at this k using interband (Kubo-like) formula.
       Returns dictionaries keyed by (mu,nu) with arrays shape (nbands,)."""
    eps, U = np.linalg.eigh(Hk)  # U[:,n] column is eigenvector of band n
    Uhc = U.conj().T
    nb = Hk.shape[0]

    # Velocity matrices in band basis: v_mu = U^† (∂H/∂k_mu) U
    v = [Uhc @ dH_mu @ U for dH_mu in dH]

    # Initialize accumulators
    g = {(0, 0): np.zeros(nb), (1, 1): np.zeros(nb), (0, 1): np.zeros(nb)}
    Omega = {(0, 1): np.zeros(nb)}  # we'll compute Omega_xy

    # Interband sum
    for n in range(nb):
        denom = (eps - eps[n])  # (nb,)
        denom[n] = np.inf       # avoid zero; will skip m=n naturally
        denom2 = denom**2

        for mu, nu in [(0, 0), (1, 1), (0, 1)]:
            num = v[mu][n, :] * v[nu][:, n]  # (nb,)
            num[n] = 0.0
            A = np.sum(num / denom2)  # complex
            if (mu, nu) != (0, 1):
                g[(mu, nu)][n] = np.real(A)
            else:
                g[(0, 1)][n] = np.real(A)
                Omega[(0, 1)][n] = -2.0 * np.imag(A)

    return g, Omega  # each entry is (nb,)


def compute_qgt_maps(hr_path: str, nkx: int, nky: int, kz: float, bands: str = "all"):
    HR = read_wannier90_hr(hr_path)
    nb = HR.num_wann

    # k-grid in fractional coords
    kxs = np.linspace(0.0, 1.0, nkx, endpoint=False)
    kys = np.linspace(0.0, 1.0, nky, endpoint=False)

    if bands == "all":
        band_list = list(range(nb))
    else:
        band_list = [int(x) for x in bands.split(",")]

    gxx = np.zeros((len(band_list), nkx, nky))
    gyy = np.zeros((len(band_list), nkx, nky))
    gxy = np.zeros((len(band_list), nkx, nky))
    omz = np.zeros((len(band_list), nkx, nky))

    for ix, kx in enumerate(kxs):
        for iy, ky in enumerate(kys):
            kvec = np.array([kx, ky, kz], dtype=float)
            Hk, dH = fourier_interpolate(HR, kvec)
            g, Om = qgt_at_k(Hk, dH)
            # Select requested bands
            gxx[:, ix, iy] = g[(0, 0)][band_list]
            gyy[:, ix, iy] = g[(1, 1)][band_list]
            gxy[:, ix, iy] = g[(0, 1)][band_list]
            omz[:, ix, iy] = Om[(0, 1)][band_list]

    return {
        "kxs": kxs, "kys": kys, "kz": kz,
        "bands": np.array(band_list, dtype=int),
        "gxx": gxx, "gyy": gyy, "gxy": gxy, "Omega_xy": omz
    }


def plot_qgt_2d(data: dict, band_sel: int, save_prefix: str = None):
    import matplotlib.pyplot as plt
    kxs = data["kxs"]; kys = data["kys"]
    extent = [kxs.min(), kxs.max(), kys.min(), kys.max()]
    ix = band_sel
    label_band = int(data["bands"][ix])

    for key, arr, ttl in [
        ("Omega_xy", data["Omega_xy"][ix], r"Berry curvature $\Omega_{xy}$"),
        ("gxx", data["gxx"][ix], r"Quantum metric $g_{xx}$"),
        ("gyy", data["gyy"][ix], r"Quantum metric $g_{yy}$"),
        ("gxy", data["gxy"][ix], r"Quantum metric $g_{xy}$"),
    ]:
        plt.figure()
        plt.imshow(arr.T, origin="lower", extent=extent, aspect="equal")
        plt.colorbar(label=key)
        plt.xlabel("kx (fractional)")
        plt.ylabel("ky (fractional)")
        plt.title(f"{ttl} — band {label_band}")
        if save_prefix is not None:
            plt.savefig(f"{save_prefix}_{key}_band{label_band}.png", dpi=50, bbox_inches="tight")
        #plt.show()

def main():
    ap = argparse.ArgumentParser(description="QGT from Wannier90 HR (interband formula).")
    ap.add_argument("--hr", required=True, help="Path to wannier90_hr.dat")
    ap.add_argument("--nk", nargs=2, type=int, default=[101, 101], help="nkx nky grid sizes")
    ap.add_argument("--kz", type=float, default=0.0, help="Fixed kz (fractional)")
    ap.add_argument("--bands", type=str, default="all", help="'all' or comma-separated band indices (0-based)")
    ap.add_argument("--out", type=str, default="qgt_maps.npz", help="Output .npz file")
    ap.add_argument("--plot", action="store_true", help="If set, plot 2D maps for one selected band.")
    ap.add_argument("--plot-band", type=int, default=0, help="Index within the chosen list of bands to plot (default: first).")
    ap.add_argument("--save-prefix", type=str, default=None, help="If provided, save PNGs with this prefix.")

    args = ap.parse_args()

    data = compute_qgt_maps(args.hr, args.nk[0], args.nk[1], args.kz, args.bands)
    np.savez_compressed(args.out, **data)
    print(f"Saved QGT maps to {args.out}")
    print("Arrays saved:", list(data.keys()))
    
    if args.plot:
        if args.plot_band < 0 or args.plot_band >= len(data["bands"]):
            raise ValueError(f"--plot-band must be in [0, {len(data['bands'])-1}] for the chosen --bands.")
        plot_qgt_2d(data, args.plot_band, save_prefix=args.save_prefix)

if __name__ == "__main__":
    main()

