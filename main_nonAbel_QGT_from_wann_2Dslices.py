#!/usr/bin/env python3
"""
nonabelian_qgt_from_wannier90.py

Slice-based, memory-friendly non-Abelian QGT for a composite band manifold
defined by an energy window [Emin, Emax], using a Wannier90 TB Hamiltonian.

We:
  - sample a 3D BZ on a uniform grid:
        kx in [0,1), ky in [0,1), kz in [0,1)
    with sizes (nkx, nky, nkz),
  - at each kz, build the projector P(k) onto all states with energies in
    [Emin, Emax],
  - compute non-Abelian metric and curvature on the (kx, ky) plane:
        g_xx, g_yy, g_xy, Omega_xy,
  - accumulate 3D BZ averages of g_mu_nu over all kx,ky,kz,
  - from these, compute geometric superfluid weight:
        D^geom_{mu nu} ≈ 2 * Delta * < g_{mu nu}(k) >_3D-BZ
    in the flat-manifold, T=0, uniform-gap approximation.

We only keep **one 2D slice** (for a chosen kz index) in memory for plotting.

Usage examples
--------------
Compute 3D QGT averages over nkz slices, save everything, no plots:

    python nonabelian_qgt_from_wannier90.py \
        --hr wannier90_hr.dat \
        --nk 81 81 \
        --nkz 21 \
        --Emin -0.02 \
        --Emax  0.03 \
        --Delta 0.002 \
        --out nonabelian_qgt_3d.npz

Same but also plot the kz-slice with index 10, and save PNGs:

    python nonabelian_qgt_from_wannier90.py \
        --hr wannier90_hr.dat \
        --nk 81 81 \
        --nkz 21 \
        --Emin -0.02 \
        --Emax  0.03 \
        --Delta 0.002 \
        --plot \
        --plot-kz-idx 10 \
        --save-prefix CoSn \
        --out nonabelian_qgt_3d.npz

Notes
-----
- kx, ky, kz are fractional coordinates in [0,1) along each reciprocal basis vector.
- Energies (Emin, Emax, Delta) are in the same units as the TB (typically eV).
"""

import argparse
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

TWOPI = 2.0 * np.pi


# ----------------------------------------------------------------------
# Wannier90 HR reader
# ----------------------------------------------------------------------

@dataclass
class HRData:
    num_wann: int
    nrpts: int
    R_list: np.ndarray          # (nrpts, 3) int
    degen: np.ndarray           # (nrpts,) int
    H_of_R: np.ndarray          # (nrpts, num_wann, num_wann) complex


def read_wannier90_hr(path: str) -> HRData:
    """Minimal parser for wannier90_hr.dat."""
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    it = 1
    num_wann = int(lines[it]); it += 1
    nrpts = int(lines[it]); it += 1

    # Degeneracies (may span multiple lines)
    degen_vals = []
    while len(degen_vals) < nrpts:
        degen_vals.extend([int(x) for x in lines[it].split()])
        it += 1
    degen = np.array(degen_vals[:nrpts], dtype=int)

    # Hopping entries: R1 R2 R3 i j Re Im
    n_expected = nrpts * num_wann * num_wann
    R_to_idx: Dict[Tuple[int, int, int], int] = {}
    R_list = []
    H_of_R = np.zeros((nrpts, num_wann, num_wann), dtype=np.complex128)

    r_count = 0
    filled = 0

    for L in lines[it:]:
        parts = L.split()
        if len(parts) != 7:
            # Assume standard formatting. If this triggers, inspect HR file.
            continue
        R1, R2, R3 = (int(parts[0]), int(parts[1]), int(parts[2]))
        i, j = int(parts[3]) - 1, int(parts[4]) - 1
        val = float(parts[5]) + 1j * float(parts[6])
        key = (R1, R2, R3)

        if key not in R_to_idx:
            if r_count >= nrpts:
                raise RuntimeError("More R-points than nrpts indicates.")
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
        H_of_R=H_of_R,
    )


# ----------------------------------------------------------------------
# k-space interpolation and projector at a single k
# ----------------------------------------------------------------------

def Hk_from_HR(HR: HRData, kvec: np.ndarray) -> np.ndarray:
    """
    Fourier-interpolate H(k) from H(R) at fractional kvec=(k1,k2,k3).

    Phase convention: exp(i 2π k · R).
    """
    phases = np.exp(1j * TWOPI * (HR.R_list @ kvec))  # (nrpts,)
    w = (phases / HR.degen).astype(np.complex128)[:, None, None]
    Hk = np.sum(w * HR.H_of_R, axis=0)  # (num_wann, num_wann)
    return Hk


def projector_from_window(Hk: np.ndarray, Emin: float, Emax: float) -> Tuple[np.ndarray, int]:
    """
    Diagonalize Hk and build the projector onto the subspace of eigenstates
    whose energies fall in [Emin, Emax].

    Returns:
        P   : (N, N) complex projector
        Nsub: dimension of the subspace at this k
    """
    eps, U = np.linalg.eigh(Hk)
    mask = (eps >= Emin) & (eps <= Emax)
    Nsub = int(np.count_nonzero(mask))

    if Nsub == 0:
        # Empty subspace: projector is zero.
        P = np.zeros_like(Hk, dtype=np.complex128)
        return P, 0

    U_sub = U[:, mask]              # (N, Nsub)
    P = U_sub @ U_sub.conj().T      # (N, N)
    return P, Nsub


# ----------------------------------------------------------------------
# Non-Abelian QGT on a single (kx, ky) plane at fixed kz
# ----------------------------------------------------------------------

def qgt_on_plane(
    HR: HRData,
    nkx: int,
    nky: int,
    kz: float,
    Emin: float,
    Emax: float,
):
    """
    Compute non-Abelian g_{xx}, g_{yy}, g_{xy}, Omega_{xy} on a
    (kx, ky) grid at fixed kz, for the composite band manifold
    defined by [Emin, Emax].

    Returns a dict with:
        kxs, kys, kz, gxx, gyy, gxy, Omega_xy, Nsub
    """
    Nw = HR.num_wann

    kxs = np.linspace(0.0, 1.0, nkx, endpoint=False)
    kys = np.linspace(0.0, 1.0, nky, endpoint=False)
    dkx = 1.0 / nkx
    dky = 1.0 / nky

    # Store projectors P(k) for this plane: (nkx, nky, Nw, Nw)
    P = np.zeros((nkx, nky, Nw, Nw), dtype=np.complex128)
    Nsub = np.zeros((nkx, nky), dtype=int)

    for ix, kx in enumerate(kxs):
        for iy, ky in enumerate(kys):
            kvec = np.array([kx, ky, kz], dtype=float)
            Hk = Hk_from_HR(HR, kvec)
            Pk, ns = projector_from_window(Hk, Emin, Emax)
            P[ix, iy] = Pk
            Nsub[ix, iy] = ns

    # Finite differences with periodic boundary conditions in plane
    dP_dx = np.zeros_like(P)
    dP_dy = np.zeros_like(P)

    for ix in range(nkx):
        ixp = (ix + 1) % nkx
        ixm = (ix - 1) % nkx
        dP_dx[ix, :, :, :] = (P[ixp, :, :, :] - P[ixm, :, :, :]) / (2.0 * dkx)

    for iy in range(nky):
        iyp = (iy + 1) % nky
        iym = (iy - 1) % nky
        dP_dy[:, iy, :, :] = (P[:, iyp, :, :] - P[:, iym, :, :]) / (2.0 * dky)

    # Allocate QGT fields
    gxx = np.zeros((nkx, nky), dtype=float)
    gyy = np.zeros((nkx, nky), dtype=float)
    gxy = np.zeros((nkx, nky), dtype=float)
    Omega_xy = np.zeros((nkx, nky), dtype=float)

    # Compute non-Abelian metric and curvature at each (kx, ky)
    for ix in range(nkx):
        for iy in range(nky):
            Pk = P[ix, iy]
            dPx = dP_dx[ix, iy]
            dPy = dP_dy[ix, iy]

            # Metric: g_{μν}(k) = Re Tr[(∂_μ P)(∂_ν P)]
            gxx[ix, iy] = np.real(np.trace(dPx @ dPx))
            gyy[ix, iy] = np.real(np.trace(dPy @ dPy))
            gxy[ix, iy] = np.real(np.trace(dPx @ dPy))

            # Curvature: Ω_{xy}(k) = -2 Im Tr[P (∂_x P)(∂_y P)]
            Omega_xy[ix, iy] = -2.0 * np.imag(np.trace(Pk @ dPx @ dPy))

    return {
        "kxs": kxs,
        "kys": kys,
        "kz": kz,
        "gxx": gxx,
        "gyy": gyy,
        "gxy": gxy,
        "Omega_xy": Omega_xy,
        "Nsub": Nsub,
    }


# ----------------------------------------------------------------------
# Geometric superfluid weight from 3D BZ average
# ----------------------------------------------------------------------

def geometric_superfluid_weight_3d(avg_gxx, avg_gyy, avg_gxy, Delta: float):
    """
    Geometric superfluid weight in the flat-manifold, T=0, uniform-gap approximation:

        D^{geom}_{μν} ≈ 2 * Delta * < g_{μν}(k) >_3D-BZ

    Parameters
    ----------
    avg_gxx, avg_gyy, avg_gxy : float
        3D BZ averages of gxx, gyy, gxy (uniform grid over kx, ky, kz).
    Delta : float
        Pairing gap amplitude (same energy units as TB, e.g. eV).

    Returns
    -------
    dict with keys 'Dxx', 'Dyy', 'Dxy'.
    """
    Dxx = 2.0 * Delta * avg_gxx
    Dyy = 2.0 * Delta * avg_gyy
    Dxy = 2.0 * Delta * avg_gxy
    return {"Dxx": Dxx, "Dyy": Dyy, "Dxy": Dxy}


# ----------------------------------------------------------------------
# Plotting for a single kz slice (optional)
# ----------------------------------------------------------------------

def plot_qgt_2d_slice(slice_data: dict, save_prefix: str = None):
    """
    Plot 2D distributions of gxx, gyy, gxy, Omega_xy for a single kz slice.

    Each quantity is plotted in its own figure. Called only when --plot is set.
    """
    import matplotlib.pyplot as plt

    kxs = slice_data["kxs"]
    kys = slice_data["kys"]
    kz = slice_data["kz"]
    extent = [kxs.min(), kxs.max(), kys.min(), kys.max()]

    fields = [
        ("gxx", slice_data["gxx"], r"Quantum metric $g_{xx}$"),
        ("gyy", slice_data["gyy"], r"Quantum metric $g_{yy}$"),
        ("gxy", slice_data["gxy"], r"Quantum metric $g_{xy}$"),
        ("Omega_xy", slice_data["Omega_xy"], r"Berry curvature $\Omega_{xy}$"),
    ]

    for key, arr, ttl in fields:
        plt.figure()
        plt.imshow(arr.T, origin="lower", extent=extent, aspect="equal")
        plt.colorbar(label=key)
        plt.xlabel("kx (fractional)")
        plt.ylabel("ky (fractional)")
        plt.title(f"{ttl} at kz = {kz}")
        if save_prefix is not None:
            out = f"{save_prefix}_{key}_kz{float(kz):.3f}.png"
            plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.show()


# ----------------------------------------------------------------------
# CLI wrapper (3D slice-based driver)
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Slice-based non-Abelian QGT from Wannier90 HR (3D, energy window [Emin, Emax])."
    )
    ap.add_argument("--hr", required=True, help="Path to wannier90_hr.dat")
    ap.add_argument("--nk", nargs=2, type=int, default=[31, 31],
                    help="nkx nky grid sizes for kx, ky")
    ap.add_argument("--nkz", type=int, default=1,
                    help="Number of kz slices (uniform in [0,1))")
    ap.add_argument("--Emin", type=float, required=True,
                    help="Lower edge of the energy window (same units as TB, e.g. eV)")
    ap.add_argument("--Emax", type=float, required=True,
                    help="Upper edge of the energy window (same units as TB, e.g. eV)")
    ap.add_argument("--Delta", type=float, required=True,
                    help="Pairing gap amplitude Delta (same energy units as TB)")

    ap.add_argument("--out", type=str, default="2D_slices_nonabelian_qgt.npz",
                    help="Output .npz file name")

    ap.add_argument("--plot", action="store_true",
                    help="If set, plot 2D maps for a chosen kz slice.")
    ap.add_argument("--plot-kz-idx", type=int, default=0,
                    help="Index (0..nkz-1) of the kz slice to plot/store.")
    ap.add_argument("--save-prefix", type=str, default='_',
                    help="If provided, save PNGs with this prefix for the slice.")

    args = ap.parse_args()



    nkx, nky = args.nk
    nkz = args.nkz
    Emin, Emax = args.Emin, args.Emax
    Delta = args.Delta

    if nkz <= 0:
        raise ValueError("nkz must be >= 1")
    if args.plot_kz_idx < 0 or args.plot_kz_idx >= nkz:
        raise ValueError(f"--plot-kz-idx must be in [0, {nkz-1}]")

    # Load HR once
    hr_path = args.hr
    HR = read_wannier90_hr(hr_path)
    input_path = hr_path.rsplit('/',1)[0]
    print('Input/Output file location : ', input_path)

    # kz grid
    kzs = np.linspace(0.0, 1.0, nkz, endpoint=False)

    # Accumulators for 3D BZ averages
    sum_gxx = 0.0
    sum_gyy = 0.0
    sum_gxy = 0.0
    total_kpts = nkx * nky * nkz

    # Track Nsub statistics
    global_Nsub_min = +np.inf
    global_Nsub_max = -np.inf

    # Slice to store/plot
    stored_slice = None

    for iz, kz in enumerate(kzs):
        plane_data = qgt_on_plane(
            HR=HR,
            nkx=nkx,
            nky=nky,
            kz=kz,
            Emin=Emin,
            Emax=Emax,
        )

        gxx = plane_data["gxx"]
        gyy = plane_data["gyy"]
        gxy = plane_data["gxy"]
        Nsub = plane_data["Nsub"]

        sum_gxx += gxx.sum()
        sum_gyy += gyy.sum()
        sum_gxy += gxy.sum()

        global_Nsub_min = min(global_Nsub_min, Nsub.min())
        global_Nsub_max = max(global_Nsub_max, Nsub.max())

        if iz == args.plot_kz_idx:
            stored_slice = plane_data  # keep one slice for plotting/saving

    avg_gxx_3d = sum_gxx / total_kpts
    avg_gyy_3d = sum_gyy / total_kpts
    avg_gxy_3d = sum_gxy / total_kpts

    D_geom = geometric_superfluid_weight_3d(
        avg_gxx=avg_gxx_3d,
        avg_gyy=avg_gyy_3d,
        avg_gxy=avg_gxy_3d,
        Delta=Delta,
    )

    # Prepare output dict
    out_data = {
        "kzs": kzs,
        "nkx": nkx,
        "nky": nky,
        "nkz": nkz,
        "Emin": Emin,
        "Emax": Emax,
        "Delta": Delta,
        "avg_gxx_3d": avg_gxx_3d,
        "avg_gyy_3d": avg_gyy_3d,
        "avg_gxy_3d": avg_gxy_3d,
        "Dxx_geom": D_geom["Dxx"],
        "Dyy_geom": D_geom["Dyy"],
        "Dxy_geom": D_geom["Dxy"],
        "Nsub_min": global_Nsub_min,
        "Nsub_max": global_Nsub_max,
    }

    if stored_slice is not None:
        # Store the 2D slice fields for convenience
        out_data["kxs"] = stored_slice["kxs"]
        out_data["kys"] = stored_slice["kys"]
        out_data["kz_slice"] = stored_slice["kz"]
        out_data["gxx_slice"] = stored_slice["gxx"]
        out_data["gyy_slice"] = stored_slice["gyy"]
        out_data["gxy_slice"] = stored_slice["gxy"]
        out_data["Omega_xy_slice"] = stored_slice["Omega_xy"]
        out_data["Nsub_slice"] = stored_slice["Nsub"]

    np.savez_compressed(input_path + '/' + args.out, **out_data)

    with open(input_path+'/2D_slices_Superfluid_weight.txt','w') as ff:
        ff.write(f"Saved 3D non-Abelian QGT info and geometric superfluid weight to {args.out}\n")
        ff.write(f"Keys in file: {list(out_data.keys())}\n")
        ff.write(f"Subspace dim Nsub(k): min ={global_Nsub_min}, max ={global_Nsub_max}\n")
        ff.write("3D BZ-averaged metric:\n")
        ff.write(f"  <gxx>_3D = {avg_gxx_3d}\n")
        ff.write(f"  <gyy>_3D = {avg_gyy_3d}\n")
        ff.write(f"  <gxy>_3D = {avg_gxy_3d}\n")
        ff.write("Geometric superfluid weight (flat-manifold approx):\n")
        ff.write(f"  Delta     = {Delta}\n")
        ff.write(f"  Dxx_geom  = {D_geom['Dxx']}\n")
        ff.write(f"  Dyy_geom  = {D_geom['Dyy']}\n")
        ff.write(f"  Dxy_geom  = {D_geom['Dxy']}\n")

    # Optional plotting for the stored slice
    if args.plot and stored_slice is not None:
        plot_qgt_2d_slice(stored_slice, save_prefix=input_path + '/2D_slices_' + args.save_prefix)


if __name__ == "__main__":
    main()
