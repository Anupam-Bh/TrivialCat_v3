import numpy as np
import re

## Calculate psi, rho along a given path in real space  ## also finds kpoint weights from OUTCAR
def charge_density_along_a_path(path, Emin, Emax ,n_samp_pts, wavecar,Fouriergrid):

    ispin_indices = np.arange(wavecar.nsdim)
    iwk_indices = np.arange(wavecar.nwdim)    ## All Kpoints are included
    iband_indices = np.arange(wavecar.nbdim)    ## choosing band indices 11-24
    
    kpts_list_with_weights = read_kpoint_weights_from_outcar('OUTCAR')
    weights = [w[3] for w in kpts_list_with_weights]
    
    psi_s_n_k_r = np.zeros((n_samp_pts, wavecar.nsdim,wavecar.nbdim,wavecar.nwdim),dtype= complex)
    rho_s_n_r = np.zeros((n_samp_pts, wavecar.nsdim,wavecar.nbdim),dtype=float)
    rho_s_r = np.zeros((n_samp_pts,wavecar.nsdim),dtype=float)
    rho_r = np.zeros((n_samp_pts),dtype=float)
    
    abspsi_s_n_r = np.zeros((n_samp_pts, wavecar.nsdim,wavecar.nbdim),dtype=complex)
    abspsi_s_r = np.zeros((n_samp_pts,wavecar.nsdim),dtype=complex)
    abspsi_r = np.zeros((n_samp_pts),dtype=complex)
    
    ## Getting real space psi_nk(r)
    for num, r in enumerate(path):
        x,y,z = r
        for ispin_index in ispin_indices:
            for iband_index in iband_indices:
                for iwk_index in iwk_indices:
                    if Emin<= wavecar.cener[ispin_index,iwk_index,iband_index] <= Emax :    ## only for bands in [Emin,Emax]
                    # if iwk_index==0 and iband_index==0 and ispin_index==0:                ## for specific band and kpt index
                        csum = 0.0 + 0.0j
                        ## Sum over all plane waves
                        npln = wavecar.nplane[ispin_index, iwk_index]
                        for iplane in range(npln):
                            ig1, ig2, ig3 = wavecar.igall[ispin_index, iwk_index, iplane, :]
                            # sumkg(j) = (wk(1,iwk,ispin)+ig1)*b1(j) + ...
                            sumkg = ( (wavecar.wk[ispin_index, iwk_index, 0] + ig1) * wavecar.b1 +
                                      (wavecar.wk[ispin_index, iwk_index, 1] + ig2) * wavecar.b2 +
                                      (wavecar.wk[ispin_index, iwk_index, 2] + ig3) * wavecar.b3 )
                            phase = sumkg[0]*x + sumkg[1]*y + sumkg[2]*z
                            csum += wavecar.coeff[ispin_index, iwk_index, iband_index, iplane] * np.exp(1j * phase)
                            #if ispin_index==0 and iwk_index==0 and iband_index==0:
                                #trial.append(coeff[ispin_index, iwk_index, iband_index, iplane] * np.exp(1j * phase))
                                #print(coeff[ispin_index, iwk_index, iband_index, iplane] * np.exp(1j * phase))
                        
                        Vcell = np.dot(wavecar.a1,np.cross(wavecar.a2,wavecar.a3))
                        csum /= np.sqrt(Vcell)
                        psi_s_n_k_r[num, ispin_index,iband_index,iwk_index] = csum
    
                        ## Summing psi_snk(r) to get rho_sn(r)
                        rho_s_n_r[num, ispin_index,iband_index] += wavecar.occ[ispin_index,iwk_index,iband_index]* np.real(np.multiply(np.conj(csum), csum))*weights[iwk_index]
                        abspsi_s_n_r[num, ispin_index,iband_index] += abs(csum)  *wavecar.occ[ispin_index,iwk_index,iband_index] *weights[iwk_index]
                ## Summing rho_sn(r) to get rho_s(r)
                rho_s_r[num, ispin_index] +=  rho_s_n_r[num, ispin_index,iband_index]
                abspsi_s_r[num, ispin_index] += abspsi_s_n_r[num, ispin_index,iband_index]
            ## Summing the spin channels: rho_s(r) to get rho(r)
            rho_r[num] += rho_s_r[num, ispin_index]
            abspsi_r[num] += abspsi_s_r[num, ispin_index]
    
    print ('  x       y        z      rho(r)      CHGCARvalue      Sum(abs(wavfunc)) \n -------------------------------------------------------------------')
    for num, r in enumerate(path):
        print(f' {r[0]:.4f}    {r[1]:.4f}    {r[2]:.4f}    {rho_r[num]:.4f}     {(rho_r[num]*Fouriergrid[0]*Fouriergrid[1]*Fouriergrid[2]):.4f}     {abspsi_r[num]} ')
    print('\n')
    return [rho_r , rho_s_n_r, psi_s_n_k_r]

def read_kpoint_weights_from_outcar(outcar_path):
    """
    Reads k-point coordinates and weights from the VASP OUTCAR file.
    Returns a list of tuples: [(kx, ky, kz, weight), ...].
    """
    kpoints = []
    inside_kpoint_block = False

    # Regex to match a line with exactly four floats
    float4_pattern = re.compile(r"^\s*([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s*$")

    with open(outcar_path, "r") as f:
        for line in f:
            # 1) Look for the heading that indicates we are entering the k-point section
            if "k-points in reciprocal lattice and weights" in line:
                inside_kpoint_block = True
                continue  # Go to the next line

            # 2) If we are inside the kpoint block, try to parse lines containing 4 floats
            if inside_kpoint_block:
                # Stop if we hit an empty line or a line that doesn't match "4 floats"
                if not line.strip():
                    # Blank line -> end of block
                    break
                match = float4_pattern.match(line)
                if match:
                    # Extract floats from capturing groups
                    kx = float(match.group(1))
                    ky = float(match.group(2))
                    kz = float(match.group(3))
                    wt = float(match.group(4))
                    kpoints.append((kx, ky, kz, wt))
                else:
                    # If it doesn't match the 4-float pattern, we assume we're out of the block
                    break
    return kpoints