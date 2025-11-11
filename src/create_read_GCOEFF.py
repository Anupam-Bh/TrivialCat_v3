import subprocess
import numpy as np
import os
from pymatgen.io.vasp import outputs
from pymatgen.core import Structure


## Create and read GCOEFF.txt file
def get_wavecar_band_data(exe_model_path):
    ##### Read Wavecar OUTCAR POSCAR
    create_GCOEFF(exe_model_path+'/WaveTrans.exe')
    wavecar = read_gcoeff()
    print(f'kpoints used in the DFT calculation:\n {wavecar.wk[0]}\n======================================\n')
    outcar = outputs.Outcar('OUTCAR')
    Fouriergrid = outcar.ngf
    fermi = outcar.efermi
    return wavecar, Fouriergrid, fermi
    
def create_GCOEFF(exe_path):
    print('Creating GCOEFF.txt ...')
    ### Runs the WaveTrans.exe . Path to the execulatble is given in the exe_path    
    try: 
        result = subprocess.run([exe_path], check=True, capture_output=True, text=True) 
        print("Output:", result.stdout)  # Print standard output 
        print("Errors:", result.stderr)    # Print standard error (if any) 
    except subprocess.CalledProcessError as e: 
        print("An error occurred while running the .exe file.") 
        print("Return Code:", e.returncode) 
        print("Output:", e.output) 
        print("Error Output:", e.stderr)
    print('------GCOEFF created------------\n')
    
def read_gcoeff():
    ## Reads GCOEFF.txt file in the current directory and returns Wavecar, 
    def vcross(b, c):
        """
        Replicates the Fortran vcross subroutine:
            a(1) = b(2)*c(3) - b(3)*c(2)
            a(2) = -(b(1)*c(3) - b(3)*c(1))
            a(3) = b(1)*c(2) - b(2)*c(1)
        """
        return np.array([
            b[1]*c[2] - b[2]*c[1],
            -(b[0]*c[2] - b[2]*c[0]),
            b[0]*c[1] - b[1]*c[0]
        ], dtype=float)
    
    #def wavefcnplot():
    # Read the entire GCOEFF.txt file into memory
    print('=================================== \nReading GCOEFF.txt ....               \n')
    with open('GCOEFF.txt', 'r') as f:
        lines = f.readlines()
    
    # A simple line counter to walk through the file
    line_index = 0
    
    # Read nspin, nwk, nband
    nspin = int(lines[line_index].strip()); line_index += 1
    nwk   = int(lines[line_index].strip()); line_index += 1
    nband = int(lines[line_index].strip()); line_index += 1
    
    print(f'no. spin values = {nspin}')
    print(f'no. k points    = {nwk}')
    print(f'no. bands       = {nband}')
    
    # Fortran code checks dimension limits. We hardcode some “max” dims
    # to replicate the Fortran logic. In practice, you might dynamically size.
    # nsdim = 1
    # nwdim = 15
    # nbdim = 120
    nsdim = nspin
    nwdim = nwk
    nbdim = nband
    npdim = 20000
    
    if nspin > nsdim:
        raise ValueError("*** error - nsdim < nspin. Increase nsdim and re-run.")
    if nwk > nwdim:
        raise ValueError("*** error - nwdim < nwk. Increase nwdim and re-run.")
    if nband > nbdim:
        raise ValueError("*** error - nbdim < nband. Increase nbdim and re-run.")
    
    # Read the real-space lattice vectors a1, a2, a3
    a1 = np.array(list(map(float, lines[line_index].split())), dtype=float); line_index += 1
    a2 = np.array(list(map(float, lines[line_index].split())), dtype=float); line_index += 1
    a3 = np.array(list(map(float, lines[line_index].split())), dtype=float); line_index += 1
    
    print(f'real space lattice vectors:')
    print(f'a1 = {a1}')
    print(f'a2 = {a2}')
    print(f'a3 = {a3}\n')
    
    # Compute volume of the unit cell
    vtmp = vcross(a2, a3)  # cross product of a2, a3
    Vcell = a1[0]*vtmp[0] + a1[1]*vtmp[1] + a1[2]*vtmp[2]
    print(f'volume unit cell = {Vcell}\n')
    
    # Read the reciprocal lattice vectors b1, b2, b3
    b1 = np.array(list(map(float, lines[line_index].split())), dtype=float); line_index += 1
    b2 = np.array(list(map(float, lines[line_index].split())), dtype=float); line_index += 1
    b3 = np.array(list(map(float, lines[line_index].split())), dtype=float); line_index += 1
    
    print(f'reciprocal lattice vectors:')
    print(f'b1 = {b1}')
    print(f'b2 = {b2}')
    print(f'b3 = {b3}\n')
    print('reading plane wave coefficients...')
    
    # We need data structures for:
    #   wk(3, nwdim, nsdim)
    #   cener(nbdim, nwdim, nsdim)
    #   occ(nbdim, nwdim, nsdim)
    #   nplane(nwdim, nsdim)
    #   igall(3, npdim, nwdim, nsdim)
    #   coeff(npdim, nbdim, nwdim, nsdim)
    #
    # In Python, let's store them as:
    wk     = np.zeros((nspin, nwk, 3),              dtype=float)                      ## Kpoints
    cener  = np.zeros((nspin, nwk, nband),          dtype=complex)  # complex*16      ##Energy eigenvalues
    occ    = np.zeros((nspin, nwk, nband),          dtype=float)                      ## Occupation
    nplane = np.zeros((nspin, nwk),                 dtype=int)                        ## number of Plane waves
    igall  = np.zeros((nspin, nwk, npdim, 3),       dtype=int)                        ## G-values
    coeff  = np.zeros((nspin, nwk, nband, npdim),   dtype=complex)  # complex*8 mapped to np.complex128         ## Plane wave coefficients
    
    # Loop over spin, k, band, plane wave coefficients
    for ispin in range(nspin):
        for iwk in range(nwk):
            print(f'k point #{iwk+1}')
            
            # read wk(1..3, iwk, ispin)
            parts = lines[line_index].split()
            wk[ispin, iwk, :] = [float(parts[0]), float(parts[1]), float(parts[2])]
            line_index += 1
            
            for iband in range(nband):
                # read "itmp, nplane" line
                # The Fortran code: read(11,*) itmp, nplane(iwk,ispin)
                parts = lines[line_index].split()
                itmp  = int(parts[0])
                npln  = int(parts[1])
                nplane[ispin, iwk] = npln
                line_index += 1
                
                # read cener(iband, iwk, ispin) and occ(iband, iwk, ispin)
                parts = lines[line_index].split()
                real_cener = float(parts[1])
                imag_cener = float(parts[3])  # if it’s stored as a complex # in file
                # If cener is real in the file, you might only have one float.
                # This example assumes 2 floats for the complex center. Adjust if needed.
                cener[ispin, iwk, iband] = complex(real_cener, imag_cener)
                occ[ispin, iwk, iband]   = float(parts[5]) if len(parts) > 2 else 0.0
                line_index += 1
                
                # read plane-wave G-vectors + coefficients
                for iplane in range(npln):
                    parts = lines[line_index].split()
                    # igall(3, iplane, iwk, ispin)
                    # Fortran: read(11,*) (igall(j,iplane,iwk,ispin), j=1,3), coeff(iplane,iband,iwk,ispin)
                    ig1 = int(parts[0])
                    ig2 = int(parts[1])
                    ig3 = int(parts[2])
                    # Next part is the plane-wave coefficient (real+imag?)
                    # For example: "0.123 0.456" => complex(0.123,0.456)
                    real_coeff = float(parts[4])
                    imag_coeff = float(parts[6]) if len(parts) > 4 else 0.0
                    igall[ispin, iwk, iplane, :] = [ig1, ig2, ig3]
                    coeff[ispin, iwk, iband, iplane] = complex(real_coeff, imag_coeff)
                    line_index += 1
    
    print('GCOEFF reading complete------------------------------------\n===================================================\n') 
    params = wavecar_params(nsdim, nwdim, nbdim, npdim, a1,a2,a3, b1,b2,b3, wk, cener, occ,nplane, igall, coeff)
    os.remove('GCOEFF.txt')
    return params
    
class wavecar_params:
    def __init__(self,nsdim, nwdim, nbdim, npdim, a1,a2,a3, b1,b2,b3, wk, cener, occ,nplane, igall, coeff):
        self.nsdim = nsdim
        self.nwdim = nwdim
        self.nbdim = nbdim
        self.npdim = npdim
        self.a1 = a1 
        self.a2 = a2
        self.a3 = a3
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.wk = wk
        self.cener = cener
        self.occ = occ
        self.nplane = nplane
        self.igall = igall
        self.coeff = coeff
