import sys
import pyscf
import scipy
from pyscf import gto, scf
from pyscf import lib, lo
from pyscf.tools import cubegen
import imp
import numpy as np
from scipy.constants import physical_constants

print(imp.find_module('pyscf'))

(a0, unit, uncertainty) = physical_constants['Bohr radius'] # in m

a0 = a0*1e10 # convert to Ang.
d  = 1.4  # Bohr
d  = d*a0 # Ang
print('d =',d)
# implicit STO-3G basis set
mol = gto.M(atom=[["H", 0., 0., 0.],
                  ["H", 0., 0., d ]],
            #charge = 1,
            basis='sto-3g')
mol.verbose = 20
mf = scf.RHF(mol)
mf.kernel()
mf.analyze()

# explicit STO-3G basis set
mol = gto.M(atom=[["H", 0., 0., 0.],
                  ["H", 0., 0., d ]], 
            basis = {'H': gto.parse('''
#from https://www.basissetexchange.org
#BASIS "ao basis" SPHERICAL PRINT
#BASIS SET: (3s) -> [1s]
H    S
      0.3425250914E+01       0.1543289673E+00
      0.6239137298E+00       0.5353281423E+00
      0.1688554040E+00       0.4446345422E+00
                                ''')})
mol.verbose = 20 
mf = scf.RHF(mol)
mf.kernel()
mf.analyze()

#by making a transformation
C = lo.orth_ao(mf, 'lowdin')
orbs = C[:,mf.mo_occ>0] # Only get occupied orbitals
#orbs = C
#print(orbs)
#for i in range(orbs.shape[1]):
#    cubegen.orbital(mol, f'H2_lowdin_mo{i+1}.cube', orbs[:,i])
#but transformation not necessary

#by using directly mo_coeff
for i in range(orbs.shape[1]):
    cubegen.orbital(mol, f'H2_lowdin_mo{i+1}.cube', mf.mo_coeff[:,i])
print('mo_coeff =',mf.mo_coeff)
#we check if c (the matrix of the coef) is the same as mo_coeff
e, c = scipy.linalg.eigh(F, S)
print('Check if mo_coeff = c')
print('matrix of the coeff=',c)
