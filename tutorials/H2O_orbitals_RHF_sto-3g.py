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

# implicit STO-3G basis set
mol = gto.M(atom=[["O", 0.000000, 0.000000, 0.1177900],
                  ["H", 0.000000, 0.755453, -0.4711610],
                  ["H", 0.000000, -0.755453, -0.471161 ]],
            symmetry= 1,
            basis='sto-3g')
mol.build()
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
    cubegen.orbital(mol, f'H2O_lowdin_mo{i+1}.cube', mf.mo_coeff[:,i])
print(mf.mo_coeff)
#we check if c (the matrix of the coef) is the same as mo_coeff
e, c = scipy.linalg.eigh(F, S)
print(c)