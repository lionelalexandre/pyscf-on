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
mol = gto.M(atom=[["C", 0.00000000  ,  0.00000000  ,  0.000000000],
                  ["C", 0.00000000  ,  0.00000000  ,  1.395160000],
                  ["C", 1.20775100  ,  0.00000000  ,  2.09269800],
                  ["C", 2.41626000  , -0.00119900  ,  1.39504400],
                  ["C", 2.41618200  , -0.00167800  ,  0.00021900],
                  ["C", 1.20797600  , -0.00068200  , -0.69738200],
                  ["H", -0.95231700  ,  0.00045000  , -0.54975900],
                  ["H", -0.95251300  ,  0.00131500  ,  1.94466800],
                  ["H", 1.20783100  ,  0.00063400  ,  3.19237800],
                  ["H", 3.36840300  , -0.00125800  ,  1.94524400],
                  ["H", 3.36846300  , -0.00263100  , -0.54990300],
                  ["H", 1.20815900  , -0.00086200  , -1.79698600]],
            basis='sto-3g')
mol.build()
mol.verbose = 2
mf = scf.RHF(mol)
mf.kernel()
mf.analyze()

#geometric optimization
from pyscf.geomopt.geometric_solver import optimize
#from pyscf.geomopt.berny_solver import optimize
mol_eq = optimize(mf, maxsteps=100)
print(mol_eq.tostring())

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
    cubegen.orbital(mol, f'benzene_opt_lowdin_mo{i+1}.cube', mf.mo_coeff[:,i])
print(mf.mo_coeff)
#we check if c (the matrix of the coef) is the same as mo_coeff
#e, c = scipy.linalg.eigh(F, S)
#print(c)