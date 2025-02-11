import sys
import pyscf
from pyscf import gto, scf
from pyscf import lib
import numpy
import importlib.util
print(importlib.util.find_spec('pyscf'))

# List of basis
basis_list = 'basis.list'

# Open basis_list
basis_list = [ ]
f = open(basis_list,'r')

#convergence settings
conv_tolerance = 1e-10

# Loop over each molecule
for basis in basis_list:
    print(f"\nProcessing basis: {basis}")

    mol = gto.Mole()
    mol.atom = open('anthracene.xyz').read()
    mol.basis = open(basis).read() # Read basis
    mol.verbose = 4 
    mol.build()

    # diagonalization
    mf = scf.RHF(mol).set(conv_tol=conv_tolerance, conv_check=True)
    mf.kernel(dmp_scf=False)
    e_tot_rhf = mf.energy_tot()

    # purification
    mf = scf.RHF_DM(mol).set(conv_tol=conv_tolerance, conv_check=True)
    mf.kernel(dmp_scf=True)
    e_tot_rhf_dm = mf.energy_tot()

    # Print results
    print(f'e_tot_rhf for {basis}: {e_tot_rhf:.6f}')
    print(f'e_tot_rhf_dm for {basis}: {e_tot_rhf_dm:.6f}')
    print(f'Matrix size (NAO): {mol.nao_nr()}')
     
