import sys
import pyscf
from pyscf import gto, scf
from pyscf import lib
import numpy
import importlib.util
import os
print(importlib.util.find_spec('pyscf'))

# List of basis
basis_list = 'basis_list.list'

# Open basis_list and store each basis name
with open(basis_list, 'r') as f:
    basis = [line.strip() for line in f]  # Read lines and remove whitespace

#convergence settings
conv_tolerance = 1e-10

# Directory for molecule files
molecule_dir = 'xyz'

# Loop over each basis
for basis in basis:
    print("Processing basis:", basis)

    mol = gto.Mole()
    molecule_path = os.path.join(molecule_dir, 'benzene.xyz')
    mol.atom = open(molecule_path).read()
    mol.basis = basis # Read basis
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
    print(f'number of contracted GTO: {mol.nao_nr()}')
    print(f'number of occupied states: {sum(mf.mo_occ > 0)}')
    print('filling factor: ', sum(mf.mo_occ > 0)/mol.nao_nr())