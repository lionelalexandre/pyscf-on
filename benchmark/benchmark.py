import sys
import pyscf
from pyscf import gto, scf
from pyscf import lib
import numpy
import importlib.util
print(importlib.util.find_spec('pyscf'))

# List of molecule xyz files
molecules = ['benzene.xyz', 'naphtalene.xyz', 'anthracene.xyz', 'tetracene.xyz', 'pentacene.xyz', 'hexacene.xyz', 'heptacene.xyz', 'octacene.xyz', 'nonacene.xyz']#,decacene.xyz]

# Basis set and convergence settings
basis_set = '6-31G'
conv_tolerance = 1e-10

# Loop over each molecule
for mol_file in molecules:
    print(f"\nProcessing molecule: {mol_file}")

    mol = gto.Mole()
    mol.atom = open(mol_file).read()  # Read atomic positions
    mol.basis = basis_set
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
    print(f'e_tot_rhf for {mol_file}: {e_tot_rhf:.6f}')
    print(f'e_tot_rhf_dm for {mol_file}: {e_tot_rhf_dm:.6f}')
    print(f'Matrix size (NAO): {mol.nao_nr()}')
     
