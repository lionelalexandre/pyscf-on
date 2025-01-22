#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run HF calculation.

.kernel() function is the simple way to call HF driver.
.analyze() function calls the Mulliken population analysis etc.
'''
import sys
import pyscf
from pyscf import gto, scf
from pyscf import lib
import imp
import numpy
print(imp.find_module('pyscf'))

#log = lib.logger.Logger(sys.stdout, 9)

#mol = pyscf.M(
#    atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
#    basis = 'ccpvdz',
#    symmetry = True,
#    verbose = 20,
#)

#myhf = mol.HF()
#myhf.kernel()
#myhf.analyze()

# No symmmetry
#mol.symmetry = False

# print overlap matrix
#print(mol.intor_symmetric('int1e_ovlp'))

# get hcore (2 ways)
#print(mol.intor_symmetric('int1e_kin')+ mol.intor_symmetric('int1e_nuc'))
#print(scf.my_hf.get_hcore(mol))

#myhf = mol.my_HF(max_cycle = 0)
#myhf.kernel()
#myhf.analyze()

#mol = gto.M(atom="my_molecule.xyz")

#mol.symmetry = True
#myhf = mol.my_HF()
#myhf.kernel()
#myhf.analyze()

# Orbital energies, Mulliken population etc.
#myhf.analyze()


#
# myhf object can also be created using the APIs of gto, scf module
#
from pyscf import gto, scf

mol = gto.Mole()
mol.atom = open('glycine.xyz').read()
mol.basis = 'ccpvtz'
mol.verbose = 20
mol.build()
#myhf = mol.HF_DM(max_cycle = 0)
#print('##########',myhf)
#myhf.kernel()
mf = scf.RHF(mol).set(conv_tol=1e-10,conv_check=True)
mf.kernel(dmp_scf=True)
e_tot_rhf = mf.energy_tot()

mf = scf.RHF_DM(mol).set(conv_tol=1e-10,conv_check=True)
mf.kernel(dmp_scf=True)
e_tot_rhf_dm = mf.energy_tot()
#myf = scf.hf_dm.SCF(mol).set(conv_tol=1e-8,conv_check=True)
#myf.max_cycle = 30
#myf.kernel(dmp_scf=True)

mol.charge = 1
mol.spin = 3
mf = scf.UHF(mol).set(conv_tol=1e-10,conv_check=True)
mf.kernel(dmp_scf=True)
e_tot_uhf = mf.energy_tot()

mol.charge = 1
mol.spin = 3
mf = scf.UHF_DM(mol).set(conv_tol=1e-10,conv_check=True)
mf.kernel(dmp_scf=True)
e_tot_uhf_dm = mf.energy_tot()

mol.charge = 1
mol.spin = 3
mf = scf.ROHF(mol).set(conv_tol=1e-10,conv_check=True)
mf.kernel()
e_tot_rohf = mf.energy_tot()

mol.charge = 1
mol.spin = 3
mf = scf.ROHF_DM(mol).set(conv_tol=1e-10,conv_check=True)
mf.kernel()
e_tot_rohf_dm = mf.energy_tot()

print('e_tot_rhf    ',e_tot_rhf)
print('e_tot_rhf_dm ',e_tot_rhf_dm)
print('e_tot_uhf    ',e_tot_uhf)
print('e_tot_uhf_dm ',e_tot_uhf_dm)
print('e_tot_rohf   ',e_tot_rohf)
print('e_tot_rohf_dm',e_tot_rohf_dm)

#print(myf.conv_tol_grad)
#print(myf.e_tot)
#print(myf.mo_energy)
#print(myf.mo_coeff)
#print(myf.mo_occ)

#myf._finalize()

#myhf.analyze()
#mol = gto.M(
#    atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
#    basis = 'ccpvdz',
#    symmetry = True,
#)
#myhf = scf.HF(mol)
#myhf.kernel()
#myhf.analyze()
#myhf.energy_elec()

#mol.symmetry = False
#myhf = scf.my_HF(mol)
#myhf.kernel()
#myhf.analyze()
#myhf.energy_elec()

#mol.symmetry = True
#myhf = scf.my_HF(mol)
#myhf.kernel()
#myhf.analyze()
#myhf.energy_elec()
