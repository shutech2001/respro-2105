# $ python3 exercise_C.py

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

class RDKit_2D_descriptors:
    def __init__(self, smiles):
        self.mols = Chem.MolFromSmiles(smiles)
        self.smiles = smiles

    def compute_2D_desc(self):
        # transform name to calculator
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors.descList])
        # calculate all 2D descroptors
        ds = calc.CalcDescriptors(self.mols)
        # transform tuple to numpy array (for after exercises)
        return np.array(ds)

    def compute_2D_desc_v2(self):
        # transform name to calculator
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors.descList])
        # calculate all 2D descroptors
        ds = calc.CalcDescriptors(self.mols)
        # tuple value cannot change
        ds = list(ds)
        # descList[40] == 'Ipc'
        # prevent larger value
        ds[40] = Descriptors.Ipc(self.mols, avg=True)
        # transform tuple to numpy array (for after exercises)
        # ds = [ds[1], ds[17], ds[23], ds[26], ds[39], ds[55], ds[60], ds[65], ds[66], ds[77], ds[99], ds[121], ds[122]]
        return np.array(ds)

def get_2D_desc():
  	# transform name to calculator
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors.descList])
    # store descriptor names
    return calc.GetDescriptorNames()

if __name__ == '__main__':
    # validate RDKit_2D_descroptors class 
    ibuprofen_smiles = 'CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O'
    rdkit_2d_desc = RDKit_2D_descriptors(ibuprofen_smiles)
    print(f'the number of 2D_descroptors is {len(rdkit_2d_desc.compute_2D_desc())}')