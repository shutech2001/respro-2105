# $ python3 exercise_A.py -out exercise_A_Ibuprofen.png

import argparse

from rdkit import Chem
from rdkit.Chem import Draw

def Draw_mol(smiles, output_file_name):
    """
    draw structural formula

    Parameters
        smiles : molecule as SMILES type
        output_file_name : output file name
    """

    # make molecule from SMILES
    mol = Chem.MolFromSmiles(smiles)
    # draw molecule and output .png file
    Draw.MolToFile(mol, output_file_name)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Draw structural formula")
    parser.add_argument("-out", help="path to structural formula image")
    args = parser.parse_args()
    if args.out is None:
        print(parser.print_help())
        exit(1)

    ibuprofen_smiles = 'CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O'
    Draw_mol(ibuprofen_smiles, args.out)