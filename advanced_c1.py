# $ python3 advanced_c1.py -small_mol SM.csv -cyclic_pep CP.csv

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
from exercise_C import RDKit_2D_descriptors
from exercise_D import calc_lnka

import mordred
from mordred import Calculator, descriptors

# class DescCalculator:
#     def __init__(self, smiles):
#         self.mols = [Chem.MolFromSmiles(i) for i in smiles]
#         self.smiles = smiles

#     def compute_desc(self, ignore_3D=True):
#         desc_value_vec = []
#         desc_values = []
#         calc = Calculator(descriptors, ignore_3D=ignore_3D)
#         for i in tqdm(range(len(self.mols))):
#             desc_dict = calc(self.mols[i])
#             for value in desc_dict:
#                 if type(value) == mordred.error.Missing:
#                     continue
#                 elif type(value) == mordred.error.Error:
#                     continue
#                 elif type(value) == bool:
#                     if value == True:
#                         value = 1
#                     else:
#                         value = 0
#                 desc_values.append(value)
#             desc_value_vec.append(desc_values)

#         return np.array(desc_value_vec)

parser = argparse.ArgumentParser(description="construct regression model from Small molecule data")
parser.add_argument("-small_mol", help="path to small molecule csv data")
parser.add_argument("-cyclic_pep", help="path to cyclic peptide drug csv data")
args = parser.parse_args()
if args.small_mol is None:
    print(parser.print_help())
    exit(1)
if args.cyclic_pep is None:
    print(parser.print_help())
    exit(1)

##### 1. data preparation #####

# train data set
# read .csv
df = pd.read_csv(args.small_mol)
# make explanatory variable
smiles = df['SMILES'].values
# apply compute_2D_desc to each molecule
X_sm = np.array([RDKit_2D_descriptors(mol).compute_2D_desc_v2() for mol in smiles])
# make response variable
# apply calc_lnka to each PPB
y_sm = df['PPB (fb)'].apply(calc_lnka).values
# test data set
# read .csv
df = pd.read_csv(args.cyclic_pep)
# make explanatory variable
smiles = df['SMILES'].values
# apply compute_2D_desc to each molecule
X_cp = np.array([RDKit_2D_descriptors(mol).compute_2D_desc_v2() for mol in smiles])
# make response variable
# apply calc_lnka to each PPB
y_cp = df['PPB(fb)'].apply(calc_lnka).values

##### 2. construct regression model (with changing input)#####

while True:
    # calculate std 
    std_array = np.std(X_sm, axis=0)

    ##### remove if std == 0 #####
    # storing remove row number
    rm_row = []
    for idx, std in enumerate(std_array):
	    if std == 0:
		    rm_row.append(idx)
    # reversed loop for preventing gap
    for row in reversed(rm_row):
	    X_sm = np.delete(X_sm, row, 1)
	    X_cp = np.delete(X_cp, row, 1)

    # Standardization of explanatory variables
    sc = StandardScaler()
    sc.fit(X_sm)
    X_sm = sc.transform(X_sm)
    X_cp = sc.transform(X_cp)

    ##### construct RF regression model #####
    reg = RFR(n_estimators = 460,
	    min_samples_split=2,
	    min_samples_leaf=1,
	    max_features=0.6400954926671139,
	    max_leaf_nodes=575,
	    ccp_alpha=1.0658736175592873e-05,
	    warm_start=True)
    reg.fit(X_sm, y_sm)

    ##### remove lower important variables #####
    importances = reg.feature_importances_
    # storing remove row number
    rm_row = []
    for idx, value in enumerate(importances):
    	# setting threshold 0.01
	    if value < 0.01:
		    rm_row.append(idx)
	# reversed loop for preventing gap
    for row in reversed(rm_row):
	    X_sm = np.delete(X_sm, row, 1)
	    X_cp = np.delete(X_cp, row, 1)

	# all variable importances greater than 0.01, then stop loop
    if len(rm_row) == 0:
	    break

##### 3. prediction Cyclic peptide drug #####
# predict (input : CP data)
y_pred = reg.predict(X_cp)

##### 4. evaluate prediction accuracy #####
# calculate RMSE
rmse = np.sqrt(mean_squared_error(y_cp, y_pred))
# calculate correlation coefficient
r = np.corrcoef(y_cp, y_pred)[0,1]
print('RMSE (ln(K_a))')
print(rmse)
print('R (ln(K_a))')
print(r)
