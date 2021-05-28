# $ python3 exercise_F.py -train SM.csv -test CP.csv

import argparse

import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from exercise_A import Draw_mol
from exercise_C import RDKit_2D_descriptors, get_2D_desc
from exercise_D import calc_lnka

parser = argparse.ArgumentParser(description="construct regression model from Small molecule data")
parser.add_argument("-train", help="path to train csv data")
parser.add_argument("-test", help="path to test csv data")
args = parser.parse_args()
if args.train is None:
    print(parser.print_help())
    exit(1)
if args.test is None:
    print(parser.print_help())
    exit(1)

##### 1. data preparation #####

# train data set
# read .csv
df = pd.read_csv(args.train)
# make explanatory variable
smiles = df['SMILES'].values
# apply compute_2D_desc to each molecule
X_train = np.array([RDKit_2D_descriptors(mol).compute_2D_desc() for mol in smiles])
# make response variable
# apply calc_lnka to each PPB
y_train = df['PPB (fb)'].apply(calc_lnka).values

# test data set
# read .csv
df = pd.read_csv(args.test)
# make explanatory variable
smiles = df['SMILES'].values
# apply compute_2D_desc to each molecule
X_test = np.array([RDKit_2D_descriptors(mol).compute_2D_desc() for mol in smiles])

# # compute_2D_desc_v2 is updated version (Ipc modified)
# X_test = np.array([RDKit_2D_descriptors(mol).compute_2D_desc_v2() for mol in smiles])

# make response variable
# apply calc_lnka to each PPB
y_test = df['PPB(fb)'].apply(calc_lnka).values

# Standardization of explanatory variables
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

##### 2. construct lasso regression model #####
reg = Lasso(alpha = 0.010001550247043559,
            max_iter = 18633,
            tol = 2.0186369425523876e-05,)

reg.fit(X_train, y_train)

##### 3. prediction Cyclic peptide drug #####

# predict (input : CP data)
y_pred = reg.predict(X_test)
# calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# calculate correlation coefficient
r = np.corrcoef(y_test, y_pred)[0,1]

print('RMSE (ln(K_a))')
print(rmse)
print('R (ln(K_a))')
print(r)
