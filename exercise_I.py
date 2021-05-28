# $ python3 exercise_I.py -small_mol SM.csv -cyclic_pep CP.csv

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from exercise_A import Draw_mol
from exercise_C import RDKit_2D_descriptors, get_2D_desc
from exercise_D import calc_lnka

import optuna

def objective(trial):
    """
    search hyper parameter based on RMSE value (5-fold CV)
    """

    # number of trees
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    # min samples for dividing node
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    # min samples for making leaf
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    # num of features (if overfitting, recommend to decrease)
    max_features = trial.suggest_loguniform('max_features', 0.5, 1.)
    # max of node (if overfitting, recommend to decrease)
    max_leaf_nodes =  trial.suggest_int('max_leaf_nodes', 100, 1000)
    # cost-complexity-pruning (if overfitting, recommend to increase)
    ccp_alpha = trial.suggest_loguniform('ccp_alpha', 1e-6, 1e-1)

    reg = RFR(n_estimators=n_estimators,
    	      min_samples_split=min_samples_split,
    	      min_samples_leaf=min_samples_leaf,
    	      max_features=max_features,
    	      max_leaf_nodes=max_leaf_nodes,
    	      oob_score=True,
    	      ccp_alpha=ccp_alpha,
              n_jobs=-1,
              random_state=0)
    
    rmse_list = cross_val_score(reg, X_sm, y_sm, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)
    # convert neg_rmse value to positive 
    return - np.array(rmse_list).mean()


if __name__ == "__main__":

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

    ##### exercise E step #####

    ##### 1. data preparation #####
    # read SM.csv
    df = pd.read_csv(args.small_mol)
    # make explanatory variable
    smiles = df['SMILES'].values
    # apply compute_2D_desc to each molecule
    X_sm = np.array([RDKit_2D_descriptors(mol).compute_2D_desc() for mol in smiles])
    # make response variable
    # apply calc_lnka to each PPB
    y_sm = df['PPB (fb)'].apply(calc_lnka).values
    # Standardization of explanatory variables
    sc = StandardScaler()
    X_sm = sc.fit_transform(X_sm)

    ##### 2. search hyper parameters using optuna #####
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    # output result of searching hyper parameters
    print('Random Forest Regressor : Best Parameters')
    for key, value in study.best_params.items():
    	print(f'{key} = {value},')
    print('==================================================')

    ##### 3. output best parameters' result #####
    reg = RFR(**study.best_params)
    # 5-fold cross validation 
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    # store each RMSE value and R value
    RMSE = []
    R = []
    for tr_index, val_index in kf.split(X_sm, y_sm):
    	# split train data and validation data
    	X_tr, X_val = X_sm[tr_index], X_sm[val_index]
    	y_tr, y_val = y_sm[tr_index], y_sm[val_index]
    	reg.fit(X_tr, y_tr)
    	# validate regressor
    	y_pr = reg.predict(X_val)
    	# root-MSE
    	RMSE.append(np.sqrt(mean_squared_error(y_val, y_pr)))
    	# not diagonal element of variance-covariance matrix
    	R.append(np.corrcoef(y_val, y_pr)[0,1])
    print('RMSE (ln(K_a))')
    print(RMSE)
    print('R (ln(K_a))')
    print(R)

    # ====== * ===== * ===== * ===== #

    ##### exercise F step #####
    
    ##### 4. prepare cyclic peptide drug data #####

    # read CP.csv
    # the same variable for saving memory
    df = pd.read_csv(args.cyclic_pep)
    # make explanatory variable
    smiles = df['SMILES'].values
    # apply compute_2D_desc to each molecule
    X_cp = np.array([RDKit_2D_descriptors(mol).compute_2D_desc() for mol in smiles])
    # make response variable
    # apply calc_lnka to each PPB
    y_cp = df['PPB(fb)'].apply(calc_lnka).values
    # apply StandardScaler
    X_cp = sc.transform(X_cp)

    # train with small molecule data
    reg = RFR(**study.best_params)
    reg.fit(X_sm, y_sm)
    # prediction Cyclic peptide drug
    y_pred = reg.predict(X_cp)
    # calculate performance and output
    rmse = np.sqrt(mean_squared_error(y_cp, y_pred))
    r = np.corrcoef(y_cp, y_pred)[0,1]
    # check important descriptors
    # store 2D descriptors list
    desc_list = get_2D_desc()
    weight_list = reg.feature_importances_
    for desc, weight in zip(desc_list, weight_list):
        # if importances is larger than 0.02, then output descriptors and weight
        if abs(weight) > 0.02:
            print(f'[2D description name] : {desc}')
            print(f'[importances]         : {weight}')
            print("===================================================")
    print('RMSE (ln(K_a))')
    print(rmse)
    print('R (ln(K_a))')
    print(r)
