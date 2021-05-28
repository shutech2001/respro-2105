# $ python3 exercise_E.py -train SM.csv

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from exercise_C import RDKit_2D_descriptors
from exercise_D import calc_lnka

import optuna
 
def objective(trial):
    """
    search hyper parameter based on RMSE value (5-fold CV)
    """	

	# Lagrange multiplier (determine weight parameter of Regularization term)
    alpha = trial.suggest_loguniform('alpha', 1e-2, 1.5)
    # the number of iteration
    max_iter = trial.suggest_int('max_iter', 1000, 100000)
    # tolerance for optimization
    tol = trial.suggest_loguniform('tol', 1e-6, 1e-4)

    reg = Lasso(alpha=alpha,
                max_iter=max_iter,
                tol=tol,)
    
    rmse_list = cross_val_score(reg, X, y, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)
    # convert neg_rmse value to positive 
    return - np.array(rmse_list).mean()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="construct regression model from Small molecule data")
    parser.add_argument("-train", help="path to train csv data")
    args = parser.parse_args()
    if args.train is None:
        print(parser.print_help())
        exit(1)

    ##### 1. data preparation #####

    # read .csv
    df = pd.read_csv(args.train)
    
    # make explanatory variable
    smiles = df['SMILES'].values
    # apply compute_2D_desc to each molecule
    X = np.array([RDKit_2D_descriptors(mol).compute_2D_desc() for mol in smiles])

    # make response variable
    # apply calc_lnka to each PPB
    y = df['PPB (fb)'].apply(calc_lnka).values

    # Standardization of explanatory variables
    sc = StandardScaler()
    X = sc.fit_transform(X)

    ##### 2. search hyper parameter using optuna #####
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    print('Lasso Regression : Best Parameters')
    for key, value in study.best_params.items():
    	print(f'{key} = {value},')
    print('==================================================')


    ##### 3. output best parameters' result #####
    reg = Lasso(**study.best_params)
    # 5-fold cross validation 
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    # store each RMSE value and R value
    RMSE = []
    R = []
    for tr_index, val_index in tqdm(kf.split(X, y)):
    	# split train data and validation data
    	X_tr, X_val = X[tr_index], X[val_index]
    	y_tr, y_val = y[tr_index], y[val_index]
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
