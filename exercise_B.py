# $ python3 exercise_B.py

# need to install xlrd and openpyxl

import pandas as pd

# read .xlsx file (specify sheet_name)
df = pd.read_excel('SupplementaryTableS1.xlsx', sheet_name='1-SM')
# convert to .csv file (index=False to remove [Unnamed : 0])
df.to_csv('SM.csv', index=False)

# the same name to save memory
df = pd.read_excel('SupplementaryTableS1.xlsx', sheet_name='2-CP')
df.to_csv('CP.csv', index=False)

# validate .csv file
df = pd.read_csv('SM.csv')
print('output SM.csv [5rows]')
print(df.head())
print('========================================')
df = pd.read_csv('CP.csv')
print('output CP.csv [5rows]')
print(df.head())