#%% 1. Data exploration
'''
Data Exploration for BMW thinktank
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

AuditHistory = pd.read_csv('./Data/AuditHistory.csv')                   # 7690 entries
SupplierPerformance = pd.read_csv('./Data/SupplierPerformance.csv')     # 5823 entries

AuditHistory.head()
SupplierPerformance.head()

Suppliers = list(AuditHistory['SupplierId'].unique())               # 818 Suppliers
Derivatives = list(AuditHistory['DerivativeName'].unique())         # 75 Derivatives
ProductionLines = list(AuditHistory['ProductionLine'].unique())     # 4 Production Lines

S_j = {j:[] for j in Derivatives}
D_i = {i:[] for i in Suppliers}

for i in AuditHistory.index: 
    if AuditHistory['DerivativeName'][i] not in D_i[AuditHistory['SupplierId'][i]]:
        S_j[AuditHistory['DerivativeName'][i]].append(AuditHistory['SupplierId'][i])
        D_i[AuditHistory['SupplierId'][i]].append(AuditHistory['DerivativeName'][i])

SupplierPerformance =  SupplierPerformance[SupplierPerformance['SupplierId'].isin(Suppliers)]


#%% 2. Preprocess



#%% 4. Predict latest results


#%% 5. 

