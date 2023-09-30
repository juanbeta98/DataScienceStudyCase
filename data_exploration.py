#%% 1. Data exploration
'''
Data Exploration for BMW thinktank
'''
from source import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import copy

AuditHistory.head()
SupplierPerformance.head()

Suppliers = list(AuditHistory['SupplierId'].unique())               # 818 Suppliers
Derivatives = list(AuditHistory['DerivativeName'].unique())         # 75 Derivatives
ProductionLines = list(AuditHistory['ProductionLine'].unique())     # 4 Production Lines
Regions = list(AuditHistory['DerivativeRegion'].unique())           # 6 Regions

################ Explore data ################
### 1. Missing data
Audits_nan_counts = AuditHistory.isna().sum()
Suppliers_nan_counts = SupplierPerformance.isna().sum()
print(Audits_nan_counts)
print('----------------')
print(Suppliers_nan_counts)

# Re-code the BadSupplierIndicator column
SupplierPerformance['BadSupplierIndicator']=SupplierPerformance['BadSupplierIndicator'].fillna(0).map({'bad':1,0:0})


new_df = copy(AuditHistory)
new_df.join(SupplierPerformance,on='SupplierId',how='left')



plt.matshow(SupplierPerformance.corr())
plt.title('Correlogram'); plt.xlabel('Features');plt.ylabel('Features')
plt.show()


# %%

# %%

#%% 2. Preprocess



#%% 4. Predict latest results


#%% 5. 

