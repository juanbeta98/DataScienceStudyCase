#%% 1. Data exploration
'''
Data Exploration for BMW thinktank
'''
from source import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import copy

################ Explore data ################
### 1. Missing data
# Audits_nan_counts = AuditHistory.isna().sum()
# Suppliers_nan_counts = SupplierPerformance.isna().sum()
# print(Audits_nan_counts)
# print('----------------')
# print(Suppliers_nan_counts)

# Re-code the BadSupplierIndicator column
SupplierPerformance['BadSupplierIndicator']=SupplierPerformance['BadSupplierIndicator'].fillna(0).map({'bad':1,0:0})
SupplierPerformance = SupplierPerformance[SupplierPerformance.index.isin(Suppliers)]




# plt.matshow(SupplierPerformance.corr())
# plt.title('Correlogram'); plt.xlabel('Features');plt.ylabel('Features')
# plt.show()


#%%
entry_counts = AuditHistory.groupby(['SupplierId', 'DerivativeName']).size().reset_index(name='EntryCount')

# Assuming 'entry_counts' is your DataFrame with EntryCount column
max_entry_count = entry_counts['EntryCount'].max()
min_entry_count = entry_counts['EntryCount'].min()
mean_entry_count = entry_counts['EntryCount'].mean()

# Displaying the results
print(f"Maximum EntryCount: {max_entry_count}")
print(f"Minimum EntryCount: {min_entry_count}")
print(f"Mean EntryCount: {mean_entry_count}")


# Assuming 'entry_counts' is your DataFrame with EntryCount column
max_entry_index = entry_counts['EntryCount'].idxmax()
supplier_id_max_count = entry_counts.loc[max_entry_index, 'SupplierId']
derivative_name_max_count = entry_counts.loc[max_entry_index, 'DerivativeName']

# Filter the original DataFrame based on the SupplierID and DerivativeName
max_entry_df = AuditHistory[(AuditHistory['SupplierId'] == supplier_id_max_count) & (AuditHistory['DerivativeName'] == derivative_name_max_count)]
max_entry_df = max_entry_df.sort_values(by='RecentMonth')

# Displaying the result
print(f"Entries for SupplierID {supplier_id_max_count} and DerivativeName {derivative_name_max_count}:")
print(max_entry_df)

# %%

#%% 2. Preprocess



#%% 4. Predict latest results


#%% 5. 

