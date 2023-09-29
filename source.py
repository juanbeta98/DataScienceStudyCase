'''
Source code for BMW thinktank interview

Juan Betancourt
juan.beta98@gmail.com
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

################ Upload Data ################
AuditHistory = pd.read_csv('./Data/AuditHistory.csv')                   # 7690 entries
SupplierPerformance = pd.read_csv('./Data/SupplierPerformance.csv',index_col='SupplierId')     # 5823 entries

def plot_categorical_distribution(col_name):
    counts = AuditHistory[col_name].value_counts()

    # Setting Seaborn style
    sns.set(style="whitegrid")

    # Creating a bar plot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.index, y=counts, palette="viridis")

    # Adding labels and title
    plt.xlabel(col_name, fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title(f'Distribution of {col_name}', fontsize=16)

    # Rotating x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Displaying the plot
    plt.show()

    
    return None
