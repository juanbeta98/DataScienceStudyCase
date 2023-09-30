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

Suppliers = list(AuditHistory['SupplierId'].unique())               # 818 Suppliers
Derivatives = list(AuditHistory['DerivativeName'].unique())         # 75 Derivatives
ProductionLines = list(AuditHistory['ProductionLine'].unique())     # 4 Production Lines
Regions = list(AuditHistory['DerivativeRegion'].unique())           # 6 Regions


################ Ploting functions ################
def plot_categorical_distribution(col_name):
    if col_name != 'BadSupplierIndicator':
        counts = AuditHistory[col_name].value_counts()
    else:
        counts = SupplierPerformance[col_name].value_counts()

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


def plot_numerical_distribution(col_name):
    if col_name != 'BadSupplierIndicator':
        # Setting Seaborn style
        sns.set(style="whitegrid")

        # Creating a histogram using Seaborn
        plt.figure(figsize=(10, 6))
        sns.histplot(SupplierPerformance[col_name], bins=20, color='skyblue', edgecolor='black')

        # Adding labels and title
        plt.xlabel(col_name, fontsize=14)
        plt.ylabel('Observations', fontsize=14)
        plt.title(f'Distribution of {col_name}', fontsize=16)

        # Displaying the plot
        plt.show()
    else:
        plot_categorical_distribution(col_name)


def code_categorical_columns(df, categorical_columns):
    encoding_dicts = {}
    for col in categorical_columns:
        if col[0]=='T' or col=='Result':
            df[col] = df[col].map({'-': 1, 'Green': 2, 'Yellow': 3, 'Red': 4})
        else:
            unique_categories = df[col].unique()
            encoding_dict = {category: code + 1 for code, category in enumerate(unique_categories)}
            encoding_dicts[col] = encoding_dict
            df[col] = df[col].map(encoding_dict)
        
    return df, encoding_dicts

def plot_audit_colors(df, SupplierId, DerivativeName):
    # Filter the DataFrame for the specified SupplierId and DerivativeName
    filtered_df = df[(df['SupplierId'] == SupplierId) & (df['DerivativeName'] == DerivativeName)]

    # Check if there is any data for the given filter
    if filtered_df.empty:
        print(f"No data found for SupplierId {SupplierId} and DerivativeName {DerivativeName}")
        return

    # Set the 'RecentMonth' column as the index for time series plotting
    filtered_df.set_index('RecentMonth', inplace=True)
    filtered_df = filtered_df.sort_intex()

    # Create a color map for mapping status to color
    color_map = {'-': 'white', 'Green': 'green', 'Yellow': 'yellow', 'Red': 'red'}

    # Plot the squares for each stage
    plt.figure(figsize=(12, 6))
    stages = ['T1Color', 'T2Color', 'T3Color']
    for stage in stages:
        plt.scatter(filtered_df.index, [stages.index(stage) + 1] * len(filtered_df), c=filtered_df[stage].map(color_map), marker='s', s=100, label=stage)

    # Customize the plot
    plt.title(f"Supplier {SupplierId} - Derivative {DerivativeName} Audit Colors Over Time")
    plt.xlabel('RecentMonth')
    plt.yticks([1, 2, 3], stages)
    plt.legend()

    # Show the plot
    plt.show()