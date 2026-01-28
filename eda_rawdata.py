import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk

print("Loading training data")
train_ds = load_from_disk("data/train_data")
df = train_ds.to_pandas()

def perform_eda(df):
    print("\nCalculating Baseline Uplift...")
    t_conv = df[df['treatment'] == 1]['conversion'].mean()
    c_conv = df[df['treatment'] == 0]['conversion'].mean()
    print(f"Global Uplift: {t_conv - c_conv:.5%}")

    print("\nGenerating Correlation Heatmap...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.drop(['conversion', 'treatment'], axis=1).corr(), cmap='viridis')
    plt.title("Feature Correlation Matrix (f0 - f11)")
    plt.savefig("data/correlation_matrix.png")
    
    print("\nChecking Treatment/Control Balance...")
    print(df['treatment'].value_counts(normalize=True))

if __name__ == "__main__":
    perform_eda(df)
    print("\nEDA Complete. Plots saved to your 'data' folder.")