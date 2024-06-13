import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def calculate_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers

def create_violin_plot(combined_df, images_path):
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=combined_df, x='Source', y='SNR', palette='Spectral')

    # Superponer los valores atípicos
    for name, group in combined_df.groupby('Source'):
        outliers = calculate_outliers(group['SNR'])
        plt.scatter([name] * len(outliers), outliers, color='r', zorder=2, label='Outliers' if name == 'Floor 1' else "")

    plt.title('Violin Plot of SNR by Source')
    plt.xlabel('Source')
    plt.ylabel('SNR (dB)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()  # Adjust layout
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(images_path, 'violin_plot.png'))
    #plt.show()
