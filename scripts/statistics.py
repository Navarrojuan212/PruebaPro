import pandas as pd

def calculate_statistics(df):
    stats = {
        'Place': [],
        'Q1 {dB}': [],
        'Mean {dB}': [],
        'Q3 {dB}': [],
        'IQR {dB}': [],
        'Lower Outlier {dB}': [],
        'Upper Outlier {dB}': [],
        'Std Dev {dB}': [],
        'Median {dB}': [],
        'Mode {dB}': [],
        'Variance {dB}': []
    }

    place = df['Source'].iloc[0] if 'Source' in df.columns else 'Unknown'
    stats['Place'].append(place)
    stats['Q1 {dB}'].append(df['SNR'].quantile(0.25))
    stats['Mean {dB}'].append(df['SNR'].mean())
    stats['Q3 {dB}'].append(df['SNR'].quantile(0.75))
    stats['IQR {dB}'].append(df['SNR'].quantile(0.75) - df['SNR'].quantile(0.25))
    stats['Lower Outlier {dB}'].append(df['SNR'].mean() - 1.5 * (df['SNR'].quantile(0.75) - df['SNR'].quantile(0.25)))
    stats['Upper Outlier {dB}'].append(df['SNR'].mean() + 1.5 * (df['SNR'].quantile(0.75) - df['SNR'].quantile(0.25)))
    stats['Std Dev {dB}'].append(df['SNR'].std())
    stats['Median {dB}'].append(df['SNR'].median())
    stats['Mode {dB}'].append(df['SNR'].mode().iloc[0] if not df['SNR'].mode().empty else None)
    stats['Variance {dB}'].append(df['SNR'].var())

    return pd.DataFrame(stats)
