import seaborn as sns
import matplotlib.pyplot as plt
import os

def create_histogram(data_frames, images_path, translate_name):
    plt.figure(figsize=(12, 6))
    for file_name, df in data_frames.items():
        translated_name = translate_name(file_name)
        sns.histplot(df['SNR'], kde=True, element='step', label=translated_name, alpha=0.6)
    plt.title('Histogram of SNR by Source', fontsize=26)
    plt.xlabel('SNR (dB)', fontsize=20)
    plt.ylabel('Frequency', fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)  # Add grid
    plt.tight_layout()  # Adjust layout
    plt.savefig(os.path.join(images_path, 'histogram.png'))
    #plt.show()


