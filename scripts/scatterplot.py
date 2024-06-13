import matplotlib.pyplot as plt
import os

def create_scatterplot(data_frames, images_path, translate_name):
    plt.figure(figsize=(12, 6))
    for file_name, df in data_frames.items():
        translated_name = translate_name(file_name)
        plt.scatter(df['Time_Index'], df['SNR'], label=translated_name, alpha=0.6)
    plt.title('Scatter Plot of SNR vs. Time', fontsize=26)
    plt.xlabel('Time (seconds)', fontsize=20)
    plt.ylabel('SNR (dB)', fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(True)  # Add grid
    plt.tight_layout()  # Adjust layout
    plt.savefig(os.path.join(images_path, 'scatterplot.png'))
    #plt.show()


