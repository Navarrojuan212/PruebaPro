import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def analyze_data(all_data, images_path):
    # Normalización de los datos SNR utilizando RobustScaler
    features = ['SNR', 'Time_Index']
    x = all_data[features]
    y = all_data['Source']

    # Normalizar las características utilizando RobustScaler
    scaler = RobustScaler()
    x_scaled = scaler.fit_transform(x)

    # Aplicar el test de Shapiro-Wilk a los datos normalizados
    stat, p_value = stats.shapiro(x_scaled[:, 0])  # Solo la columna SNR

    # Mostrar los resultados del test de Shapiro-Wilk
    print('Shapiro-Wilk Test:')
    print(f'Statistics={stat:.3f}, p-value={p_value:.3f}')

    # Interpretación del resultado
    alpha = 0.05
    if p_value > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')

    # PCA
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_scaled)

    # Visualización PCA
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=y, palette='tab10', s=100)
    plt.title('PCA of SNR Data', fontsize=26, fontweight='bold')
    plt.xlabel('Principal Component 1', fontsize=22)
    plt.ylabel('Principal Component 2', fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.legend(loc='best')
    plt.savefig(os.path.join(images_path, 'pca.png'))
    plt.show()

    # LDA
    lda = LDA(n_components=2)
    x_lda = lda.fit_transform(x_scaled, y)

    # Verificar el número de componentes resultantes de LDA
    if x_lda.shape[1] == 1:
        lda_df = pd.DataFrame(data=x_lda, columns=['LD1'])
        lda_df['Source'] = y

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Source', y='LD1', data=lda_df, palette='tab10')
        plt.title('LDA of SNR Data (Single Component)', fontsize=26, fontweight='bold')
        plt.xlabel('Source', fontsize=22)
        plt.ylabel('LDA Component 1', fontsize=22)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.grid(True)
        plt.xticks(rotation=90)
        plt.savefig(os.path.join(images_path, 'lda_single.png'))
        plt.show()
    else:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x_lda[:, 0], y=x_lda[:, 1], hue=y, palette='tab10', s=100)
        plt.title('LDA of SNR Data', fontsize=26, fontweight='bold')
        plt.xlabel('LDA Component 1', fontsize=22)
        plt.ylabel('LDA Component 2', fontsize=22)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(os.path.join(images_path, 'lda.png'))
        plt.show()

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    x_tsne = tsne.fit_transform(x_scaled)

    # Visualización t-SNE
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_tsne[:, 0], y=x_tsne[:, 1], hue=y, palette='tab10', s=100)
    plt.title('t-SNE of SNR Data', fontsize=26, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=22)
    plt.ylabel('t-SNE Component 2', fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(images_path, 'tsne.png'))
    plt.show()
