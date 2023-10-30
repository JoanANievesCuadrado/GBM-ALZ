import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from scipy.stats.mstats import gmean
from typing import List, Tuple, Optional


gbm_sample_path: Path = Path(r'data/TCGA-GBM')
gbm_path: Path = gbm_sample_path / 'data'
alz_path: Path = Path(r'data/ALZ')

outputpath: Path = Path(r'output')
if not outputpath.is_dir():
    outputpath.mkdir()


def read_gbm_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # The "sample.xls" file has the information about each sample
    sample: pd.DataFrame = pd.read_excel(gbm_sample_path / 'sample.xls')

    # Masks to get the normal and tumor samples
    normal_mask: pd.Series = sample['Sample Type'] == 'Solid Tissue Normal'
    tumor_mask: pd.Series = sample['Sample Type'] != 'Solid Tissue Normal'

    # Reading first file to get the ensembl ids list
    column_names: List = ['gene_id', 'value']
    file: pd.DataFrame = pd.read_table(gbm_path / sample['File Name'][0], names=column_names)

    # Identifier of each gene in ensembl format
    genes_id: np.ndarray = file['gene_id'].to_numpy()
    genes_id = np.vectorize(lambda x: x.split('.')[0])(genes_id)

    # Loading data
    def get_data(row: pd.Series) -> pd.Series:
        filename = row['File Name']
        file = pd.read_table(gbm_path / filename, names=['gene_id', 'value'])
        return file['value']

    data: np.ndarray = sample.apply(get_data, axis=1).to_numpy()
    data += 0.1

    return data[normal_mask], data[tumor_mask], genes_id


def read_alz_data() -> Tuple[np.ndarray, np.ndarray]:
    # Loading DonorInformation
    DonorInformation = pd.read_csv(alz_path / 'DonorInformation.csv', sep=';')

    # Masks to get no dementia and Alzheimer disease samples
    no_dementia_mask = DonorInformation.dsm_iv_clinical_diagnosis == 'No Dementia'
    alz_mask = DonorInformation.dsm_iv_clinical_diagnosis == "Alzheimer's Disease Type"

    # Getting donor_id
    NoDementia_donor_id = DonorInformation.loc[no_dementia_mask, 'donor_id']
    Alz_donor_id = DonorInformation.loc[alz_mask, 'donor_id']

    # Loading ColumnSample
    ColumnsSample = pd.read_csv(alz_path / 'columns-samples.csv', sep=';', index_col=1)

    NoDementia_CS = ColumnsSample.loc[NoDementia_donor_id]
    Alz_CS = ColumnsSample.loc[Alz_donor_id]

    # Masks to get only Forewhite matter (FWM)
    nd_fwm_mask = NoDementia_CS.structure_acronym == 'FWM'
    alz_fwm_mask = Alz_CS.structure_acronym == 'FWM'

    # Getting rnaseq_profile_id
    NoDementia_FWM_id = NoDementia_CS.loc[nd_fwm_mask, 'rnaseq_profile_id'].astype(str)
    Alz_FWM_id = Alz_CS.loc[alz_fwm_mask, 'rnaseq_profile_id'].astype(str)

    # Loading FPKM file
    fpkm = pd.read_csv(alz_path / 'fpkm_table_normalized.csv', index_col=0)

    # Getting no dementia and alz data
    old = fpkm[NoDementia_FWM_id].to_numpy().T + 0.1
    alz = fpkm[Alz_FWM_id].to_numpy().T + 0.1

    return old, alz


def get_filters(genes_id: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Loading the modified rows_genes file with the ensembl ids
    rows_genes = pd.read_excel(alz_path / 'rows_genes2.xlsx')

    # Getting valids ensembl ids
    ensembls_id = rows_genes.ensembl.dropna()

    # Getting the index for alz data
    index_oa = ensembls_id.index.to_numpy()

    # Getting the index for gbm data
    _, index_nt = np.where(genes_id == ensembls_id.to_numpy()[:, np.newaxis])

    # Getting corresponding symbols
    symbols = rows_genes.loc[index_oa, 'gene_symbol']

    return index_nt, index_oa, symbols.to_numpy()


def save_n_comp(fname: str, vect: np.ndarray, genes: np.ndarray, n: Optional[int] = None) -> None:
    n = n or len(vect)    
    folder: Path = outputpath / f'comp_{n}'
    if not folder.is_dir():
        folder.mkdir()

    args: np.ndarray = np.argsort(-np.abs(vect))
    file = open(folder / f'{fname}_{n}.txt', 'w')

    for arg in args[:n]:
        file.write('{:<10}\t{}\n'.format(genes[arg], str(vect[arg])))
    file.close()


def geometry_analysis(tumor: np.ndarray, old: np.ndarray, alz: np.ndarray, genes: np.array) -> None:
    # Vector to the center of each cloud
    T_mean: np.ndarray = np.mean(tumor, axis=0)
    O_mean: np.ndarray = np.mean(old, axis=0)
    A_mean: np.ndarray = np.mean(alz, axis=0)

    # Distance to the center of each cloud
    T_norm: float = np.linalg.norm(T_mean)
    O_norm: float = np.linalg.norm(O_mean)
    A_norm: float = np.linalg.norm(A_mean)

    # Angle between clouds
    T_O_angle: float = np.arccos(np.dot(T_mean, O_mean)/(T_norm * O_norm))
    T_A_angle: float = np.arccos(np.dot(T_mean, A_mean)/(T_norm * A_norm))
    A_O_angle: float = np.arccos(np.dot(A_mean, O_mean)/(A_norm * O_norm))

    # Vector and distance from old to alz cloud
    OA_vec = A_mean - O_mean
    OA_norm = np.linalg.norm(OA_vec)
    OA_vec = OA_vec / OA_norm

    # Constructing an orthonormal basis with the vectors of the center of each cloud (Gram-Schmidt)
    T_mean = T_mean/T_norm

    dummy: np.ndarray = A_mean - np.dot(A_mean, T_mean)*T_mean
    dummy_norm: float = np.linalg.norm(dummy)
    A_mean = dummy/dummy_norm

    dummyV: np.ndarray = O_mean - np.dot(O_mean, T_mean)*T_mean - np.dot(O_mean, A_mean)*A_mean
    dummyV_norm: float = np.linalg.norm(dummyV)
    O_mean = dummyV/dummyV_norm

    # Project the data onto the new base
    N_T_point: np.ndarray = np.dot(normal, T_mean)
    T_T_point: np.ndarray = np.dot(tumor, T_mean)
    O_T_point: np.ndarray = np.dot(old, T_mean)
    A_T_point: np.ndarray = np.dot(alz, T_mean)

    N_A_point: np.ndarray = np.dot(normal, A_mean)
    T_A_point: np.ndarray = np.dot(tumor, A_mean)
    O_A_point: np.ndarray = np.dot(old, A_mean)
    A_A_point: np.ndarray = np.dot(alz, A_mean)

    N_O_point: np.ndarray = np.dot(normal, O_mean)
    T_O_point: np.ndarray = np.dot(tumor, O_mean)
    V_O_point: np.ndarray = np.dot(old, O_mean)
    A_O_point: np.ndarray = np.dot(alz, O_mean)

    # Plots
    # Alzheimer vs. Tumor
    fig, ax = plt.subplots()

    ax.scatter(N_T_point, N_A_point, label = 'N')
    ax.scatter(T_T_point, T_A_point, label = 'GBM')
    ax.scatter(O_T_point, O_A_point, label = 'O')
    ax.scatter(A_T_point, A_A_point, label = 'Alz')

    ax.set_title('Alz vs. GBM')
    ax.grid()
    ax.set_xlabel('GBM')
    ax.set_ylabel('Alz')
    ax.legend()

    # Alzheimer vs. Old
    fig2, ax2 = plt.subplots()
    ax2.scatter(N_O_point, N_A_point, label = 'N')
    ax2.scatter(T_O_point, T_A_point, label = 'GBM')
    ax2.scatter(V_O_point, O_A_point, label = 'O')
    ax2.scatter(A_O_point, A_A_point, label = 'Alz')

    ax2.set_title('Alz vs. O')
    ax2.grid()
    ax2.set_xlabel('O')
    ax2.set_ylabel('Alz')
    ax2.legend()

    # Old vs. Tumor
    fig3, ax3 = plt.subplots()
    ax3.scatter(N_T_point, N_O_point, label = 'N')
    ax3.scatter(T_T_point, T_O_point, label = 'GBM')
    ax3.scatter(O_T_point, V_O_point, label = 'O')
    ax3.scatter(A_T_point, A_O_point, label = 'Alz')

    ax3.set_title('O vs. GBM')
    ax3.grid()
    ax3.set_xlabel('GBM')
    ax3.set_ylabel('O')
    ax3.legend()

    # 3D plot
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(projection='3d')

    ax4.scatter(N_T_point, N_A_point, N_O_point, label = 'N')
    ax4.scatter(T_T_point, T_A_point, T_O_point, label = 'GBM')
    ax4.scatter(O_T_point, O_A_point, V_O_point, label = 'O')
    ax4.scatter(A_T_point, A_A_point, A_O_point, label = 'Alz')

    ax4.set_title('Alz vs. GBM vs. O')
    ax4.grid()
    ax4.set_xlabel('GBM')
    ax4.set_ylabel('Alz')
    ax4.set_zlabel('O')
    ax4.legend()

    # Export data
    outputpath2: Path = outputpath / 'geometry_analysis'
    if not outputpath2.is_dir():
        outputpath2.mkdir()

    np.savetxt(outputpath2 / 'T_mean.txt', T_mean)
    np.savetxt(outputpath2 / 'O_mean.txt', O_mean)
    np.savetxt(outputpath2 / 'A_mean.txt', A_mean)
    np.savetxt(outputpath2 / 'OA_vec.txt', OA_vec)

    np.savetxt(outputpath2 / 'T_norm.txt', [T_norm])
    np.savetxt(outputpath2 / 'O_norm.txt', [O_norm])
    np.savetxt(outputpath2 / 'A_norm.txt', [A_norm])
    np.savetxt(outputpath2 / 'OA_norm.txt', [OA_norm])

    np.savetxt(outputpath2 / 'T_O_angle.txt', [T_O_angle])
    np.savetxt(outputpath2 / 'T_A_angle.txt', [T_A_angle])
    np.savetxt(outputpath2 / 'A_O_angle.txt', [A_O_angle])

    np.savetxt(outputpath2 / 'N.txt', normal)
    np.savetxt(outputpath2 / 'T.txt', tumor)
    np.savetxt(outputpath2 / 'O.txt', old)
    np.savetxt(outputpath2 / 'A.txt', alz)

    np.savetxt(outputpath2 / 'T_mean_no_normalized.txt', T_norm*T_mean)
    np.savetxt(outputpath2 / 'A_mean_no_normalized.txt', dummy)
    np.savetxt(outputpath2 / 'O_mean_no_normalized.txt', dummyV)

    fig.savefig(outputpath2 / 'A_vs_T.png')
    fig2.savefig(outputpath2 / 'A_vs_V.png')
    fig3.savefig(outputpath2 / 'V_vs_T.png')
    fig4.savefig(outputpath2 / 'T_V_A.png')

    save_n_comp('T_mean', T_mean, genes, 100)
    save_n_comp('O_mean', O_mean, genes, 100)
    save_n_comp('A_mean', A_mean, genes, 100)
    save_n_comp('OA_vec', OA_vec, genes, 100)


def pca_analysis(normal: np.ndarray, tumor: np.ndarray, old: np.ndarray, alz: np.ndarray, genes: np.ndarray) -> None:
    data: np.ndarray = np.concatenate((normal, tumor, old, alz))
    U, s, Vt = np.linalg.svd(data, full_matrices=False)
    eigenvalues: np.ndarray = s**2/data.shape[0]
    eigenvalues_normalized: np.ndarray = eigenvalues / eigenvalues.sum()
    eigenvectors: np.ndarray= Vt
    projection: np.ndarray = np.dot(Vt, data.T)

    # Figures
    fig_pca, ax_pca = plt.subplots()

    a1 = normal.shape[0]
    a2 = a1 + tumor.shape[0]
    a3 = a2 + old.shape[0]
    ax_pca.scatter(projection[0, :a1], projection[1, :a1], s=15, label="N")
    ax_pca.scatter(projection[0, a1:a2], projection[1, a1:a2], s=15, label="GBM")
    ax_pca.scatter(projection[0, a2:a3], projection[1, a2:a3], s=15, label="O")
    ax_pca.scatter(projection[0, a3:], projection[1, a3:], s=15, label="Alz")

    ax_pca.set_title('PC1 vs. PC2')
    ax_pca.grid()
    ax_pca.set_xlabel(f'PC1 ({eigenvalues_normalized[0]*100:.2f} %)')
    ax_pca.set_ylabel(f'PC2 ({eigenvalues_normalized[1]*100:.2f} %)')
    ax_pca.legend()

    # 3D plot
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(projection='3d')

    ax3d.scatter(projection[0, :a1], projection[1, :a1], projection[2, :a1], label = 'N')
    ax3d.scatter(projection[0, a1:a2], projection[1, a1:a2], projection[2, a1:a2], label = 'GBM')
    ax3d.scatter(projection[0, a2:a3], projection[1, a2:a3], projection[2, a2:a3], label = 'O')
    ax3d.scatter(projection[0, a3:], projection[1, a3:], projection[2, a3:], label = 'Alz')

    ax3d.set_title('PC1 vs. PC2 vs. PC3')
    ax3d.grid()
    ax3d.set_xlabel(f'PC1 ({eigenvalues_normalized[0]*100:.2f} %)')
    ax3d.set_ylabel(f'PC2 ({eigenvalues_normalized[1]*100:.2f} %)')
    ax3d.set_zlabel(f'PC3 ({eigenvalues_normalized[2]*100:.2f} %)')
    ax3d.legend()

    # Exporting data
    outputpath2: Path = outputpath / 'pca_analysis'
    if not outputpath2.is_dir():
        outputpath2.mkdir()
    fig_pca.savefig(outputpath2 / 'PC1_PC2.png')
    fig_pca.savefig(outputpath2 / 'PC1_PC2.pdf')
    fig3d.savefig(outputpath2 / 'PC1_PC2_PC3.png')
    fig3d.savefig(outputpath2 / 'PC1_PC2_PC3.pdf')
    np.savetxt(outputpath2 / 'eigenvalues.txt', eigenvalues)
    np.savetxt(outputpath2 / 'eigenvalues_normalized.txt', eigenvalues_normalized)
    np.savetxt(outputpath2 / 'eigenvectors.txt', eigenvectors)
    np.savetxt(outputpath2 / 'projection.txt', projection)

    save_n_comp('PC1', eigenvectors[0], genes, 100)
    save_n_comp('PC2', eigenvectors[1], genes, 100)
    save_n_comp('PC3', eigenvectors[2], genes, 100)


if __name__ == "__main__":
    # Loading data
    normal, tumor, genes_id = read_gbm_data()
    old, alz = read_alz_data()

    # Getting filters and gene symbols
    index_nt, index_oa, symbols = get_filters(genes_id)
    np.savetxt(outputpath / 'index_nt.txt', index_nt, fmt='%d')
    np.savetxt(outputpath / 'index_oa.txt', index_oa, fmt='%d')
    np.savetxt(outputpath / 'symbols.txt', symbols, fmt='%s')

    normal = normal[:, index_nt]
    tumor = tumor[:, index_nt]
    old = old[:, index_oa]
    alz = alz[:, index_oa]

    # Normalizing data
    normal = 1_000_000 * (normal / normal.sum(axis=1)[:, np.newaxis])
    tumor = 1_000_000 * (tumor / tumor.sum(axis=1)[:, np.newaxis])
    old = 1_000_000 * (old / old.sum(axis=1)[:, np.newaxis])
    alz = 1_000_000 * (alz / alz.sum(axis=1)[:, np.newaxis])

    # Fold change
    ref = gmean(normal)
    normal = np.log2(normal / ref)
    tumor = np.log2(tumor / ref)
    old = np.log2(old / ref)
    alz = np.log2(alz / ref)

    geometry_analysis(tumor, old, alz, symbols)
    pca_analysis(normal, tumor, old, alz, symbols)
    plt.show()
