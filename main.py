import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path
from scipy.stats.mstats import gmean
from typing import List, Tuple, Optional, Union


normal_color = 'dodgerblue'
tumor_color = 'orangered'
old_color = 'blue'
alz_color = 'red'

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

    # Get valids ensembl ids
    ensembls_id = rows_genes.ensembl.dropna()

    # Auxuliar dataframe for to get the the normal and tumor indices
    df1 = pd.DataFrame(data={'i': range(genes_id.size)}, index=genes_id)

    # Get the indices
    index_oa = ensembls_id.index.to_numpy()
    index_nt = df1.loc[ensembls_id, 'i'].to_numpy()

    # Getting corresponding symbols
    symbols = rows_genes.loc[index_oa, 'gene_symbol']

    return index_nt, index_oa, symbols.to_numpy()


def save_n_comp(fname: str, vect: np.ndarray, genes: np.ndarray, n: Optional[int] = None) -> None:
    n = n or len(vect)  

    args: np.ndarray = np.argsort(-np.abs(vect))
    file = open(fname, 'w')

    for arg in args[:n]:
        file.write(f'{genes[arg]}\t{vect[arg]}\n')
    file.close()


def figure_1b(normal: np.ndarray, tumor: np.ndarray, old: np.ndarray, alz: np.ndarray, outputpath: Path):
    normal_center = normal.mean(axis=1)
    tumor_center = tumor.mean(axis=1)
    old_center = old.mean(axis=1)
    alz_center = alz.mean(axis=1)

    fig1b, ax1b = plt.subplots()

    ax1b.scatter(*normal_center, marker='o', c=normal_color, s=50, label='N')
    ax1b.scatter(*tumor_center, marker='o', c=tumor_color, s=50, label='GBM')
    ax1b.scatter(*old_center, marker='o', c=old_color, s=50, label='O')
    ax1b.scatter(*alz_center, marker='o', c=alz_color, s=50, label='AD')

    # Arrow from normal to tumor center
    ax1b.annotate('', xy=tumor_center, xytext=normal_center,
                  arrowprops=dict(arrowstyle='->', shrinkA=4.5, shrinkB=3.9,
                                  connectionstyle="arc, angleA=-1, armA=0,"
                                  "angleB=100, armB=60, rad=50"))

    # Arrow from normal to old center
    ax1b.annotate('', xy=old_center, xytext=normal_center,
                  arrowprops=dict(arrowstyle='->', shrinkA=4.5, shrinkB=3.9,
                                  connectionstyle="arc, angleA=1, armA=0,"
                                  "angleB=-100, armB=90, rad=80"))

    # Arrow from normal to alzheimer center
    ax1b.annotate('', xy=alz_center, xytext=normal_center,
                  arrowprops=dict(arrowstyle='->', shrinkA=4.5, shrinkB=3.9,
                                  connectionstyle="arc, angleA=2, armA=0,"
                                  "angleB=-115, armB=90, rad=90"))
    
    # Arrow from old to alzheimer center
    ax1b.annotate('', xy=alz_center, xytext=old_center,
                  arrowprops=dict(arrowstyle='->', shrinkA=4.5, shrinkB=3.9))
    
    ax1b.annotate('N', xy=normal_center + (-4, -10),
                  c='k',  size=12, va='top', ha='right', weight='bold')

    ax1b.annotate('GBM', xy=tumor_center + (-10, 25),
                  c='k', size=12, va='top', ha='right', weight='bold')

    ax1b.annotate('O', xy=old_center + (10, -12),
                  c='k',  size=12, va='center', ha='left',  weight='bold')

    ax1b.annotate('AD', xy=alz_center + (-15, 0),
                  c='k', size=12, va='top', ha='right', weight='bold')

    ax1b.annotate('Early\nAD',  xy=alz_center/2 + (50, 15), weight='bold')

    # ax1b.text(285, 75, "Aging", ha='left', va='top', weight='bold')
    ax1b.annotate('Aging', xy=old_center/2 + (130, -5), weight='bold', ha='right', va='bottom')

    ax1b.text(*((old_center + alz_center)/2 + (-5, 15)), 
              'Late AD', va='bottom', ha='center', weight='bold')

    ax1b.set_xlabel('PC1')
    ax1b.set_ylabel('PC2')
    ax1b.set_title('c')
    ax1b.set(xlim=(-20, 270), ylim=(-180, 290))

    fig1b.savefig(outputpath / 'Fig_1b.pdf',  bbox_inches='tight')


def normal_dist(x: Union[float, np.ndarray],
                        x_mean: Union[float, np.ndarray],
                        sigma: Union[float, np.ndarray]
                        ) -> Union[float, np.ndarray]:
    return np.exp(-np.power(x - x_mean, 2) / (2 * sigma*2))


def figure_1c(normal: np.ndarray, tumor: np.ndarray, alz: np.ndarray, outputpath: Path):
    x = np.linspace(-180, 400, 100)
    y = np.linspace(-250, 330, 100)
    X, Y = np.meshgrid(x, y)
    normal_center, normal_std = normal.mean(axis=1), 150*normal.std(axis=1)
    tumor_center, tumor_std = tumor.mean(axis=1), 70*tumor.std(axis=1)
    alz_center, alz_std = alz.mean(axis=1), 100*alz.std(axis=1)

    Z_normal = normal_dist(X, normal_center[0], normal_std[0]) * normal_dist(Y, normal_center[1], normal_std[1])
    Z_tumor = normal_dist(X, tumor_center[0], tumor_std[0]) * normal_dist(Y, tumor_center[1], tumor_std[1])
    Z_alz = normal_dist(X, alz_center[0], alz_std[0]) * normal_dist(Y, alz_center[1], alz_std[1])

    Z = 15 * Z_normal + 30 * Z_tumor + 3 * Z_alz

    fig1c, ax1c = plt.subplots()

    ax1c.contour(X, Y, Z, levels=80, cmap='gray')
    
    ax1c.set_title('d')
    ax1c.set_xlabel('PC1')
    ax1c.set_ylabel('PC2')

    fig1c.savefig(outputpath / 'Fig_1c.pdf', bbox_inches='tight')


def pca_analysis(normal: np.ndarray, tumor: np.ndarray,
                 old: np.ndarray, alz: np.ndarray, genes: np.ndarray) -> None:
    data: np.ndarray = np.concatenate((normal, tumor, old, alz))
    U, s, Vt = np.linalg.svd(data, full_matrices=False)
    eigenvalues: np.ndarray = s**2/data.shape[0]
    eigenvalues_normalized: np.ndarray = eigenvalues / eigenvalues.sum()
    eigenvectors: np.ndarray= Vt
    projection: np.ndarray = np.dot(Vt, data.T)

    i1 = normal.shape[0]
    i2 = i1 + tumor.shape[0]
    i3 = i2 + old.shape[0]
    
    # Figures
    # Fig 1a (PC1-PC2)
    fig1a, ax1a = plt.subplots()

    ax1a.scatter(projection[0, :i1], projection[1, :i1], s=15, label="N", c=normal_color)
    ax1a.scatter(projection[0, i1:i2], projection[1, i1:i2], s=15, label="GBM", c=tumor_color, marker='s')
    ax1a.scatter(projection[0, i2:i3], projection[1, i2:i3], s=15, label="O", c=old_color, marker='s')
    ax1a.scatter(projection[0, i3:], projection[1, i3:], s=15, label="AD", c=alz_color)

    ax1a.set_title('a')
    ax1a.set_xlabel(f'PC1 ({eigenvalues_normalized[0]*100:.2f} %)')
    ax1a.set_ylabel(f'PC2 ({eigenvalues_normalized[1]*100:.2f} %)')
    ax1a.legend()

    # Fig 1b
    figure_1b(projection[:2, :i1], projection[:2, i1:i2],
              projection[:2, i2:i3], projection[:2, i3:], outputpath)
    
    # Fig 1c
    figure_1c(projection[:2, :i1], projection[:2, i1:i2], projection[:2, i3:],
              outputpath)

    # Exporting data
    fig1a.savefig(outputpath / 'Fig_1a.pdf', bbox_inches='tight')

    save_n_comp(outputpath / 'PC1.txt', eigenvectors[0], genes, 100)
    save_n_comp(outputpath / 'PC2.txt', eigenvectors[1], genes, 100)


def figure_1d(normal: np.ndarray, tumor: np.ndarray, alz: np.ndarray):
    fig1d, ax1d = plt.subplots()
    ube2c, mmp9 = 29900, 29913

    groups = (['GBM'] * tumor.shape[0] * 2
              + ['N'] * normal.shape[0] * 2
              + ['AD'] * alz.shape[0] * 2)
    
    columns = np.concatenate([tumor[:, ube2c], tumor[:, mmp9], 
                              normal[:, ube2c], normal[:, mmp9], 
                              alz[:, ube2c], alz[:, mmp9]])
    
    categories = (['UBE3C'] * tumor.shape[0] + ['MMP9'] * tumor.shape[0]
                + ['UBE3C'] * normal.shape[0] + ['MMP9'] * normal.shape[0]
                + ['UBE3C'] * alz.shape[0] + ['MMP9'] * alz.shape[0])

    sns.violinplot(x=groups, y=columns, hue=categories, inner='quart', ax=ax1d)
    ax1d.set_ylabel('$log_2(e/e_{ref})$')

    fig1d.savefig(outputpath / 'Fig_1d.pdf', bbox_inches='tight')


def main():
    # Loading data
    normal, tumor, genes_id = read_gbm_data()
    old, alz = read_alz_data()

    # Getting filters and gene symbols
    index_nt, index_oa, symbols = get_filters(genes_id)

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

    # geometry_analysis(normal, tumor, old, alz, symbols)
    pca_analysis(normal, tumor, old, alz, symbols)
    figure_1d(normal, tumor, alz)
    plt.show()


if __name__ == "__main__":
    main()
