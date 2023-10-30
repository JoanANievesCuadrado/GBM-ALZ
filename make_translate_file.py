import json
import mygene
import numpy as np
import pandas as pd
import requests

from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Union


alz_path: Path = Path(r'data/ALZ/')
gbm_sample_path: Path = Path(r'data/TCGA-GBM/')
gbm_data_path: Path = gbm_sample_path / 'data'


def get_notfound_from_ensembl_api(df: pd.DataFrame) -> pd.DataFrame:
    mask: np.ndarray = df.entrezgene.isna() & df.symbol.isna()
    notfound: np.ndarray = df.loc[mask].index.to_numpy()

    server: str = "https://rest.ensembl.org"
    lookup: str = "/lookup/id?expand=0;species=human;object_type=gene"
    uri: str = server + lookup
    headers: Dict = {"Content-Type" : "application/json",
                     "Accept" : "application/json"}
    data_dict: Dict = {}

    for i in tqdm(range(0, notfound.size, 1000)):
        data: str = json.dumps({'ids': list(notfound[i:i+1000])})
        r: requests.Response = requests.post(uri, headers=headers, data=data)
        data_dict.update(r.json())
    
    for key, value in tqdm(data_dict.items()):
        if not value or not (symbol := value.get('display_name')):
            continue
        df.loc[key, 'symbol'] = symbol

    return df


def fill_rows_genes(rows_genes: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:

    def get_ensembl(row: pd.Series) -> Union[str, List, None]:
        # Match by gene entrez id
        ensembls: pd.Series = df.loc[df.entrezgene == row.gene_entrez_id, 'ensembl.gene']

        if ensembls.size > 1:
            return ensembls.index.to_list()
        if ensembls.size == 1:
            return ensembls.iat[0]

        # Match by gene symbol
        ensembls1: pd.Series = df.loc[df.symbol == row.gene_symbol,
                                      'ensembl.gene']
        if ensembls1.size > 1:
            return ensembls1.index.to_list()
        if ensembls1.size == 1:
            return ensembls1.iat[0]
    
    rows_genes['ensembl'] = rows_genes.apply(get_ensembl, axis=1)

    return rows_genes


def main():
    rows_genes = pd.read_csv(alz_path / 'rows-genes.csv', sep=';', index_col=0)
    sample = pd.read_excel(gbm_sample_path / 'sample.xls')
    
    fpkm = pd.read_table(gbm_data_path / sample['File Name'][0], names=['ensembl_id', 'value'])
    ensembl_ids = fpkm['ensembl_id'].to_numpy()
    ensembl_ids = np.vectorize(lambda x: x.split('.')[0])(ensembl_ids)

    # Quering data from MyGene database
    mg = mygene.MyGeneInfo()
    df = mg.querymany(ensembl_ids, species='human',
                      fields=['entrezgene', 'symbol', 'name', 'ensembl.gene'],
                      as_dataframe=True)
    
    df = get_notfound_from_ensembl_api(df)
    rows_genes = fill_rows_genes(rows_genes, df)

    rows_genes.to_excel(alz_path / 'rows-genes2.xlsx')


if __name__ == '__main__':
    main()
