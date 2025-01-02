import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats.mstats import gmean
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


data_path = Path('./data/Mouse/data/')

output = Path(r'./Figures and tables/Mouse')
if not output.is_dir():
    output.mkdir()


metadata = pd.read_table(data_path / 'BulkSeq_Aging_metadata.txt')
df = pd.read_table(data_path / 'BulkSeq_Aging_Counttable.txt')

tissues = set(metadata['tissueLong'])
ages = set(metadata['age'])
tissue = 'Corpus callosum'

tissue_mask = (metadata['tissueLong'] == tissue)
ref_id  = metadata[(metadata['age'] == 3) & tissue_mask]['sampleID'].to_numpy()
index  = metadata[tissue_mask]['sampleID'].to_numpy()

data = df[index].to_numpy() + 1
ref = df[ref_id].to_numpy() + 1
ref = gmean(ref.T)
data = np.log2(data.T/ref)

U, s, Vt = np.linalg.svd(data, full_matrices=False)

eigenvalues = s**2/data.shape[0]
eigenvalues_normalized = eigenvalues/eigenvalues.sum()
eigenvectors = Vt
projection = np.dot(Vt, data.T)

i2 = pd.DataFrame({'index': index,'i': range(len(index))})
i2 = i2.set_index('index')

groups = {}
for age in ages:
    sampleID = metadata[(metadata.tissueLong == tissue) & (metadata.age == age)]['sampleID'].to_numpy()
    groups[age] = i2.loc[sampleID, 'i'].to_numpy()

y = []
for value in groups.values():
    v = data[value].mean(axis=0)
    norm = np.linalg.norm(v)
    y.append(norm)

x = np.array(list(ages))
y = np.array(y)

model = LinearRegression()
model.fit(x.reshape((-1, 1)), y)
y_fit = model.predict(x.reshape((-1, 1)))
slope, = model.coef_
intercept = model.intercept_
r2 = model.score(x.reshape((-1, 1)), y)

markers = ['o', 's', '^', 'D', 'p', 'X', '1']
marker = {i: j for i, j in zip(groups.keys(), markers)}

fig, (ax, ax1) = plt.subplots(1, 2, figsize=(13, 5))


for i, j, m in zip(groups.keys(), groups.values(), markers):
    ax.scatter([projection[0, j].mean()], [projection[1, j].mean()], marker=marker[i], label=f'age {i} (n: {len(j)})')


ax.set_xlabel(f'PC1 ({eigenvalues_normalized[0]*100:.2f} %)')
ax.set_ylabel(f'PC2 ({eigenvalues_normalized[1]*100:.2f} %)')
ax.legend()
ax.set_xlim((-100, 100))
ax.set_ylim((-10, 50))

ax1.scatter(x, y, c='r', label='centers')
ax1.plot(x, y_fit, 'k', label=f'fit ($R^2$= {r2:.2f}): {slope:.2f} * age {"+" if intercept >= 0 else "-"} {abs(intercept):.2f}')
ax1.set_xlabel('Age (Months)')
ax1.set_ylabel('Distance')
ax1.legend()

fig.savefig(output / f'suppl_fig_1.pdf', bbox_inches='tight')

plt.show()