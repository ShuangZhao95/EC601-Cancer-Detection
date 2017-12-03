#%matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from sklearn.feature_extraction.text import TfidfVectorizer

train_variants = pd.read_csv('training_variants.csv')
print('Number of training variants: %d' % (len(train_variants)))
print(train_variants.head())

test_variants = pd.read_csv('test_variants.csv')
print('Number of test variants: %d' % (len(test_variants)))
print(test_variants.head())

def read_textfile(filename):
    return pd.read_csv(filename, sep='\|\|', header=None, names=['ID', 'Text'], skiprows=1, engine='python')

train_text = read_textfile('training_text.csv')
print('Number of train samples: %d' % (len(train_text)))
print(train_text.head())

test_text = read_textfile('test_text.csv')
print('Number of test samples: %d' % (len(test_text)))
print(test_text.head())

train_df = pd.concat([train_text, train_variants.drop('ID', axis=1)], axis=1)
print(train_df.head())

test_df = pd.concat([test_text, test_variants.drop('ID', axis=1)], axis=1)
print(test_df.head())

#frequency of each classes
plt.figure(figsize=(12,8))
sns.countplot(x="Class", data=train_df, palette="Reds_d")
plt.ylabel('Frequency', fontsize=14)
plt.xlabel('Class', fontsize=14)
plt.title("Distribution of genetic mutation classes", fontsize=18)
plt.show()

gene_group = train_df.groupby("Gene")['Gene'].count()
gene_group_test = test_df.groupby("Gene")['Gene'].count()
minimal_occ_genes = gene_group.sort_values(ascending=True)[:10]
minimal_occ_genes_test = gene_group_test.sort_values(ascending=True)[:10]
print("Train Genes with maximal occurences\n", gene_group.sort_values(ascending=False)[:10])
print("Test Genes with maximal occurences\n", gene_group_test.sort_values(ascending=False)[:10])
print("\nTrain Genes with minimal occurences\n", minimal_occ_genes)
print("\nTest Genes with minimal occurences\n", minimal_occ_genes_test)

fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(15,15))

for i in range(3):
    for j in range(3):
        gene_count_grp = train_df[train_df["Class"]==((i*3+j)+1)].groupby('Gene')["ID"].count().reset_index()
        sorted_gene_group = gene_count_grp.sort_values('ID', ascending=False)
        sorted_gene_group_top_7 = sorted_gene_group[:7]
        sns.barplot(x="Gene", y="ID", data=sorted_gene_group_top_7, ax=axs[i][j])

#text Count        
train_df.loc[:, 'Text_count']  = train_df["Text"].apply(lambda x: len(x.split()))
print(train_df.head())

#Frequently occuring terms for each class
fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(15,15))

for i in range(3):
    for j in range(3):
        gene_count_grp = train_df[train_df["Class"]==((i*3+j)+1)].groupby('Gene')["Text_count"].mean().reset_index()
        sorted_gene_group = gene_count_grp.sort_values('Text_count', ascending=False)
        sorted_gene_group_top_7 = sorted_gene_group[:7]
        sns.barplot(x="Gene", y="Text_count", data=sorted_gene_group_top_7, ax=axs[i][j])
        
#tfidf
def top_tfidf_feats(row, features, top_n=10):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_feats_in_doc(Xtr, features, row_id, top_n=10):
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=10):
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()
    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=10):
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

def plot_tfidf_classfeats_h(dfs):
    fig = plt.figure(figsize=(12, 100), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        #z = int(str(int(i/3)+1) + str((i%3)+1))
        ax = fig.add_subplot(9, 1, i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=16)
        ax.set_ylabel("Gene", labelpad=16, fontsize=16)
        ax.set_title("Class = " + str(df.label), fontsize=18)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()


#
tfidf = TfidfVectorizer(
	min_df=5, max_features=16000, strip_accents='unicode',lowercase =True,
	analyzer='word', token_pattern=r'\w+', use_idf=True, 
	smooth_idf=True, sublinear_tf=True, stop_words = 'english').fit(train_df["Text"])
        
Xtr = tfidf.fit_transform(train_df["Text"])
y = train_df["Class"]
features = tfidf.get_feature_names()
top_dfs = top_feats_by_class(Xtr, y, features)

plot_tfidf_classfeats_h(top_dfs)
