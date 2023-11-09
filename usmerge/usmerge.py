import pandas as pd
from minisom import MiniSom  
import sys
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


def equal_wid_merge(df,n):
	if isinstance(df, pd.Series):
	    df1 = df
	else:
	    print("data is not series")
	    sys.exit()

	dfconv = np.array(df1.tolist()).reshape(-1,1)
	dis = KBinsDiscretizer(n_bins=n,
                       encode="ordinal",
                       strategy="uniform"
                      )

	label_uniform = dis.fit_transform(dfconv)
	label = pd.DataFrame(label_uniform.astype(int))[0].tolist()
	edge_uniform = dis.bin_edges_
	return label,edge_uniform

def equal_fre_merge(df,n):
	if isinstance(df, pd.Series):
	    df1 = df
	else:
	    print("data is not series")
	    sys.exit()

	dfconv = np.array(df1.tolist()).reshape(-1,1)
	dis = KBinsDiscretizer(n_bins=n,
                       encode="ordinal",
                       strategy="quantile"
                      )

	label_uniform = dis.fit_transform(dfconv)
	label = pd.DataFrame(label_uniform.astype(int))[0].tolist()
	edge_uniform = dis.bin_edges_
	return label,edge_uniform

def kmeans_merge(df,n):
	if isinstance(df, pd.Series):
	    df1 = df
	else:
	    print("data is not series")
	    sys.exit()

	dfconv = np.array(df1.tolist()).reshape(-1,1)
	dis = KBinsDiscretizer(n_bins=n,
                       encode="ordinal",
                       strategy="kmeans"
                      )

	label_uniform = dis.fit_transform(dfconv)
	label = pd.DataFrame(label_uniform.astype(int))[0].tolist()
	edge_uniform = dis.bin_edges_
	return label,edge_uniform

def som_k_merge(df,n,sig=0.5,lr=0.5,echo=500):
	if isinstance(df, pd.Series):
	    df1 = df
	else:
	    print("data is not series")
	    sys.exit()

	dfconv = np.array(df1.tolist()).reshape(-1,1)
	som_shape = (1,n)
	som = MiniSom(som_shape[0], som_shape[1], dfconv.shape[1], sigma=sig, learning_rate=lr)#,neighborhood_function='gaussian', random_seed=10

	som.train_batch(dfconv, echo, verbose=True)
	weight = som.get_weights()
	w2 = weight.reshape(n,1)
	kmeans = KMeans(n_clusters=n,init=w2).fit(dfconv)
	label_k = kmeans.labels_
	label = pd.DataFrame(label_k.astype(int))[0].tolist()
	center = kmeans.cluster_centers_[:, 0]
	center.sort()
	edge = (center[1:] + center[:-1]) * 0.5
	som_k_edge = np.r_[dfconv.min(), edge,dfconv.max()]

	return label,som_k_edge
















