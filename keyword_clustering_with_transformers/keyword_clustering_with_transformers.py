"""Main module."""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.manifold import MDS
from matplotlib import pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

#Code from
#https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/clustering/fast_clustering.py
#Two parameters to tune:
#min_cluster_size: Only consider cluster that have at least 25 elements
#threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
#clusters = util.community_detection(embeddings, min_community_size=1, threshold=0.7)
"""
clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5, compute_full_tree=True, linkage='ward') #, affinity='cosine', linkage='average', distance_threshold=0.4)
clustering_model.fit(embeddings)
cluster_assignment = clustering_model.labels_
print(len(cluster_assignment))
"""
# Create empty lists to store the BIC and AIC values
#dimensions = range(1,12)
#bic_score = []
#aic_score = []
#for n in dimensions:
#    clustering_model = GaussianMixture(n_components=n, random_state=123, n_init=10)
#    clustering_model.fit(embeddings)
#    bic_score.append(clustering_model.bic(embeddings))
#    aic_score.append(clustering_model.aic(embeddings))
#
## Plot the BIC and AIC values together
#fig, ax = plt.subplots(figsize=(12,8),nrows=1)
#ax.plot(dimensions, bic_score, '-o', color='orange')
#ax.plot(dimensions, aic_score, '-o', color='green')
#ax.set(xlabel='Number of Clusters', ylabel='Score')
#ax.set_xticks(dimensions)
#ax.set_title('BIC and AIC Scores Per Number Of Clusters')
"""
#Print for all clusters the top 3 and bottom 3 elements
for i, cluster in enumerate(clusters):
    print("\nCluster {}, #{} Elements ".format(i+1, len(cluster)))
    for sentence_id in cluster[0:3]:
        print("\t", keywords[sentence_id])
    print("\t", "...")
    for sentence_id in cluster[-3:]:
        print("\t", keywords[sentence_id])
"""


# From minimized AIC
##n_components = aic_score.index(min(aic_score)) + 1
##
##clustering_model = GaussianMixture(n_components=n_components, random_state=123, n_init=15)
##clustering_model.fit(embeddings)
##cluster_assignment = clustering_model.predict(embeddings)

# Print
#clustered_sentences = {}
#for sentence_id, cluster_id in enumerate(cluster_assignment):
#    if cluster_id not in clustered_sentences:
#        clustered_sentences[cluster_id] = []
#
#    clustered_sentences[cluster_id].append(keywords[sentence_id])
#
#for i, cluster in clustered_sentences.items():
#    print("Cluster ", i+1)
#    print(cluster)
#    print("")
#
#print("Found", len(clustered_sentences), "clusters")
