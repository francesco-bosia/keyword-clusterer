from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.manifold import MDS
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError


def get_model(model_name: str = ""):
    if model_name:
        model = SentenceTransformer(model_name)
    else:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model


def calculate_embeddings(
    keywords: List[str],
    model: SentenceTransformer,
    batch_size: int = 64,
    normalize: bool = True,
):
    # Compute embedding for the keywords list
    if not model:
        raise RuntimeError("No sentence transformer was given. Please run 'get_model'.")
    return model.encode(
        keywords,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=normalize,
    )


def calculate_similarity(embeddings: List[float]):
    return util.pytorch_cos_sim(embeddings, embeddings).numpy()

def calculate_clustering(clustering_model_type: str = "SpectralClustering", number_of_clusters: int = 3, embeddings = None, **kwargs):
    """ Calculates the clustering with a given model """
    if embeddings is None:
        raise RuntimeError("No embeddings were given")

    if clustering_model_type == "GaussianMixture":
        clustering_model = GaussianMixture(n_components=number_of_clusters, **kwargs)
    elif clustering_model_type == "SpectralClustering":
        clustering_model = SpectralClustering(n_clusters=number_of_clusters, **kwargs)
    elif clustering_model_type == "AgglomerativeClustering":
        clustering_model = AgglomerativeClustering(n_clusters=number_of_clusters, **kwargs)

    clustering_model.fit(embeddings)
    return clustering_model

def get_aic(clustering_model, embeddings):
    if isinstance(clustering_model, GaussianMixture):
        return clustering_model.aic(embeddings)
    else:
        try:
            check_is_fitted(clustering_model)
        except NotFittedError:
            clustering_model.fit(embeddings)

        clusters = clustering_model.labels_

        df = pd.DataFrame(embeddings)
        print(df.head())
        df["cluster"] = clusters
        print(df.head())
        #for sentence_id, cluster_id in enumerate(clustering_model.predict(embeddings)):
        return 0



def find_correct_cluster_number(clustering_model, embeddings, min_n: int = 1, max_n: int = 12, **kwargs):
    dimensions = range(min_n, max_n)
    bic_score = []
    aic_score = []
    for n in dimensions:
        clustering_model = GaussianMixture(n_components=n, random_state=123, n_init=10)
        clustering_model.fit(embeddings)
        bic_score.append(clustering_model.bic(embeddings))
        aic_score.append(clustering_model.aic(embeddings))
    n_components = aic_score.index(min(aic_score)) + 1
    pass
