#!/usr/bin/env python

"""Tests for `keyword_clustering_with_transformers` package."""


import unittest
import sys
import os

from keyword_clustering_with_transformers import keyword_clustering_with_transformers
import keyword_clustering_with_transformers.parser as parser
import keyword_clustering_with_transformers.cluster as cluster

from sentence_transformers import util
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


class TestKeyword_clustering_with_transformers(unittest.TestCase):
    """Tests for `keyword_clustering_with_transformers` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.should_values = ["AA", "BB", "CC", "DD", "EE", "FF"]
        self.should_keywords = [
            "business model development",
            "business model development process",
            "business model development template",
            "what is business model innovation",
            "business model innovation case study",
            "business model innovation examples",
            "business model innovation framework",
            "business model innovation",
            "business innovation alignment model",
            "business innovation model",
            "business model innovation questions",
            "business model innovation canvas",
            "business model canvas channels",
            "competitive advantage",
            "cost structure business model canvas",
            "customer journey",
            "customer journey canvas",
            "customer journey map",
            "customer relationship",
            "customer relationship business model canvas",
            "customer segments",
            "customer segments business model canvas",
        ]
        self.should_keywords_short = [
            "business model development",
            "business model development template",
            "what is business model innovation",
            "customer journey map",
            "customer relationship business model canvas",
            "customer segments business model canvas",
        ]
        self.fake_embeddings = np.array(
            [[0, 0, 0, 1], [0, 0, 0, 0.9], [0, 0, 0, 0.95], [1, 0, 0, 0], [0, 1, 0, 0]]
        )
        self.fake_embeddings_clusters = [0, 0, 0, 1, 2]

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def compare_small_clustering(self, labels):
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[0], labels[2])
        self.assertNotEqual(labels[0], labels[3])
        self.assertNotEqual(labels[0], labels[4])
        self.assertNotEqual(labels[3], labels[4])

    def test_000_correctly_parse_csv(self):
        """Test that csv files are read in correctly."""

        # Header=0 reads the first line as header
        table1 = parser.parse_excel("tests/test_table.csv", header=0)
        # Header=None (default) reads the first line as part of the table
        table2 = parser.parse_excel("tests/test_table.csv")
        for i, j in zip(self.should_values, table1.columns):
            assert i == j
        for idx, i in enumerate(self.should_values):
            assert i == table2.iloc[0][idx]

        # test that second line is used as header if header is set.
        table3 = parser.parse_excel("tests/test_table.csv", header=1)
        should_values = [
            "business model development",
            "0.483363012337224",
            "0.884805763949552",
            "Unnamed: 3",
            "0.448984198059888",
            "0.359366484473102",
        ]
        for i, j in zip(should_values, table3.columns):
            assert i == j

    def test_001_correctly_parse_xlsx(self):
        """Test that xlsx files are read in correctly."""
        # Need to cut away the title!
        # Otherwise the line is interpreted as header.
        table1 = parser.parse_excel("tests/test_table.xlsx", header=0, skiprows=2)
        for i, j in zip(self.should_values, table1.columns):
            assert i == j
        # If header is not given (= None), the column names will be
        # interpreted as part of the table!
        table2 = parser.parse_excel("tests/test_table.xlsx", skiprows=2)
        for idx, i in enumerate(self.should_values):
            assert i == table2.iloc[0, idx]

    def test_002_correctly_parse_one_column(self):
        """Test that one-column files are read in correctly."""
        # Need to cut away the title!
        # Otherwise the line is interpreted as header.
        table1 = parser.parse_excel("tests/test_list.csv")
        for i, j in zip(self.should_keywords_short, table1[0]):
            assert i == j

    def test_003_raises_exception_if_filename_empty(self):
        """Test that parser throws if no file is given."""
        self.assertRaises(FileNotFoundError, parser.parse_excel, "")

    def test_004_can_read_keywords_from_file(self):
        """Tests that the keywords are read in correctly."""
        keywords = parser.get_keywords(
            "tests/test_table.xlsx", "AA", skiprows=2, header=0
        )

        for i, j in zip(self.should_keywords, keywords):
            assert i == j

    def test_005_can_load_sentence_transformer(self):
        """Tests that a Sentence Transformer can be loaded."""
        model = cluster.get_model()
        assert model._get_name() == "SentenceTransformer"
        model = cluster.get_model("sentence-transformers/all-MiniLM-L6-v2")
        assert model._get_name() == "SentenceTransformer"

    def test_006_can_calculate_embeddings(self):
        """Tests that a Sentence Transformer can be loaded."""
        model = cluster.get_model()
        embeddings = cluster.calculate_embeddings(
            self.should_keywords_short, model, normalize=True
        )

        test_embedding = model.encode(
            self.should_keywords_short[3],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        similarity = util.pytorch_cos_sim(embeddings[3], test_embedding)
        self.assertAlmostEqual(similarity.numpy()[0][0], 1.0, 3)

    def test_007_can_calculate_self_similarity(self):
        model = cluster.get_model()
        embeddings = cluster.calculate_embeddings(
            self.should_keywords_short[0], model, normalize=True
        )
        self.assertAlmostEqual(cluster.calculate_similarity(embeddings)[0][0], 1.0, 3)

    def test_008_can_calculate_clustering_with_gaussianmixture(self):
        """Tests that the clustering can be calculate with Gaussian Mixture """
        model = cluster.calculate_clustering("GaussianMixture", 3, self.fake_embeddings)
        labels = model.predict(self.fake_embeddings)
        self.compare_small_clustering(labels)

    def test_009_can_calculate_clustering_with_spectralclustering(self):
        """Tests that the clustering can be calculated with spectral clustering """
        model = cluster.calculate_clustering(
            "SpectralClustering", 3, self.fake_embeddings, affinity="rbf", assign_labels="kmeans"
        )
        self.compare_small_clustering(model.labels_)

    def test_010_can_calculate_clustering_with_agglomerativeclustering(self):
        """Tests that the clustering can be calculated with spectral clustering """
        model = cluster.calculate_clustering(
            "AgglomerativeClustering", 3, self.fake_embeddings
        )
        self.compare_small_clustering(model.labels_)

    def test_011_can_calculate_aic_with_gaussianmixture(self):
        """Tests that the AIC can be calculate with Gaussian Mixture """
        aic_scores = []
        for i in range(1,5):
            model = cluster.calculate_clustering("GaussianMixture", i, self.fake_embeddings)
            aic_scores.append(cluster.get_aic(model, self.fake_embeddings))
        # Check that the minimal AIC is with 3 clusters (fake data are tripartite)
        self.assertEqual(aic_scores.index(min(aic_scores)) + 1, 3)

    def test_012_can_calculate_aic_with_spectralclustering(self):
        """Tests that the AIC can be calculate with Spectral Clustering """
        aic_scores = []
        for i in range(1,5):
            model = cluster.calculate_clustering("SpectralClustering", i, self.fake_embeddings)
            aic_scores.append(cluster.get_aic(model, self.fake_embeddings))
        # Check that the minimal AIC is with 3 clusters (fake data are tripartite)
        self.assertEqual(aic_scores.index(min(aic_scores)) + 1, 3)

    def test_013_can_identify_elbow(self):
        rss = [100, 60, 40, 20, 18, 17, 16]
        self.assertEqual(cluster.find_elbow(rss[0]), 1)
        self.assertEqual(cluster.find_elbow(rss[:1]), 1)
        self.assertEqual(cluster.find_elbow(rss[:2]), 2)
        self.assertEqual(cluster.find_elbow(rss[:3]), 2)
        self.assertEqual(cluster.find_elbow(rss[:4]), 3)
        self.assertEqual(cluster.find_elbow(rss), 4)
