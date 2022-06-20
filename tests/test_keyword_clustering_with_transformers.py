#!/usr/bin/env python

"""Tests for `keyword_clustering_with_transformers` package."""


import unittest
import sys
import os

from keyword_clustering_with_transformers import keyword_clustering_with_transformers
import keyword_clustering_with_transformers.parser as parser

sys.path.insert(0, os.path.dirname(__file__))


class TestKeyword_clustering_with_transformers(unittest.TestCase):
    """Tests for `keyword_clustering_with_transformers` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_correctly_parse_csv(self):
        """Test that csv files are read in correctly."""
        should_values = ["AA", "BB", "CC", "DD", "EE", "FF"]

        # Header=0 reads the first line as header
        table1 = parser.parse_excel("tests/test_table.csv", header=0)
        # Header=None (default) reads the first line as part of the table
        table2 = parser.parse_excel("tests/test_table.csv")
        for i, j in zip(should_values, table1.columns):
            assert i == j
        for idx, i in enumerate(should_values):
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
        should_values = ["AA", "BB", "CC", "DD", "EE", "FF"]
        for i, j in zip(should_values, table1.columns):
            assert i == j
        # If header is not given (= None), the column names will be
        # interpreted as part of the table!
        table2 = parser.parse_excel("tests/test_table.xlsx", skiprows=2)
        for idx, i in enumerate(should_values):
            assert i == table2.iloc[0, idx]

    def test_002_raises_exception_if_filename_empty(self):
        """Test that parser throws if no file is given."""
        self.assertRaises(FileNotFoundError, parser.parse_excel, "")

    def test_003_can_read_keywords_from_file(self):
        keywords = parser.get_keywords(
            "tests/test_table.xlsx", "AA", skiprows=2, header=0
        )

        should_keywords = [
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
        for i, j in zip(should_keywords, keywords):
            assert i == j
