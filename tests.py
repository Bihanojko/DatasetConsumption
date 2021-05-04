###############################################################################
# Assignment: The dataset consumption
# tests.py
# Unit tests testing inidivdual functions
# Date: 4. 5. 2020
# Author: Nikola Valesova
###############################################################################

import unittest
import os
import random
import pandas as pd
import inspect
import sys
from pandas.testing import assert_frame_equal


from dataset_consumption import summarize_data, aggregate_per_country, save_on_local
from countries import *


# data = {
#     "country": countries,
#     "average_outstanding": [random.randrange(0, 10_000_000) for _ in range(len(countries))],
#     "total_completed": [random.randrange(0, 100_000_000) for _ in range(len(countries))],
#     "critical_rate": [random.random() * 100 for _ in range(len(countries))],
#     "error_rate": [random.random() * 100 for _ in range(len(countries))],
# }

# df = pd.DataFrame(data) 

# df.to_csv("test_save_on_local.csv", index=False, header=True)


class TestDataSummarization(unittest.TestCase):
    def setUp(self):
        self.test_folder = "Tests"
        self.test_input_filename = "test_summarization.csv"
        self.test_solution_filename = "test_summarization_result.csv"

        try:
            self.correct_result = pd.read_csv(os.path.join(self.test_folder, self.test_solution_filename))
        except IOError:
            print(f"Cannot open solution test file {self.test_solution_filename}")

    def test_summarization(self):
        """Test that the data summarization logic is correct"""
        try:
            data = pd.read_csv(os.path.join(self.test_folder, self.test_input_filename))
        except IOError:
            print(f"Cannot open input test file {self.test_input_filename}")

        result = summarize_data(data)

        assert_frame_equal(self.correct_result, result)


class TestAggregationByCountry(unittest.TestCase):
    def setUp(self):
        self.test_folder = "Tests"
        self.test_input_filename = "test_aggregation.csv"
        self.test_solution_filename = "test_aggregation_result.csv"

        try:
            self.correct_result = pd.read_csv(os.path.join(self.test_folder, self.test_solution_filename))
        except IOError:
            print(f"Cannot open solution test file {self.test_solution_filename}")

    def test_aggregation(self):
        """Test that aggregation by country is correct"""
        try:
            data = pd.read_csv(os.path.join(self.test_folder, self.test_input_filename))
        except IOError:
            print(f"Cannot open input test file {self.test_input_filename}")

        result = aggregate_per_country(data)

        assert_frame_equal(self.correct_result, result)


class TestSavingOnLocal(unittest.TestCase):
    def setUp(self):
        self.test_folder = "Tests"
        self.test_input_filename = "test_save_on_local.csv"
        self.test_solution_filename = "test_save_on_local_result.csv"

        try:
            self.correct_result = pd.read_csv(os.path.join(self.test_folder, self.test_solution_filename))
        except IOError:
            print(f"Cannot open solution test file {self.test_solution_filename}")

    def test_file_created(self):
        """Test that the file is created and is in the right location"""
        # delete the file if it exists
        if "transactions_per_country.csv" in os.listdir(os.curdir):
            os.remove("transactions_per_country.csv")

        data = pd.read_csv(os.path.join(self.test_folder, "test_save_on_local.csv"))
        save_on_local(data)

        self.assertIn("transactions_per_country.csv", os.listdir(os.curdir))


if __name__ == '__main__':
    unittest.main()
