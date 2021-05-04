###############################################################################
# Assignment: The dataset consumption
# generate_datasets.py
# Generate files with mock data
# Date: 4. 5. 2020
# Author: Nikola Valesova
###############################################################################

import pandas as pd
import numpy as np
import random
import os

from countries import *


# number of rows per one file of the dataset
row_count = 100_000
# number of files to generate
file_count = 10
# status types
statuses = ["pending", "completed", "failed"]
# destination of the generated dataset
export_file_path = os.path.join("Data", "dataset")


for page in range(file_count):
    # generate a dataframe with random data
    df = pd.DataFrame()
    df["country"] = np.random.choice(countries, row_count)
    df["status"] = np.random.choice(statuses, df.shape[0])
    df["amount"] = np.random.uniform(0, 10_000_000, df.shape[0])
    df["id"] = df.index

    df = df[["id", "country", "status", "amount"]]

    df.to_csv(f"{export_file_path}_{page}.csv", index=False, header=True)
