###############################################################################
# Assignment: The dataset consumption
# dataset_consumption.py
# Date: 4. 5. 2020
# Author: Nikola Valesova
###############################################################################

import pandas as pd
import numpy as np
import argparse
import os
import requests
import psycopg2
import functools
import time
import logging

from countries import *


# address of the API endpoint for data loading
API_endpoint = "http://domain.com/new_api"


def allow_logging(func):
    """Wrapper for logging - announces function start with its name and function end including its duration"""
    functools.wraps(func)

    def wrapper(*args, **kwargs):
        logger = logging.getLogger()
        logger.setLevel("DEBUG")
        start = time.perf_counter()
        logging.info(f"### Starting function: {func.__name__}")
        value = func(*args, **kwargs)
        end = time.perf_counter()
        logging.info(f"### Execution time of function {func.__name__}: {end - start:.3f} seconds")

        return value

    return wrapper


@allow_logging
def retrieve_dataset(page: int) -> requests.Response():
    """Retrieve a given page of a dataset from API endpoint"""
    response = requests.get(f"{API_endpoint}?page={page}")
    assert response.ok, "Error: Request failed"

    return response


@allow_logging
def retrieve_dataset_local(input_file: str) -> pd.DataFrame:
    """Retrieve a given page of a dataset from local storage"""
    try:
        response = pd.read_csv(os.path.join(os.curdir, "Data", input_file))
    except IOError:
        print(f"Cannot open input file: {input_file}")
        return None

    return response


@allow_logging
def summarize_data(response_data: pd.DataFrame) -> pd.DataFrame:
    """Summarize data from multiple pages into final statistics"""
    # filter out rows with no or invalid status
    response_data = response_data.loc[response_data["status"].isin(["pending", "completed", "failed"])]
    # filter out rows with amount equal to none and an invalid value in amount
    response_data = response_data.assign(amount=pd.to_numeric(response_data["amount"], errors='coerce'))
    response_data = response_data.dropna(subset=["amount"])

    # create flags of individual types of transactions
    response_data = response_data.assign(amount_pending=np.where(response_data["status"] == "pending", response_data["amount"], 0))
    response_data = response_data.assign(count_pending=np.where(response_data["status"] == "pending", 1, 0))
    response_data = response_data.assign(amount_completed=np.where(response_data["status"] == "completed", response_data["amount"], 0))
    response_data = response_data.assign(failed_over_1M=np.where((response_data["status"] == "failed") & (response_data["amount"] > 1_000_000), 1, 0))
    response_data = response_data.assign(failed_under_1M=np.where((response_data["status"] == "failed") & (response_data["amount"] <= 1_000_000), 1, 0))

    # get total counts of given types of transactions per country
    counts_per_country = (
        response_data
        .groupby(["country"])
        .agg({
            "amount_pending": ["sum"],
            "count_pending": ["sum"],
            "amount_completed": ["sum"],
            "failed_over_1M": ["sum"],
            "failed_under_1M": ["sum"],
            "status": ["count"],
            })
        .reset_index()
    )

    # rename columns, add the type of aggregation to each column name
    counts_per_country.columns = ['_'.join(colnames) if "country" not in colnames else "country" for colnames in counts_per_country.columns.values]

    return counts_per_country


@allow_logging
def load_and_process_data_local() -> pd.DataFrame:
    """Load data from local storage page per page and compute transaction statistics per country from them"""
    input_datasets = sorted(os.listdir(os.path.join(os.curdir, "Data")))

    for page, input_file in enumerate(input_datasets):
        response = retrieve_dataset_local(input_file)

        if response is None:
            return None

        if page == 0:
            transactions_per_country = summarize_data(response)
        else:
            transactions_per_country = pd.concat([transactions_per_country, summarize_data(response)])
    
    return transactions_per_country


def load_and_process_data_rest_api() -> pd.DataFrame:
    """Load data from REST API page per page and compute transaction statistics per country from them"""
    page = 0

    while page is not None:
        response = retrieve_dataset(page)

        response_data = response.json().get("data")

        if response_data is not None:
            if page == 0:
                transactions_per_country = summarize_data(response_data)
            else:
                transactions_per_country = pd.concat([transactions_per_country, summarize_data(response_data)])
        
        page = response.json().get("next_page")
    
    return transactions_per_country


@allow_logging
def aggregate_per_country(transactions_per_country: pd.DataFrame) -> pd.DataFrame:
    """Aggregate counts from individual pages to statistics of the entire dataset"""
    # get total counts of given types of transactions per country
    transactions_per_country = (
        transactions_per_country
        .groupby("country")
        .agg({
            "amount_pending_sum": ["sum"],
            "count_pending_sum": ["sum"],
            "amount_completed_sum": ["sum"],
            "failed_over_1M_sum": ["sum"],
            "failed_under_1M_sum": ["sum"],
            "status_count": ["sum"],
        })
        .reset_index()
    )

    transactions_per_country.columns = [colnames[0] for colnames in transactions_per_country.columns.values]

    # compute statistics per country
    transactions_per_country = transactions_per_country.assign(average_outstanding=transactions_per_country["amount_pending_sum"] / transactions_per_country["count_pending_sum"])
    transactions_per_country = transactions_per_country.assign(total_completed=transactions_per_country["amount_completed_sum"])
    transactions_per_country = transactions_per_country.assign(critical_rate=transactions_per_country["failed_over_1M_sum"] / transactions_per_country["status_count"] * 100)
    transactions_per_country = transactions_per_country.assign(error_rate=transactions_per_country["failed_under_1M_sum"] / transactions_per_country["status_count"] * 100)

    # select columns
    transactions_per_country = transactions_per_country[["country", "average_outstanding", "total_completed", "critical_rate", "error_rate"]]

    return transactions_per_country


@allow_logging
def save_on_local(transactions_per_country: pd.DataFrame) -> None:
    """Save output to local storage"""
    transactions_per_country.to_csv("transactions_per_country.csv", index=False, header=True)


@allow_logging
def save_to_s3(transactions_per_country: pd.DataFrame, file_key: str) -> None:
    """Save output to S3 bucket"""
    pass


@allow_logging
def save_pg_table(country_transactions):
    """Create table in the PostgreSQL database and store the resulting dataframe"""
    drop_table_command = "DROP TABLE country_transactions"
    create_table_command = """
        CREATE TABLE IF NOT EXISTS country_transactions (
            country VARCHAR(255) PRIMARY KEY,
            average_outstanding DECIMAL NOT NULL,
            total_completed DECIMAL NOT NULL,
            critical_rate DECIMAL NOT NULL,
            error_rate DECIMAL NOT NULL
        )
    """
    insert_command = "INSERT INTO country_transactions(country, average_outstanding, total_completed, critical_rate, error_rate) VALUES(%s)"

    conn = None

    try:
        # read the connection parameters
        params = config()
        # connect to the PostgreSQL server
        conn = psycopg2.connect(**params)
        # create a new cursor
        cur = conn.cursor()

        # drop table if it already exists
        # currently, no historization is implemented, only the most recent version is stored
        cur.execute(drop_table_command)

        # create the table
        cur.execute(create_table_command)

        # execute the INSERT statement
        cur.executemany(insert_command, country_transactions)

        # commit the changes to the database
        conn.commit()
        # close communication with the PostgreSQL database server
        cur.close()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


@allow_logging
def save_to_pg(transactions_per_country: pd.DataFrame) -> None:
    """Save output to PostgreSQL database"""
    # transform dataframe to list of tuples
    transactions_per_country_pg = [tuple(x) for x in transactions_per_country.to_numpy()]
    save_pg_table(transactions_per_country_pg)


@allow_logging
def save_result(transactions_per_country: pd.DataFrame, output_type: str, file_key: str = None) -> None:
    """Save the final dataframe to the selected destination"""
    if output_type == "local":
        save_on_local(transactions_per_country)
    elif output_type == "s3":
        save_to_s3(transactions_per_country, file_key)
    elif output_type == "pg":
        save_to_pg(transactions_per_country)


def main(args: argparse.Namespace) -> None:
    input_type = args.input_type
    output_type = args.output_type

    if input_type == "local":
        transactions_per_country = load_and_process_data_local()
            
    elif input_type == "rest-api":
        transactions_per_country = load_and_process_data_rest_api()

    if transactions_per_country is None:
        return None

    transactions_per_country = aggregate_per_country(transactions_per_country)

    # save the result to the specified destination 
    save_result(transactions_per_country, output_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-type', dest='input_type', type=str, required=False, choices=["local", "rest-api"], default="rest-api",
                    help="Destination of the resulting dataset")
    parser.add_argument('--output-type', dest='output_type', type=str, required=True, choices=["local", "s3", "pg"],
                    help="Destination of the resulting dataset")
    args = parser.parse_args()

    main(args)
