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
# import psycopg2
from timeit import default_timer as timer

from countries import *


# address of the API endpoint for data loading
API_endpoint = "domain.com/new_api"


def start_timer() -> float:
    """Start timer prior to function call"""
    start = timer()
    return start


def end_timer(start: float, function_name: str) -> None:
    """End timer right after a function and print out its duration"""
    end = timer()
    print(f"Execution time of function {function_name}:\t{end - start}")


def retrieve_dataset(page: int) -> requests.Response():
    """Retrieve a given page of a dataset from API endpoint"""
    response = requests.get(f"{API_endpoint}?page={page}")
    assert response.ok, "Error: Request failed"

    return response


def retrieve_dataset_local(input_file: str) -> pd.DataFrame:
    """Retrieve a given page of a dataset from local storage"""
    try:
        response = pd.read_csv(os.path.join(os.curdir, "Data", input_file))
    except IOError:
        print(f"Cannot open input file: {input_file}")

    return response


def summarize_data(response_data: pd.DataFrame) -> pd.DataFrame:
    """Summarize data from multiple pages into final statistics"""
    # filter out rows with no or invalid status
    response_data = response_data.loc[response_data["status"].isin(["pending", "completed", "failed"])]
    # filter out rows with mnount equal to none and an invalid value in amount
    response_data["amount"] = response_data.to_numeric(response_data["amount"], errors='coerce')
    response_data = response_data.dropna(subset=["amount"])

    # create flags of individual types of transactions
    response_data["amount_pending"] = np.where(response_data["status"] == "pending", response_data["amount"], 0)
    response_data["count_pending"] = np.where(response_data["status"] == "pending", 1, 0)
    response_data["amount_completed"] = np.where(response_data["status"] == "completed", response_data["amount"], 0)
    response_data["failed_over_1M"] = np.where((response_data["status"] == "failed") & (response_data["amount"] > 1_000_000), 1, 0)
    response_data["failed_under_1M"] = np.where((response_data["status"] == "failed") & (response_data["amount"] <= 1_000_000), 1, 0)

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
    transactions_per_country["average_outstanding"] = transactions_per_country["amount_pending_sum"] / transactions_per_country["count_pending_sum"]
    transactions_per_country["total_completed"] = transactions_per_country["amount_completed_sum"]
    transactions_per_country["critical_rate"] = transactions_per_country["failed_over_1M_sum"] / transactions_per_country["status_count"] * 100
    transactions_per_country["error_rate"] = transactions_per_country["failed_under_1M_sum"] / transactions_per_country["status_count"] * 100

    # select columns
    transactions_per_country = transactions_per_country[["country", "average_outstanding", "total_completed", "critical_rate", "error_rate"]]

    return transactions_per_country


def save_on_local(transactions_per_country: pd.DataFrame) -> None:
    """Save output to local storage"""
    transactions_per_country.to_csv("transactions_per_country.csv", index=False, header=True)


def save_to_s3(transactions_per_country: pd.DataFrame, file_key: str) -> None:
    """Save output to S3 bucket"""
    pass


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


def save_to_pg(transactions_per_country: pd.DataFrame) -> None:
    """Save output to PostgreSQL database"""
    # transform dataframe to list of tuples
    transactions_per_country_pg = [tuple(x) for x in transactions_per_country.to_numpy()]
    save_pg_table(transactions_per_country_pg)


def save_result(transactions_per_country: pd.DataFrame, output_type: str) -> None:
    print(transactions_per_country)
    """Save the final dataframe to the selected destination"""
    if output_type == "local":
        save_on_local(transactions_per_country)
    elif output_type == "s3":
        save_to_s3(transactions_per_country, file_key)
    elif output_type == "pg":
        save_to_pg(transactions_per_country)


def main(args: argparse.Namespace) -> None:
    output_type = args.output_type
    input_datasets = sorted(os.listdir(os.path.join(os.curdir, "Data")))
    page = 0

    # while page < 9:
    for page, input_file in enumerate(input_datasets):
    # while page is not None:
        start = start_timer()
        response = retrieve_dataset_local(input_file)
        end_timer(start, "retrieve_dataset_local")

        response_data = response

        # response = retrieve_dataset(page)
        # response_data = response.json().get("data")

        if response_data is not None:
            if page == 0:
                start = start_timer()
                transactions_per_country = summarize_data(response_data)
                end_timer(start, "summarize_data")
            else:
                start = start_timer()
                transactions_per_country = pd.concat([transactions_per_country, summarize_data(response_data)])
                end_timer(start, "summarize_data")
        
        page += 1
        # page = response.json().get("next_page")

    start = start_timer()
    transactions_per_country = aggregate_per_country(transactions_per_country)
    end_timer(start, "aggregate_per_country")

    # save the result to the specified destination 
    start = start_timer()
    save_result(transactions_per_country, output_type)
    end_timer(start, "save_result")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-type', dest='output_type', type=str, required=True, choices=["local", "s3", "pg"],
                    help="Destination of the resulting dataset")
    args = parser.parse_args()

    main(args)
