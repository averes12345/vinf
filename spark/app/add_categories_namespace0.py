#!/usr/bin/env python3
"""
Read Wikipedia Parquet (partitioned by namespace), load only namespace=0
(main articles), extract [[Category:...]] tags from wikitext using Spark SQL
functions, and store them as a `categories` column.

Usage inside container:

  spark-submit --master local[12] --driver-memory 32g /app/src/add_categories_namespace0_sql.py \
      --input-root /data/enwiki_parquet \
      --output /data/enwiki_ns0_with_categories
"""

import argparse
from pyspark.sql import SparkSession


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        required=True,
        help=(
            "Root directory of Wikipedia parquet with namespace=... subdirs, "
            "e.g. /data/enwiki_parquet"
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output directory for the enriched parquet, "
            "e.g. /data/enwiki_ns0_with_categories"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    spark = (
        SparkSession.builder.appName("WikipediaNamespace0WithCategoriesSQL")
        # smaller file splits -> smaller tasks -> less peak memory per task
        .config("spark.sql.files.maxPartitionBytes", "64m")
        .config("spark.sql.shuffle.partitions", "256")
        .getOrCreate()
    )

    ns0_path = f"{args.input_root}/namespace=0"

    # only load columns we actually care about
    base_df = spark.read.parquet(ns0_path).select(
        "page_id", "title", "rev_timestamp", "wikitext"
    )

    # Expression:
    #  - regexp_extract_all(...)       -> array of raw category strings
    #  - transform(...)                -> clean each:
    #       * trim
    #       * drop fragment after '#'
    #       * replace '_' with ' '
    #  - array_distinct(...)           -> deduplicate per page

    categories_expr = r"""
    array_distinct(
      transform(
        regexp_extract_all(
          wikitext,
          '\\[\\[\\s*Category\\s*:\\s*([^|\\]]+)',
          1
        ),
        cat -> trim(
          regexp_replace(
            regexp_replace(cat, '#.*$', ''),
            '_',
            ' '
          )
        )
      )
    )
    """

    df_with_categories = base_df.selectExpr(
        "page_id",
        "title",
        "rev_timestamp",
        f"{categories_expr} as categories",
        "wikitext",
    )

    (df_with_categories.write.mode("overwrite").parquet(args.output))

    spark.stop()


if __name__ == "__main__":
    main()
