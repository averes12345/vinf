#!/usr/bin/env python3
"""
Convert wikipedia xml dump to a compact parquet (and mainly paralelizable) dataset

Usage:
  spark-submit wikipedia_xml_to_parquet.py \
      --input /path/to/enwiki-latest-pages-articles.xml.bz2 \
      --output /path/to/wikipedia_parquet
"""

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

NUM_OUTPUT_PARTITIONS = 256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Path to Wikipedia XML dump (can be .xml or .xml.bz2)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for Parquet",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    spark = (
        SparkSession.builder.appName("WikipediaXmlToParquet")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.files.maxPartitionBytes", "256m")
        .getOrCreate()
    )

    # read xml: each <page> is a row
    raw_df = (
        spark.read.format("xml")  # use the built-in xml datasource
        .option("rowTag", "page")  # every <page>...</page> = one row
        .load(args.input)
    )

    pages_df = raw_df.select(
        col("id").alias("page_id"),
        col("title").alias("title"),
        col("ns").cast("int").alias("namespace"),
        # redirect element looks like <redirect title="Foo" />
        col("redirect._title").alias("redirect_title"),
        # revision metadata
        col("revision.timestamp").alias("rev_timestamp"),
        # main article body (wikitext)
        col("revision.text._VALUE").alias("wikitext"),
    )

    (
        pages_df.repartition(NUM_OUTPUT_PARTITIONS)
        .write.mode("overwrite")
        .partitionBy(
            "namespace"
        )  # so we can later filter by namespace, but also have acces to other type of pages without rereading the xml
        .parquet(args.output)
    )

    spark.stop()


if __name__ == "__main__":
    main()
