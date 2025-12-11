#!/usr/bin/env python3
"""
Join NHS medicines (JSONL) with Wikipedia medical subset via normalized titles.

How to run:


spark-submit \
  --master local[12] \
  --driver-memory 32g \
  /app/src/join_nhs_with_wiki.py \
    --nhs /data/extracted_v2.jsonl \
    --wiki /data/enwiki_ns0_medical \
    --output-root /data/nhs_wiki_join_v2


create TWO joins:

  1. NHS medicine page  <=>  Wikipedia article
     - using a normalized key derived from the NHS URL slug
     when,
     concat, 
     coalesce
       (/medicines/aciclovir/  -> "aciclovir")
       and from the Wikipedia title ("Aciclovir" -> "aciclovir")

  2. NHS "Related conditions"  <=>  Wikipedia article
     - using the pre-extracted `related_conditions` array from the NHS JSONL.

Outputs (under --output-root):

  <root>/medicines    – per-medicine joins
  <root>/conditions   – per-condition joins
  <root>/search_docs  – unified search corpus (meds + conditions)
"""

import argparse
import re

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    regexp_extract,
    regexp_replace,
    udf,
    explode_outer,
    lit,
    collect_list,
    struct,
)
from pyspark.sql.types import StringType, ArrayType, StructType, StructField

IMPORTANT_DRUG_PARAMS = {
    # identita
    "name",
    "drug_name",
    "tradename",
    "synonyms",
    "class",
    # použitie / regulácia
    "routes_of_administration",
    "pregnancy_au",
    "pregnancy_us",
    "pregnancy_category",
    "legal_status",
    "legal_au",
    "legal_uk",
    "legal_us",
    "legal_eu",
    "legal_ca",
    # farmakokinetika
    "bioavailability",
    "onset",
    "elimination_half-life",
    "protein_bound",
    "metabolism",
    "excretion",
    # kódy
    "atc_prefix",
    "atc_suffix",
    "atc_supplemental",
    "cas_number",
    "cas_number2",
    "unii",
    "drugbank",
    "chemspiderid",
    "chemspiderid2",
    "pubchem",
    "pubchem_cid",
    "kegg",
    "chebi",
    "chebi2",
    "chembl",
    "chembl2",
    "stdinchikey",
    "stdinchi",
    "smiles",
    # linky
    "medlineplus",
    "dailymedid",
    "iuphar_ligand",
    "ahfs",
}

IMPORTANT_DISEASE_PARAMS = {
    "name",
    "field",
    "specialty",
    "symptoms",
    "signs",
    "duration",
    "causes",
    "cause",
    "risk_factors",
    "diagnosis",
    "diagnostic_method",
    "treatment",
    "management",
    "medication",
    "complications",
    "prevention",
    "frequency",
    "deaths",
    "onset",
}

IMPORTANT_INFOBOX_KEYS = {
    k.lower() for k in (IMPORTANT_DRUG_PARAMS | IMPORTANT_DISEASE_PARAMS)
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nhs",
        required=True,
        help="Path to NHS JSONL (one JSON object per line).",
    )
    parser.add_argument(
        "--wiki",
        required=True,
        help="Path to Wikipedia medical subset parquet (e.g. /data/enwiki_ns0_medical).",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Root directory for output (will create /medicines, /conditions, /search_docs).",
    )
    return parser.parse_args()


def normalize_for_join_py(s):
    """
    Normalize a name/title so that:

      - 'Aciclovir'                 -> 'aciclovir'
      - 'Aciclovir (medication)'    -> 'aciclovir'
      - 'alendronic-acid'           -> 'alendronic acid'
      - 'Chickenpox'                -> 'chickenpox'

    do NOT strip words like 'disease' here; this works both for drugs
    and condition names as long as they match on both sides.
    """
    if s is None:
        return None

    s = s.lower()

    # remove trailing " - nhs" if present
    s = re.sub(r"\s*-\s*nhs$", "", s)

    # remove anything in parentheses
    s = re.sub(r"\([^)]*\)", " ", s)

    # replace non-alphanumeric with spaces
    s = re.sub(r"[^a-z0-9]+", " ", s)

    # collapse multiple spaces and trim
    s = re.sub(r"\s+", " ", s).strip()

    return s or None


normalize_for_join = udf(normalize_for_join_py, StringType())


def _split_infobox_and_body(text: str) -> tuple[str | None, str]:
    """
    Find the first {{Infobox ...}} template and split it out.

    Returns (infobox_text_or_None, remaining_text_without_infobox).
    If no infobox is found, returns (None, original_text).
    """
    if text is None:
        return None, ""

    s = text
    start = s.find("{{Infobox")
    if start == -1:
        return None, s

    pos = start
    depth = 0
    end = None
    n = len(s)

    # crude brace matching for {{ ... }} with nesting
    while pos < n - 1:
        two = s[pos : pos + 2]
        if two == "{{":
            depth += 1
            pos += 2
        elif two == "}}":
            depth -= 1
            pos += 2
            if depth == 0:
                end = pos
                break
        else:
            pos += 1

    if end is None:
        # malformed, just give up and treat as no infobox
        return None, s

    infobox = s[start:end]
    remaining = s[:start] + s[end:]
    return infobox, remaining


def _clean_wikitext_fragment(text: str) -> str:
    """
    Very simple wikitext → plain text cleanup.
    Enough for intro / infobox values, not a full parser.
    """
    if text is None:
        return ""

    # remove refs
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.S)
    text = re.sub(r"<ref[^/>]*/>", " ", text, flags=re.S)

    # remove comments
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)

    # external links [http://... label]
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)

    # internal links [[page|label]] or [[page]]
    text = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)

    # remove simple templates {{...}} (best-effort)
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)

    # strip bold/italics markup
    text = re.sub(r"'{2,}", "", text)

    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_wiki_intro_py(wikitext: str | None) -> str | None:
    """
    Extract the first reasonably clean paragraph from the article body
    (ignoring the infobox).
    """
    if not wikitext:
        return None

    try:
        infobox, body = _split_infobox_and_body(wikitext)
        if not body:
            body = wikitext

        # split into paragraphs by blank lines
        paragraphs = re.split(r"\n\s*\n", body)
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            # skip templates / headings
            if p.startswith("{{") or p.startswith("=="):
                continue
            cleaned = _clean_wikitext_fragment(p)
            if cleaned:
                return cleaned
        return None
    except Exception:
        return None


def extract_wiki_infobox_kv_py(wikitext: str | None):
    """
    Extract the infobox as a list of {'key': ..., 'value': ...} dicts,
    where both key and value are plain text, **filtered** to only important keys.
    """
    if not wikitext:
        return []

    try:
        infobox, _ = _split_infobox_and_body(wikitext)
        if not infobox:
            return []

        kv_list = []
        lines = infobox.split("\n")
        for line in lines:
            line = line.strip()
            if not line.startswith("|"):
                continue

            line = line.lstrip("|").strip()
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip().lower()
            if not key:
                continue

            #  only keep whitelisted keys
            if key not in IMPORTANT_INFOBOX_KEYS:
                continue

            value = _clean_wikitext_fragment(value.strip())
            if not value:
                continue

            kv_list.append({"key": key, "value": value})

        return kv_list
    except Exception:
        return []


infobox_schema = ArrayType(
    StructType(
        [
            StructField("key", StringType(), True),
            StructField("value", StringType(), True),
        ]
    )
)

extract_wiki_intro = udf(extract_wiki_intro_py, StringType())
extract_wiki_infobox = udf(extract_wiki_infobox_kv_py, infobox_schema)


def main():
    args = parse_args()

    spark = (
        SparkSession.builder.appName("JoinNhsWithWikipedia")
        .config("spark.sql.files.maxPartitionBytes", "64m")
        .config("spark.sql.shuffle.partitions", "256")
        .getOrCreate()
    )

    nhs_raw = spark.read.json(args.nhs)

    nhs_pages = nhs_raw.select(
        col("meta.canonical_url").alias("canonical_url"),
        col("data.title").alias("nhs_title"),
        col("data.description").alias("nhs_description"),
        col("data.body_text").alias("nhs_body"),
        col("data.related_conditions").alias("related_conditions"),
        col("meta.date_modified").alias("nhs_date_modified"),
        col("meta.source_path").alias("source_path"),
        col("meta.status").alias("http_status"),
    )

    nhs_meds = nhs_pages.withColumn(
        "slug",
        regexp_extract(
            col("canonical_url"),
            r"/medicines/([^/]+)/?",  # capture the first path segment after /medicines/
            1,
        ),
    )

    # human-friendly medicine name from slug: "alendronic-acid" -> "alendronic acid"
    nhs_meds = nhs_meds.withColumn(
        "nhs_medicine_name",
        regexp_replace(col("slug"), "-", " "),
    )

    # join key for the medicine
    nhs_meds = nhs_meds.withColumn(
        "medicine_key",
        normalize_for_join(col("nhs_medicine_name")),
    )

    wiki_raw = spark.read.parquet(args.wiki)

    wiki = (
        wiki_raw.select(
            "page_id",
            "title",
            "rev_timestamp",
            "wikitext",
            "categories",
        )
        # normalized title key for joining
        .withColumn("title_key", normalize_for_join(col("title")))
        # hort intro paragraph
        .withColumn("wiki_intro", extract_wiki_intro(col("wikitext")))
        # flattened infobox key/value text
        .withColumn("wiki_infobox", extract_wiki_infobox(col("wikitext")))
    )

    # join 1: nhs medicine to wikipedia article

    medicine_join = nhs_meds.join(
        wiki,
        nhs_meds.medicine_key == wiki.title_key,
        how="left",
    ).withColumn(
        "join_type",
        lit("medicine"),
    )

    # join 2: nhs related conditions to wikipedia article

    nhs_conditions = (
        nhs_meds.select(
            "canonical_url",
            "nhs_title",
            "nhs_description",
            "nhs_body",
            "nhs_medicine_name",
            "related_conditions",
        )
        .withColumn("condition_name", explode_outer(col("related_conditions")))
        .withColumn("condition_key", normalize_for_join(col("condition_name")))
    )

    condition_join = nhs_conditions.join(
        wiki,
        nhs_conditions.condition_key == wiki.title_key,
        how="left",
    ).withColumn(
        "join_type",
        lit("condition"),
    )

    root = args.output_root.rstrip("/")
    med_out = f"{root}/medicines"
    cond_out = f"{root}/conditions"

    medicine_join.write.mode("overwrite").parquet(med_out)
    condition_join.write.mode("overwrite").parquet(cond_out)

    # build search_docs (ONE document per NHS page)

    # flatten wiki matches from both joins into a single table
    medicine_wiki = medicine_join.select(
        "canonical_url",
        "page_id",
        "title",
        "wiki_intro",
        "wiki_infobox",
        "categories",
        lit("medicine").alias("wiki_join_type"),
        lit(None).cast("string").alias("condition_name"),
    )

    condition_wiki = condition_join.select(
        "canonical_url",
        "page_id",
        "title",
        "wiki_intro",
        "wiki_infobox",
        "categories",
        lit("condition").alias("wiki_join_type"),
        col("condition_name"),
    )

    wiki_all = medicine_wiki.unionByName(condition_wiki, allowMissingColumns=True)

    # drop exact duplicate wiki rows if they occur
    wiki_all = wiki_all.dropDuplicates(
        ["canonical_url", "page_id", "wiki_join_type", "condition_name"]
    )

    # aggregate into an array of wiki link objects per nhs page
    wiki_agg = wiki_all.groupBy("canonical_url").agg(
        collect_list(
            struct(
                col("page_id"),
                col("title"),
                col("wiki_intro"),
                col("wiki_infobox"),
                col("categories"),
                col("wiki_join_type"),
                col("condition_name"),
            )
        ).alias("wiki")
    )

    base_docs = nhs_meds.select(
        "canonical_url",
        "nhs_title",
        "nhs_description",
        "nhs_body",
        "nhs_medicine_name",
        "related_conditions",
        "nhs_date_modified",
        "source_path",
        "http_status",
    )

    # 4) join aggregated wiki info onto base nhs docs
    search_docs = base_docs.join(wiki_agg, on="canonical_url", how="left")

    search_out = f"{root}/search_docs"
    search_docs.write.mode("overwrite").parquet(search_out)

    json_df = search_docs.withColumn("doc_id", col("canonical_url"))

    json_out_dir = f"{root}/search_docs_jsonl"

    (json_df.coalesce(1).write.mode("overwrite").json(json_out_dir))

    print(f"Wrote search_docs parquet to   : {search_out}")
    print(f"Wrote search_docs JSONL under : {json_out_dir}")

    spark.stop()


if __name__ == "__main__":
    main()
