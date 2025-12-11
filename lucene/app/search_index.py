#!/usr/bin/env python3
import sys

import lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import (
    BooleanClause,
    BooleanQuery,
    BoostQuery,
    IndexSearcher,
)
from org.apache.lucene.store import FSDirectory

INDEX_DIR = "/data/lucene_index"
MAX_HITS = 10


def open_searcher():
    """Initialize JVM and open Lucene index + analyzer."""
    lucene.initVM()
    directory = FSDirectory.open(Paths.get(INDEX_DIR))
    reader = DirectoryReader.open(directory)
    searcher = IndexSearcher(reader)
    analyzer = EnglishAnalyzer()
    return searcher, analyzer, reader


def doc_to_dict(searcher, score_doc):
    stored_fields = searcher.storedFields()
    doc = stored_fields.document(score_doc.doc)

    categories = doc.getValues("category") or []

    return {
        "score": score_doc.score,
        "id": doc.get("id"),
        "join_type": doc.get("join_type"),
        "canonical_url": doc.get("canonical_url"),
        "nhs_title": doc.get("nhs_title"),
        "nhs_description": doc.get("nhs_description"),
        "nhs_medicine_name": doc.get("nhs_medicine_name"),
        "condition_name": doc.get("condition_name"),
        "wiki_title": doc.get("wiki_title"),
        "wiki_intro": doc.get("wiki_intro"),
        "wiki_infobox": doc.get("wiki_infobox"),
        "categories": list(categories),
    }


def print_hit(hit):
    print(f"- score:          {hit['score']:.3f}")
    print(f"  canonical_url:  {hit['canonical_url']}")
    # print(f"  id:             {hit['id']}")
    print(f"  join_type:      {hit['join_type']}")
    print(f"  nhs_title:      {hit['nhs_title']}")
    print(f"  nhs_medicine:   {hit['nhs_medicine_name']}")
    print(f"  condition_name: {hit['condition_name']}")
    print(f"  wiki_title:     {hit['wiki_title']}")

    if hit["categories"]:
        print(f"  categories:     {', '.join(hit['categories'])}")

    if hit.get("wiki_intro"):
        intro = hit["wiki_intro"]
        if len(intro) > 200:
            intro = intro[:200] + "..."
        print(f"  wiki_intro:     {intro}")

    if hit.get("wiki_infobox"):
        ib = hit["wiki_infobox"]
        if len(ib) > 200:
            ib = ib[:200] + "..."
        print(f"  infobox:        {ib}")

    print()


def run_query(searcher, analyzer, query_str, limit=MAX_HITS):
    """
    Vytvoríme pre každý field samostatnú Query pomocou QueryParser(field, analyzer),
    obalíme ju BoostQuery(boost) a všetko spojíme cez BooleanQuery SHOULD.
    """

    # Zoznam polí, ktoré chceme prehľadávať
    fields = [
        "nhs_title",
        "nhs_medicine_name",
        "condition_name",
        "wiki_title",
        "category",
        "nhs_description",
        "nhs_body",
        "wiki_intro",
        "wiki_infobox",
        "combined",
    ]

    # Boosty pre jednotlivé polia
    boosts = {
        "nhs_title": 4.0,
        "nhs_medicine_name": 3.0,
        "condition_name": 2.5,
        "wiki_title": 2.0,
        "category": 1.5,
        "nhs_description": 1.2,
        "nhs_body": 0.8,
        "wiki_intro": 0.8,
        "wiki_infobox": 0.5,
        "combined": 0.5,
    }

    bq_builder = BooleanQuery.Builder()

    for field in fields:
        try:
            parser = QueryParser(field, analyzer)
            sub_query = parser.parse(query_str)
        except Exception:
            # ak sa pre nejaké pole nepodarí query spracovať, proste ho preskočíme
            continue

        boost = boosts.get(field, 1.0)
        if boost != 1.0:
            sub_query = BoostQuery(sub_query, float(boost))

        bq_builder.add(sub_query, BooleanClause.Occur.SHOULD)

    query = bq_builder.build()

    hits = searcher.search(query, limit).scoreDocs
    return [doc_to_dict(searcher, sd) for sd in hits]


def interactive_loop(searcher, analyzer, reader):
    print("Interactive PyLucene search")
    print("Index directory:", INDEX_DIR)
    print()
    print("Type Lucene queries and press Enter.")
    print("Examples:")
    print("  diabetes treatment")
    print('  nhs_title:"type 2 diabetes"          (phrase query)')
    print("  wiki_title:aspiriin~                   (fuzzy query)")
    print("  combined:insulin AND category:diabetes (boolean query)")
    print()
    print("Commands:")
    print("  :q   or :quit   → exit")
    print()

    while True:
        try:
            line = input("lucene query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not line:
            continue
        if line in {":q", ":quit", "q", "quit", "exit"}:
            break

        try:
            hits = run_query(searcher, analyzer, line, limit=MAX_HITS)
        except Exception as e:
            print(f"Error parsing/executing query: {e}")
            continue

        print(f"Found {len(hits)} hits.")
        for h in hits:
            print_hit(h)

    reader.close()


def main():
    searcher, analyzer, reader = open_searcher()

    # If user passed a query on the CLI, run once and exit
    if len(sys.argv) > 1:
        query_str = " ".join(sys.argv[1:])
        print(f"[One-off search] {query_str}")
        hits = run_query(searcher, analyzer, query_str)
        print(f"Found {len(hits)} hits.")
        for h in hits:
            print_hit(h)
        reader.close()
        return

    # Otherwise, interactive mode
    interactive_loop(searcher, analyzer, reader)


if __name__ == "__main__":
    main()
