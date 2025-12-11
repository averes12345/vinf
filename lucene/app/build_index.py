#!/usr/bin/env python3
import json
import os
import lucene

from java.nio.file import Paths
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.document import (
    Document,
    Field,
    StringField,
    TextField,
    StoredField,
    IntPoint,
)
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import FSDirectory


def add_text_field(doc, name, value, store=True):
    if value is None:
        return
    value = str(value).strip()
    if not value:
        return
    doc.add(TextField(name, value, Field.Store.YES if store else Field.Store.NO))


def add_string_field(doc, name, value, store=True):
    if value is None:
        return
    value = str(value).strip()
    if not value:
        return
    doc.add(StringField(name, value, Field.Store.YES if store else Field.Store.NO))


def main():
    lucene.initVM()

    # JSONL produced by join_nhs_with_wiki.py

    jsonl_path = "/data/nhs_wiki_join_v2/search_docs_jsonl/part-00000-07e69562-6371-4456-bd51-c00277c4b1a7-c000.json"
    index_dir = "/data/lucene_index"

    if not os.path.exists(jsonl_path):
        raise SystemExit(f"JSONL file not found: {jsonl_path}")

    os.makedirs(index_dir, exist_ok=True)

    directory = FSDirectory.open(Paths.get(index_dir))
    analyzer = EnglishAnalyzer()
    config = IndexWriterConfig(analyzer)
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)

    writer = IndexWriter(directory, config)

    num_docs = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            doc = Document()

            # --------- ID / metadata ---------
            canonical_url = obj.get("canonical_url")
            doc_id = obj.get("doc_id") or canonical_url

            add_string_field(doc, "id", doc_id, store=True)
            add_string_field(doc, "canonical_url", canonical_url, store=True)

            # --------- NHS fields ----------
            nhs_title = obj.get("nhs_title")
            nhs_description = obj.get("nhs_description")
            nhs_body = obj.get("nhs_body_short") or obj.get("nhs_body")
            nhs_medicine_name = obj.get("nhs_medicine_name")

            add_text_field(doc, "nhs_title", nhs_title)
            add_text_field(doc, "nhs_description", nhs_description)
            add_text_field(doc, "nhs_body", nhs_body)
            add_text_field(doc, "nhs_medicine_name", nhs_medicine_name)

            # NHS related conditions (top-level)
            rel_conds = obj.get("related_conditions") or []
            rel_cond_list = []
            if isinstance(rel_conds, list):
                for rc in rel_conds:
                    add_text_field(doc, "related_condition", rc)
                    rel_cond_list.append(str(rc))

            # We'll collect condition names from both NHS and wiki
            condition_names = set(rel_cond_list)

            # --------- Wikipedia fields from nested `wiki` array ----------
            wiki_entries = obj.get("wiki") or []

            wiki_titles = []
            wiki_intros = []
            wiki_infoboxes = []
            wiki_categories = set()
            join_types = set()
            page_ids = set()

            for w in wiki_entries:
                jt = w.get("wiki_join_type")
                if jt:
                    join_types.add(str(jt))

                cn = w.get("condition_name")
                if cn:
                    condition_names.add(str(cn))

                t = w.get("title")
                if t:
                    wiki_titles.append(str(t))

                wi = w.get("wiki_intro")
                if wi:
                    wiki_intros.append(str(wi))

                infobox_items = w.get("wiki_infobox") or []
                if isinstance(infobox_items, list):
                    for item in infobox_items:
                        if not isinstance(item, dict):
                            continue
                        key = item.get("key")
                        val = item.get("value")
                        if not val:
                            continue
                        # build a "key: value" text snippet for Lucene
                        if key:
                            wiki_infoboxes.append(f"{key}: {val}")
                        else:
                            wiki_infoboxes.append(str(val))

                cats = w.get("categories") or []
                if isinstance(cats, list):
                    for cat in cats:
                        wiki_categories.add(str(cat))

                pid = w.get("page_id")
                if pid is not None:
                    page_ids.add(str(pid))

            # --------- Numeric stats for range queries ---------
            num_related_conditions = len(condition_names)  # NHS + wiki-derived
            num_wiki_categories = len(wiki_categories)
            num_wiki_articles = len(page_ids)

            # Index as IntPoint (for range queries) + StoredField (for display)
            doc.add(IntPoint("num_related_conditions", num_related_conditions))
            doc.add(StoredField("num_related_conditions", num_related_conditions))

            doc.add(IntPoint("num_wiki_categories", num_wiki_categories))
            doc.add(StoredField("num_wiki_categories", num_wiki_categories))

            doc.add(IntPoint("num_wiki_articles", num_wiki_articles))
            doc.add(StoredField("num_wiki_articles", num_wiki_articles))

            # Index aggregated condition_name(s)
            for cn in condition_names:
                add_text_field(doc, "condition_name", cn)

            # Aggregated wiki_title(s)
            if wiki_titles:
                add_text_field(doc, "wiki_title", "; ".join(wiki_titles))

            # Aggregated intro / infobox (truncated)
            if wiki_intros:
                wiki_intro_joined = " ".join(wiki_intros)[:1000]
                add_text_field(doc, "wiki_intro", wiki_intro_joined)

            if wiki_infoboxes:
                wiki_infobox_joined = " ".join(wiki_infoboxes)[:1000]
                add_text_field(doc, "wiki_infobox", wiki_infobox_joined)

            # Categories from all matched wiki articles
            for cat in wiki_categories:
                add_text_field(doc, "category", cat)

            # Store aggregated page_ids (optional, for debugging / analysis)
            if page_ids:
                doc.add(StoredField("page_ids", ",".join(sorted(page_ids))))

            # Derive a simple join_type summary for this NHS page
            join_type_val = None
            if join_types:
                if join_types == {"medicine"}:
                    join_type_val = "medicine"
                elif join_types == {"condition"}:
                    join_type_val = "condition"
                else:
                    join_type_val = "+".join(sorted(join_types))
            add_string_field(doc, "join_type", join_type_val, store=True)

            # --------- Combined field for searching ---------
            combined_pieces = []

            for v in [
                nhs_title,
                nhs_description,
                nhs_body,
                nhs_medicine_name,
                " ".join(condition_names),
                " ".join(wiki_titles),
                " ".join(wiki_intros),
                " ".join(wiki_infoboxes),
            ]:
                if v:
                    combined_pieces.append(str(v))

            combined_pieces.extend(rel_cond_list)
            combined_pieces.extend(wiki_categories)

            combined_text = " ".join(combined_pieces)
            add_text_field(doc, "combined", combined_text, store=False)

            writer.addDocument(doc)
            num_docs += 1

            if num_docs % 500 == 0:
                print(f"Indexed {num_docs} documents...")

    writer.commit()
    writer.close()

    print(f"Index built in {index_dir}")
    print(f"Total documents: {num_docs}")


if __name__ == "__main__":
    main()
