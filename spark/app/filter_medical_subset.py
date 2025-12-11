#!/usr/bin/env python3
"""
Filter Wikipedia namespace=0 (articles) with categories into a smaller
'medical-ish' subset using positive/negative keyword heuristics on
category names.

Logic:
  - For each category c:
      * if doesntmatch any negative keyword -> keep
      * else -> drop
  - For each page:
      * keep only the categories that passed
      * drop the page if it ends up with 0 kept categories

Example usage inside container:

  spark-submit \
    --master local[12] \
    --driver-memory 32g \
    /app/src/filter_medical_subset.py \
      --input /data/enwiki_ns0_with_categories \
      --output /data/enwiki_ns0_medical \
      --json-output /data/enwiki_ns0_medical_json
"""

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, size, col


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Input parquet with namespace=0 and a `categories` array column.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output parquet path for the filtered subset.",
    )
    parser.add_argument(
        "--json-output",
        required=False,
        help=(
            "Optional: output directory for JSON version of the filtered subset "
            "(written uncompressed, to approximate plaintext size)."
        ),
    )
    return parser.parse_args()


positive_keywords = [
    "disease",
    "diseases",
    "disorder",
    "disorders",
    "syndrome",
    "syndromes",
    "condition",
    "conditions",
    "signs",
    "symptoms",
    "infection",
    "infections",
    "infectious",
    "cancer",
    "menopause",
    "pregnancy",
    "fertility",
    "obstetrics",
    "gynecology",
    "allergology",
    "endocrine",
    "endocrinology",
    "immunology",
    "rheumatology",
    "dermatology",
    "gastroenterology",
    "nephrology",
    "pulmonology",
    "psychiatry",
    "neurology",
    "pediatrics",
    "geriatrics",
    "cardiovascular",
    "autoimmune",
    "amyloidosis",
    "spondylitis",
    "abdominal pain",
    "ailments",
    "abdomen",
    "acneiform eruptions",
    "acute pain",
    "adverse childhood experiences",
    "arthritis",
    "arthropod attacks",
    "asthma",
    "aphasias",
    "audiology",
    "bacterial vaginosis",
    "vitamin",
    "medicine",
    "medicines",
    "medical",
    "drug",
    "drugs",
    "pharmacology",
    "pharmaceutical",
    "antibiotic",
    "antibiotics",
    "analgesic",
    "analgesics",
    "vaccine",
    "vaccines",
    "chemotherapy",
    "therapy",
    "treatment",
    "therapeutic",
    "toxicology",
    "public health",
    "diabetes association",
    "heart association",
    "bioethics",
    "biologically based therapies",
    "inhibitor",
    "inhibitors",
    "agonist",
    "agonists",
    "antagonist",
    "antagonists",
    "modulator",
    "modulators",
    "receptor",
    "abortifacients",
    "ace inhibitors",
    "antiviral",
    "antivirals",
    "antifungal",
    "antifungals",
    "antacids",
    "anthelmintics",
    "antidepressants",
    "anticonvulsants",
    "antidiarrhoeals",
    "antihypertensive agents",
    "antihypotensive agents",
    "anti-inflammatory agents",
    "antianginals",
    "antiemetics",
    "antidotes",
    "antidyskinetic agents",
    "antigout agents",
    "antifibrinolytics",
    "antifolates",
    "antimigraine",
    "anti-acne",
    "anti-aging substances",
    "antidementia agents",
    "antidiuretics",
    "antihistamines",
    "antimalarial agents",
    "antimetabolites",
    "antimicrobials",
    "antimineralocorticoids",
    "antioxidants",
    "antiparkinsonian agents",
    "antiprotozoal agents",
    "antipruritics",
    "antipyretics",
    "antirheumatic products",
    "antiseptics",
    "antitussives",
    "anxiolytics",
    "aphrodisiacs",
    "appetite stimulants",
    "atypical antipsychotics",
    "atypical pneumonias",
    "b vitamins",
    "betaherpesvirinae",
    "alkaloids",
    "amino acids",
    "amino acid derivatives",
    "amides",
    "acetamides",
    "acetic acids",
    "acetals",
    "anthranilic acids",
    "alcohols",
    "esters",
    "compounds",
    "derivatives",
    "tetralins",
    "piperazines",
    "piperidines",
    "morphinans",
    "oxazolidinones",
    "tryptamines",
    "androstane",
    "amphenicols",
    "adamantanes",
    "amidines",
    "amines",
    "aromatic amines",
    "aromatic ethers",
    "aromatic ketones",
    "amphetamine",
    "amphetamines",
    "benzoic acids",
    "benzodiazepines",
    "benzimidazoles",
    "benzothiadiazines",
    "benzonitriles",
    "beta blockers",
    "beta-lactams",
    "biguanides",
    "biphenyls",
    "antibiotic",
    "antiviral",
    "antifungal",
    "antiparasitic",
    "anticoagulant",
    "anticonvulsant",
    "antipsychotic",
    "anesthetic",
    "diuretic",
    "sedative",
    "stimulant",
    "opioid",
    "opiates",
    "antidepressant",
    "disease",
    "syndrome",
    "condition",
    "infection",
    "inflammation",
    "injuries",
    "neoplasm",
    "cancer",
    "carcinogen",
    "endocrine",
    "hormone",
    "metabolism",
    "palliative care",
    "nursing",
    "diagnoses",
    "pharma",
    "pharmaceutical",
    "biotech",
    "toxin",
    "toxicant",
    "hepatotoxins",
    "nephrotoxins",
    "syndrome",
    "disease",
    "disorder",
    "infection",
    "arthritis",
    "glaucoma",
    "migraine",
    "pneumonia",
    "hepatitis",
    "cystic fibrosis",
    "ulcerative colitis",
    "diabetes",
    "stroke",
    "psychosis",
    "schizophrenia",
    "insomnia",
    "narcolepsy",
    "dementia",
    "epilepsy",
    "hypertension",
    "constipation",
    "diarrhea",
    "vomiting",
    "cough",
    "pain",
    "headache",
    "bleeding",
    "contact dermatitis",
    "seborrheic dermatitis",
    "lichenoid",
    "urticaria",
    "erythema",
    "antiarrhythmic",
    "decongestant",
    "hypnotic",
    "vasodilator",
    "diuretic",
    "sympathomimetic",
    "neuroprotective",
    "muscle relaxant",
    "statins",
    "beta blockers",
    "calcium channel blockers",
    "potassium channel blockers",
    "glucocorticoids",
    "mineralocorticoids",
    "progestogens",
    "prolactin",
    "pancreatic hormones",
    "psychosis",
    "hallucinogens",
    "monoclonal antibodies",
    "recombinant proteins",
    "oncology",
    "otorhinolaryngology",
    "rhinology",
    "laryngology",
    "gynaecology",
    "pedodontology",
    "palliative care",
    "oral mucosal pathology",
]

negative_keywords = [
    "cedar fair attractions",
    "cedar point",
    "knott's berry farm",
    "fishkeeping",
    "internet memes",
    "crust and d-beat groups",
    "ska groups",
    "slowcore groups",
    "one-man bands",
    "ryan adams",
    "inventions",
    "record labels",
    "artists",
    "musical",
    "music videos",
    "music",
    "quartets",
    "trios",
    "punk groups",
    "metal",
    "rock",
    "rap",
    "rappers",
    "singers",
    "albums",
    "magazines",
    "books by",
    "books",
    "fiction",
    "novels",
    "comics",
    "characters created by",
    "games",
    "video games",
    "switch games",
    "macos games",
    "windows games",
    "interactive entertainment",
    "symmetry orchestra",
    "opera",
    "cultural depictions of",
    "eddic poetry",
    "neo-noir",
    "verse contests",
    "works by george f. kerr",
    "culture",
    "in popular culture",
    "mythology",
    "mythological",
    "monsters",
    "constellations",
    "ptolemy",
    "astronomy",
    "genera",
    "taxonomy",
    "gastropod",
    "decapod",
    "crabs in culture",
    "dance companies",
    "companies in",
    "artists",
    "record labels",
    "broadcast",
    "radio",
    "sports teams",
    "kraken",
    "rowing",
    "awards",
    "grammy",
    "press books",
    "technology",
    "programming",
    "ipod",
    "consumer brands",
    "video game",
    "video games",
    "action games",
    "road movies",
    "films",
    "film",
    "movies",
    "drama films",
    "thriller films",
    "romantic thriller films",
    "television",
    "tv series",
    "episodes",
    "episode",
    "plays",
    "album",
    "albums",
    "soundtrack",
    "soundtrack albums",
    "song",
    "songs",
    "single",
    "singles",
    "ep ",
    " eps",
    "musical groups",
    "rock groups",
    "metal musical groups",
    "death metal",
    "post-rock",
    "indie rock",
    "quintets",
    "shoegaze",
    "grindcore",
    "novel",
    "novels",
    "non-fiction books",
    "horror novels",
    "fantasy novels",
    "comics",
    "comics debuts",
    "comics endings",
    "in fiction",
    "female supervillains",
    "amusement rides",
    "roller coasters",
    "rides closed",
    "rides introduced",
    "robots",
    "academic journals established",
    "academic journals associated with learned and professional societies",
    "academic journals published by",
    "bimonthly journals",
    "8 times per year journals",
    "bmj group academic journals",
    "journal",
    "journals",
    "dupont–columbia university award winners",
    "award winners",
    "establishments in",
    "disestablishments in",
    "introductions",
    "neologisms",
    "births",
    "in science",
    "in biology",
    "in the united states",
    "in germany",
    "in west bengal",
    "articles containing video clips",
    "original programming",
    "broadcasting corporation original programming",
    "ballads",
    "compositions",
    "radio stations",
    "radio station",
    "sports",
    "football",
    "footballers",
    "basketball",
    "hockey",
    "clubs",
    "teams",
    "astronomical myths",
    "berkshire in fiction",
    "american horror novels",
    "constellation",
    "fauna of",
    "ludvika",
    "people from",
    "family",  # biology
    "taxa named by",
    "extant miocene first appearances",
    "fauna of",
    "insects",
    "cancroidea",
    "prayidae",
    "psenidae",
    "vertiginidae",
    "cookham",
    "mytholog",
    "heracles",
    "ritual",
    "magical thinking",
    "illusions",
    "phenomena",
    "middle age",
    "fear",
    "non-sexuality",
    "products introduced",
    "living people",
    "taxa named by",
    "hairdressing",
    "latin words and phrases",
    "food additives",
    "homelessness",
    "cnidarians",
    "horse anatomy",
    "oligohymenophorea",
    "william osler",
    "unprintworthy redirects",
    "redirects from unnecessary disambiguation",
    "history of psychology",
    "emotional issues",
    "learning disabilities",
    "mass media portrayals",
    "military sociology",
    "military veterans",
    "underwater diving hazards",
    "effects of external causes",
]


def escape_sql_like_literal(kw: str) -> str:
    return kw.replace("'", "''")


def main():
    args = parse_args()

    spark = (
        SparkSession.builder.appName("FilterWikipediaMedicalSubsetHeuristic")
        .config("spark.sql.files.maxPartitionBytes", "64m")
        .config("spark.sql.shuffle.partitions", "256")
        .getOrCreate()
    )

    df = spark.read.parquet(args.input)

    # sanity check to do the filtering we need categories
    if "categories" not in df.columns:
        raise ValueError("Input data must have a 'categories' array column.")

    # pos_parts = [
    #     f"lower(c) LIKE '%{escape_sql_like_literal(kw)}%'" for kw in positive_keywords
    # ]
    # pos_cond_sql = " OR ".join(pos_parts)

    neg_parts = [
        f"lower(c) LIKE '%{escape_sql_like_literal(kw)}%'" for kw in negative_keywords
    ]
    neg_cond_sql = " OR ".join(neg_parts)

    # keep only categories where:
    # filter out negative keyword matches
    # already filtered the categories to contain only sensible information, but did not want to inlude another xyz keywords in positive

    keep_categories_sql = f"""
      filter(
        categories,
        c -> NOT ({neg_cond_sql})
      )
    """

    df_with_med_cats = df.withColumn("categories_medical", expr(keep_categories_sql))

    # keep only pages that still have at least one "medical" category
    filtered = df_with_med_cats.where(size(col("categories_medical")) > 0)

    # sanity check
    total = df.count()
    kept = filtered.count()
    print(f"Total ns0 rows: {total}")
    print(f"Kept medical-ish rows: {kept} (~{kept/total:.2%})")

    # write Parquet
    filtered.write.mode("overwrite").parquet(args.output)

    # JSON output (uncompressed-ish so we know aproximate size (would be higher in xml, but I cant be bothered to again decompress and filter the wikipedia dump))
    if args.json_output:
        (
            filtered.write.mode("overwrite")
            .option("compression", "none")
            .json(args.json_output)
        )

    spark.stop()


if __name__ == "__main__":
    main()
