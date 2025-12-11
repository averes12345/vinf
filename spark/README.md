# Proces práce s wiki dátami
Najprv som zkonvertoval dump wikipedie do niecoho s cim vie spark pekne pracovat a teda som vytvorili zo vsetkých stránok Parquetes ktoré boli roztriedené na základe wikipedia ns.
ls wiki_data/enwiki_parquet/
'namespace=0'  'namespace=10'  'namespace=100'  'namespace=118'  'namespace=12'  'namespace=126'  'namespace=14'  'namespace=1728'  'namespace=4'  'namespace=6'  'namespace=710'  'namespace=8'  'namespace=828'   _SUCCESS
Z wikipédie som sa rozhodol vytiahnuť polia:
- 
- 
- 
- 
- 

We first decided not to filter infoboxes but take all of them. Then in spark we display all the infoboxes and pick those, which are relevant
how do we find, which infoboxes we care about?

# Filtrovanie kategorii
+------------------+-----+                                                      
|         relevance|count|
+------------------+-----+
|likely_non_medical|  739|
|    likely_medical|  276|
|           unknown|  975|
+------------------+-----+

# explode wiki array, then infobox array, then take distinct keys
infobox_keys_df = (
search_docs
.select(explode(col("wiki")).alias("w"))
.select(explode(col("w.wiki_infobox")).alias("kv"))
.select(col("kv.key").alias("key"))
.distinct()
)

infobox_keys_df.show(10000, truncate=False)

# Ako máme veľa distinct wiki stránok sa nám podarilo pripojiť?
# Cód
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, countDistinct

spark = SparkSession.builder.getOrCreate()

search_docs = spark.read.parquet("/data/nhs_wiki_join_v2/search_docs")

# Each row = NHS page; "wiki" is an array of joined wiki pages
wiki_links = search_docs.select(explode("wiki").alias("w"))

num_unique_wiki = (
    wiki_links
    .where(col("w.page_id").isNotNull())
    .agg(countDistinct("w.page_id").alias("num_unique_wiki"))
    .collect()[0]["num_unique_wiki"]
)

print(f"Unique Wikipedia pages linked to NHS pages: {num_unique_wiki}")

# Odpoveď

# Ako veľa nhs stránok majú pripojenú aspoň jednu wiki stránku?

# Cód
num_nhs_with_wiki = (
    search_docs
    .where((col("wiki").isNotNull()) & (col("wiki")[0].isNotNull()))
    .count()
)

num_nhs_total = search_docs.count()

print(f"NHS pages with at least one wiki link: {num_nhs_with_wiki} / {num_nhs_total}")

# Odpoveď

# Aké máme distinctívne kategórie?

# Cód
from pyspark.sql.functions import explode, desc

wiki_links = search_docs.select(explode("wiki").alias("w"))

categories_df = (
    wiki_links
    .where(col("w.categories").isNotNull())
    .select(explode("w.categories").alias("category"))
)

## Number of distinct categories:
num_distinct_categories = categories_df.select("category").distinct().count()
print(f"Distinct Wikipedia categories among linked pages: {num_distinct_categories}")

## top 50 
categories_df.groupBy("category") \
    .count() \
    .orderBy(desc("count")) \
    .show(50, truncate=False)
# Odpoveď

# Aké redirecty ukazujú na naše stránky?
# Cód
wiki_all = spark.read.parquet("/data/enwiki_ns0_with_categories")
from pyspark.sql.functions import regexp_extract, lower

redirects = (
    wiki_all
    .where(
        lower(col("wikitext")).startswith("#redirect")
    )
    .withColumn(
        "redirect_target_title",
        regexp_extract("wikitext", r"(?i)#redirect\s*\[\[(.*?)\]\]", 1),
    )
    .where(col("redirect_target_title") != "")
)

redirects.select("page_id", "title", "redirect_target_title", "categories") \
         .show(20, truncate=False)
# Odpoveď



+++++++++++++++++++++++++cashe+++++++++++++++++++++++

... 
+------------------------+
|key                     |
+------------------------+
|cas_number              |
|imager                  |
|pregnancy_au_comment    |
|altr                    |
|atc_suffix              |
|index2_label            |
|verifiedfields          |
|subject                 |
|oclc                    |
|unii                    |
|management              |
|metabolites             |
|chemspiderid2           |
|smiles2                 |
|domicile                |
|pregnancy_au            |
|field                   |
|pubchem                 |
|cl                      |
|length                  |
|numbernearbystars       |
|element                 |
|name                    |
|nlm                     |
|link2                   |
|starring                |
|image_class2            |
|brighteststarname       |
|causes                  |
|width2                  |
|chemspiderid            |
|legal_au                |
|producer                |
|molecular_weight        |
|editing                 |
|drugbank2               |
|month                   |
|pub_date                |
|smiles                  |
|album                   |
|ra                      |
|protein_bound           |
|next_year               |
|stdinchikey2            |
|addiction_liability     |
|duration                |
|isbn_note               |
|legal_us_comment        |
|legal_status            |
|title                   |
|genitive                |
|named after             |
|image_class             |
|legal_un_comment        |
|neareststarname         |
|legal_uk_comment        |
|writer                  |
|impact                  |
|border                  |
|caption2                |
|f                       |
|image_classl            |
|caption1                |
|watchedfields           |
|chebi2                  |
|sol_units               |
|media_type              |
|director                |
|n                       |
|image_style             |
|author                  |
|cause                   |
|latmin                  |
|symbol                  |
|k                       |
|image2                  |
|chembl                  |
|link1                   |
|widthr                  |
|dailymedid              |
|types                   |
|isbn                    |
|melting_high            |
|density                 |
|coden                   |
|image symbol            |
|source                  |
|iupac_name              |
|areatotal               |
|stardistancely          |
|family                  |
|legal_nz                |
|duration_of_action      |
|bordering               |
|screenplay              |
|prev_title              |
|licence_eu              |
|diagnosis               |
|o                       |
|legal_eu                |
|usan                    |
|target                  |
|h                       |
|chembl2                 |
|cover                   |
|link2-name              |
|elimination_half-life   |
|pphrases                |
|image4                  |
|metabolism              |
|numbermainstars         |
|altl                    |
|medlineplus             |
|constellation           |
|stdinchikey             |
|drugbank                |
|chirality               |
|history                 |
|verifiedrevid           |
|specialty               |
|caption4                |
|imagel                  |
|label                   |
|year                    |
|alt2                    |
|legal_br_comment        |
|legal_ca                |
|atc_supplemental        |
|differential            |
|eissn                   |
|file                    |
|studio                  |
|image3                  |
|legal_uk                |
|chebi                   |
|discipline              |
|followed_by             |
|iuphar_ligand2          |
|synonym                 |
|unii2                   |
|dewey                   |
|arearank                |
|country                 |
|artist                  |
|editor                  |
|dependency_liability    |
|b-side                  |
|latmax                  |
|language                |
|symbolism               |
|treatment               |
|jmol                    |
|melting_notes           |
|legal_de                |
|routes_of_administration|
|tradename               |
|image1                  |
|c                       |
|pubchem2                |
|numberbrightstars       |
|nr                      |
|gross                   |
|type                    |
|congress                |
|fall                    |
|i                       |
|legal_ca_comment        |
|cinematography          |
|legal_br                |
|class                   |
|website                 |
|url                     |
|onset                   |
|risks                   |
|detriment               |
|legal_us                |
|melting_point           |
|misc                    |
|cas_supplemental        |
|exaltation              |
|abbreviation            |
|stardistancepc          |
|pdb_ligand2             |
|notes                   |
|numberbfstars           |
|starmagnitude           |
|li                      |
|released                |
|complications           |
|pregnancy_category      |
|prev_year               |
|issn                    |
|link1-name              |
|bioavailability         |
|kegg2                   |
|legal_un                |
|frequency               |
|meteorshowers           |
|drug_name               |
|pdb_ligand              |
|dec                     |
|genre                   |
|boiling_point           |
|caption3                |
|medication              |
|pages                   |
|runtime                 |
|stdinchi2               |
|solubility              |
|stdinchi                |
|prevention              |
|image_thumbtime         |
|recorded                |
|biosimilars             |
|legal_au_comment        |
|synonyms                |
|jan                     |
|mab_type                |
|numbermessierobjects    |
|iuphar_ligand           |
|cas_number2             |
|next_title              |
|widthl                  |
|image_classr            |
|atc_prefix              |
|quality                 |
|prognosis               |
|release_date            |
|excretion               |
|publisher               |
|s                       |
|lccn                    |
|symptoms                |
|deaths                  |
|niaid_chemdb            |
|impact-year             |
|kegg                    |
+------------------------+
