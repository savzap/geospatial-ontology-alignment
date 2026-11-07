# Geospatial Entity Alignment in the Semantic Web

A Python-based semantic web alignment pipeline designed to query, process, and link geospatial data across heterogeneous knowledge bases. This project demonstrates how to remotely query SPARQL endpoints, filter geospatial data by bounding boxes, translate multilingual attributes, and apply hybrid matching techniques (string similarity + deep learning sentence embeddings) to discover identical entities and link them using the `owl:sameAs` property.

The core implementation uses **Lesvos Island, Greece** as the geographic Area of Interest (AOI) to align instances between the **University of the Aegean** database and **LinkedGeoData (LGD)**.

## 🚀 Features

- **Remote SPARQL Execution**: Directly queries semantic repositories using POST requests via `sparql_dataframe`.
- **Geospatial Filtering**: Parses Well-Known Text (WKT) geometries and applies a geographic bounding box filter using `shapely`.
- **Automated Coordinate Transformation**: Handles coordinate transformations across spatial reference systems (EPSG:4326, EPSG:2100, EPSG:3857) via `pyproj`.
- **Multilingual Label Translation**: Automatically identifies Greek text strings using `langid` and translates them into English using `googletrans`.
- **Hybrid Data Alignment**:
  - **Lexical Matching**: Employs Jaro-Winkler and Levenshtein distance metrics.
  - **Semantic Matching**: Utilizes Transformer models (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `paraphrase-MiniLM-L6-v2`) via `sentence-transformers` to compute cosine similarity scores.
- **Geographical Validation**: Filters potential matches using a distance threshold (e.g., within 100 meters) calculated using `geopy`'s geodesic distance.
- **RDF Graph Generation**: Outputs discovered identity pairs into valid RDF graph files serialized in **Turtle (.ttl)** format using `rdflib`.

## 📦 Prerequisites & Dependencies

To execute this script, ensure you have Python 3.8+ installed along with the following libraries:

```bash
pip install sparql-dataframe jarowinkler Levenshtein numpy pyproj pandas rdflib langid googletrans==4.0.0-rc1 shapely geopy rapidfuzz sentence-transformers torch
```

## 🛠️ Pipeline Architecture

1. **Class Alignment**: Extracts ontology classes from both endpoints inside the designated bounding box and tests them against 10 combinations of algorithms and thresholds (Jaro-Winkler, Levenshtein, and 3 Sentence Transformer models at 0.7 and 0.8 thresholds).
2. **Instance Retrieval**: Fetches up to 800 individual spatial instances for each matching class from both databases.
3. **Data Cleaning & Normalization**: Translates Greek labels into English and derives centroids for lines and polygons to create a unified tracking format.
4. **Entity Resolution**: Evaluates physical distance ($\le 100\text{ meters}$) and string similarity ($\ge 0.7$) between instances to declare an explicit match.
5. **Knowledge Graph Output**: Serializes results into individual `.rdf` files utilizing the `owl:sameAs` property.

## 🌐 SPARQL Endpoints

- **LinkedGeoData Endpoint**: "http://linkedgeodata.org/sparql"
- **University of the Aegean Endpoint**: "http://semantics.aegean.gr:3030/data/sparql" *(Note: This endpoint is currently offline; however, the pipeline logic remains fully functional for architectural evaluation and alternative endpoints).*
