import sparql_dataframe
from jarowinkler import jarowinkler_similarity
import Levenshtein 


import re
from shapely import wkt
from shapely.wkt import loads as wkt_loads
from shapely.geometry import Polygon
import numpy as np

from pyproj import CRS, Transformer
import pandas as pd
from rdflib import Graph, URIRef
from rdflib.namespace import OWL
import langid
from googletrans import Translator
from geopy.distance import geodesic
from rapidfuzz.fuzz import ratio
from sentence_transformers import SentenceTransformer, util 

# Definition of the SPARQL Endpoints used to communicate with each database

lgd_endpoint = "http://linkedgeodata.org/sparql"
aegean_endpoint = "http://semantics.aegean.gr:3030/data/sparql"

# First SPARQL query, on the Aegean database

aegean_classes_query = """
    PREFIX geo: <http://www.opengis.net/ont/geosparql#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?feature ?class ?wkt WHERE {
        ?feature a ?class ;
               geo:hasGeometry ?geometry .
  
        ?geometry geo:asWKT ?wkt .
        FILTER(CONTAINS(str(?class), "ontology")) 
        }
     
"""

aegean_classes_query_df = sparql_dataframe.get(aegean_endpoint,aegean_classes_query, post=True)

# Definition of the study area polygon, surrounding Lesvos Island

aoi = Polygon([
    (25.749074757308364, 38.92321311071993), 
    (26.653739441512016, 38.92321311071993),  
    (26.653739441512016, 39.415505904985196),  
    (25.749074757308364, 39.415505904985196),  
    (25.749074757308364, 38.92321311071993)   
])

# Data processing

def clean_wkt(wkt_string):
    try:
        cleaned_wkt = re.sub(r'<.*?>', '', wkt_string)
        return cleaned_wkt
    except Exception as e:
        print(f"Error cleaning WKT: {wkt_string} - {e}")
        return None
    
aegean_classes_query_df['cleaned_w'] = aegean_classes_query_df['wkt'].apply(clean_wkt)

aegean_classes_query_df['geometry'] = aegean_classes_query_df['cleaned_w'].apply(lambda w: wkt.loads(w) if w else None)


aegean_classes_filtered_df = aegean_classes_query_df[aegean_classes_query_df["geometry"].apply(lambda g: g.within(aoi) if g else False)]

aegean_classes_filtered_df['name'] = aegean_classes_filtered_df['class'].str.extract(r'ontology\/([a-zA-Z]+)') 

aegean_classes = aegean_classes_filtered_df["name"].unique().tolist()

# First SPARQL query, on the LinkedGeoData database

lgd_classes_query = """
     Prefix ogc: <http://www.opengis.net/ont/geosparql#>
     Prefix geom: <http://geovocab.org/geometry#>

    SELECT DISTINCT ?class (COUNT(?s) AS ?instanceCount)
    WHERE {
    ?s a ?class.
    ?s geom:geometry [
        ogc:asWKT ?g
    ] .
    FILTER (bif:st_intersects (?g, bif:st_geomFromText("POLYGON ((25.749074757308364 38.92321311071993, 26.653739441512016 38.92321311071993, 26.653739441512016 39.415505904985196, 25.749074757308364 39.415505904985196, 25.749074757308364 38.92321311071993))")))
    Filter contains(str(?class),"ontology") 
    }
    GROUP BY ?class
    HAVING (COUNT(?s) > 0)
    
"""
# Data processing

lgd_classes_query_df = sparql_dataframe.get(lgd_endpoint, lgd_classes_query, post=True)

lgd_classes_query_df['classes'] = lgd_classes_query_df['class'].str.extract(r'ontology\/([a-zA-Z]+)') 

lgd_classes = lgd_classes_query_df['classes'].unique().tolist()


def split_phrase(value):
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', value)

aegean_phrases = [split_phrase(x) for x in aegean_classes]
aegean_phrases[62] = "NATURA Area"

lgd_phrases = [split_phrase(x) for x in lgd_classes]

aegean_classes_df = pd.DataFrame(aegean_classes)
aegean_classes_df['phrases'] = aegean_phrases
aegean_classes_df.columns.values[0] = "names"

lgd_classes_df = pd.DataFrame(lgd_classes)
lgd_classes_df['phrases'] = lgd_phrases
lgd_classes_df.columns.values[0] = "names"


# The dataframes aegean_classes_df, lgd_classes_df contain the class names compared during the first comparison step 


# Jaro-Winkler and Levenshtein criteria results, for thresholds 0.7 and 0.8

class_matches_jarowinkler_07 = []
class_matches_jarowinkler_08 = [] 
class_matches_levenshtein_07 = []
class_matches_levenshtein_08 = []


for index1, row1 in aegean_classes_df.iterrows():
    for index2, row2 in lgd_classes_df.iterrows():
        similarity_jw = jarowinkler_similarity(row1['phrases'], row2['phrases'])
        similarity_lev = Levenshtein.ratio(row1['phrases'], row2['phrases'])
        
        if similarity_jw > 0.7 :
            class_matches_jarowinkler_07.append((row1['names'], row2['names']))
            
        if similarity_jw > 0.8 :
            class_matches_jarowinkler_08.append((row1['names'], row2['names']))
        
        if similarity_lev > 0.7  :
            class_matches_levenshtein_07.append((row1['names'], row2['names']))
            
        if similarity_lev > 0.8  :
            class_matches_levenshtein_08.append((row1['names'], row2['names']))

# Cosine similarity matching results, for the three selected models, for thresholds 0.7 and 0.8

class_matches_MiniLM_07 = []
class_matches_MiniLM_08 = []
class_matches_mpnet_07 = []
class_matches_mpnet_08 = []
class_matches_paraphrase_07 = []
class_matches_paraphrase_08 = []



model_MiniLM = SentenceTransformer('all-MiniLM-L6-v2')

model_mpnet = SentenceTransformer('all-mpnet-base-v2')

model_paraphrase = SentenceTransformer('paraphrase-MiniLM-L6-v2')

for index1, row1 in aegean_classes_df.iterrows():
    for index2, row2 in lgd_classes_df.iterrows():
        phrase1 = row1['phrases']
        phrase2 = row2['phrases']

        embedding1 = model_MiniLM.encode(phrase1, convert_to_tensor=True)
        embedding2 = model_MiniLM.encode(phrase2, convert_to_tensor=True)
        
        embedding3 = model_mpnet.encode(phrase1, convert_to_tensor=True)
        embedding4 = model_mpnet.encode(phrase2, convert_to_tensor=True)
        
        embedding5 = model_paraphrase.encode(phrase1, convert_to_tensor=True)
        embedding6 = model_paraphrase.encode(phrase2, convert_to_tensor=True)

        similarity_1 = util.pytorch_cos_sim(embedding1, embedding2).item()
        similarity_2 = util.pytorch_cos_sim(embedding3, embedding4).item()
        similarity_3 = util.pytorch_cos_sim(embedding5, embedding6).item()

        if similarity_1 > 0.7: 
            class_matches_MiniLM_07.append((row1['names'], row2['names'])) 
        
        if similarity_1 > 0.8:
            class_matches_MiniLM_08.append((row1['names'], row2['names'])) 
        
        if similarity_2 > 0.7:
            class_matches_mpnet_07.append((row1['names'], row2['names'])) 
            
        if similarity_2 > 0.8:
            class_matches_mpnet_08.append((row1['names'], row2['names'])) 
            
        if similarity_3 > 0.7:
            class_matches_paraphrase_07.append((row1['names'], row2['names'])) 
            
        if similarity_3 > 0.8:
            class_matches_paraphrase_08.append((row1['names'], row2['names'])) 

# Creation of a set of indexes

index_list = ["jarowinkler_07","jarowinkler_08","levenshtein_07","levenshtein_08","MiniLM_07","MiniLM_08","mpnet_07","mpnet_08","paraphrase_07","paraphrase_08"]
for i in index_list:
    print(i)
    
class_matches_sets = []
       
for i in index_list:
    variable_name = f"class_matches_{i}"
    if variable_name in globals():  
        class_matches_sets.append(globals()[variable_name])

# The set class_matches_sets contains all the sets of results from the class comparison     

# Creation of lists to store the data from the LinkedGeoData database, according to the sets of results of the first comparison
 
for i in index_list:
    globals()[f"list_lgd_{i}"] = []  


for i in index_list:
    class_matches_var = f"class_matches_{i}"
    
    if class_matches_var in globals():  
        for j in globals()[class_matches_var]:  
            globals()[f"list_lgd_{i}"].append(j[1])  

for i in index_list:
    globals()[f"dataframes_lgd_{i}"] = []
    
# SPARQL query to retrieve the data from the LinkedGeoData database to be compared in the next step

for i in index_list:
    list_name = f"list_lgd_{i}"
    
    if list_name in globals(): 
        for element in globals()[list_name]: 

            query_lgd = """
                Prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                Prefix ogc: <http://www.opengis.net/ont/geosparql#>
                Prefix geom: <http://geovocab.org/geometry#>
                Prefix lgdo: <http://linkedgeodata.org/ontology/>

                Select *
                Where {
                    ?s a lgdo:%s ;
                    rdfs:label ?l ;    
                    geom:geometry [
                        ogc:asWKT ?g
                    ] .

                    Filter(bif:st_intersects (?g, bif:st_geomFromText("POLYGON ((25.749074757308364 38.92321311071993, 26.653739441512016 38.92321311071993, 26.653739441512016 39.415505904985196, 25.749074757308364 39.415505904985196, 25.749074757308364 38.92321311071993))"))) .
                }
                Limit 200
            """ % element

            
            df = sparql_dataframe.get(lgd_endpoint, query_lgd, post=True)

            longitude = []
            latitude = []
            
            
            for g in df['g']:
                match = re.search(r'(POINT|LINESTRING|POLYGON|MULTIPOLYGON)\s*\(.*\)', g)
                if match:
                    wkt_geometry = match.group(0) 
                    try:
                        geom = wkt_loads(wkt_geometry) 
                        centroid = geom.centroid 
                        longitude.append(centroid.x)
                        latitude.append(centroid.y)
                    except Exception as e:
                        print(f"Error processing geometry: {e}")
                        longitude.append(None)
                        latitude.append(None)
                else:
                    longitude.append(None)
                    latitude.append(None)
                        
            df['longitude'] = longitude
            df['latitude'] = latitude
            
            globals()[f"dataframes_lgd_{i}"].append((element,df))

# Data processing
            
def is_english(name):
    lang, _ = langid.classify(name)
    return lang == 'en'
def is_greek(text):
    return bool(re.search(r'[α-ωΑ-Ω]', text))


def translate_name(name):
    if pd.isna(name):  
        return name
    if is_greek(name):
        try:
            translation = translator.translate(name, src='el', dest='en')
            return translation.text
        except Exception as e:
            print(f"Error translating {name}: {e}")
            return name  
    else:
        return name
    
translator = Translator()
    
for i in index_list:
    dataframes_lgd_i = globals().get(f"dataframes_lgd_{i}", [])

    for idx, (class_label, df) in enumerate(dataframes_lgd_i):
        df['language'] = df['l'].apply(is_english)
        df_sorted = df.sort_values(by='language', ascending=False)
        df_deduplicated = df_sorted.drop_duplicates(subset='s', keep='first')
        df_deduplicated = df_deduplicated.drop(columns=['language'])

        df_deduplicated['english_labels'] = df_deduplicated['l'].apply(translate_name)

        dataframes_lgd_i[idx] = (class_label, df_deduplicated)
        globals()[f"dataframes_lgd_{i}"] = dataframes_lgd_i

# Creation of lists to store the data from the Aegean database, according to the sets of results of the first comparison 

for i in index_list:
    globals()[f"list_aegean_{i}"] = []
    
for i in index_list:
    class_matches_var = f"class_matches_{i}"
    
    if class_matches_var in globals():  
        for j in globals()[class_matches_var]:  
            globals()[f"list_aegean_{i}"].append(j[0])    

for i in index_list:
    globals()[f"dataframes_aegean_{i}"] = []

# SPARQL query to retrieve the data from the Aegean database to be compared in the next step 

for i in index_list:
    list_name = f"list_aegean_{i}"
    
    if list_name in globals(): 
        for element in globals()[list_name]: 
            query_aegean = """
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    
                    SELECT ?s ?w ?label ?name
                    WHERE {
                        ?s <http://www.opengis.net/ont/geosparql#hasGeometry> ?o.
                        ?o <http://www.opengis.net/ont/geosparql#asWKT> ?w.
                        OPTIONAL{?s rdfs:label ?label.}
                        OPTIONAL{?s <http://semantics.aegean.gr/ontology/hasName> ?name.} 
                        FILTER (CONTAINS(STR(?s), "%s"))
                        }
                    LIMIT 800
                    """ % element
            
            
            df = sparql_dataframe.get(aegean_endpoint, query_aegean, post=True)
            
            
            
            
            longitude = []
            latitude = []
            
            epsg_transformers = {
            "4326": None,  
            "2100": Transformer.from_crs(CRS("EPSG:2100"), CRS("EPSG:4326"), always_xy=True),
            "3857": Transformer.from_crs(CRS("EPSG:3857"), CRS("EPSG:4326"), always_xy=True),
            }
            
            for idx, w in enumerate(df['w']):
                try:
                    lon, lat = None, None
                    match = re.search(r'(POINT|LINESTRING|MULTILINESTRING|POLYGON|MULTIPOLYGON)\s*\(.*\)', w)
                    if match:
                        wkt_geometry = match.group(0)
                        epsg_match = re.search(r'EPSG/0/(\d+)', w)
                        if epsg_match:
                            epsg_code = epsg_match.group(1)

                            if epsg_code in epsg_transformers:
                                if epsg_code == "4326":
                                    geom = wkt_loads(wkt_geometry)
                                    lon, lat = geom.centroid.x, geom.centroid.y
                                else:
                                    coords = re.findall(r"[-+]?\d*\.\d+|\d+", wkt_geometry)
                                    x, y = map(float, coords[:2])
                                    transformer = epsg_transformers[epsg_code]
                                    lon, lat = transformer.transform(x, y)
                            else:
                                print(f"Unsupported EPSG code at row {idx}: {epsg_code}")
                        else:
                            print(f"No EPSG code detected at row {idx}.")
                    
                    
                    longitude.append(lon)
                    latitude.append(lat)

                except Exception as e:
                    print(f"Error at row {idx}: {e}")
                    longitude.append(None)
                    latitude.append(None) 
           

            df['longitude'] = longitude
            df['latitude'] = latitude
            
            
            globals()[f"dataframes_aegean_{i}"].append((element,df))

# Data processing 

for i in index_list:
    globals()[f"dataframes_aegean_filtered_{i}"] = []
    
def clean_wkt(wkt_string):
    try:
        cleaned_wkt = re.sub(r'<.*?>', '', wkt_string)
        return cleaned_wkt
    except Exception as e:
        print(f"Error cleaning WKT: {wkt_string} - {e}")
        return None


for i in index_list:
    dataframes_aegean_i = globals().get(f"dataframes_aegean_{i}", [])
    dataframes_aegean_filtered_i = globals().get(f"dataframes_aegean_filtered_{i}", [])

    for idx, (class_label, df) in enumerate(dataframes_aegean_i):
        df['cleaned_w'] = df['w'].apply(clean_wkt)
        
        df['geometry'] = df['cleaned_w'].apply(lambda w: wkt.loads(w) if w else None)
        
        filtered_df = df[df['geometry'].apply(lambda geom: geom.within(aoi) if geom else False)]
        filtered_df = filtered_df.drop(columns=['cleaned_w', 'geometry'], errors='ignore')
        
        dataframes_aegean_filtered_i.append((class_label, filtered_df))
        globals()[f"dataframes_aegean_filtered_{i}"] = dataframes_aegean_filtered_i
        

for i in index_list:
    dataframes_aegean_filtered_i = globals().get(f"dataframes_aegean_filtered_{i}", [])

    for idx, (class_label, df) in enumerate(dataframes_aegean_filtered_i):
        df = df.copy()
        df.loc[:, 'translated_name'] = df['name'].apply(translate_name)
        
        df.loc[:, 'translated_label'] = df['label'].apply(translate_name)
        
        df.loc[:, 'english_labels'] = df['translated_name'].combine_first(df['translated_label'])
        
        df = df.drop(columns=['translated_name', 'translated_label'], errors='ignore')
        
        dataframes_aegean_filtered_i[idx] = (class_label, df)
        
        
    globals()[f"dataframes_aegean_filtered_{i}"] = dataframes_aegean_filtered_i
        

for i in index_list:
    dataframes_aegean_filtered_i = globals().get(f"dataframes_aegean_filtered_{i}", [])

    for idx, (class_label, df) in enumerate(dataframes_aegean_filtered_i):
        df_clear = df.drop_duplicates(subset='s', keep='first')


        dataframes_aegean_filtered_i[idx] = (class_label, df_clear)
        
    globals()[f"dataframes_aegean_filtered_{i}"] = dataframes_aegean_filtered_i


# The entity alignment comparison process is executed

threshold_km = 0.1
threshold_label = 0.7


for i in index_list:
    dataframes_aegean_filtered_i = globals().get(f"dataframes_aegean_filtered_{i}", [])
    dataframes_lgd_i = globals().get(f"dataframes_lgd_{i}", [])

    matches_i = []

    for idx, ((x, df1), (z, df2)) in enumerate(zip(dataframes_aegean_filtered_i, dataframes_lgd_i)):
        print(f"Processing index {i}, pair {idx}: Aegean ID={x}, LGD ID={z}")

        aegean_coords = df1[['latitude', 'longitude']].to_numpy()
        aegean_labels = df1['english_labels'].to_numpy()
        aegean_ids = df1['s'].to_numpy()

        lgd_coords = df2[['latitude', 'longitude']].to_numpy()
        lgd_labels = df2['english_labels'].to_numpy()
        lgd_ids = df2['s'].to_numpy()

        for a_idx, coord in enumerate(aegean_coords):
            distances = np.array([geodesic(coord, lgd_coord).kilometers for lgd_coord in lgd_coords])
            
            if pd.isna(aegean_labels[a_idx]):  
                valid_matches = distances < threshold_km
            else:
                label_similarities = np.array([
                    ratio(aegean_labels[a_idx], lgd_label) if not pd.isna(lgd_label) else 0
                    for lgd_label in lgd_labels
                ])
                valid_matches = (distances < threshold_km) & (label_similarities > threshold_label)

            for match_idx in np.where(valid_matches)[0]:
                matches_i.append((aegean_ids[a_idx], lgd_ids[match_idx]))

    globals()[f"matches_{i}"] = matches_i


# Duplicate entity pairs are removed from the sets of results

for i in index_list:
    match_list = globals()[f"matches_{i}"]  
    seen = set()

    for j in range(len(match_list) - 1, -1, -1):
        if match_list[j] in seen:
            del match_list[j]  
        else:
            seen.add(match_list[j])

### Creation of RDF/TURTLE files, for each of the sets of results 

for i in index_list:
    g = Graph()
    match_list = globals()[f"matches_{i}"]  

    for s0, s1 in match_list:
        g.add((URIRef(s0), OWL.sameAs, URIRef(s1)))

    file_name = f"graph_matches_{i}.rdf"
    g.serialize(file_name, format="ttl")
    
    g.parse(file_name, format="ttl")
    print(f"Contents of {file_name}:")
    for subj, pred, obj in g:
        print(f"Subject: {subj}, Predicate: {pred}, Object: {obj}")


