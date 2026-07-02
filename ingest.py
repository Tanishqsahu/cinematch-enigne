import pandas as pd
import json
import ast
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
load_dotenv()

# Safe list extraction helper function
def safe_parse_column(cell_value, key_target="name"):
    if pd.isna(cell_value) or not str(cell_value).strip():
        return []
    try:
        # ast.literal_eval handles single and double-quoted evaluation securely
        parsed_nodes = ast.literal_eval(str(cell_value))
        return [item[key_target] for item in parsed_nodes if key_target in item]
    except:
        return []

if __name__ == '__main__':
    print("📖 Reading movie datasets...")
    movies_df = pd.read_csv("sample_data/tmdb_5000_movies.csv")
    credits_df = pd.read_csv("sample_data/tmdb_5000_credits.csv")

    # Merge datasets on core keys
    df = pd.merge(movies_df, credits_df, left_on='id', right_on='movie_id')
    df = df.dropna(subset=['id', 'title_x', 'overview', 'release_date', 'genres', 'keywords', 'cast'])

    print("🧼 Cleaning and parsing text matrices across all columns...")
    # Pre-process columns safely using optimized Pandas vector mapping
    df['clean_genres'] = df['genres'].apply(lambda x: safe_parse_column(x, "name"))
    df['clean_keywords'] = df['keywords'].apply(lambda x: safe_parse_column(x, "name"))
    df['clean_cast'] = df['cast'].apply(lambda x: safe_parse_column(x, "name"))

    print("🧠 Loading local Embedding Model...")
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

    print("⚙️ Waking up CPU multi-processing pool...")
    pool = model.start_multi_process_pool()

    # Cloud DB Setup Connection
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "cinematch-movies"

    if index_name not in pc.list_indexes().names():
        print(f"Creating a new vector index named '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index = pc.Index(index_name)

    batch_size = 100
    print(f"\n🚀 Processing and uploading {len(df)} movies in blocks of {batch_size}...")

    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i : i + batch_size]

        enriched_chunks = []
        metadata_list = []
        id_list = []

        for _, row in batch_df.iterrows():
            movie_id = int(row['id'])

            # Extract primitive arrays pre-cleaned above
            genres_list = row['clean_genres']
            keywords_list = row['clean_keywords']
            cast_list = row['clean_cast'][:3] # Slice top 3 actors cleanly

            # Parse Date elements securely
            try:
                year = int(pd.to_datetime(row['release_date']).year)
            except:
                year = 0

            # Format language settings securely
            lang_code = str(row['original_language']).strip().lower() if pd.notna(row['original_language']) else "en"

            # Structure string blocks for the dense vector semantic field
            genre_str = ", ".join(genres_list)
            keyword_str = ", ".join(keywords_list)
            actor_str = f"Starring: {', '.join(cast_list)}" if cast_list else ""

            enriched_text = f"Title: {row['title_x']}. Language: {lang_code}. Genres: {genre_str}. {actor_str}. Plot Summary: {row['overview']} Tags: {keyword_str}"

            enriched_chunks.append(enriched_text)
            id_list.append(str(movie_id))

            metadata_list.append({
                "title": str(row['title_x']),
                "overview": str(row['overview']),
                "release_year": year,
                "genres": genres_list, # Native list of strings perfectly matching metadata requirements
                "language": lang_code
            })

        # --- FIX: Embedding computation and API calls sit OUTSIDE the inner row loop ---
        embeddings = model.encode_multi_process(enriched_chunks, pool).tolist()

        payload = []
        for idx in range(len(id_list)):
            payload.append({
                "id": id_list[idx],
                "values": embeddings[idx],
                "metadata": metadata_list[idx]
            })

        index.upsert(vectors=payload)

    # Shutdown parallel computation processes cleanly
    model.stop_multi_process_pool(pool)
    print("\n✨ Database pipeline execution completed successfully! Your metadata filters will work flawlessly.")