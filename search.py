import sys
import json
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone
from google import genai
import os
from dotenv import load_dotenv
load_dotenv()

class QueryBlueprint(BaseModel):
    clean_semantic_query: str  
    target_genre: str

if __name__ == '__main__':
   
    print("Loading Pinecone Bi-Encoder (Paraphrase-L3)...")
    bi_encoder = SentenceTransformer('paraphrase-MiniLM-L3-v2')

    print("Loading Stage 2: Cross-Encoder (Local BGE Reranker)...")
    reranker = CrossEncoder('BAAI/bge-reranker-base')

    # 2. Connect to Pinecone
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("cinematch-movies")

    # 3. Connect to Gemini 
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    ai_client = genai.Client(api_key=GEMINI_API_KEY)

    
    user_query = input("\n🎬 How are you feeling / What kind of movie do you want? ")

    router_prompt = f"""
    Analyze this user movie query: "{user_query}"

    Task: Extract two things:
    1. clean_semantic_query: Strip away filler words like 'movie', 'film', or actor names that aren't in plots. Keep purely the core environmental/situational plot vibes (e.g., "spacecraft malfunction distant colony planet relationship").
    2. target_genre: Identify the dominant high-level genre classification. It MUST match one of the standard TMDB genres capitalized exactly like this: 'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'Thriller', 'TV Movie', 'War', 'Western'.
    """

    print("Routing query through Gemini metadata parser...")

    try:routing_response = ai_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=router_prompt,
        config={'response_mime_type': 'application/json', 'response_schema': QueryBlueprint}
    )

    except Exception as e:
         st.warning("⚠️ High traffic on the AI live synthesis engine. Here are your raw database matches:")

    structured_data = json.loads(routing_response.text)
    clean_query = structured_data["clean_semantic_query"]
    filter_genre = structured_data["target_genre"]

    print(f"📊 Gemini Routing Strategy -> Detected Genre: '{filter_genre}' | Dense Query: '{clean_query}'")

    
    print("\n📡 Fetching top 100 raw candidates from Pinecone...")
    query_vector = bi_encoder.encode(clean_query).tolist() # Embedded the clean text query here!

 
    raw_results = index.query(
        vector=query_vector, 
        top_k=100,
        filter={"genres": {"$in": [filter_genre]}}, 
        include_metadata=True
    )

    if not raw_results['matches']:
        print(f"❌ No movies found in your index matching the genre '{filter_genre}'.")
        print("Trying fallback search without genre restrictions...")
        raw_results = index.query(vector=query_vector, top_k=100, include_metadata=True)
        
        if not raw_results['matches']:
            print("Still nothing found. Your index might be completely empty.")
            sys.exit()

    # ---- STAGE 2: LOCAL CONTEXTUAL RERANKING ----
    print("Reranker is analyzing sentence attention patterns...")

    pairs = []
    candidates = []

    for match in raw_results['matches']:
        metadata = match.get('metadata', {})
        movie_info = {
            "title": metadata.get('title', 'Unknown Title'),
            "overview": metadata.get('overview', 'No overview available.')
        }
        candidates.append(movie_info)
        
        pairs.append([user_query, movie_info['overview']])

    
    rerank_scores = reranker.predict(pairs)

    for idx, score in enumerate(rerank_scores):
        candidates[idx]['rerank_score'] = float(score)

   
    reranked_list = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

   
    best_movie = reranked_list[0]
    print(f"\n[🏆 Reranker chose: {best_movie['title']} as the absolute #1 match]")

    rag_prompt = f"""
    You are an advanced movie curation assistant.
    The user's emotional prompt is: "{user_query}"

    Our high-precision pipeline retrieved this specific movie matching their criteria:
    Title: {best_movie['title']}
    Plot: {best_movie['overview']}

    Task: Write a highly custom recommendation. Tell the user directly why this movie outranked other candidates to perfectly fit their exact vibe. Keep it punchy and engaging.
    """

    print("Gemini is generating the final recommendation text...")
    try:response = ai_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=rag_prompt,
    )
    except Exception as e:
        st.warning("⚠️ High traffic on the AI live synthesis engine. Here are your raw database matches:")

    print("\n✨ CineMatch Precision Response: ✨")
    print("=" * 60)
    print(response.text)
    print("=" * 60)