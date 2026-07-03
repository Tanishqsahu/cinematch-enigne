import sys
import json
import streamlit as st
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


st.title("🎬 CineMatch AI Engine")
st.write("Two-Stage Retrieval + RAG Recommendation System")
st.write("---")

# --- 2. INITIALIZE AI MODELS (Cached so they only load once) ---
@st.cache_resource
def load_models():
    bi_encoder = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    reranker = CrossEncoder('BAAI/bge-reranker-base')
    return bi_encoder, reranker

bi_encoder, reranker = load_models()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("cinematch-movies")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
ai_client = genai.Client(api_key=GEMINI_API_KEY)


user_query = st.text_input("🎬 How are you feeling / What kind of movie do you want?")

if user_query:
    st.write("⏳ Processing your request through the pipeline...")
    
    # --- STAGE 1: METADATA ROUTING ---
    router_prompt = f"""
    Analyze this user movie query: "{user_query}"

    Task: Extract two things:
    1. clean_semantic_query: Strip away filler words like 'movie', 'film', or actor names that aren't in plots. Keep purely the core environmental/situational plot vibes (e.g., "spacecraft malfunction distant colony planet relationship").
    2. target_genre: Identify the dominant high-level genre classification. It MUST match one of the standard TMDB genres capitalized exactly like this: 'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'Thriller', 'TV Movie', 'War', 'Western'.
    """

    try:
        routing_response = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=router_prompt,
            config={'response_mime_type': 'application/json', 'response_schema': QueryBlueprint}
        )
    except Exception as e:
         st.warning("⚠️ High traffic on the AI live synthesis engine. Here are your raw database matches:")

    structured_data = json.loads(routing_response.text)
    clean_query = structured_data["clean_semantic_query"]
    filter_genre = structured_data["target_genre"]

    st.info(f"📊 **Gemini Router Matrix:** Detected Genre: `{filter_genre}` | Cleaned Query: *'{clean_query}'*")

    # --- VECTOR RETRIEVAL ---
    query_vector = bi_encoder.encode(clean_query).tolist()

    raw_results = index.query(
        vector=query_vector, 
        top_k=100,
        filter={"genres": {"$eq": filter_genre}}, 
        include_metadata=True
    )

    # Fallback if genre filter yields 0 matches
    if not raw_results['matches']:
        st.warning(f"No movies found matching '{filter_genre}' in your database. Initializing fallback scan...")
        raw_results = index.query(vector=query_vector, top_k=100, include_metadata=True)

    if not raw_results['matches']:
        st.error("Your index appears completely empty.")
        st.stop()

    # --- STAGE 2: LOCAL CONTEXTUAL RERANKING ---
    pairs = []
    candidates = []

    for match in raw_results['matches']:
        metadata = match.get('metadata', {})
        movie_info = {
            "title": metadata.get('title', 'Unknown Title'),
            "overview": metadata.get('overview', 'No overview available.'),
            # Pull poster path from Pinecone metadata space
            "poster_path": metadata.get('poster_path', '')
        }
        candidates.append(movie_info)
        pairs.append([user_query, movie_info['overview']])

    rerank_scores = reranker.predict(pairs)

    for idx, score in enumerate(rerank_scores):
        candidates[idx]['rerank_score'] = float(score)

    reranked_list = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
    top_3_winners = reranked_list[:3]

    # --- STAGE 3: GENERATING THE RAG REASONING ---
    movies_text = ""
    for i, movie in enumerate(top_3_winners):
        movies_text += f"\n🔥 Option #{i+1}:\nTitle: {movie['title']}\nPlot: {movie['overview']}\n"

    rag_prompt = f"""
    You are an advanced movie curation assistant.
    The user's emotional prompt is: "{user_query}"

    Our high-precision pipeline analyzed 100 candidates and narrowed it down to these top 3 absolute best matches:
    {movies_text}

    Task: Write a highly custom recommendation list for these 3 movies. For each movie, tell the user directly why it outranked the other candidates to fit their exact vibe. Keep it punchy, engaging, and professional.
    """

    try:
        response = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=rag_prompt,
        )
    except Exception as e:
        st.warning("⚠️ High traffic on the AI live synthesis engine. Here are your raw database matches:")

    # --- RENDER RESULTS CLEARLY ---
    titles_string = " | ".join(m.get('title', 'Unknown Title') for m in top_3_winners)

    st.success(f"🏆 **Top 3 Match Selections:** {titles_string}")

    st.subheader("✨ CineMatch Precision Response:")
    st.markdown(response.text)
    
    st.write("---")
    
   
# --- 📸 HIGH-FIDELITY DIRECT OMDb IMAGE ENGINE ---
    st.subheader("🖼️ Featured Recommendations Gallery")
    
    # Initialize 3 responsive horizontal containers
    cols = st.columns(3)
    
    
    def get_omdb_poster(movie_name):
        encoded_title = movie_name.replace(" ", "+")
        OMDB_API_KEY=os.getenv("OMDB_API_KEY")
        url = f"http://www.omdbapi.com/?t={encoded_title}&apikey={OMDB_API_KEY}"
        
        try:
            import requests
            res = requests.get(url, timeout=4)
            res.raise_for_status()
            data = res.json()
            
           
            poster_url = data.get('Poster')
            rating_url = data.get('imdbRating','N/A')
            if poster_url == "N/A" :
                
                poster_url = None

            return posted_url,imdb_rating
        except:
            pass
        return None, "N/A"


    for i, movie in enumerate(top_3_winners):
        title = movie.get('title', 'Unknown Title')
      
        encoded_title = title.replace(" ", "+")
        working_image_url = f"https://placehold.co/500x750/0e1117/ffffff?text={encoded_title}"
        
        
        omdb_poster, movie_rating = get_omdb_poster(title)
        
        # 3. 🎯 ONLY overwrite if the API actually returned a valid, non-empty URL string
        if omdb_poster and isinstance(omdb_poster, str):
            working_image_url = omdb_poster

        # 4. Render directly onto the interface layout
        with cols[i]:
            # This will now NEVER receive a NoneType object!
            st.image(working_image_url, use_container_width=True)
            st.markdown(f"**🎬 {title}** \n⭐ IMDb: `{movie_rating}/10`")