import requests

def get_poster(movie_name):
    movie_url=f"http://www.omdbapi.com/?i=tt3896198&apikey=dff3a6a4"

    try:
        response=requests.get(movie_url,timeout=5)
        response.raise_for_status
        data=response.json()

        results=data.get('Poster')
        if not results:
            print("Not found")
            return 



    except Exception as e:
        print(f"Network error finding cities")
        return None

for i,movie in enumerate(top_3_winners):
            
            title = movie.get('title', 'Unknown Title')
            st.image(data,use_container_width=True)
            st.markdown(f"**🎬 {title}**")



