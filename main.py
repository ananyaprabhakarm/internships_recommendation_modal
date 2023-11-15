import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from fastapi import FastAPI
import requests

app = FastAPI()


response = requests.get('http://workshala.onrender.com/jobs')
data = response.json()  
df = pd.DataFrame(data)
df.isnull().sum()
df.head()

vectorizer = TfidfVectorizer()
df['features'] = df['_id'] + ' ' + df['companyName'] + ' ' + df['jobType'] + ' ' + df['description'] + ' ' + df['startDate'] + ' ' + df['stipend'] + ' ' + df['duration'] 


df_vector = vectorizer.fit_transform(df['features'])

similarity = linear_kernel(df_vector, df_vector)

def recommend(user_preferences, new_df):
  intern_matrix = vectorizer.transform([user_preferences])
  cosine_similarities = linear_kernel(intern_matrix, vectorizer.transform(new_df['features']))
  similarity_scores = list(enumerate(cosine_similarities[0]))
  sorted_internships = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
  top_n_recommendations = [(new_df.iloc[idx]['_id'], score) for idx, score in sorted_internships[1:6]]
  return top_n_recommendations

@app.get("/recommendations/{user_preferences}")

async def read_recommendations(user_preferences: str):
    recommendations = recommend(user_preferences, df)
    return {"user_preferences": user_preferences, "recommendations":recommendations}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8002)














    