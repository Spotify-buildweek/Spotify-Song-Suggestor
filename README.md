![Spotify_Logo_CMYK_Green](https://user-images.githubusercontent.com/86321131/137929977-3d67a2a3-96ce-4078-8fcc-2a331e03321a.png)
# Spotify Song Recommender
***What is Spotify?
Spotify is a audio and media streaming service that offer a variety of plans to suit your listening needs. This service has over 365 million monthly active users with over 70 million songs to stream.***

Our goal with this project is to create an interactive app where a user inputs their favorite song/artist. Based on that input feedback is given to recommend similar songs to their input....to the nearest 5 songs.
# **[Our Website](https://spotify-suggest-it.herokuapp.com/)**
Click the link above to view the spotify song suggestor...***Music awaits get groovy!***

# App Instructions
After clicking the link to the app, navigate to the search engine, from that search engine you will then enter your favorite artist or song. After inputing your results you will now discover new songs to the nearest five as well check the metrics of each song.
# Methods
Data was collected via kaggle from this **[link](https://www.kaggle.com/geomack/spotifyclassification)**.
With the use of the Nearest Neigbor model, and an API, our app will make predictions to the nearest five similar songs.
# Nearest Neighbor Model
Libraries used:
Pandas,
Numpy,
Sklearn,
Pickle,

How the dataset was wrangled:
```sh
def wrangle(df):
    # Drop columns not in use by nearest-neighbors 
    df.drop(columns=['type', 'id', 'track_href', 'analysis_url', 'title', 'Unnamed: 0'], inplace=True)
    
    # Drop unuseful audio features
    df.drop(columns=['instrumentalness', 'time_signature'], inplace=True)
    
    # Drop genre, might not work well with nearest-neighbors and does not appear in spotify api request
    df.drop(columns=['genre'], inplace=True)
    
    # Making a pool of songs to use as query items
    test_df = df[df['song_name'].isna() == True]
    
    # Dropping rows without song_names, maybe we can keep them if we implement the api calls
    df = df[df['song_name'].isna() == False]
    
    # Drop song-name, not used in nearest-neighbors
    df.drop(columns=['song_name'], inplace=True)
    test_df.drop(columns=['song_name'], inplace=True)
    
    
    return df
   ```
Use of standard scalar to remove mean and scale it to unit variance for each feature
```sh
scaler = Normalizer()
scaler.fit(uq_df)
scaled_df = scaler.transform(uq_df)
```
Initiate Nearest Neighbor Estimator
```sh
# Instantiate nearest-neighbors estimator, n_neighbors is Number of neighbors to use by default for kneighbors queries.
nn = NearestNeighbors(n_neighbors=5)
# fit to our song's audio features
nn.fit(scaled_df)
```
Nearest Neighbor Model Trained and Fitted
```sh
# NearestNeighbors is the model that is going to give us a list of the most 
# similar songs to the searched song
nn = NearestNeighbors(n_neighbors=6)
nn.fit(X_train)

# This shows us what songs are similar by index
doc_index = 5
doc = [X_train[doc_index]]

# Query using kneighbors 
neigh_dist, neigh_index = nn.kneighbors(doc)
```

# Licenense
MIT License

Copyright (c) 2021 Fadil Shaikh, Jafar Sakha, Mikayla Kosmala, Mohamed Mosaed, Royce Roberts

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
