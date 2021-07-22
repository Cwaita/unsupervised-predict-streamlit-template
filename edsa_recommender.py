"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System", "Insights","About Recommenders","Our Team"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('First Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                        We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            try:
                if st.button("Recommend"):
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
            except:
                st.error("Oops! Looks like this algorithm does't work.\
                        We'll need to fix it!")

                    


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "About Recommenders":
  
        st.title("About Recommenders")
        """The internet has changed how we consume content. We have moved from using compact disks 
        to access content such as movies and series to having access to a catalog with thousands of 
        movies/series through streaming services such as Netflix and Showmax. This is where the paradox 
        of choice kicks in, because we have so many options to choose from it has become increasingly
        difficult to decide what to watch. To address this issue, we built 2 recommender engines, a content 
        based filtering algorithm and a collaborative filtering algorithm.
        """

        st.header("Our Similarity Measure:")
        """
        For both collaborative filtering and content based filtering we used cosine similarity to make our recommendations.
        This a similarity measure that uses the angle between vectors. The cosine similarity score is in the range  [0,1] (angle can not be better
        90$^\circ$ since our arrays in both cases do not contain negative values). Values close 
        to 1 represent a high degree of similarity and values close to zero represent the opposite. The cosine 
        similarity between two vectors in p dimensional space is:
        """
        st.image('resources/imgs/eq4.png',use_column_width=True)

    
        r"""
        where:
        - $\bar{x}.\bar{y}$ is the dot product of $\bar{x}$ and $\bar{y}$
        - $|\bar{x}|$ and $|\bar{y}|$ are the euclidean norms (euclidean distance from origin point) of $\bar{x}$ and $\bar{y}$ respectively
        """
        st.image('resources/imgs/cosine_sim.png',use_column_width=True)


        st.header("Collaborative Based Filtering")
        """
        Collaborative filtering is a technique that can filter out items that a user might like on the basis of reactions by similar users.
        
        It works by searching a large group of people and finding a smaller set of users with tastes similar to a particular user. 
        It looks at the items they like and combines them to create a ranked list of suggestions.
        """
        """
        **Advantages:**

        * Can be used for any item-no feature selection needed
        
         * Produces more diverse recommendations that content based approaches, it is better for when a user wants to have a wide range of suggestions.
        
        **Disadvantages:**
        
        * Cold start problem - needs enough users in the system to find a match
        
        * User/rating matrix is sparse - it is hard to find users that have rated the same items
        
        * First Rater - can not generate recommendations for unrated items 
        
        * Popularity bias - tends to recommend the most popular items
        
        """
        
        st.subheader("Singular Value Decomposition (SVD)")
        """
        he singular value decomposition (SVD) provides another way to factorize a matrix, into singular vectors and singular values.
         The SVD is used widely both in the calculation of other matrix operations, such as matrix inverse, 
         but also as a data reduction method in machine learning. For the case of recommender system it finds latent features between the
         movies and use those features to predict what the likely rating that a particular user is going to give.


        The mathematics of this approach is quite messy but we used the python surprise package to train our model on over 10 million examples. 
        For making recommendations, given the 3 titles we used svd to predict the top 10 users that rated those movies highly and the obtained
        all the movies that those users rated and their ratings and used that to find the similarities between the movies and then proceeded to find the 
        top 10 similarities"""
        

        st.header("Content Based Filtering")
        """
        As stated earlier on, collaborative filtering algorithms suffer from the cold start phenomenon. 
        Content based algorithms address this issue by looking at common attributes between movies and 
        ranking them according to their relative similarities. The downside to using a content based is 
        reduced diversity in suggestions compared to collaborative based algorithm.

        We are going to be using the following features:
        """

        """
         * genres
         * 3 leading actors
         * director
         * plot_keywords
         * tag
         * the year the movie was released
        """
        st.subheader("How We Prepared Our Data")
        """
        
        
        We noticed that the tags dataframe contained multiple tags on the same movie but at different 
        timestamps and so we joined all the tags with the same movie id. Before doing this, we converted all the tags to lowercase.
        We merged merged df_movies,df_imdb and df_grouped_tags to have title,genre,title_cast, director, 
        plot_keywords, and tags in one dataframe. After merging the dataframes we proceeded to clean the individual columns:
        - For the genres column, we first ensured that everything is in string format, replaced vertical bars with spaces and finally converted all the words to lowercase.
        - For the title_cast column, we ensured that everything is in string format, we joined the names and surnames of the title cast members. Created a list containing the cast members names+surname using vertical bars as separators. We then obtained the first 3 elements of the list created above and joined them using space and finally we converted everything to lowercase.
        - For the directors column, we joined the names and surnames together, then removed commas and dashes from names (for names like David O. Russel and Kim Ki-duk). Finally, converted everything to lowercase.
        - For plot_keywords, words were already lowercase so we replaced the vertical bars with spaces.
        - We extracted the year for the title column
        - We created a column that joins (soups) the columns above using spaces.
        """
        st.subheader("Feature Extraction : Term Frequency Inverse Document Frequency (TF-IDF)")
        """
        

        Term Frequency Inverse Document Frequency is a statistic that measures how important a a word is in a document while taking into consideration how many times it appears in other documents.

        The reason for doing this is:
        * If a word appears in a document then it is more likely to contain information about that document
        * If a word is scattered throughout the whole document it is unlikely to contain information distinguishing the various documents


        The calculation tf-idf can be broken into 4 parts:

        1. Calculate the term frequency:""" 
        st.image('resources/imgs/eq1.png',use_column_width=True)

     

        
        r"""
        Where: 

        - $f(w,d)$ measures how frequently a word $w$ appears in document $d$

        2. Calculate the inverse term frequency:
        """

        st.image('resources/imgs/eq2.png',use_column_width=True)

        """

        Where:
        * $N$ is the number of documents in our corpus
        * $D$ is a set containing all the documents in our corpus
        * $f(w,D)$ is the number documents that contain $w$.
        3. We combine the above to form tf-idf - Term Frequency - Inverse Document Frequency:
        """
        st.image('resources/imgs/eq3.png',use_column_width=True)
        """
        4. Finally, we get a vectorized vesion of all the documents in our corpus


        Drawbacks:
        * Not able to capture semantics
        """
        st.image('resources/imgs/tfidf.png',use_column_width=True)
        st.subheader("Making Recommendatins")
        """
        We used the vector representations of the "documents" associated with the movies to find the 10 most similar movies using cosine similarity.
        """
    
    if page_selection == "Insights":
        st.title("Insights")
        st.subheader("Distribution of Ratings")

        st.image('resources/imgs/stat.png',use_column_width=True)
        """
        - The average rating of all the movies in our dataset is 3.53.
        -The median (middle value) is 3.5 which is slighly less than the mean.
        -The standard deviation (spread around the average) is  1.06.
        - The lowest rating is 0.5 and the highest rating is 5.
        - The lower quantile (bottom 25 cut-off point) is 3 and the upper quantile (top 25 cut-off point) is 4.
        """
        st.image('resources/imgs/dist.png',use_column_width=True)
        """
        - The most common rating given by users (26.5%) is 4. 
        - The ratings appear to be skewed to the left (evidenced by the long tail to the left). This means that there are a few movies who received lower ratings compared to those that received higher ratings
        """
        """
        Then we calculated the coefficient of skewness to check our above findings. We found a skewness of 0.93. The positive skewness contradicts with our suspicion that the distribution is negatively skewed. 
        The contradiction can be explained by the fact that the distribution appears to have 2 peaks (slightly bimodal).
        """
        st.subheader("Rating by Number of Ratings")
        st.write("We plotted the average ratings of movies as a functions of how many times they were rated")
        st.image('resources/imgs/ratnum.png',use_column_width=True)
        """
        The ratings of movies that have a smaller number of ratings are widely spread. A possible explanation 
        for this would be law of large numbers, it dictates that movies that are frequently rated produce more 
        stable estimates of the true average rating."""

        st.subheader("Most common Genres")
        st.image('resources/imgs/mcg.png',use_column_width=True)
        """
        - The top 3 occuring genres are Drama, Comedy, and Thrillers. This makes sense because these are relatively broad genres.
        - Musical, Film-Noir and IMAX movies were the least represented in our dataset. These are niche genres, an example being IMAX 
        which is company that has its own line of high-resolution cameras, film formats, projectors, and theaters
        """
        st.subheader("Rating by Genre")
        st.image('resources/imgs/rbg.png',use_column_width=True)
        """
        - The average ratings according to genre were in a narrow range
        - Film-Noirs, War Movies , and Documentaries recieved the highest average ratings.
        - Film-Noirs received the highest rating even though they are the second lowest reviewed movies.
        - Comedies and Horrors received the lowest ratings. Possible explanation: too predictable?
        - Comedies recieved the second lowest average rating even though they were in the top 2 reviewed movies.
        """
        st.subheader("Do Budget and/or Runtime Affect Ratings?")
        st.image('resources/imgs/brtr.png',use_column_width=True)
        """
        - It seems that ther is no obvious relationship between the budget of the movie and the rating
        - With the exception of one point, as runtime increases the rating also increase in a linear fashion
        """
        st.subheader("Most Rated Movies")
        """
        This can be used as a proxy for how many people have watched the movies/ how popular the movies are amongst the users.
        """
        st.image('resources/imgs/mrm.png',use_column_width=True)

        """
        Most of the movies on this list are critically acclaimed. The most reviewed movie "The Shawshank Redemption" holds the number one 
        spot on the imdb top 250 movies of all time. The same applies all the titles on this list, most of them are in the 30. Here's a link to the full list click
        """
        """
        **N.B. For top movies, top directors and top lead actors we limited our analysis to movies that received 500 
        or more ratings so that the comparisons are fair, a five star rating from one user is not as significant as a 5
         star from 100 or more**
        """
        st.subheader("Top Rated Movies")
        st.image('resources/imgs/trm.png',use_column_width=True)
        """
        These movies correspond to those that are part of the imdb top 250 movies, they are critically acclaimed and our 
        data confirms that. Movies like The Shawshank Redemption,The Godfather, The Usual Suspects,12 Angry Men and Schindler's 
        List are examples of movies that appear on the top 20 of both lists. Here's a [link](https://www.imdb.com/search/title/?groups=top_250&sort=user_rating)  to view the imdb top 250 movies.
        """
        st.subheader("Worst Rated Movies")
        st.image('resources/imgs/wrm.png',use_column_width=True)
        """
        The bottom movies, since we limited our analysis to movies that recieved 500 ratings or more we get a 
        list of familiar movies, an example "Stop! Or My Mom Will Shoot" which is the 3rd worst rated movies 
        and is one of the lesser popular Sylvester Stallone movies, the comment that received the highest number 
        on imdb of upvotes on imdb is "The story was corny, the plot was predictable. The supporting cast was lacking. 
        This is not a thriller or an intelligent movie. It is a B comedy., at best." [link](https://www.imdb.com/title/tt0105477/?ref_=nv_sr_srsg_0)
         and this is reflected in the rating that it received. Other movies appearing on the list are titles such as Catwoman and Fifty Shades of Grey 
         that are based on novels/comic books received lower ratings because the public/critics felt that they didn't capture 
         the essence of the books/comics that they are based on.
        """
        st.subheader("50 Most Frequent Cast Members")
        st.image('resources/imgs/wc1.png',use_column_width=True)
        """
        Household names such as Steve Buscemi, who is credited with 166 acting roles accoring to imdb, 
        click [here](https://www.imdb.com/name/nm0000114/) for full bio. Keith David is credited with 366 roles 
        according to the same source, for more info : [here](https://www.imdb.com/name/nm0202966/?ref_=fn_al_nm_1). 
        Richard Jenkins has 115 acting roles under his belt, full bio [here](https://www.imdb.com/name/nm0420955/?ref_=nv_sr_srsg_0). 
        Samuel L. Jackson has 195 acting roles, full bio [here](https://www.imdb.com/name/nm0000168/?ref_=nv_sr_srsg_0).
        """
        """
        **N.B. For the following 2 diagrams we are assuming that the lead actor/actress is the first name mentioned on title credits. And the rating that the movie gets will be used as a proxy for the rating of the lead actor.**
        """
        st.subheader("Top Rated Lead Actors")
        st.image('resources/imgs/lta.png',use_column_width=True)
        """
        Stephen Baldwin highest rated lead actor according proxy that we used and is primarily known for the 
        movie "The Usual Suspect", it has the 5th highest average rating (4.28/5) in our filtered dataset. 
        Rumi Hiiragi is known for "Spirited Away" which is the 12th highest rated movie in our filtered dataset. 
        Tim Roth appeared in "Pulp Fiction" which is the 19th highest rated movie.
        """

        st.subheader("Bottom Rated Lead Actors")
        st.image('resources/imgs/bta.png',use_column_width=True)
        """
        Shaquile O'Neal is the second bottom 10 rated actor according to the proxy and known for the movie "Kazaam", 
        which the 4th worst rated movie in our filtered dataset. Mel B is the third bottom rated actor and known for the
         movie "Spice World", which received the 6th worst rating. Mark Addy is the forth bottom rated actor and known for 
         "Flintstones in  Viva Rock in Vegas", which part of the 20 bottom rated movies.
        """
        st.subheader("Most Frequently Occurring Directors")
        st.image('resources/imgs/tod.png',use_column_width=True)
        """We get prominent directors such as Woody Allen, who has directed 55 movies over the span of his career according to imdb, for boi :[link](https://www.imdb.com/name/nm0000095/).
        Luc Besson is has directed over 65 movies/series according to imdb, for bio : [link](https://www.imdb.com/name/nm0000108/?ref_=nv_sr_srsg_0). 
        Stephen King is also a household name and has directed 15 movies/screenplays and has 325 credits as a writer.
        """
        st.subheader("Top Rated Directors")
        st.image('resources/imgs/trd.png',use_column_width=True)
        """
        We get critically acclaimed directors such as Steven Spielberg, who is knowned for the "Schindler's List" which is 
        number 8 on the imdb 250 movies of all time and the 9th highest rated movie according to our filtered dataset. The top spot is held 
        by Chuck Palahniuk who is known for "Fight Club" which is also a critically acclaimed movie and is the 11th highest rated movie in our 
        dataset
        """
        st.subheader("Worst Rated Directors")
        st.image('resources/imgs/brd.png',use_column_width=True)
        """
        The worst rated director is Corey Mandell who is known for "Battlefield Earth" which is the worst rated movie according to our filtered dataset. Steven Paul is known for the movie "Baby Geniuses" and is the second worst rated director.
        The same applies to most of the names on this list, there's a near perfect association between the worst movies obtained earlier and the worst directors.
        """

    
        st.subheader("Most Common Tags")
        st.image('resources/imgs/mct.png',use_column_width=True)

        """
        It is not surprising that the most common tags are attributes of movies that correspond to general themes such as
        "twist ending" and "based on a book" and features such as genres are expected to be common across multiple movies.
        """
        st.subheader("Least Common Tags")
        st.image('resources/imgs/lct.png',use_column_width=True)

        """
        Least common tags are associations with the author which could possibly 
        apply only movies that are based on a book. And that the author is most 
        likely to correspond to only one movie.
        """

        
    if page_selection=="Our Team":
        st.title("Our Team")
        st.write("**Hlulani Nkonyani**")
        col1, mid, col2 = st.beta_columns([1.25,3.25,25])
        with col1:
            st.image('resources/imgs/hlux.jpg', width=100)
        with col2:
            st.write("Coordinator")


        st.write("**Kholofelo Ngoepe**")
        col1, mid, col2 = st.beta_columns([1.25,3.25,25])
        with col1:
            st.image('resources/imgs/kholo.jpeg', width=100)
        with col2:
            st.write("Insights")

        st.write("**Sbusiso Phakathi**")
        col1, mid, col2 = st.beta_columns([1.25,3.25,25])
        with col1:
            st.image('resources/imgs/hh.jpg', width=100)
        with col2:
            st.write("Comet and AWS")
        
        st.write("**Chwayita Happiness Sidzumo**")
        col1, mid, col2 = st.beta_columns([1.25,3.25,25])
        with col1:
            st.image('resources/imgs/happy.jpeg', width=100)
        with col2:
            st.write("Streamlit")
        
        st.write("**Raveena Ramlakan**")
        col1, mid, col2 = st.beta_columns([1.25,3.25,25])
        with col1:
            st.image('resources/imgs/raveena.png', width=100)
        with col2:
            st.write("Marketing")











        


        

        




    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()

