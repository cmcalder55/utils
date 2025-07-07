from sklearn.feature_extraction.text import CountVectorizer

from twitter_scraper import get_all_tweets, clean_tweet
from plotter import show_semantic_network


def subdf(df, term):
    """ get a subset of tweets """
    sens = []
    for i in df.index:
        sen = df['full text'][i]

        if term in sen.lower():
            sens.extend([i])

    return df.loc[sens]

def vec(x):
    """ vectorizer to get the desired features and adj list """

    vectorizer = CountVectorizer(
        ngram_range=(1,1),              # default unigram model
        min_df=2, 
        stop_words="english",
        max_features=50
    ) 
    X = vectorizer.fit_transform(x)
    feature_names = vectorizer.get_feature_names_out()
    return X, feature_names

def cov_mat(X, threshold=1):
    """ get the feature to feature co-occurrence matrix 

    :param int threshold: threshold to reduce the density of the network

    :returns: covariance matrix
    """
    co = (X.T * X)              # co-occurrence matrix in sparse csr format
    co.setdiag(0)               # fill same word cooccurence to 0
    return np.where(co.todense()>threshold, 1, 0)

def build_semantic_network(x):
    """ build the semantic network graph 
    - use the dichotomized cooccurrence matrix to build the network
    - make node size proportional to the eigenvector centrality of words
    """

    df = pd.Series(list(x))
   # get features and adjacency list
    X, feats = vec(df)
    cooccurence_matrix = cov_mat(X)

    # Build a network out of this co-occurrence matrix from adjacency matrix
    G = nx.from_numpy_matrix(cooccurence_matrix, create_using=nx.Graph())
    node_name = dict(zip(range(0,len(list(G.nodes()))), feats))
    G = nx.relabel_nodes(G, node_name)
    return G

def run(context, title, term):
    corpus = get_all_tweets(context)                            # scrape the tweets for the specified accounts and compile a corpus
    df = subdf(corpus, term)                                    # get a subset of tweets for the target term
    df["cleaned_tweet"] = df.iloc[:,3].apply(clean_tweet)       # clean the tweets
    G = build_semantic_network(set(df["cleaned_tweet"]))
    show_semantic_network(G, title)
    return df, G
