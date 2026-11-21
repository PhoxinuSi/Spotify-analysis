import pandas as pd


#Data Preparation
def prepare_data(data):
    df_clean = data.dropna()
    df = df_clean.drop_duplicates(subset='track_id')
    df['first_artist'] = df['artists'].str.split(";",n=1).str[0]
    df['duration_min'] = df['duration_ms']/60_000
    df = df.drop('duration_ms',axis=1)
    df['explicit'] = df['explicit'].apply(int)
    df = df.drop(columns=['Unnamed: 0','track_id', 'album_name','artists'])
    #data = df[df['duration_min'] < 10]
    data = df
    data['mood_score'] = data['valence']*data['energy']
    data['acoustic_sadness'] = data['acousticness'] * (1 - data['valence'])
    data['is_popular'] = data['popularity'] >= 70
    data['is_popular'] = data['is_popular'].apply(int)
    return data 

def drop_names(df):
    data = df.drop(columns=['track_name'])
    return data


def drop_long_songs(df):
    data =  df[df['duration_min'] < 10]
    return data

