import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import preparations as md
import pickle


data = pd.read_csv("dataset.csv")
df = md.prepare_data(data)
categorical_cols = [ 'track_genre'
                    ]
numerical_cols = ['danceability','energy','loudness',
                  'acousticness','instrumentalness',
                  'valence'
                  ]
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols),
        ('num', numerical_transformer, numerical_cols)
    ]
)

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('reg', LinearRegression())
    ])
y = df['mood_score']
selected_features = ['valence', 'energy', 'acousticness', 'danceability', 
                     'instrumentalness', 'loudness', 'track_genre']
X = df[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.33)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)


print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

model_filename_1 = "predict_mood_score.pkl"
with open (model_filename_1, "wb") as file:
   pickle.dump(pipeline, file)
print("Линейная регрессия загружена")