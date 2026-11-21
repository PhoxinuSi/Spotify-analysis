import streamlit as st
import pandas as pd
import preparations as md
import altair as alt
import pickle
# prepare data
data = pd.read_csv("dataset.csv")
data = md.prepare_data(data)


st.markdown(
    "<h2 style='text-align: center; color: #2b66c9;'>АНАЛИЗ ТРЕКОВ SPOTIFY</h2>", unsafe_allow_html=True
    )

MENU_OPTIONS = {
    "graphs" : "Исследовательский анализ данных EDA",
    "predict" : "Давайте предскажем настроение?"
}

with st.sidebar:
    st.write("**Выберете, что вы хотите сделать** ")
    option = st.radio(
        "",
    options=list(MENU_OPTIONS.values()),
    label_visibility="collapsed"
)

selected_key = [key for key, value in MENU_OPTIONS.items() if value == option][0]

if selected_key == "graphs":
    df = md.drop_names(data)
    st.markdown(
        "<h4 style='text-align: center; color: black;'>📊 ГРАФИКИ 📊</h4>", unsafe_allow_html=True
        )
    col1, col2 = st.columns(2)
    with col1:
        # Продолжительность треков в датасете
        st.markdown('###### Насколько длинные треки в датасете?')
        duration_info = df['duration_min'].round(0).value_counts().reset_index()
        duration_info.columns = ['minutes', 'counts']
        chart_duration = alt.Chart(duration_info).mark_bar().encode(
            x = alt.X('minutes:O',title=' Продолжительность (минуты)'),
            y = alt.Y('counts:Q',title='Количество треков (тысячи)', axis=alt.Axis(format="~s")))
        st.altair_chart(chart_duration)
        st.markdown("<p style='text-align: justify;'>В основном длительность треков находится между 1 и 8 минутами, при этом "
        "есть композиции, чья продолжительность достигает 80 минут 🤯 </p>", unsafe_allow_html=True)
        
    with col2:
        # популярность треков в датасете
        st.markdown("###### Какова популярность у треков в датасете?")
        popularity_info = df['popularity'].value_counts().reset_index()
        popularity_info.columns = ['popularity','counts']
        chart_popualrity = alt.Chart(popularity_info).mark_bar().encode(
            x = alt.X("popularity:O", title="Популярность трека"),
            y = alt.Y("counts:Q", title='Количество треков (тысячи)', axis=alt.Axis(format="~s"))
        )
        st.altair_chart(chart_popualrity)
        st.markdown("<p style='text-align: justify;'>Популярность высчитывается на основе многих критериев, как мы видим, большинство "
        "треков не особо известны 😊 </p>", unsafe_allow_html=True)
       
    col1, col2 = st.columns(2)
    with col1:
        # Соотношение треков со словами и без
        st.markdown('###### В каком соотношении находятся треки с матом и без?')
        explicit_info = df['explicit'].value_counts().reset_index()
        explicit_info.columns = ['criteria', 'counts']
        explicit_info['criteria'] = explicit_info['criteria'].map({
            0: "Без мата",
            1: "C матом"
        })
        chart_explicit = alt.Chart(explicit_info).mark_arc().encode(
            theta=alt.Theta('counts:Q'),
            color=alt.Color('criteria:N', 
                            title='Тип трека')).properties(
                                  width=400,
                                  height=300
                              )
        st.altair_chart(chart_explicit)
        st.markdown("<p style='text-align: justify;'>Большинство треков не содержат ненормативной лексики, "
        "что объяснимо, так как треки с матом могут иметь сложности в продвижении 🤬 </p>", unsafe_allow_html=True)
        
    with col2:
        #тепловая карта
        st.markdown("###### Какова корреляция числовых признаков датасета?")
        correlation_matrix = df[['danceability', 'energy', 'loudness','speechiness', 'acousticness', 'instrumentalness', 
                              'liveness', 'valence', 'tempo']].corr()
        corr_long = (
            correlation_matrix.reset_index().melt(id_vars='index')
        )
        corr_long.columns = ['feature_1','feature_2', 'correlation']

        heatmap = alt.Chart(corr_long).mark_rect().encode(
            x = alt.X('feature_1',title=''),
            y = alt.Y('feature_2',title=''),
            color = alt.Color('correlation:Q',
                              scale=alt.Scale(
                                  scheme='blues',
                                  domain=[-1,1]
                              ))).properties(
                                  width=400,
                                  height=300
                              )
        
        st.altair_chart(heatmap, use_container_width=False)
        st.markdown("<p style='text-align: justify;'> Наиболее выражена взаимосвязь между "
        "громкостью и энергичностью, а слабее всего между energy и acousticness. В целом же корреляция средняя 🤓 </p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        #связь популярности и продолжительности
        st.markdown("######  Какова продолжительность самых популярных треков?")
        pop_dur_info = df[df['popularity'] > 80][['duration_min','popularity']]
        pop_dur_chart = alt.Chart(pop_dur_info).mark_point().encode(
            x = alt.X("duration_min:Q",title="Продолжительность треков (минуты)"),
            y = alt.Y("popularity:Q",title='Популярность',
            scale=alt.Scale(domain=[80,100]))
        ).properties(
                                  width=400,
                                  height=300
                              )
        st.altair_chart(pop_dur_chart)
        st.markdown(
        "<p style='text-align: justify;'>Самые популярные треки находятся в диапазоне от двух до пяти "
        "минут, что в принципе логично, так как слушать одно и тоже слишком долго люди устают ⏳ </p>", unsafe_allow_html=True
        )
    
    with col2:
        st.markdown("######  Треки какого жанра наиболее популярны?")
        genre_info = df[df['popularity'] > 80]['track_genre'].value_counts()
        genre_info = genre_info[genre_info > 10].reset_index()
        genre_info.columns = ['genre','counts']
        genre_info = genre_info.sort_values('counts', ascending=False)
        chart_genre = alt.Chart(genre_info).mark_bar().encode(
            x = alt.X('genre:N',title=' Жанры'),
            y = alt.Y('counts:Q',title='Количество треков')).properties(
                                  width=400,
                                  height=320
                              )
        st.altair_chart(chart_genre)
        st.markdown(
            "<p style='text-align: justify;'>Наиболее популярные треки относятся к танцевальному "
        "и латиноамериканскому жанру, это не удивительно, так как такие треки часто ставят на фон 💃 </p>", unsafe_allow_html=True
        )
    
    st.write("<p style='text-align: justify;'>После подобного анализа было принято решение оставить в датасете только треки" \
    " до 10 минут, так как остальных ничтожно мало 🤝 </p>", unsafe_allow_html=True)

# СТРАНИЦА С ПРЕДСКАЗАНИЕМ НАСТРОЕНИЯ ПЕСНИ
elif selected_key == "predict":
    df = md.drop_names(data)
    df = md.drop_long_songs(df)
    st.markdown(
        "<h4 style='text-align: center; color: black;'> 😥 УГАДАЙКА НАСТРОЕНИЯ 🤪 </h4>", unsafe_allow_html=True
        )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        valence_info = st.number_input(
    "Valence?",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="""
    Valence показывает эмоциональную окраску трека.
    
    • 0 — грустный, мрачный, депрессивный

    • 0.5 — нейтральный

    • 1 — радостный, позитивный, счастливый
    """)

    with col2:
        energy_info = st.number_input(
    "Energy?",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="""
    Energy показывает ощущение интенсивности и активности трека.
    
    • 0 — малоактивный

    • 0.5 — нейтральный

    • 1 — энергичный, зажигательный
    """
        )

    with col3:
        acousticness_info = st.number_input(
    "Acousticness",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="""
    Acousticness показывает с какой степенью трек является акустическим
    """)

    with col4:
        danceability_info = st.number_input(
    "Danceability?",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="""
    Danceability показывает насколько трек подходит для танца.
    
    • 0 — совсем не подходит

    • 0.5 — нейтрально

    • 1 — идеальный трек для танцев
    """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        instrumental_info = st.number_input(
    "Instrumentalness?",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="""
    Instrumentalness показывает есть ли в треке вокал.
    
    • 0 — очень много разговорных слов

    • 0.5 — нейтрально

    • 1 — музыка без слов
    """
    )

    with col3:
        loudness_info = st.number_input(
    "Loudness?",
    min_value = -60.0,  
        max_value=0.0,
        value=-10.0,
        step=1.0,
        help="""
    Loudness показывает громкость трека в децибелах.
    """
    )
    with col2:
        options = data['track_genre'].unique().tolist()
        selected_option = st.selectbox(
        'Какой жанр вашего трека?',
        options
        )
    flag = False
    with col2:
        if st.button("Предсказать настроение",width="stretch"):
            flag = True
    if flag:
        user_input = {
            'valence': [valence_info],
            'energy': [energy_info],
            'acousticness': [acousticness_info],
            'danceability': [danceability_info],
            'instrumentalness': [instrumental_info],
            'loudness': [loudness_info],
            'track_genre': [selected_option]
        }
        user_df = pd.DataFrame(user_input)
        with open ("predict_mood_score.pkl", "rb") as file:
            model = pickle.load(file)
            prediction = float(model.predict(user_df)[0].round(4))
            st.write(f"В количественном выражение настроение песни составляет {prediction}")
            if prediction <= 0.3:
                st.write("Трек скорее грустный 😭")
            elif prediction < 0.7:
                st.write("Трек скорее нейтральный 🫣")
            else:
                st.write("Трек скорее радостный 😃")

