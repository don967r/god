The user wants to add a function to save the results of the analysis in the provided Python script. I will add a section for saving dataframes to CSV files, specifically for the "Таблица судов-кандидатов" and the "Дополнительная аналитика" dataframes.

Here's how I'll modify the script:

1.  **Add a new header for download options.**
2.  **Add a `to_csv` function to convert dataframes to CSV and provide a download button.**
3.  **Apply this function to the `display_df` (Суда-кандидаты).**
4.  **Apply this function to the `ship_incident_counts`, `ship_area_sum`, `spill_candidate_counts`, and `prime_suspects_df` dataframes from the "Дополнительная аналитика" section.**

<!-- end list -->

```python
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
import plotly.express as px
from streamlit_folium import st_folium
from datetime import timedelta

# --- 1. Конфигурация страницы и Заголовок ---
st.set_page_config(layout="wide", page_title="Анализ 'Судно-Пятно'")

st.title("🚢 Анализ связи 'Судно-Пятно' 💧")
st.write("""
Загрузите GeoJSON с полигонами разливов и CSV-файл с данными AIS.
Приложение автоматически найдет суда, которые находились в зоне разлива незадолго до его обнаружения,
и предоставит расширенную аналитику по инцидентам.
""")

# --- 2. Боковая панель с загрузчиками и параметрами ---
st.sidebar.header("Параметры анализа")

spills_file = st.sidebar.file_uploader(
    "1. Загрузите GeoJSON с полигоонами разливов",
    type=["geojson", "json"]
)
ais_file = st.sidebar.file_uploader(
    "2. Загрузите CSV с данными AIS",
    type=["csv"]
)

time_window_hours = st.sidebar.slider(
    "3. Временное окно поиска (часы до обнаружения):",
    min_value=1, max_value=168, value=24, step=1,
    help="Искать суда, которые были в зоне разлива за указанное количество часов ДО его обнаружения."
)

# --- 3. Функции для обработки и анализа данных ---

@st.cache_data
def load_spills_data(uploaded_file):
    st.info("Обработка GeoJSON с разливами...")
    try:
        gdf = gpd.read_file(uploaded_file)
    except Exception as e:
        st.error(f"Не удалось прочитать GeoJSON файл. Ошибка: {e}")
        return None

    required_cols = ['slick_name', 'area_sys']
    if not all(col in gdf.columns for col in required_cols):
        missing = [col for col in required_cols if col not in gdf.columns]
        st.error(f"В GeoJSON отсутствуют обязательные поля: {', '.join(missing)}")
        return None

    gdf.rename(columns={'slick_name': 'spill_id', 'area_sys': 'area_sq_km'}, inplace=True)

    if 'date' in gdf.columns and 'time' in gdf.columns:
        st.success("Обнаружен формат с колонками 'date' и 'time'.")
        gdf['detection_date'] = pd.to_datetime(gdf['date'] + ' ' + gdf['time'], errors='coerce')
    else:
        st.success("Обнаружен формат с датой в ID пятна. Парсинг 'spill_id'...")
        gdf['detection_date'] = pd.to_datetime(gdf['spill_id'], format='%Y-%m-%d_%H:%M:%S', errors='coerce')

    if gdf['detection_date'].isnull().any():
        failed_count = gdf['detection_date'].isnull().sum()
        st.warning(f"Не удалось распознать дату в {failed_count} записях о разливах. Эти записи будут проигнорированы.")
        gdf.dropna(subset=['detection_date'], inplace=True)

    if gdf.empty:
        st.error("После обработки не осталось ни одной записи о разливах с корректной датой.")
        return None

    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    else:
        gdf = gdf.to_crs("EPSG:4326")

    st.success("Данные о разливах успешно загружены и обработаны.")
    return gdf

@st.cache_data
def load_ais_data(uploaded_file):
    st.info("Обработка CSV с данными AIS...")
    try:
        # Указываем low_memory=False для избежания ошибок с типами данных в больших файлах
        df = pd.read_csv(uploaded_file, low_memory=False)
    except Exception as e:
        st.error(f"Не удалось прочитать CSV файл. Ошибка: {e}")
        return None

    required_cols = ['mmsi', 'latitude', 'longitude', 'BaseDateTime']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        st.error(f"В CSV файле отсутствуют обязательные колонки: {', '.join(missing)}")
        return None

    df['timestamp'] = pd.to_datetime(df['BaseDateTime'], errors='coerce')
    df.dropna(subset=['timestamp', 'latitude', 'longitude'], inplace=True)

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )

    st.success("Данные AIS успешно загружены.")
    return gdf

def find_candidates(spills_gdf, vessels_gdf, time_window_hours):
    if spills_gdf is None or vessels_gdf is None:
        return gpd.GeoDataFrame()

    candidates = gpd.sjoin(vessels_gdf, spills_gdf, predicate='within')

    if candidates.empty:
        return gpd.GeoDataFrame()

    time_delta = timedelta(hours=time_window_hours)
    candidates = candidates[
        (candidates['timestamp'] <= candidates['detection_date']) &
        (candidates['timestamp'] >= candidates['detection_date'] - time_delta)
    ]

    return candidates

# --- Function to convert DataFrame to CSV for download ---
@st.cache_data
def convert_df_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

# --- 4. Основная логика приложения ---
if spills_file and ais_file:
    spills_gdf = load_spills_data(spills_file)
    vessels_gdf = load_ais_data(ais_file)

    if spills_gdf is None or vessels_gdf is None or spills_gdf.empty or vessels_gdf.empty:
        st.stop()

    candidates_df = find_candidates(spills_gdf, vessels_gdf, time_window_hours)

    st.header("Карта разливов и судов-кандидатов")

    map_center = [spills_gdf.unary_union.centroid.y, spills_gdf.unary_union.centroid.x]
    m = folium.Map(location=map_center, zoom_start=8, tiles="CartoDB positron")

    # Слой с пятнами
    spills_fg = folium.FeatureGroup(name="Пятна разливов").add_to(m)
    for _, row in spills_gdf.iterrows():
        folium.GeoJson(
            row['geometry'],
            style_function=lambda x: {'fillColor': '#B22222', 'color': 'black', 'weight': 1.5, 'fillOpacity': 0.6},
            tooltip=f"<b>Пятно:</b> {row.get('spill_id', 'N/A')}<br>"
                    f"<b>Время:</b> {row['detection_date'].strftime('%Y-%m-%d %H:%M')}<br>"
                    f"<b>Площадь:</b> {row.get('area_sq_km', 0):.2f} км²"
        ).add_to(spills_fg)

    if not candidates_df.empty:
        candidate_vessels_fg = folium.FeatureGroup(name="Суда-кандидаты").add_to(m)
        for _, row in candidates_df.iterrows():
            vessel_name = row.get('vessel_name', 'Имя не указано')
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                tooltip=f"<b>Судно:</b> {vessel_name} (MMSI: {row['mmsi']})<br>"
                        f"<b>Время прохода:</b> {row['timestamp'].strftime('%Y-%m-%d %H:%M')}<br>"
                        f"<b>Внутри пятна:</b> {row['spill_id']}",
                icon=folium.Icon(color='blue', icon='ship', prefix='fa')
            ).add_to(candidate_vessels_fg)

    folium.LayerControl().add_to(m)
    st_folium(m, width=1200, height=500)

    st.header(f"Таблица судов-кандидатов (найдено в пределах {time_window_hours} часов)")

    if candidates_df.empty:
        st.info("В заданном временном окне суда-кандидаты не найдены.")
    else:
        # Убираем дубликаты, если одно судно несколько раз попало в одно пятно
        report_df = candidates_df.drop_duplicates(subset=['spill_id', 'mmsi'])
        desired_cols = ['spill_id', 'mmsi', 'vessel_name', 'timestamp', 'detection_date', 'area_sq_km']
        existing_cols = [col for col in desired_cols if col in report_df.columns]
        display_df = report_df[existing_cols].copy()

        rename_dict = {
            'spill_id': 'ID Пятна',
            'mmsi': 'MMSI Судна',
            'vessel_name': 'Название судна',
            'timestamp': 'Время прохода судна',
            'detection_date': 'Время обнаружения пятна',
            'area_sq_km': 'Площадь пятна, км²'
        }
        display_df.rename(columns=rename_dict, inplace=True)
        st.dataframe(display_df.sort_values(by='Время обнаружения пятна', ascending=False).reset_index(drop=True))

        # --- DOWNLOAD BUTTON FOR CANDIDATE VESSELS TABLE ---
        csv_display_df = convert_df_to_csv(display_df)
        st.download_button(
            label="⬇️ Скачать таблицу судов-кандидатов как CSV",
            data=csv_display_df,
            file_name='candidate_vessels_report.csv',
            mime='text/csv',
        )

        # --- НОВЫЙ БЛОК С РАСШИРЕННОЙ АНАЛИТИКОЙ ---
        st.markdown("---")
        st.header("Дополнительная аналитика")

        # Готовим данные для аналитики - уникальные пары "судно-пятно"
        unique_incidents = candidates_df.drop_duplicates(subset=['mmsi', 'spill_id'])

        tab1, tab2, tab3 = st.tabs(["📊 Аналитика по судам", "📍 Горячие точки (Hotspots)", "🔍 Аналитика по инцидентам"])

        with tab1:
            st.subheader("Антирейтинг по количеству связанных пятен")
            ship_incident_counts = unique_incidents.groupby('mmsi').size().reset_index(name='incident_count') \
                .sort_values('incident_count', ascending=False).reset_index(drop=True)
            if 'vessel_name' in unique_incidents.columns:
                ship_names = unique_incidents[['mmsi', 'vessel_name']].drop_duplicates()
                ship_incident_counts = pd.merge(ship_incident_counts, ship_names, on='mmsi', how='left')
            st.dataframe(ship_incident_counts)
            csv_ship_incident_counts = convert_df_to_csv(ship_incident_counts)
            st.download_button(
                label="⬇️ Скачать антирейтинг судов по количеству пятен",
                data=csv_ship_incident_counts,
                file_name='ship_incident_counts.csv',
                mime='text/csv',
                key='ship_incident_counts_dl'
            )
            
            st.subheader("Антирейтинг по суммарной площади связанных пятен (км²)")
            ship_area_sum = unique_incidents.groupby('mmsi')['area_sq_km'].sum().reset_index(name='total_area_sq_km') \
                .sort_values('total_area_sq_km', ascending=False).reset_index(drop=True)
            if 'vessel_name' in unique_incidents.columns:
                ship_area_sum = pd.merge(ship_area_sum, ship_names, on='mmsi', how='left')
            st.dataframe(ship_area_sum)
            csv_ship_area_sum = convert_df_to_csv(ship_area_sum)
            st.download_button(
                label="⬇️ Скачать антирейтинг судов по площади пятен",
                data=csv_ship_area_sum,
                file_name='ship_area_sum.csv',
                mime='text/csv',
                key='ship_area_sum_dl'
            )

        with tab2:
            st.subheader("Карта 'горячих точек' разливов")
            m_heatmap = folium.Map(location=map_center, zoom_start=8, tiles="CartoDB positron")
            heat_data = [[point.xy[1][0], point.xy[0][0], row['area_sq_km']] for index, row in spills_gdf.iterrows() for point in [row['geometry'].centroid]]
            HeatMap(heat_data, radius=15, blur=20, max_zoom=10).add_to(m_heatmap)
            st_folium(m_heatmap, width=1200, height=500)

        with tab3:
            st.subheader("Пятна с наибольшим количеством судов-кандидатов")
            spill_candidate_counts = candidates_df.groupby('spill_id')['mmsi'].nunique().reset_index(name='candidate_count') \
                .sort_values('candidate_count', ascending=False).reset_index(drop=True)
            st.dataframe(spill_candidate_counts)
            csv_spill_candidate_counts = convert_df_to_csv(spill_candidate_counts)
            st.download_button(
                label="⬇️ Скачать пятна по количеству судов-кандидатов",
                data=csv_spill_candidate_counts,
                file_name='spill_candidate_counts.csv',
                mime='text/csv',
                key='spill_candidate_counts_dl'
            )

            st.subheader("Главные подозреваемые (минимальное время до обнаружения)")
            candidates_df['time_to_detection'] = candidates_df['detection_date'] - candidates_df['timestamp']
            prime_suspects_idx = candidates_df.groupby('spill_id')['time_to_detection'].idxmin()
            prime_suspects_df = candidates_df.loc[prime_suspects_idx]

            display_cols = ['spill_id', 'mmsi', 'vessel_name', 'time_to_detection', 'area_sq_km']
            existing_display_cols = [col for col in display_cols if col in prime_suspects_df.columns]
            st.dataframe(prime_suspects_df[existing_display_cols].sort_values('area_sq_km', ascending=False))
            csv_prime_suspects_df = convert_df_to_csv(prime_suspects_df[existing_display_cols])
            st.download_button(
                label="⬇️ Скачать таблицу главных подозреваемых",
                data=csv_prime_suspects_df,
                file_name='prime_suspects_report.csv',
                mime='text/csv',
                key='prime_suspects_dl'
            )

        # Опциональный блок для анализа по типам судов
        if 'VesselType' in unique_incidents.columns:
            with st.expander("🚢 Аналитика по типам судов"):
                vessel_type_analysis = unique_incidents.groupby('VesselType').agg(
                    incident_count=('spill_id', 'count'),
                    total_area_sq_km=('area_sq_km', 'sum')
                ).sort_values('incident_count', ascending=False).reset_index()

                st.dataframe(vessel_type_analysis)
                csv_vessel_type_analysis = convert_df_to_csv(vessel_type_analysis)
                st.download_button(
                    label="⬇️ Скачать аналитику по типам судов",
                    data=csv_vessel_type_analysis,
                    file_name='vessel_type_analysis.csv',
                    mime='text/csv',
                    key='vessel_type_analysis_dl'
                )

                fig = px.pie(vessel_type_analysis, names='VesselType', values='incident_count',
                             title='Распределение инцидентов по типам судов',
                             labels={'VesselType':'Тип судна', 'incident_count':'Количество инцидентов'})
                st.plotly_chart(fig)

else:
    st.info("⬅️ Пожалуйста, загрузите оба файла на боковой панели, чтобы начать анализ.")

```
