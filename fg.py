import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap, TimestampedGeoJson
from streamlit_folium import st_folium
from datetime import timedelta
import requests
from io import BytesIO
import os

# --- 1. Конфигурация страницы и заголовок ---
st.set_page_config(layout="wide", page_title="Анализ судов и нефтяных пятен")
st.title("🛰️ Анализ судов и нефтяных пятен 🛢️")
st.markdown("""
Приложение автоматически загружает данные о нефтяных пятнах и судах, анализирует потенциальную связь между ними,
и визуализирует маршруты судов, которые могли оставить пятна.
""")

# --- 2. Функции для загрузки данных ---
@st.cache_data(ttl=86400)  # Кеширование на 24 часа
def load_spills_data():
    """Загрузка данных о нефтяных пятнах"""
    url = "https://raw.githubusercontent.com/your-account/oil-spills-data/main/fields2.geojson"
    try:
        response = requests.get(url)
        response.raise_for_status()
        gdf = gpd.read_file(BytesIO(response.content))
        st.success("Данные о нефтяных пятнах успешно загружены")
        return gdf
    except Exception as e:
        st.error(f"Ошибка при загрузке данных о пятнах: {str(e)}")
        return gpd.GeoDataFrame()

@st.cache_data(ttl=3600)
def load_ais_data():
    """Загрузка данных AIS (пример, замените на реальный источник)"""
    # В реальном приложении замените URL на актуальный источник данных
    url = "https://marine-api.example.com/ais_data.csv"
    try:
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(BytesIO(response.content))
        st.success("Данные AIS успешно загружены")
        return df
    except Exception as e:
        st.error(f"Ошибка при загрузке данных AIS: {str(e)}")
        return pd.DataFrame()

# --- 3. Функции для анализа данных ---
def preprocess_data(spills_gdf, ais_df):
    """Предварительная обработка данных"""
    # Обработка пятен
    spills_gdf = spills_gdf.rename(columns={'slick_name': 'spill_id', 'area_sys': 'area_sq_km'})
    
    if 'date' in spills_gdf.columns and 'time' in spills_gdf.columns:
        spills_gdf['detection_date'] = pd.to_datetime(
            spills_gdf['date'] + ' ' + spills_gdf['time'], 
            errors='coerce'
        )
    else:
        spills_gdf['detection_date'] = pd.to_datetime(
            spills_gdf['spill_id'], 
            format='%Y-%m-%d_%H:%M:%S', 
            errors='coerce'
        )
    
    spills_gdf = spills_gdf.dropna(subset=['detection_date'])
    
    # Обработка данных AIS
    ais_gdf = gpd.GeoDataFrame(
        ais_df,
        geometry=gpd.points_from_xy(ais_df.longitude, ais_df.latitude),
        crs="EPSG:4326"
    )
    ais_gdf['timestamp'] = pd.to_datetime(ais_gdf['BaseDateTime'], errors='coerce')
    ais_gdf = ais_gdf.dropna(subset=['timestamp'])
    
    return spills_gdf, ais_gdf

def find_ship_spill_connections(spills_gdf, ais_gdf, time_window_hours):
    """Поиск связей между судами и нефтяными пятнами"""
    connections = []
    time_delta = timedelta(hours=time_window_hours)
    
    for _, spill in spills_gdf.iterrows():
        spill_geom = spill.geometry
        spill_time = spill.detection_date
        
        # Фильтрация судов во временном окне
        vessels_in_window = ais_gdf[
            (ais_gdf['timestamp'] >= spill_time - time_delta) &
            (ais_gdf['timestamp'] <= spill_time)
        ]
        
        # Поиск судов внутри полигона пятна
        vessels_in_spill = vessels_in_window[vessels_in_window.geometry.within(spill_geom)]
        
        for _, vessel in vessels_in_spill.iterrows():
            connections.append({
                'spill_id': spill.spill_id,
                'detection_date': spill.detection_date,
                'area_sq_km': spill.area_sq_km,
                'mmsi': vessel.mmsi,
                'vessel_name': vessel.get('vessel_name', 'N/A'),
                'vessel_type': vessel.get('vessel_type', 'N/A'),
                'timestamp': vessel.timestamp,
                'time_to_detection': (spill.detection_date - vessel.timestamp).total_seconds() / 3600
            })
    
    return pd.DataFrame(connections)

# --- 4. Функции для визуализации ---
def create_base_map(default_location=[60.0, 70.0], default_zoom=5):
    """Создание базовой карты"""
    return folium.Map(
        location=default_location,
        zoom_start=default_zoom,
        tiles='CartoDB positron',
        control_scale=True
    )

def plot_spills(spills_gdf, map_object):
    """Добавление нефтяных пятен на карту"""
    spills_layer = folium.FeatureGroup(name='Нефтяные пятна', show=True)
    
    for _, row in spills_gdf.iterrows():
        spill_popup = folium.Popup(
            f"<b>ID:</b> {row.spill_id}<br>"
            f"<b>Дата:</b> {row.detection_date.strftime('%Y-%m-%d %H:%M')}<br>"
            f"<b>Площадь:</b> {row.area_sq_km:.2f} км²",
            max_width=300
        )
        
        folium.GeoJson(
            row.geometry,
            style_function=lambda x: {
                'fillColor': '#ff0000',
                'color': '#ff0000',
                'weight': 1,
                'fillOpacity': 0.4
            },
            popup=spill_popup
        ).add_to(spills_layer)
    
    spills_layer.add_to(map_object)
    return map_object

def plot_ship_routes(connections_df, ais_gdf, map_object):
    """Визуализация маршрутов судов"""
    if connections_df.empty:
        return map_object
    
    routes_layer = folium.FeatureGroup(name='Маршруты судов', show=True)
    
    for mmsi in connections_df['mmsi'].unique():
        ship_data = ais_gdf[ais_gdf['mmsi'] == mmsi].sort_values('timestamp')
        ship_name = ship_data.iloc[0].get('vessel_name', f'Судно {mmsi}')
        
        # Создание маршрута
        route_points = []
        for _, point in ship_data.iterrows():
            route_points.append([point.geometry.y, point.geometry.x])
        
        if len(route_points) > 1:
            folium.PolyLine(
                locations=route_points,
                color='#1f77b4',
                weight=3,
                opacity=0.7,
                popup=f"Маршрут: {ship_name} (MMSI: {mmsi})"
            ).add_to(routes_layer)
        
        # Добавление точек маршрута с временными метками
        for _, point in ship_data.iterrows():
            folium.CircleMarker(
                location=[point.geometry.y, point.geometry.x],
                radius=4,
                color='#1f77b4',
                fill=True,
                fill_color='#1f77b4',
                popup=f"{ship_name}<br>Время: {point.timestamp.strftime('%Y-%m-%d %H:%M')}"
            ).add_to(routes_layer)
    
    routes_layer.add_to(map_object)
    return map_object

def plot_connections(connections_df, map_object):
    """Добавление связей судно-пятно на карту"""
    if connections_df.empty:
        return map_object
    
    connections_layer = folium.FeatureGroup(name='Связи судно-пятно', show=True)
    
    for _, row in connections_df.iterrows():
        spill_center = spills_gdf[spills_gdf['spill_id'] == row.spill_id].geometry.iloc[0].centroid
        vessel_point = ais_gdf[
            (ais_gdf['mmsi'] == row.mmsi) & 
            (ais_gdf['timestamp'] == row.timestamp)
        ].geometry.iloc[0]
        
        folium.PolyLine(
            locations=[
                [vessel_point.y, vessel_point.x],
                [spill_center.y, spill_center.x]
            ],
            color='#ff9900',
            weight=2,
            dash_array='5, 10',
            popup=f"Судно: {row.vessel_name} (MMSI: {row.mmsi})<br>Пятно: {row.spill_id}"
        ).add_to(connections_layer)
        
        folium.Marker(
            location=[spill_center.y, spill_center.x],
            icon=folium.Icon(color='red', icon='tint', prefix='fa'),
            popup=f"Пятно: {row.spill_id}"
        ).add_to(connections_layer)
        
        folium.Marker(
            location=[vessel_point.y, vessel_point.x],
            icon=folium.Icon(color='blue', icon='ship', prefix='fa'),
            popup=f"Судно: {row.vessel_name}<br>Время: {row.timestamp.strftime('%Y-%m-%d %H:%M')}"
        ).add_to(connections_layer)
    
    connections_layer.add_to(map_object)
    return map_object

# --- 5. Основной интерфейс приложения ---
time_window_hours = st.sidebar.slider(
    "Временное окно поиска (часы до обнаружения):",
    min_value=1, max_value=72, value=24, step=1
)

analysis_type = st.sidebar.radio(
    "Тип анализа:",
    ["Все суда", "Только подозрительные"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Инструкция:**
1. Данные автоматически загружаются при запуске
2. Используйте временное окно для настройки чувствительности
3. Выберите тип анализа для фильтрации результатов
""")

# Загрузка данных
spills_gdf = load_spills_data()
ais_df = load_ais_data()

if spills_gdf.empty or ais_df.empty:
    st.warning("Не удалось загрузить данные. Проверьте подключение к интернету.")
    st.stop()

# Обработка данных
with st.spinner("Обработка данных..."):
    spills_gdf, ais_gdf = preprocess_data(spills_gdf, ais_df)
    connections_df = find_ship_spill_connections(spills_gdf, ais_gdf, time_window_hours)

# Отображение статистики
st.subheader("📊 Статистика данных")
col1, col2, col3 = st.columns(3)
col1.metric("Нефтяных пятен", len(spills_gdf))
col2.metric("Записей AIS", len(ais_gdf))
col3.metric("Найденных связей", len(connections_df))

# Отображение карты
st.subheader("🗺️ Карта нефтяных пятен и маршрутов судов")

if not spills_gdf.empty:
    map_center = [spills_gdf.geometry.centroid.y.mean(), spills_gdf.geometry.centroid.x.mean()]
else:
    map_center = [60.0, 70.0]  # Координаты по умолчанию

m = create_base_map(default_location=map_center, default_zoom=6)
m = plot_spills(spills_gdf, m)

if analysis_type == "Все суда":
    m = plot_ship_routes(ais_gdf, ais_gdf, m)
else:
    if not connections_df.empty:
        m = plot_ship_routes(connections_df, ais_gdf, m)
        m = plot_connections(connections_df, m)

folium.LayerControl().add_to(m)
st_folium(m, width=1200, height=600)

# Отображение таблиц с результатами
st.subheader("🔍 Результаты анализа")

if not connections_df.empty:
    st.write("### Судья с потенциальной связью с нефтяными пятнами")
    connections_display = connections_df[[
        'mmsi', 'vessel_name', 'vessel_type', 'spill_id', 
        'timestamp', 'detection_date', 'time_to_detection', 'area_sq_km'
    ]]
    
    connections_display = connections_display.rename(columns={
        'mmsi': 'MMSI',
        'vessel_name': 'Название судна',
        'vessel_type': 'Тип судна',
        'spill_id': 'ID пятна',
        'timestamp': 'Время судна',
        'detection_date': 'Время обнаружения',
        'time_to_detection': 'Время до обнаружения (ч)',
        'area_sq_km': 'Площадь (км²)'
    })
    
    st.dataframe(connections_display.sort_values('Время до обнаружения (ч)'))

    # Аналитика по судам
    st.write("### Статистика по судам")
    ship_stats = connections_df.groupby(['mmsi', 'vessel_name']).agg(
        spill_count=('spill_id', 'nunique'),
        total_area=('area_sq_km', 'sum'),
        avg_time_to_detection=('time_to_detection', 'mean')
    ).reset_index().sort_values('spill_count', ascending=False)
    
    ship_stats = ship_stats.rename(columns={
        'mmsi': 'MMSI',
        'vessel_name': 'Название судна',
        'spill_count': 'Количество пятен',
        'total_area': 'Суммарная площадь (км²)',
        'avg_time_to_detection': 'Среднее время до обнаружения (ч)'
    })
    
    st.dataframe(ship_stats)
else:
    st.info("Не найдено связей между судами и нефтяными пятнами в заданном временном окне")

# Отображение сырых данных
with st.expander("📁 Просмотр исходных данных"):
    tab1, tab2 = st.tabs(["Данные о нефтяных пятнах", "Данные AIS"])
    
    with tab1:
        st.write(spills_gdf.drop(columns=['geometry']))
    
    with tab2:
        st.write(ais_gdf.drop(columns=['geometry']).head(1000))

st.markdown("---")
st.caption("© 2023 Система мониторинга нефтяных разливов | Данные обновляются ежедневно")
