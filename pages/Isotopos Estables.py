import streamlit as st
import matplotlib.pyplot as plt
import streamlit_folium as st_folium
import folium
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas  as pd
import seaborn as sns
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from scipy.stats import linregress
from PIL import Image


logo = Image.open("images/PageImage.png")  # Reemplaza con la ruta de tu logo

# Mostrar el logo en la barra lateral
st.sidebar.image(logo, use_container_width=True)


with st.sidebar:
    st.markdown("""
        # Uso de isótopos estables en Hidrogeología
    """)
    st.markdown("<br><br><br>", unsafe_allow_html=True)  # Ajusta el número de <br> según necesites

    st.sidebar.image("images/Profile.png", width=80)
    # Texto en la barra lateral
    st.sidebar.markdown(
        """
        <div style="text-align: left;">
            <p><strong>Prof. Adriana Piña</strong><br> Universidad Nacional de Colombia</p>
        </div>
        """,
        unsafe_allow_html=True
    )
# Inicializar la variable de página en el estado de sesión si no existe
if "page" not in st.session_state:
    st.session_state.page = 1

# Definir una función para cambiar de página
def next_page():
    if st.session_state.page < total_pages:
        st.session_state.page += 1

def prev_page():
    if st.session_state.page > 1:
        st.session_state.page -= 1

# Definir el número total de páginas
total_pages = 6  # Cambia esto según el número de páginas que necesites

# Mostrar el contenido según la página actual
# ----------------------------------------------------------------------------------------------
if st.session_state.page == 1:
#----------------------------------------------------------------------------------------------
    st.markdown("""

    # USO DE ISÓTOPOS ESTABLES EN HIDROGEOLOGÍA

    Prof. Adriana Piña

    Este taller práctico se centrará en el uso de isótopos estables (δ¹⁸O y δ²H) aplicados a la hidrogeología, proporcionando a los participantes una comprensión teórica y práctica de su relevancia en el estudio de los recursos hídricos. A través de actividades guiadas, los participantes aprenderán a manejar y analizar datos isotópicos utilizando herramientas computacionales como Excel y Python, con el fin de construir gráficos e interpretar patrones isotópicos que permitan identificar procesos hidrogeológicos clave, como la recarga de acuíferos, mezcla de aguas y evaporación.
    El taller combinará presentaciones teóricas breves con actividades prácticas enfocadas en la interpretación de datos reales o simulados. Los asistentes trabajarán directamente con conjuntos de datos isotópicos para generar gráficas de δ¹⁸O vs. δ²H, relacionar los resultados con líneas de meteóricas global y local, y discutir las implicaciones hidrogeológicas de sus análisis. Además, se abordará la relación entre las firmas isotópicas y factores como la altitud de las estaciones de muestreo.


    ## Objetivo

    Este recurso permite cargar, visualizar y analizar información isotópica, contruir líneas meteóricas locales y calcular parámetros isotópicos como en Exceso de Deuterio.
    """)
#----------------------------------------------------------------------------------------------
elif st.session_state.page == 2:
#----------------------------------------------------------------------------------------------
    st.markdown("""
                # Análisis de Isótopos Estables
                ## Datos de prueba
                Algunas fuentes de información del contenido isotópico en el agua.
                """)
    col1,col2 = st.columns([0.3,0.3])
    with col1:
        with st.expander("SIAMS-UNAL"):
            st.write("Descripción base de datos unal")
    with col2:
        with st.expander("MEGIA Isotopes"):
            st.write("Base de datos deL Proyecto MEGIA") 

    df_selection = st.radio(
        "Selecciona la base de datos de prueba que quieras usar",
        ["SIAMS-UNAL","MEGIA Isotopes","Otro"],
        captions=[
            "Chingaza,Bogotá,Guaymaral-Torca",
            "VMM",
            "Archivo propio o datos ingresados"
        ]
    )
    st.session_state.df_selection = df_selection  # Asignar df a session_state
    if df_selection == "Otro":
        st.markdown("""
        En esta sección podras ingresar tus datos **manualmente**, para realizar la representación visual de sus cantidad isotópicas de $\delta O^{18}$ y $\delta 2^H$ 
        """)
        #Cargar archivo de ejemplo de excel
        edf = pd.read_excel("datos.xlsx").to_csv(index=False)

        #Ingreso de datos de forma manual
        df = pd.DataFrame(
            {
                "Nombre": ["Punto1"],
                "Latitud": [10.123],
                "Longitud": [-74.123],
                "18O": [2.1],
                "2H": [-8.5],
                "Tipo":["Subterraneo"],
                "Fecha":["12/02/2024"]
            }
        )
        writed_df = st.data_editor(df,num_rows="dynamic")

        st.markdown("O descargar la plantilla y colocar tus datos en ella en formato CSV")
        col1,col2 = st.columns([1,2.5])
        with col1:
            st.write("*Descarga el archivo plantilla*")
            st.download_button( label="Descargar CSV", data=edf, file_name="data.csv", mime="text/csv")
            st.warning("Si no deseas descargar la plantilla asegurate de que los nombres de las columnas correspondan con los de la tabla presentada previamente")
        with col2:
            C1,C2,C3 = st.columns([1,.5,4])
            with C1:
                st.write("Indica el separador de tu CSV")
            with C2:
                sep = st.text_input(label="",value=',')
            with C3:
                # Carga de datos desde archivo excel
                uploaded_df = st.file_uploader("Sube tu archivo",type=["csv"])
                # Procesar el archivo cuando el usuario lo suba
            if uploaded_df is not None:
                df = pd.read_csv(uploaded_df, sep=sep)
                df["Fecha"] = pd.to_datetime(df["Fecha"], format="%m/%d/%Y")
                data = df.dropna(subset=['2H','18O','Nombre','Latitud','Longitud'])
                data = df.dropna(subset=['2H'])
                st.session_state.df = df  # Asignar df a session_state
            else:
                st.session_state.df = df  # Asignar df a session_state si se ingresan datos manualmente
            if uploaded_df is not None:
                st.success("Archivo cargado correctamente")
            else:
                st.error("No has subido ningún archivo aún")


    elif df_selection == "SIAMS-UNAL":
        df = pd.read_csv("BDIsotopos.csv",sep=",") 
        st.session_state.df = df  # Asignar df a session_state
    elif df_selection == "MEGIA Isotopes":
        df = pd.read_csv("Insumos\MEGIA_ISOTOPOS_ESTABLES (version 1).csv",sep=",")
        st.session_state.df = df  # Asignar df a session_state

    # df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True,format='mixed')
    df["Latitud"] = pd.to_numeric(df["Latitud"], errors="coerce")
    df["Longitud"] = pd.to_numeric(df["Longitud"], errors="coerce")
    df = df.dropna(subset=['Latitud'])
    df = df.dropna(subset=['Longitud'])
    df["Fecha"] = pd.to_datetime(df["Fecha"], infer_datetime_format=True)
    data = df.dropna(subset=['2H','18O','Nombre','Latitud','Longitud'])
    data = df.dropna(subset=['2H'])

    st.divider()
        # Crear el mapa
    m = folium.Map(location=[df["Latitud"].mean(),df["Longitud"].mean()])

    # Agregar capas y controles
    marker_cluster = MarkerCluster(name="Nombre").add_to(m)
    folium.TileLayer("cartodbpositron").add_to(m)
    folium.LayerControl().add_to(m)

    # Calcular límites del mapa
    bounds = [[df["Latitud"].min(), df["Longitud"].min()],
            [df["Latitud"].max(), df["Longitud"].max()]]

    # Configurar los límites para ajustar el mapa a los puntos de monitoreo
    m.fit_bounds(bounds)

    # Agregar marcadores
    def plot_station(row):
        html = row[["Nombre","2H","18O","Tipo","Fecha"]].to_frame("Información").to_html(classes="table table-striped table-hover table-condensed table-responsive")
        popup = folium.Popup(html, max_width=1000)
        folium.Marker(location=[row.Latitud, row.Longitud], popup=popup).add_to(marker_cluster)

    df.apply(plot_station, axis=1)

    st.markdown("""
    ### Visualizemos la ubicación de los datos
    """)
    # Mostrar el mapa
    folium_static(m)

#----------------------------------------------------------------------------------------------
elif st.session_state.page == 3:
#----------------------------------------------------------------------------------------------
    if "df" in st.session_state:
        df = st.session_state.df
    # Continuar con el uso de df en esta página
    else:
        st.warning("Por favor, ingresa o carga los datos en la página 2.")
#----------------------------------------------------------------------------------------------
    st.markdown("""

    ## Actividad 1: Análisis de Datos Isotópicos 

    ### Construcción de Líneas Meteóricas

    Las líneas meteóricas son referencias fundamentales en el estudio de isótopos estables de oxígeno y hidrógeno en el agua. Estas líneas representan la relación isotópica entre δ¹⁸O y δ²H en el agua de lluvia y fueron identificadas a partir de observaciones globales y locales de precipitaciones. La Línea de las Aguas Meteoricas Global (GMWL) fue establecida por Craig en 1961 (Craig, 1961) y se expresa generalmente como:

    $$\delta^{2}H = 8 \cdot \delta^{18}O + 10$$

    Esta línea se construyó al observar que la mayoría de las precipitaciones alrededor del mundo siguen esta relación lineal debido a fraccionamientos isotópicos en los procesos de evaporación y condensación en la atmósfera. Además de la GMWL, en regiones específicas existen Líneas de Aguas Meteoricas Locales (LMWL), que reflejan variaciones particulares del clima, altitud y humedad en esas zonas. Estas líneas son clave para interpretar la procedencia y los procesos que afectan las aguas subterráneas y superficiales, ya que permiten diferenciar si un agua ha sufrido procesos como evaporación o mezcla con otras fuentes (Clark, 1997).


    El Organismo Internacional de Energía Atómica (IAEA) mediante su aplicativo WISER - Water Isotope System
    for Electronic Retrieval dispone de una base de datos a nivel mundial de la información isotópica del agua en el mundo (https://nucleus.iaea.org/wiser/explore/).

    La GNIP - Global Network of Isotopes in Precipitation,  es una red global de monitoreo de isótopos en la precipitación, establecida y coordinada por la Agencia Internacional de Energía Atómica (IAEA) y la Organización Meteorológica Mundial (OMM). Esta información es la base para  establecer Líneas de Agua Meteórica Local (LMWL), las cuales se usan como referencias para identificar la procedencia y procesos del agua subterránea.
    """)
    # Generar los datos para la línea meteórica global
    delta18O = np.linspace(-30, 10, num=50)
    delta2H = delta18O * 8.00 + 10

    # Crear la figura y graficar la línea meteórica global
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(delta18O, delta2H, label='GMWL: δ²H = 8·δ¹⁸O + 10', color='black')

    # Configurar etiquetas, título, leyenda y cuadrícula
    ax.set_xlabel('δ¹⁸O (‰)')
    ax.set_ylabel('δ²H (‰)')
    ax.set_title('Línea Meteórica Global')
    ax.legend()
    ax.grid(True)

    # Mostrar la gráfica en Streamlit
    st.pyplot(fig)
    st.divider()

    df['18O'] = pd.to_numeric(df['18O'], errors='coerce')
    df['2H'] = pd.to_numeric(df['2H'], errors='coerce')
    df = df.dropna(subset=['18O', '2H'])
    fig2, ax1 = plt.subplots(figsize=(6,6))
    # Graficar la GMWL
    ax1.plot(delta18O, delta2H, label='Línea Meteorológica Global (GMWL)', color='gray')
    ax1.scatter(df['18O'], df['2H'], label='Datos de Isótopos', color='r')
    ax1.set_xlabel('δ¹⁸O (‰)')
    ax1.set_ylabel('δ²H (‰)')
    ax1.set_title('Datos de Isótopos sobre la Línea Meteorológica Global')
    ax1.axhline(0, color='black', linewidth=0.5, linestyle='--')  # Línea horizontal en y=0
    ax1.axvline(0, color='black', linewidth=0.5, linestyle='--')  # Línea vertical en x=0
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig2)

    st.divider()

    #Graficar datos sobre la línea meteórica global diferenciando por estación
    unique_stations = df['Nombre'].unique()
    num_stations = len(unique_stations)
    colors = plt.cm.viridis(np.linspace(0, 1, num_stations))

    # Crear la gráfica
    fig3, ax2 = plt.subplots(figsize=(6, 6))

    # Graficar la GMWL
    ax2.plot(delta18O, delta2H, label='Línea Meteorológica Global (GMWL)', color='gray')

    # Graficar los datos
    for i, nombre in enumerate(unique_stations):
        subset = df[df['Nombre'] == nombre]
        plt.scatter(subset['18O'], subset['2H'], label=nombre, color=colors[i])


    # Etiquetas y título
    ax2.set_xlabel('δ¹⁸O vi(‰)')
    ax2.set_ylabel('δ²H (‰)')
    ax2.set_title('Datos de Isótopos sobre la Línea Meteorológica Global')
    ax2.axhline(0, color='black', linewidth=0.5, linestyle='--')  # Línea horizontal en y=0
    ax2.axvline(0, color='black', linewidth=0.5, linestyle='--')  # Línea vertical en x=0
    # plt.legend()
    ax2.grid(True)
    
    st.pyplot(fig3)


    st.divider()

    st.markdown("### Gráficos interactivos")
    Omax,Omin = [np.max(df["18O"]),np.min(df["18O"])]
    # Hmax,Hmin = [np.max(df["2H"]),np.min(df["2H"])]

    delta18O = np.linspace(Omin, Omax, num=50)
    Sadelta2H = delta18O * 8.02 + 12.12  # Saylor
    delta2H = delta18O * 8.03 + 9.66

    fig = px.scatter(df, y="2H", x="18O", color="Tipo",hover_name="Nombre")
    fig.update_traces(textposition='top center', textfont=dict(size=11))
    fig.update_layout(
    xaxis_title='δ¹⁸O (‰)',
    yaxis_title='δ²H (‰)',
    showlegend=True
    )
    fig.add_trace(go.Scatter(y=Sadelta2H, x=delta18O, mode='lines', name='CML - (Rodriguez, 2004)', line=dict(color='darkgray')))
    fig.add_trace(go.Scatter(y=delta2H, x=delta18O, mode='lines', name='CML - (Saylor et al., 2009)', line=dict(color='gray')))

    # Mostrar el gráfico isotópico
    st.plotly_chart(fig)

    st.divider()

    st.markdown("### Línea meteorica local")
    col1,col2,col3,col4 = st.columns([0.25,.25,.25,.25])
    with col1:
        Filtro = st.selectbox(
            "Seleccione la categoria para filtrar",
            options=["Tipo","Nombre","Fecha"],
            # default="Nombre"
            )
    with col2:
            # Crear un widget multiselect para que el usuario seleccione los puntos de interés
        selected_points1 = st.multiselect(
            "Seleccione puntos para el grupo 1",
            options=df[Filtro].unique(),
            format_func=lambda x: f"{x}",
        )

    with col3:
            # Crear un widget multiselect para que el usuario seleccione los puntos de interés
        selected_points2 = st.multiselect(
            "Seleccione puntos para el grupo 2",
            options=df[Filtro].unique(),
            format_func=lambda x: f"{x}",
        )
    with col4:
            # Crear un widget multiselect para que el usuario seleccione los puntos de interés
        selected_points3 = st.multiselect(
            "Seleccione puntos para el grupo 3",
            options=df[Filtro].unique(),
            format_func=lambda x: f"{x}",
        )# Filtrar el DataFrame según los puntos seleccionados

    if selected_points1:
        df_selected1 = df[df[Filtro].isin(selected_points1)].copy()
        df_str_selected1 = ",".join(df_selected1[Filtro].unique())
        df_selected1["Grupo"] = df_str_selected1
    else:
        df_selected1 = pd.DataFrame(columns=df.columns)

    if selected_points2:
        df_selected2 = df[df[Filtro].isin(selected_points2)].copy()
        df_str_selected2 = ",".join(df_selected2[Filtro].unique())
        df_selected2["Grupo"] = df_str_selected2
    else:
        df_selected2 = pd.DataFrame(columns=df.columns)

    if selected_points3:
        df_selected3 = df[df[Filtro].isin(selected_points3)].copy()
        df_str_selected3 = ",".join(df_selected3[Filtro].unique())
        df_selected3["Grupo"] = df_str_selected3
    else:
        df_selected3 = pd.DataFrame(columns=df.columns)

    df_selected = pd.concat([df_selected1,df_selected2,df_selected3])

    with col2:
        if selected_points1:
            slope1, intercept1, r_value, p_value, std_err = linregress(df_selected1["18O"], df_selected1["2H"])
            st.markdown(f"LMG= {np.round(slope1,2)} δ18O + {np.round(intercept1,2)}")
    with col3:
        if selected_points2:
            slope2, intercept2, r_value, p_value, std_err = linregress(df_selected2["18O"], df_selected2["2H"])
            st.markdown(f"LMG= {np.round(slope2,2)} δ18O + {np.round(intercept2,2)}")
    with col4:
        if selected_points3:
            slope3, intercept3, r_value, p_value, std_err = linregress(df_selected3["18O"], df_selected3["2H"])
            st.markdown(f"LMG= {np.round(slope3,2)} δ18O + {np.round(intercept3,2)}")

    if not any([selected_points1, selected_points2, selected_points3]):
        st.warning("Debe seleccionar al menos una opción en los desplegables para generar el gráfico.")
    else:
        nfig = px.scatter(df_selected, y="2H", x="18O", color="Grupo",hover_name="Nombre")
        nfig.update_traces(textposition='top center', textfont=dict(size=11))
        nfig.update_layout(
        xaxis_title='δ¹⁸O (‰)',
        yaxis_title='δ²H (‰)',
        showlegend=True
        )

        Omax,Omin = [np.max(df_selected["18O"]),np.min(df_selected["18O"])]
        delta18O = np.linspace(Omin, Omax, num=50)
        Sadelta2H = delta18O * 8.02 + 12.12  # Saylor
        delta2H = delta18O * 8.03 + 9.66
        if selected_points1:
            delta2H1 = delta18O * slope1 + intercept1
        if selected_points2:
            delta2H2 = delta18O * slope2 + intercept2
        if selected_points3:
            delta2H3 = delta18O * slope3 + intercept3
        nfig.add_trace(go.Scatter(y=Sadelta2H, x=delta18O, mode='lines', name='CML - (Rodriguez, 2004)', line=dict(color='darkgray')))
        fig.add_trace(go.Scatter(y=delta2H, x=delta18O, mode='lines', name='CML - (Saylor et al., 2009)', line=dict(color='gray')))

        if selected_points1:
            nfig.add_trace(go.Scatter(y=delta2H1, x=delta18O, mode='lines', name=f"LMG= {np.round(slope1,2)} δ18O + {np.round(intercept1,2)}", line=dict(color='#83c9ff',dash="dash")))
        if selected_points2:
            nfig.add_trace(go.Scatter(y=delta2H2, x=delta18O, mode='lines', name=f"LMG= {np.round(slope2,2)} δ18O + {np.round(intercept2,2)}", line=dict(color='#0068c9',dash="dash")))
        if selected_points3:
            nfig.add_trace(go.Scatter(y=delta2H3, x=delta18O, mode='lines', name=f"LMG= {np.round(slope3,2)} δ18O + {np.round(intercept3,2)}", line=dict(color='#ffabab',dash="dash")))
        # Mostrar el gráfico isotópico
        st.plotly_chart(nfig)

#----------------------------------------------------------------------------------------------
elif st.session_state.page == 4:
#----------------------------------------------------------------------------------------------
    if "df" in st.session_state:
        df = st.session_state.df
    # Continuar con el uso de df en esta página
    else:
        st.warning("Por favor, ingresa o carga los datos en la página 2.")


    if "df_selection" in st.session_state:
        df_selection  = st.session_state.df_selection
#----------------------------------------------------------------------------------------------
    st.markdown("""
    ### Exceso de deuterio (d-excess)
    Es un parámetro que proporciona información sobre las condiciones climáticas y atmosféricas durante la formación del agua de lluvia. Se calcula a partir de las relaciones isotópicas de hidrógeno y oxígeno en el agua, usando la siguiente fórmula:

    $$d = \delta^{2}H - 8 \cdot \delta^{18}O$$

    El exceso de deuterio refleja las desviaciones respecto a esta pendiente global (8), y es útil para interpretar condiciones de evaporación y humedad relativas en la zona de origen del vapor de agua.

    Condiciones Húmedas: Un valor de d-excess cercano a 10 indica condiciones de equilibrio isotópico en un ambiente de humedad relativa típica (~85%), como las que suelen encontrarse en áreas marítimas.

    Condiciones Secas: Valores más altos de d-excess (mayores a 10) suelen indicar que el vapor de agua se formó en condiciones de baja humedad relativa, típicas de climas áridos.

    Evaporación y Recarga: En el agua subterránea, valores anómalos de d-excess pueden sugerir procesos de evaporación previos a la infiltración.
    """)
    df = df.dropna(subset=['2H','18O','Nombre','Latitud','Longitud'])
    df = df.dropna(subset=['2H'])
    df['exceso_deuterio'] = df['2H'] - 8 * df['18O']
    st.divider()
    col1,col2 = st.columns(2)
    with col1:
         st.markdown("### Resultados")
         df[["Nombre","18O","2H","exceso_deuterio"]]
    with col2:
        st.markdown("#### Estadísticos descriptores")
        st.write(df[["Nombre","18O","2H","exceso_deuterio"]].describe())
        
    st.divider()
    st.markdown("### Agrupación de los resultados")
    col1, col2 = st.columns([1,4])
    with col1:
        Filtro = st.selectbox(
            "Seleccione la categoria para filtrar",
            options=["Tipo","Nombre","Fecha"],
            # default="Nombre"
            )
        df_selected1 = df[df[Filtro].isin(df)].copy()
        df_str_selected1 = ",".join(df_selected1[Filtro].unique())
        df_selected1["Grupo"] = df_str_selected1
    with col2:
        if df_selection == "IAEA Isotopes" and Filtro == "Nombre":
            Estaciones = st.multiselect("Selecciona las estaciones que quieres ver",
                                        options=df[Filtro].unique(),
                                        default=["MANAUS", "SAN GABRIEL", "BENJAMIN CONSTANT", "VILLAVICENCIO", "BOGOTA2"]
                                        )
        elif df_selection == "SIAMS-UNAL" and Filtro == "Tipo":
            Estaciones = st.multiselect("Selecciona las estaciones que quieres ver",
                                        options=df[Filtro].unique(),
                                        default=["Humedal ","Lluvia","Pozo","Piezometro"]
                                        )
        else:
            Estaciones = st.multiselect("Selecciona las estaciones que quieres ver",
                                        options=df[Filtro].unique(),
                                        )

        
    if Estaciones:
        ndf_selected = df[df[Filtro].isin(Estaciones)].copy()

        fig, ax = plt.subplots()
        sns.violinplot(x=Filtro,y="exceso_deuterio",data=ndf_selected,hue=Filtro)
        ax.tick_params(axis='x', rotation=90)
        ax.set_xlabel('')

        st.pyplot(fig)
    else:
        st.warning("Debe seleccionar al menos una opción en los desplegables para generar el gráfico.")
#----------------------------------------------------------------------------------------------
elif st.session_state.page == 5:
#----------------------------------------------------------------------------------------------
    if "df" in st.session_state:
        df = st.session_state.df
    # Continuar con el uso de df en esta página
    else:
        st.warning("Por favor, ingresa o carga los datos en la página 2.")


    if "df_selection" in st.session_state:
        df_selection  = st.session_state.df_selection
#----------------------------------------------------------------------------------------------
    st.markdown("""
    ## Pruebas de Bombeo
    El análisis de la cantidad isotópica de muestras de agua obtenidas durante un ensayo o prueba de bombeo, es una herramienta que permite identificar la posible época de recarga del agua que esta siendo extraida de los acuíferos, la comparación de estas relaciones isotópicas con los registros históricos ponderados para el agua meteórica establece la base conceptual para formular su epoca de recarga. 
    """)
    col1,col2 = st.columns(2)
    with col1:
        st.image("images/Posible Groundwater behaviors(2).jpg",use_column_width=True)
    with col2:
        st.image("images/Posible Groundwater behaviors.jpg",use_column_width=True)
    st.divider()
    if df_selection == "MEGIA Isotopes":
        st.markdown("### Resultados Bombeo Pozo MEGIA")
        # df = pd.read_csv("BDIsotopos.csv",sep=",")
        ndf = df.loc[df["Informacion"].notna()]
        ndf["Informacion"] = ndf["Informacion"].astype(float)
        ndf = ndf.rename(columns={"Informacion":"Horas de Bombeo"})
        ndf = ndf.dropna(subset=['18O', '2H'])
        st.write(ndf)
        pump_time = st.select_slider("Horas de bombeo",options=ndf["Horas de Bombeo"])
        pdf = ndf.loc[ndf["Horas de Bombeo"]<=pump_time]

        fig, ax = plt.subplots()

        # Límites para los datos de δ18O
        Omax, Omin = [np.max(ndf["18O"]), np.min(ndf["18O"])]
        delta18O = np.linspace(Omin, Omax, num=50)

        # Definición de las líneas basadas en las ecuaciones de Saylor y Rodriguez
        Sadelta2H = delta18O * 8.02 + 12.12  # Saylor
        delta2H = delta18O * 8.03 + 9.66  # Rodriguez

        # Trazar las líneas con etiquetas para la primera leyenda
        line1, = ax.plot(delta18O, Sadelta2H, label="CML - (Saylor et al., 2009)", color='black',linestyle='-',zorder=1)
        line2, = ax.plot(delta18O, delta2H, label="CML - (Rodriguez, 2004)", color='black',linestyle='--',zorder=1)

        # Primera leyenda solo con las líneas
        legend1 = ax.legend(handles=[line1, line2], loc='upper left')
        ax.add_artist(legend1)  # Añadir la primera leyenda al gráfico

        # Trazar los puntos de dispersión y asignar etiquetas para la segunda leyenda
        scat1 = sns.scatterplot(x="18O", y="2H", hue="Horas de Bombeo", data=pdf, ax=ax, s = 100,zorder=2)
        # scat2 = sns.scatterplot(x=[-10.17],y=[-70.2],ax=ax,s=150,zorder=2,color='blue')
        # ax.text(-9.1,-48.9,f"δ18O (Anual)",ha='left')
        # Colocar etiquetas de horas de bombeo encima de cada punto
        for i in range(len(pdf["18O"])):
            ax.text(pdf["18O"].iloc[i], pdf["2H"].iloc[i] + 0.5, f"{pdf['Horas de Bombeo'].iloc[i]:.0f}", ha='center')

        # Segunda leyenda solo para los puntos de dispersión
        handles, labels = scat1.get_legend_handles_labels()
        scatter_legend = ax.legend(handles=handles[2:], labels=labels[2:], title="Horas de Bombeo", loc="upper left", bbox_to_anchor=(1.05, 1))

        # Etiquetas y límites de los ejes
        ax.set_xlabel('δ¹⁸O (‰)')
        ax.set_ylabel('δ²H (‰)')
        ax.set_xlim(np.min(ndf["18O"]) - 0.05, np.max(ndf["18O"]) + 0.05)
        ax.set_ylim(np.min(ndf["2H"]) - 0.5, np.max(ndf["2H"]) + 2)
        ax.set_title("Contenido Isotópico vs Horas de Bombeo")

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)
    elif df_selection == "SIAMS-UNAL":
        st.markdown("### Resultados Bombeo Pozo")
        # df = pd.read_csv("BDIsotopos.csv",sep=",")
        ndf = df.loc[df["Informacion"]=="Bombeo2"][["Nombre","18O","2H","Informacion","Comentarios"]]
        ndf["Comentarios"] = ndf["Comentarios"].astype(float)
        ndf = ndf.rename(columns={"Comentarios":"Horas de Bombeo"})
        ndf = ndf.dropna(subset=['18O', '2H'])
        st.write(ndf)
        st.divider()
        pump_time = st.select_slider("Horas de bombeo",options=ndf["Horas de Bombeo"])
        pdf = ndf.loc[ndf["Horas de Bombeo"]<=pump_time]

        fig, ax = plt.subplots()

        # Límites para los datos de δ18O
        Omax, Omin = [np.max(ndf["18O"]), np.min(ndf["18O"])]
        delta18O = np.linspace(Omin, Omax, num=50)

        # Definición de las líneas basadas en las ecuaciones de Saylor y Rodriguez
        Sadelta2H = delta18O * 8.02 + 12.12  # Saylor
        delta2H = delta18O * 8.03 + 9.66  # Rodriguez

        # Trazar las líneas con etiquetas para la primera leyenda
        line1, = ax.plot(delta18O, Sadelta2H, label="CML - (Saylor et al., 2009)", color='black',linestyle='-',zorder=1)
        line2, = ax.plot(delta18O, delta2H, label="CML - (Rodriguez, 2004)", color='black',linestyle='--',zorder=1)

        # Primera leyenda solo con las líneas
        legend1 = ax.legend(handles=[line1, line2], loc='upper left')
        ax.add_artist(legend1)  # Añadir la primera leyenda al gráfico

        # Trazar los puntos de dispersión y asignar etiquetas para la segunda leyenda
        scat1 = sns.scatterplot(x="18O", y="2H", hue="Horas de Bombeo", data=pdf, ax=ax, s = 100,zorder=2)
        scat2 = sns.scatterplot(x=[-10.17],y=[-70.2],ax=ax,s=150,zorder=2,color='blue')
        ax.text(-9.1,-48.9,f"δ18O (Anual)",ha='left')
        # Colocar etiquetas de horas de bombeo encima de cada punto
        for i in range(len(pdf["18O"])):
            ax.text(pdf["18O"].iloc[i], pdf["2H"].iloc[i] + 0.5, f"{pdf['Horas de Bombeo'].iloc[i]:.0f}", ha='center')

        # Segunda leyenda solo para los puntos de dispersión
        handles, labels = scat1.get_legend_handles_labels()
        scatter_legend = ax.legend(handles=handles[2:], labels=labels[2:], title="Horas de Bombeo", loc="upper left", bbox_to_anchor=(1.05, 1))

        # Etiquetas y límites de los ejes
        ax.set_xlabel('δ¹⁸O (‰)')
        ax.set_ylabel('δ²H (‰)')
        ax.set_xlim(np.min(ndf["18O"]) - 2, np.max(ndf["18O"]) + 2)
        ax.set_ylim(np.min(ndf["2H"]) - 5, np.max(ndf["2H"]) + 5)
        ax.set_title("Contenido Isotópico vs Horas de Bombeo")

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)
        st.divider()
        st.image("images/FlujosBog.png")
#----------------------------------------------------------------------------------------------
elif st.session_state.page == 6:
    #----------------------------------------------------------------------------------------------
    if "df" in st.session_state:
        df = st.session_state.df
    # Continuar con el uso de df en esta página
    else:
        st.warning("Por favor, ingresa o carga los datos en la página 2.")


    if "df_selection" in st.session_state:
        df_selection  = st.session_state.df_selection
#----------------------------------------------------------------------------------------------
    st.markdown("""
    ## Altura de Recarga
    ---
    ### Caso de estudio
    Altura de muestras de infiltración dentro de tunel
                
     EL uso de isótopos para la identificación de posibles zonas de recarga puede estudiarse por medio de la medición del $\delta^{18}O$ relacionando la cantidad isotópica de aguas de infiltración o subterráneas con la elevación las muestras superficiales del sector. 
    """)
    st.image("images/MetElevacion.png")
    df = pd.read_excel("data.xlsx",sheet_name="data")
    df = df.dropna(subset=["18O","2H","Elevacion","Tipo"])
    st.divider()
    st.markdown("### Datos del caso de estudio")
    def highlight_cells(val):
        color = 'background-color: yellow' if val == 'Superficial' else ''
        return color
    col1,col2 = st.columns([0.7,0.3])
    with col1:
        df = df[["Nombre","18O","2H","Elevacion","Tipo"]]
        styled_df = df.style.applymap(highlight_cells, subset=['Tipo']) 
        st.dataframe(styled_df)
    with col2:
        slope1, intercept1, r_value, p_value, std_err = linregress(df.loc[df["Tipo"]=="Superficial"]["Elevacion"], df.loc[df["Tipo"]=="Superficial"]["18O"])
        st.write(f"La ecuación de la cantidad isotópica de $\delta$18O vs la altura de recarga en superficie:")
        st.write(f"$\delta$18O = {intercept1:.2f} + Elev*{slope1:0.4f}")
    Emin,Emax = [1600,1900]#np.min(df.loc[df["Tipo"]=="Superficial"]["Elevacion"]),np.max(df.loc[df["Tipo"]=="Superficial"]["Elevacion"])]
    deltaElev= np.linspace(Emin, Emax, num=50)
    O18 = intercept1 + deltaElev*slope1
    # IO18 = -0.00261*
    fig = px.scatter(df, x="Elevacion", y="18O", color="Tipo",hover_name="Nombre",text="Nombre")
    fig.update_traces(textposition='top center', textfont=dict(size=11))
    fig.add_trace(go.Scatter(y=O18,x=deltaElev,mode='lines',name="Linea meteorica de la recarga superficial"))
    fig.update_traces(textposition='top center', textfont=dict(size=11))
    fig.update_layout(
    yaxis_title='δ¹⁸O (‰)',
    xaxis_title='Elevación (msnm)',
    showlegend=True
    )   
    st.divider()
    st.plotly_chart(fig)
    st.divider()
    st.markdown("### Visualización del análisis")
    st.image("images/Elevacion.jpg")
#----------------------------------------------------------------------------------------------
# Añadir espacio extra
st.markdown("<br><br><br>", unsafe_allow_html=True)  # Ajusta el número de <br> según necesites
#----------------------------------------------------------------------------------------------
# Crear botones de navegación
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    # Usar el comando "on_click" para capturar de forma confiable los cambios de página
    st.button("Anterior", on_click=prev_page)
with col3:
    st.button("Siguiente️", on_click=next_page)
# #----------------------------------------------------------------------------------------------