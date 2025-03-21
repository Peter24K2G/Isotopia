import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import matplotlib.dates as mdates

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
    st.markdown(""" # Datación con Tririo 3H
    ## Datación con Tritio en Aguas Subterráneas

    El **tritio (³H)** es un isótopo radiactivo del hidrógeno con una vida media de **12.32 años**, útil para fechar aguas subterráneas recientes (**<60 años**). Se origina en la atmósfera y entra al ciclo hidrológico a través de la precipitación.

    ### Aplicaciones
    - Determinar la edad de aguas modernas.
    - Identificar mezclas entre aguas jóvenes y antiguas.
    - Validar modelos de flujo subterráneo.

    ### Método de Datación
    El tritio se mide en **Unidades de Tritio (TU)** y su decaimiento sigue:

    
    $C_t = C_0 e^{-\lambda t}$
    

    donde **$C_t$** es la concentración actual y **$\lambda$** la constante de decaimiento.
    """)
# ----------------------------------------------------------------------------------------------
if st.session_state.page == 2:
#----------------------------------------------------------------------------------------------
    st.markdown("""
    ## Mediciones de Tritio en Colombia
    ### LA Intertional Atomic Energy Agency (IAEA)
    La Agencia Internacional de Energía Atómica (IAEA, por sus siglas en inglés) realiza un monitoreo mensual de tritio (³H) en la atmósfera como parte de sus esfuerzos para comprender los procesos del ciclo hidrológico y evaluar el impacto de fuentes naturales y antropogénicas de radionúclidos. Estas mediciones permiten rastrear la variabilidad temporal y espacial del tritio, un isótopo radiactivo del hidrógeno, y son fundamentales para estudios de datación de aguas subterráneas, modelación de flujos hídricos y seguimiento de emisiones nucleares a nivel global.
    """)
    df = pd.read_excel("file-937497206062955.xlsx")

    df = df.loc[df["Measurand Symbol"]=="H3"][["Sample Date","Sample Site Name","Measurand Amount"]]
    # df.loc[df["Sample Site Name"]=="BOGOTA"]
    df["Sample Date"] = pd.to_datetime(df["Sample Date"],utc=True)
    df["Data"] = np.where(df["Measurand Amount"].notna(), 1, np.nan)
    df.rename(columns={"Sample Site Name":"Estación"},inplace=True)

    # fig.ax = plt.subplots(figsize = (12,4))
    g = sns.FacetGrid(data=df,hue="Estación",col="Estación",col_wrap=1,height=1,aspect=4)
    g.map(plt.plot,"Sample Date","Data")
    for ax in g.axes.flat:
        ax.set_yticks([])  # Quita las etiquetas del eje Y
        ax.spines['left'].set_visible(False)  # Eliminar la línea del eje Y

    g.set_axis_labels("Año", "")
    st.pyplot(g)


    df = pd.read_excel("file-937497206062955.xlsx")
    df = df.loc[df["Measurand Symbol"]=="H3"][["Sample Date","Sample Site Name","Measurand Amount"]]
    df["Sample Date"] = pd.to_datetime(df["Sample Date"],utc=True)
    df['Año'] = df['Sample Date'].dt.year
    df.dropna(axis=1,how='all',inplace=True)
    dfs = df.groupby(["Sample Site Name","Año"])["Measurand Amount"].mean().reset_index()
    ndfs = dfs.pivot(index="Año",columns="Sample Site Name",values="Measurand Amount")
    st.write(ndfs)

    st.divider()


    Bog = ndfs["BOGOTA"].values
    Barr= ndfs["BARRANQUILLA"].values
    # 1.61*np.exp(-Lambda*tavg)
    PFMBog = pd.DataFrame({"Año":ndfs["BOGOTA"].index,"H3avg":Bog})#,"PFM":Bog*np.exp(-Lambda*tavg)})
    PFMBog.set_index("Año",inplace=True)

    PFMBarr = pd.DataFrame({"Año":ndfs["BARRANQUILLA"].index,"H3avg":Barr})#,"PFM":Barr*np.exp(-Lambda*tavg)})
    PFMBarr.set_index("Año",inplace=True)

    # PFMTul = pd.DataFrame({"Año":ndfs["TULENAPA"].index,"H3avg":Tul,"PFM":Tul*np.exp(-Lambda*tavg)})
    # PFMTul.set_index("Año",inplace=True)

    fig, ax = plt.subplots()
    # H3 = [2.6,1.8,1.4,1.2,0.8]
    # i=0
    # for h3 in H3:
    #     y_line = h3
    #     plt.axhline(y=y_line, color='gray', linestyle='--', linewidth=0.5)

        # # Add annotation above the line
        # plt.annotate(
        #     f'{y_line}',  # Text to display
        #     xy=(1974+i, y_line),  # Point of annotation (x, y)
        #     xytext=(1974+i, y_line),  # Position of the text
        #     textcoords='data',  # Coordinate system
        #     ha='center',  # Horizontal alignment
        #     color='gray',
        #     fontsize=5,
        #     # bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')  # Background box
        # )
        # i += 2
    # [ax.axhline(y=x,color="gray",linewidth=0.5,linestyle='--') for x in [1.8,2.6,1.4,1.2,0.8]]

    PFMBog["H3avg"].plot(ax=ax, label="$^3H$ Promedio Bogotá", color="blue")
    # PFMBog["PFM"].plot(ax=ax, label="MFP Bogotá", color="red")
    PFMBarr["H3avg"].plot(ax=ax, label="$^3H$ Promedio Barranquilla", color="red")
    # PFMBarr["PFM"].plot(ax=ax, label="MFP Barranquilla", color="red",linestyle='--')
    # PFMTul["H3avg"].plot(ax=ax, label="H3 Promedio Tulenapa", color="blue",linestyle=':')
    # PFMTul["PFM"].plot(ax=ax, label="MFP Tulenapa", color="red",linestyle=':')

    # Add legend
    ax.legend()

    # Add labels and title
    ax.set_title("H3 Promedio Bogotá y Barranquilla")
    ax.set_xlabel("Año")
    ax.set_ylabel("$^3H$ [U.T.]")
    ax.grid()
    st.pyplot(fig)
# ----------------------------------------------------------------------------------------------
if st.session_state.page == 3:
#----------------------------------------------------------------------------------------------
    st.markdown("""
    ## Proyección de las mediciones a 2024
    Considerando una muestra tomada en 2024, se realiza una proyección de los datos a este año para estimar los valores de Tritio, para este caso se usará la serie de datos más completa, la de Bogotá. 
    """)
    # Función exponencial para ajuste
    def exp_model(x, a, b, c):
        return a * np.exp(b * (x - 1970)) + c  # Normalización al año base para estabilidad numérica

    # Cargar los datos (Asegúrate de reemplazar 'file_path' con tu archivo real si es necesario)
    df = pd.read_csv("2025-03-21T10-55_export.csv")
    df_bogota = df[['Año', 'BOGOTA']].dropna()

    # Extraer años y valores de Bogotá
    years_bogota = df_bogota['Año'].values.reshape(-1, 1)
    values_bogota = df_bogota['BOGOTA'].values

    # Ajustar el modelo exponencial
    popt, _ = curve_fit(exp_model, years_bogota.flatten(), values_bogota, p0=(1, -0.1, 1))

    # Generar predicción con modelo exponencial
    future_years = np.arange(df_bogota['Año'].min(), 2025).reshape(-1, 1)
    bogota_exp_pred = exp_model(future_years.flatten(), *popt)

    # Graficar los datos históricos y la proyección exponencial
    fig = plt.figure(figsize=(10, 5))
    plt.plot(df_bogota['Año'], df_bogota['BOGOTA'], 'bo', label="Bogotá Datos Reales")
    plt.plot(future_years, bogota_exp_pred, 'g--', label="Proyección Exponencial")

    plt.xlabel("Año")
    plt.ylabel(r"$^3H$ [U.T.]")
    plt.title("Proyección Exponencial de Tritio ($^3H$) en Bogotá hasta 2024")
    plt.legend()
    plt.grid()
    st.write(fig)
    
    st.divider()
    st.markdown("""
    ## Modelo de flujo a pistón para el decaimiento del tritio
    El modelo de flujo a pistón (MFP) se utiliza para estimar la evolución del tritio $^3H$ en el tiempo, considerando su decaimiento radiactivo y el tiempo promedio de tránsito del agua en el sistema. Para ello, se calcula el coeficiente de decaimiento $lambda$ basado en la vida media del tritio (12.43 años) y se determina el tiempo promedio de tránsito a partir de la porosidad efectiva $n$, el espesor del acuífero $d$ y la recarga específica $Q_{rec}$. La nueva curva de decaimiento se obtiene aplicando la ecuación exponencial de atenuación,$MFP = ^3H_{avg} \cdot e^{-\lambda t_{avg}}$, lo que permite modelar la concentración de tritio en función del tiempo y evaluar la influencia del tránsito del agua en la atenuación isotópica.

    """)
    # Reemplazar valores faltantes (2021-2024) con la proyección
    df_future = pd.DataFrame({'Año': future_years.flatten(), 'BOGOTA': bogota_exp_pred})
    df_bogota = pd.concat([df_bogota, df_future[df_future['Año'] >= 2021]], ignore_index=True)

    df_bogota.set_index("Año",inplace=True)
    #Coeficiente de decaimiento Lambda
    Lambda = np.log(2)/12.43

    col1, col2, col3 = st.columns(3)
    with col1:
        Qrec = st.number_input("Recarga (m/año)", min_value=0.01, max_value=1.0, value=0.155, step=0.01)

    with col2:
        n = st.number_input("Porosidad efectiva", min_value=0.01, max_value=0.5, value=0.02, step=0.01)

    with col3:
        d = st.number_input("Espesor (m)", min_value=1, max_value=500, value=150, step=5)
    # n = 0.02 # Porosidad
    # d = 150 # Espesor (Metros)
    # Qrec = 0.155 # Recarga (m/año)
    tavg = n*d/Qrec #Tiempo promedio 

    Bog = df_bogota["BOGOTA"].values
    PFMBog = pd.DataFrame({"Año":df_bogota["BOGOTA"].index,"H3avg":Bog,"PFM":Bog*np.exp(-Lambda*tavg)})
    PFMBog.set_index("Año",inplace=True)

    fig, ax = plt.subplots()
    PFMBog["H3avg"].plot(ax=ax, label="$^3H$ Promedio Bogotá", color="blue")
    PFMBog["PFM"].plot(ax=ax, label="MFP Bogotá", color="red")
    # Add legend
    ax.legend()

    # Add labels and title
    # ax.set_title("H3 Promedio and MFP")
    ax.set_xlabel("Año")
    ax.set_ylabel("$^3H$ [U.T.]")
    ax.grid()
    st.pyplot(fig)

    st.divider()
    # Asegurar que el índice es datetime o numérico
    if not isinstance(PFMBog.index, pd.DatetimeIndex):
        try:
            PFMBog.index = pd.to_datetime(PFMBog.index, format="%Y")  # Si son años
        except:
            PFMBog.index = pd.to_numeric(PFMBog.index)  # Si son solo números

    # Input del usuario
    input_tritio = st.number_input("Ingrese un valor de tritio ($^3H$) [U.T.]", min_value=0.0, step=0.1,value=2.15)

    if not PFMBog.empty and input_tritio is not None:
        pfm_values = PFMBog["PFM"].values
        idx_cercano = np.abs(pfm_values - input_tritio).argmin()
        valor_cercano = pfm_values[idx_cercano]
        fecha_cercana = PFMBog.index[idx_cercano]

        fig, ax = plt.subplots()
        PFMBog["H3avg"].plot(ax=ax, label="$^3H$ Promedio Bogotá", color="blue")
        PFMBog["PFM"].plot(ax=ax, label="MFP Bogotá", color="red")

        ax.axhline(y=valor_cercano, color='gray', linestyle='--', label=f"$^3H$ más cercano: {valor_cercano:.2f}")
        ax.axvline(x=fecha_cercana, color='green', linestyle='--', label=f"Año: {fecha_cercana.year if hasattr(fecha_cercana, 'year') else fecha_cercana}")

        # Formatear el eje X
        if isinstance(PFMBog.index, pd.DatetimeIndex):
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        else:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.xticks(rotation=45)
        ax.set_xlabel("Año")
        ax.set_ylabel("$^3H$ [U.T.]")
        ax.legend()
        ax.grid()

        st.pyplot(fig)
        st.write(f"El valor de $^3H$ ingresado ({input_tritio:0.2f}) se aproxima al año **{fecha_cercana.year if hasattr(fecha_cercana, 'year') else fecha_cercana}** en la curva del Modelo de Flujo a Pistón.")
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