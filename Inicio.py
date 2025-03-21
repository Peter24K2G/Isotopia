import streamlit as st
from PIL import Image

# Set the title of the app
st.title("Isótopos en hidrogeología")

logo = Image.open("images/PageImage.png")  # Reemplaza con la ruta de tu logo

# Mostrar el logo en la barra lateral
st.sidebar.image(logo, use_container_width=True)


with st.sidebar:
    st.markdown("""
        # Isotopía en Hidrogeología
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
st.markdown("""
## Introducción a la Isotopía en Hidrogeología

El uso de isótopos en hidrogeología permite rastrear el origen, la edad y los procesos que afectan al agua subterránea. Los isótopos pueden ser estables, como el **oxígeno-18 (¹⁸O) y el deuterio (²H)**, o radiactivos, como el **tritio (³H) y el carbono-14 (¹⁴C)**, cada uno con aplicaciones específicas en la evaluación de recursos hídricos.

Estos trazadores isotópicos proporcionan información clave sobre procesos como la **recarga acuífera, la evaporación, la mezcla de masas de agua y la contaminación**. La relación entre la composición isotópica y factores ambientales como la altitud, la latitud y el clima permite interpretar la evolución y la dinámica de los sistemas hídricos.

La hidrogeología isotópica es una herramienta fundamental en la gestión del agua, contribuyendo a la identificación de fuentes de abastecimiento, la detección de sobreexplotación y la evaluación de la contaminación tanto natural como antrópica.

""")
