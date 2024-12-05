import pandas as pd
import plotly.express as px
import streamlit as st


@st.cache_data
def load_data():
    df = pd.read_csv("detections.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


def main():
    st.set_page_config(layout="wide")

    st.title("Análisis de Seguimiento de Vehículos")
    df = load_data()

    # 1. Statistics
    with st.sidebar:
        st.subheader("Estadísticas Generales")
        st.metric("Total de Vehículos Únicos", len(df["Track ID"].unique()))

        # Group by minute and calculate averages
        df["Minute"] = df["Timestamp"].dt.floor("T")  # Round to minute
        vehicles_per_min = df.groupby("Minute")["Track ID"].nunique()

        # Calculate true per-minute metrics
        st.metric("Promedio de Vehículos por Minuto", round(vehicles_per_min.mean(), 2))

        # Get maximum vehicles in any minute
        st.metric("Máximo de Vehículos por Minuto", vehicles_per_min.max())

    col1, col2 = st.columns(2)

    # 2. Object Count Over Time
    with col1:
        st.subheader("Conteo de Vehículos en el Tiempo")
        df["Minute"] = df["Timestamp"].dt.floor("T")
        count_df = df.groupby("Minute").size().reset_index(name="count")

        fig_count = px.line(
            count_df,
            x="Minute",
            y="count",
            title="Número de Vehículos Detectados por Minuto",
        )
        fig_count.update_xaxes(title="Tiempo")
        fig_count.update_yaxes(title="Cantidad de Vehículos")
        st.plotly_chart(fig_count)

    # 3. Detection Heatmap
    with col2:
        st.subheader("Zonas de Detección de Vehículos")
        fig_heat = px.density_heatmap(
            df,
            x="X",
            y="Y",
            title="Mapa de Densidad de Detección de Vehículos",
            labels={
                "X": "Posición Izquierda a Derecha (píxeles)",
                "Y": "Posición Arriba a Abajo (píxeles)",
            },
            color_continuous_scale="Viridis",
        )

        fig_heat.update_layout(
            yaxis_autorange="reversed",
            coloraxis_colorbar_title="Frecuencia<br>de Detección",
        )

        st.plotly_chart(fig_heat)

    # 4. Track ID Movement
    st.subheader("Análisis de Trayectoria Individual de Vehículos")
    st.write(
        "Esta visualización muestra la ruta de cada vehículo detectado a través de la vista de la cámara"
    )
    st.write(
        "- Los cambios de color de rojo a azul muestran la dirección del movimiento"
    )
    st.write(
        "- Pase el cursor sobre los puntos para ver el tiempo y la posición exactos"
    )

    selected_track = st.selectbox(
        "Seleccione ID del Vehículo a Seguir", sorted(df["Track ID"].unique())
    )
    track_df = df[df["Track ID"] == selected_track]

    fig_track = px.scatter(
        track_df,
        x="X",
        y="Y",
        color="Frame",
        color_continuous_scale="Reds",
        text="Frame",
        hover_data={"Timestamp": True, "X": ":.1f", "Y": ":.1f", "Frame": True},
        title=f"Ruta del Vehículo #{selected_track}",
        labels={
            "X": "Posición Izquierda a Derecha (píxeles)",
            "Y": "Posición Arriba a Abajo (píxeles)",
            "Frame": "Progresión de Tiempo",
        },
    )

    fig_track.update_layout(
        yaxis_autorange="reversed", showlegend=True, hovermode="closest"
    )

    for i in range(len(track_df) - 1):
        fig_track.add_annotation(
            x=track_df.iloc[i]["X"],
            y=track_df.iloc[i]["Y"],
            ax=track_df.iloc[i + 1]["X"],
            ay=track_df.iloc[i + 1]["Y"],
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="#888",
        )

    st.plotly_chart(fig_track)

    st.write("### Estadísticas del Vehículo Seleccionado")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Tiempo en Cuadro", f"{len(track_df)} frames")
    with col2:
        distance = (
            (track_df["X"].diff() ** 2 + track_df["Y"].diff() ** 2) ** 0.5
        ).sum()
        st.metric("Distancia Total", f"{distance:.1f} píxeles")

    # 5. Size Distribution
    st.subheader("Distribución de Tamaños de Vehículos")
    df["Area"] = df["W"] * df["H"]
    fig_size = px.histogram(df, x="Area", title="Distribución de Tamaños de Vehículos")
    fig_size.update_xaxes(title="Área (píxeles²)")
    fig_size.update_yaxes(title="Frecuencia")
    st.plotly_chart(fig_size)


if __name__ == "__main__":
    main()
