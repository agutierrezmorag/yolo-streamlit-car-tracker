import os

import pandas as pd
import plotly.express as px
import streamlit as st


@st.cache_data
def load_data(model_name):
    """Load data for specific YOLO model"""
    csv_path = os.path.join(f"output_{model_name}", "data", "detections.csv")
    df = pd.read_csv(csv_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


# Get available YOLO model outputs
def get_available_models():
    """Get list of available model outputs"""
    return [
        d.replace("output_", "") for d in os.listdir() if d.startswith("output_yolo")
    ]


def main():
    st.set_page_config(layout="wide")
    st.title("Análisis de Seguimiento de Vehículos")

    # Model selection in sidebar
    with st.sidebar:
        available_models = get_available_models()
        selected_model = st.selectbox(
            "Seleccionar Modelo YOLO", available_models, format_func=lambda x: x.upper()
        )
        st.write("---")

    # Load data for selected model
    try:
        df = load_data(selected_model)
    except FileNotFoundError:
        st.error("No se encontraron datos para el modelo seleccionado")
        return

    # Statistics in sidebar
    with st.sidebar:
        st.subheader("Estadísticas Generales")
        st.metric("Total de Vehículos Únicos", len(df["Track ID"].unique()))

        # Group by minute and calculate averages
        df["Minute"] = df["Timestamp"].dt.floor("min")
        vehicles_per_min = df.groupby("Minute")["Track ID"].nunique()

        # Calculate true per-minute metrics
        st.metric("Promedio de Vehículos por Minuto", round(vehicles_per_min.mean(), 2))

        # Get maximum vehicles in any minute
        st.metric("Máximo de Vehículos por Minuto", vehicles_per_min.max())

    col1, col2 = st.columns(2)

    # 2. Object Count Over Time
    with col1:
        st.subheader("Conteo de Vehículos en el Tiempo")
        df["Minute"] = df["Timestamp"].dt.floor("min")
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
    col1, col2 = st.columns(2)
    with col1:
        selected_track = st.selectbox(
            "Seleccione ID del Vehículo a Seguir", sorted(df["Track ID"].unique())
        )
        track_df = df[df["Track ID"] == selected_track]

        # Plot trajectory
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

        # Add trajectory arrows
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

    with col2:
        # Display detected car image
        st.write("### Imagen del Vehículo")
        # Update path to use model-specific directory
        image_path = os.path.join(
            f"output_{selected_model}", "detected_cars", f"car_id{selected_track}.jpg"
        )
        try:
            st.image(
                image_path,
                caption=f"Vehículo ID #{selected_track} - Modelo {selected_model.upper()}",
            )
        except FileNotFoundError:
            st.warning("Imagen no disponible para este vehículo")

        # Show additional details if available
        st.write("### Detalles Adicionales")
        st.write("Dimensiones promedio:")
        avg_w = track_df["W"].mean()
        avg_h = track_df["H"].mean()
        area = avg_w * avg_h

        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.write(f"- Ancho: {avg_w:.1f} píxeles")
            st.write(f"- Alto: {avg_h:.1f} píxeles")
        with metrics_col2:
            st.write(f"- Área: {area:.1f} píxeles²")
            st.write(f"- Ratio: {(avg_w/avg_h):.2f}")

        # Display car metrics
        st.write("### Estadísticas del Vehículo")
        subcol1, subcol2, subcol3 = st.columns(3)
        with subcol1:
            st.metric("Frames", f"{len(track_df)}")
        with subcol2:
            time_visible = len(track_df) / 30  # Assuming 30 fps
            st.metric("Tiempo Visible", f"{time_visible:.1f}s")
        with subcol3:
            distance = (
                (track_df["X"].diff() ** 2 + track_df["Y"].diff() ** 2) ** 0.5
            ).sum()
            st.metric("Distancia", f"{distance:.1f}px")

    # 5. Size Distribution
    st.subheader("Distribución de Tamaños de Vehículos")
    df["Area"] = df["W"] * df["H"]
    fig_size = px.histogram(df, x="Area", title="Distribución de Tamaños de Vehículos")
    fig_size.update_xaxes(title="Área (píxeles²)")
    fig_size.update_yaxes(title="Frecuencia")
    st.plotly_chart(fig_size)


if __name__ == "__main__":
    main()
