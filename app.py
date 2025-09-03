import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# =====================
# 1. Cargar Excel con todas las hojas
# =====================
excel_path = "WDIEXCEL.xlsx"

data = pd.read_excel(excel_path, sheet_name="Data")
series = pd.read_excel(excel_path, sheet_name="Series")
country = pd.read_excel(excel_path, sheet_name="Country")

# =====================
# 2. Seleccionar indicadores relevantes (económicos y sociales)
# =====================
# Palabras clave de indicadores numéricos importantes
keywords = ["GDP", "Inflation", "Population", "Employment", "Unemployment",
            "Poverty", "Life expectancy", "CO2", "School enrollment"]

relevant_codes = series[
    series["Indicator Name"].str.contains("|".join(keywords), case=False, na=False)
]["Series Code"].tolist()

# Filtrar datos de esos indicadores
data_filtered = data[data["Indicator Code"].isin(relevant_codes)]

# =====================
# 3. Transformar a formato largo (Country, Year, Value)
# =====================
data_long = data_filtered.melt(
    id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
    var_name="Year",
    value_name="Value"
)

# Limpiar datos
data_long = data_long.dropna(subset=["Value"])
data_long = data_long[data_long["Year"].str.isnumeric()]
data_long["Year"] = data_long["Year"].astype(int)

# =====================
# 4. Unir con metadatos
# =====================
data_long = data_long.merge(country, on="Country Code", how="left")
data_long = data_long.merge(series[["Series Code", "Topic"]],
                            left_on="Indicator Code", right_on="Series Code", how="left")

# =====================
# 5. Interfaz Streamlit
# =====================
st.title("🌍 Predicción de Indicadores Económicos y Sociales (WDI)")

# Selector de indicador
selected_indicator = st.selectbox(
    "Selecciona un indicador",
    data_long["Indicator Name"].unique()
)

# Selector de país
selected_country = st.selectbox(
    "Selecciona un país",
    data_long["Country Name"].unique()
)

# Filtrar dataset según selección
subset = data_long[
    (data_long["Country Name"] == selected_country) &
    (data_long["Indicator Name"] == selected_indicator)
]

if subset.empty:
    st.warning("No hay datos disponibles para esta combinación.")
else:
    st.line_chart(subset.set_index("Year")["Value"])

    # =====================
    # 6. Preparar dataset predictivo
    # =====================
    data_model = subset[["Country Name", "Country Code", "Region", "Income Group", "Year", "Value"]].copy()

    # Codificar categóricas
    data_model = pd.get_dummies(data_model, columns=["Region", "Income Group"], drop_first=True)

    # Entrenar modelo
    X = data_model.drop(columns=["Value", "Country Name", "Country Code"])
    y = data_model["Value"]

    if len(X) > 5:  # aseguramos datos suficientes
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)

        st.metric("Error RMSE del modelo", f"{rmse:,.2f}")

        # =====================
        # 7. Predicción futura
        # =====================
        st.subheader("Predicción futura")

        future_years = [2030, 2035, 2040]
        last_row = data_model.iloc[-1:].copy()

        if not last_row.empty:
            for fy in future_years:
                row = last_row.copy()
                row["Year"] = fy

                # Asegurar columnas correctas
                row = row.drop(columns=["Value", "Country Name", "Country Code"], errors="ignore")
                row = row.reindex(columns=X.columns, fill_value=0)

                pred_future = model.predict(row)[0]
                st.write(f"{selected_indicator} estimado en {fy}: {pred_future:,.2f}")
    else:
        st.warning("No hay suficientes datos para entrenar el modelo.")
