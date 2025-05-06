
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuración de la página
st.set_page_config(page_title="IA para Reciclaje de Subproductos", layout="centered")

st.title("♻️ Asistente IA para Reciclaje Químico")
st.markdown("Este asistente utiliza un modelo de *Random Forest* para predecir si un subproducto químico puede ser reutilizado.")

# Datos simulados
data = {
    'pH': [2.1, 6.5, 3.3, 7.0, 1.5, 8.2, 4.8, 5.6, 9.0, 3.8],
    'metales_pesados_ppm': [80, 10, 50, 5, 100, 2, 30, 15, 1, 60],
    'organico': [1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
    'temperatura_C': [45, 60, 50, 70, 55, 40, 65, 58, 42, 62],
    'proceso_acido': [1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
    'reutilizable': [0, 1, 0, 1, 0, 1, 1, 1, 1, 0]
}
df = pd.DataFrame(data)

# Preparar modelo
X = df.drop('reutilizable', axis=1)
y = df['reutilizable']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)

# Formulario de entrada de usuario
st.subheader("🧪 Ingrese los datos del subproducto:")

pH = st.slider("pH", 0.0, 14.0, 7.0)
metales = st.slider("Metales pesados (ppm)", 0, 150, 20)
organico = st.selectbox("¿Contiene compuestos orgánicos?", ["Sí", "No"])
organico_val = 1 if organico == "Sí" else 0
temp = st.slider("Temperatura (°C)", 20, 100, 50)
proceso = st.selectbox("¿Proviene de un proceso ácido?", ["Sí", "No"])
proceso_val = 1 if proceso == "Sí" else 0

# Clasificación
if st.button("🔍 Evaluar reutilización"):
    entrada = pd.DataFrame([[pH, metales, organico_val, temp, proceso_val]],
                           columns=['pH', 'metales_pesados_ppm', 'organico', 'temperatura_C', 'proceso_acido'])
    entrada_scaled = scaler.transform(entrada)
    pred = modelo_rf.predict(entrada_scaled)[0]
    proba = modelo_rf.predict_proba(entrada_scaled)[0][pred]

    if pred == 1:
        st.success(f"✅ El subproducto puede ser reutilizado (confianza: {proba*100:.2f}%)")
    else:
        st.error(f"❌ El subproducto no es adecuado para reutilización (confianza: {proba*100:.2f}%)")

    # Mostrar importancia de variables
    st.subheader("📊 Variables más influyentes en la decisión del modelo:")
    importancias = modelo_rf.feature_importances_
    nombres = X.columns
    importancia_df = pd.DataFrame({'Variable': nombres, 'Importancia': importancias})
    fig, ax = plt.subplots()
    sns.barplot(x="Importancia", y="Variable", data=importancia_df.sort_values("Importancia", ascending=False), ax=ax)
    st.pyplot(fig)
