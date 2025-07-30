# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from google.colab import files


uploaded = files.upload()
filename = list(uploaded.keys())[0]


df = pd.read_excel(filename)
df.columns = ['Fecha', 'Pronosticado', 'Observado']  # Asegura nombres estándar
df['Pronosticado'] = pd.to_numeric(df['Pronosticado'], errors='coerce')
df['Observado'] = pd.to_numeric(df['Observado'], errors='coerce')
df.dropna(inplace=True)


scaler = MinMaxScaler()
datos = df['Pronosticado'].values.reshape(-1, 1)
datos_esc = scaler.fit_transform(datos)


def crear_secuencias(data, pasos=10):
    X, y = [], []
    for i in range(pasos, len(data)):
        X.append(data[i - pasos:i])
        y.append(data[i])
    return np.array(X), np.array(y)

pasos = 10
X, y = crear_secuencias(datos_esc, pasos)


split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


model = Sequential([
    GRU(64, return_sequences=True, input_shape=(X.shape[1], 1)),
    GRU(32),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=1)


pred = model.predict(X_test)
pred_inv = scaler.inverse_transform(pred)
y_test_inv = scaler.inverse_transform(y_test)


observado = df['Observado'].values.reshape(-1, 1)[-len(pred_inv):]
forecasting = df['Pronosticado'].values.reshape(-1, 1)[-len(pred_inv):]
fechas = df['Fecha'].values[-len(pred_inv):]


print("== Forecasting vs Observado ==")
print(f"MSE: {mean_squared_error(observado, forecasting):.4f}")
print(f"MAE: {mean_absolute_error(observado, forecasting):.4f}")
print(f"R²: {r2_score(observado, forecasting):.4f}")

print("\n== GRU vs Observado ==")
print(f"MSE: {mean_squared_error(observado, pred_inv):.4f}")
print(f"MAE: {mean_absolute_error(observado, pred_inv):.4f}")
print(f"R²: {r2_score(observado, pred_inv):.4f}")


comparacion_df = pd.DataFrame({
    'Fecha': fechas,
    'Observado': observado.flatten(),
    'Forecasting (original)': forecasting.flatten(),
    'Predicción GRU': pred_inv.flatten(),
    'Error Forecast': np.abs(observado.flatten() - forecasting.flatten()),
    'Error GRU': np.abs(observado.flatten() - pred_inv.flatten())
})


plt.figure(figsize=(14, 6))
plt.plot(comparacion_df['Fecha'], comparacion_df['Observado'], label='Observado', marker='o')
plt.plot(comparacion_df['Fecha'], comparacion_df['Forecasting (original)'], label='Forecasting', marker='x')
plt.plot(comparacion_df['Fecha'], comparacion_df['Predicción GRU'], label='GRU', marker='s')
plt.xlabel("Fecha")
plt.ylabel("Humedad del Suelo (kPa)")
plt.title("Comparación: Observado vs Forecasting vs GRU")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("comparacion_3_series.png")
plt.show()


comparacion_df.to_excel("comparacion_3_columnas.xlsx", index=False)
files.download("comparacion_3_columnas.xlsx")


model.save("modelo_gru_entrenado.h5")
files.download("modelo_gru_entrenado.h5")

