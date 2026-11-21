import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('share-of-cumulative-co2.csv')
df_china = df[(df['entity'] == 'China') & (df['year'] >= 1950)].dropna(subset=['cumulative_emissions_total_as_share_of_global'])

X = df_china[['year']]
y = df_china['cumulative_emissions_total_as_share_of_global']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

print("Filas de China (1950+):", len(df_china))
print("Pendiente:", modelo.coef_[0])
print("R2:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

plt.figure(figsize=(8, 4))
plt.scatter(df_china['year'], y, color='gray', alpha=0.6, label='Datos reales')
plt.plot(df_china['year'], modelo.predict(X), color='red', label='Regresion lineal')
plt.title('China: Participacion en emisiones acumuladas de CO2')
plt.xlabel('Ano')
plt.ylabel('Participacion global')
plt.legend()
plt.savefig('grafica_china.png')
print("Grafica guardada como grafica_china.png")
