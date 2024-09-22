#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Cargar el dataset
data = pd.read_csv('AppleStock2024.csv')

# Convertir la columna 'Date' a formato datetime y establecerla como índice
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Filtrar datos desde 2019
data = data[data.index >= '2019-01-01']

# Seleccionar la columna de 'Price' para la predicción
adj_close_prices = data['Price']

# Crear el modelo ARIMA
model = ARIMA(adj_close_prices, order=(5, 1, 0))  # Elige (p,d,q) basándote en la serie temporal
model_fit = model.fit()

# Predecir los próximos 30 días
forecast = model_fit.forecast(steps=30)

# Crear una nueva serie temporal para la predicción
forecast_dates = pd.date_range(start=adj_close_prices.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
forecast_series = pd.Series(forecast, index=forecast_dates)

# Graficar el precio y la predicción
plt.figure(figsize=(14, 7))
plt.plot(adj_close_prices, label='Historical Price', color='#5A9BD4', linewidth=2)
plt.plot(forecast_series, label='Predicted Price', color='#007FFF', linestyle='--', linewidth=2)
plt.title('Apple Stock Price Prediction', fontsize=16, fontweight='bold', color='#003366')
plt.xlabel('Date', fontsize=14, color='#003366')
plt.ylabel('Price (USD)', fontsize=14, color='#003366')
plt.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='black', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5, color='#D3D3D3')
plt.show()


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt

# Cargar el dataset
data = pd.read_csv('AppleStock2024.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Filtrar datos desde 2019
data = data[data.index >= '2019-01-01']

# Gráfica de la serie temporal
plt.figure(figsize=(14, 7))
plt.plot(data['Price'], label='Historical Price', color='#5A9BD4', linewidth=2)
plt.title('Historical Apple Stock Prices', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price (USD)', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


# In[4]:


import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(data['Price'], bins=30, kde=True, color='#5A9BD4')
plt.title('Distribution of Apple Stock Prices', fontsize=16, fontweight='bold')
plt.xlabel('Price (USD)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


# In[5]:


# Filtrar datos para el año 2024
data_2024 = data['2024']

from sklearn.linear_model import LinearRegression
import numpy as np

# Preparar los datos
data_2024['Date_Ordinal'] = data_2024.index.map(pd.Timestamp.toordinal)
X = data_2024['Date_Ordinal'].values.reshape(-1, 1)
y = data_2024['Price'].values

# Ajustar el modelo de regresión
model = LinearRegression()
model.fit(X, y)

# Predecir valores
predictions = model.predict(X)

# Gráfica de regresión
plt.figure(figsize=(14, 7))
plt.scatter(data_2024.index, y, color='blue', label='Actual Prices')
plt.plot(data_2024.index, predictions, color='red', linewidth=2, label='Regression Line')
plt.title('Regression of Apple Stock Prices for 2024', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price (USD)', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Cargar el dataset
data = pd.read_csv('AppleStock2024.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Filtrar datos para los últimos 2 años
data_last_2_years = data[data.index >= '2022-01-01']

# Preparar los datos
data_last_2_years['Date_Ordinal'] = data_last_2_years.index.map(pd.Timestamp.toordinal)
X = data_last_2_years['Date_Ordinal'].values.reshape(-1, 1)
y = data_last_2_years['Price'].values

# Ajustar un modelo de regresión polinómica (de grado 3)
poly_features = PolynomialFeatures(degree=3)
X_poly = poly_features.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

# Predecir valores
predictions = model.predict(X_poly)

# Gráfica de regresión polinómica
plt.figure(figsize=(14, 7))
plt.scatter(data_last_2_years.index, y, color='blue', label='Actual Prices', alpha=0.5)
plt.plot(data_last_2_years.index, predictions, color='red', linewidth=2, label='Polynomial Regression Line', alpha=0.8)
plt.title('Polynomial Regression of Apple Stock Prices (Last 2 Years)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price (USD)', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar el dataset
data = pd.read_csv('AppleStock2024.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Filtrar datos desde 2023
data_2023 = data[data.index >= '2023-01-01']

# Preparar los datos
data_2023['Date_Ordinal'] = data_2023.index.map(pd.Timestamp.toordinal)
X = data_2023['Date_Ordinal'].values.reshape(-1, 1)
y = data_2023['Price'].values

# Probar diferentes grados de polinomio
degrees = [1, 2, 3, 4, 5]
best_degree = 0
lowest_mse = float('inf')
best_predictions = None

plt.figure(figsize=(14, 7))

for degree in degrees:
    # Ajustar un modelo de regresión polinómica
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Predecir valores
    predictions = model.predict(X_poly)
    
    # Calcular el error cuadrático medio (MSE)
    mse = mean_squared_error(y, predictions)
    
    # Guardar el mejor modelo
    if mse < lowest_mse:
        lowest_mse = mse
        best_degree = degree
        best_predictions = predictions
    
    # Graficar la regresión polinómica
    plt.plot(data_2023.index, predictions, label=f'Polynomial Degree {degree}')

# Graficar los precios reales
plt.scatter(data_2023.index, y, color='blue', label='Actual Prices', alpha=0.5)
plt.title('Polynomial Regression of Apple Stock Prices (2023 Onwards)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price (USD)', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Mostrar el mejor grado de polinomio
print(f'Best Polynomial Degree: {best_degree} with MSE: {lowest_mse}')


# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar el dataset
data = pd.read_csv('AppleStock2024.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Preparar los datos
data['Date_Ordinal'] = data.index.map(pd.Timestamp.toordinal)
X = data['Date_Ordinal'].values.reshape(-1, 1)
y = data['Price'].values

# Probar diferentes grados de polinomio
degrees = [1, 2, 3, 4, 5]
best_degree = 0
lowest_mse = float('inf')
best_predictions = None

for degree in degrees:
    # Ajustar un modelo de regresión polinómica
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Predecir valores
    predictions = model.predict(X_poly)
    
    # Calcular el error cuadrático medio (MSE)
    mse = mean_squared_error(y, predictions)
    
    # Guardar el mejor modelo
    if mse < lowest_mse:
        lowest_mse = mse
        best_degree = degree
        best_predictions = predictions

# Gráfica profesional
plt.figure(figsize=(14, 7))
plt.scatter(data.index, y, color='blue', label='Actual Prices', alpha=0.5, s=10)
plt.plot(data.index, best_predictions, color='red', linewidth=2, label=f'Polynomial Degree {best_degree}', alpha=0.8)
plt.title('Best Polynomial Regression of Apple Stock Prices', fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price (USD)', fontsize=14)
plt.xticks(rotation=45)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.annotate(f'Best Degree: {best_degree}\nMSE: {lowest_mse:.2f}', 
             xy=(0.05, 0.95), 
             xycoords='axes fraction', 
             fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightgrey'))
plt.show()


# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns

# Establecer el estilo de Seaborn
sns.set(style="whitegrid")

# Cargar el dataset
data = pd.read_csv('AppleStock2024.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Filtrar datos desde 2010
data = data[data.index >= '2010-01-01']

# Preparar los datos
data['Date_Ordinal'] = data.index.map(pd.Timestamp.toordinal)
X = data['Date_Ordinal'].values.reshape(-1, 1)
y = data['Price'].values

# Probar diferentes grados de polinomio
degrees = [1, 2, 3, 4, 5]
best_degree = 0
lowest_mse = float('inf')
best_predictions = None

for degree in degrees:
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)
    
    predictions = model.predict(X_poly)
    mse = mean_squared_error(y, predictions)
    
    if mse < lowest_mse:
        lowest_mse = mse
        best_degree = degree
        best_predictions = predictions

# Gráfica profesional
plt.figure(figsize=(14, 8))
plt.scatter(data.index, y, color='dodgerblue', label='Actual Prices', alpha=0.6, s=20, edgecolor='w')
plt.plot(data.index, best_predictions, color='darkorange', linewidth=2.5, label=f'Polynomial Degree {best_degree}', alpha=0.9)

# Estilo del título y etiquetas
plt.title('Best Polynomial Regression of Apple Stock Prices (2010 - Present)', fontsize=20, fontweight='bold', color='navy')
plt.xlabel('Date', fontsize=16, color='darkslategray')
plt.ylabel('Price (USD)', fontsize=16, color='darkslategray')
plt.xticks(rotation=45, fontsize=12, color='dimgray')
plt.yticks(fontsize=12, color='dimgray')
plt.legend(loc='upper left', fontsize=14, frameon=True, facecolor='white', edgecolor='black')
plt.grid(True, linestyle='--', alpha=0.7)

# Anotaciones
plt.annotate(f'Best Degree: {best_degree}\nMSE: {lowest_mse:.2f}', 
             xy=(0.05, 0.95), 
             xycoords='axes fraction', 
             fontsize=14, 
             bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightgrey'))

plt.tight_layout()
plt.show()


# In[11]:


plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Price'], color='skyblue', linewidth=2)
plt.title('Apple Stock Prices Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price (USD)', fontsize=14)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[12]:


data['Change %'] = data['Price'].pct_change() * 100
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Change %'], color='orange', linewidth=1)
plt.title('Daily Percentage Change of Apple Stock Prices', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Change %', fontsize=14)
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[13]:


plt.figure(figsize=(14, 7))
plt.bar(data.index, data['Vol.'], color='lightgreen')
plt.title('Trading Volume of Apple Stock', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Volume', fontsize=14)
plt.grid(axis='y', linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[14]:


plt.figure(figsize=(8, 6))
sns.boxplot(y=data['Price'], color='lightblue')
plt.title('Boxplot of Apple Stock Prices', fontsize=16, fontweight='bold')
plt.ylabel('Price (USD)', fontsize=14)
plt.grid(axis='y', linestyle='--')
plt.show()


# In[15]:


correlation = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=.5)
plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
plt.show()


# In[16]:


data['MA20'] = data['Price'].rolling(window=20).mean()
data['MA50'] = data['Price'].rolling(window=50).mean()

plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Price'], label='Actual Price', color='skyblue', linewidth=2)
plt.plot(data.index, data['MA20'], label='20-Day MA', color='orange', linewidth=1)
plt.plot(data.index, data['MA50'], label='50-Day MA', color='red', linewidth=1)
plt.title('Apple Stock Prices with Moving Averages', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price (USD)', fontsize=14)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:




