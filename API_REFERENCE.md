# Guia de Referência Rápida - API do Sistema

Este guia documenta a API correta dos principais módulos do sistema.

## 📊 Preprocessing - StationarityTester

### Inicialização
```python
from preprocessing.stationarity import StationarityTester

# Alpha é definido no construtor
tester = StationarityTester(alpha=0.05, max_diff=2)
```

### Teste de Estacionaridade
```python
# CORRETO ✓
result = tester.test_stationarity(df['coluna'])  # Aceita pd.Series
is_stationary = result['is_stationary']
adf_pval = result['adf']['pvalue']
kpss_pval = result['kpss']['pvalue']

# INCORRETO ✗
# is_stationary, adf_pval, kpss_pval = tester.test_stationarity(df['coluna'], alpha=0.05)
```

**Retorno**: Dicionário com estrutura:
```python
{
    'adf': {
        'statistic': float,
        'pvalue': float,
        'is_stationary': bool
    },
    'kpss': {
        'statistic': float,
        'pvalue': float,
        'is_stationary': bool
    },
    'is_stationary': bool,  # Ambos concordam
    'agreement': bool
}
```

### Transformação Automática
```python
# Torna série estacionária automaticamente
series_stat, info = tester.make_stationary(
    df['coluna'],
    name='nome_coluna',
    try_seasonal=True,
    period=12,
    verbose=True
)

print(info)
# {'type': 'regular', 'order': 1, 'seasonal_order': 0, 'period': None}
```

## 🔍 Preprocessing - GrangerSelector

### Inicialização e Seleção
```python
from preprocessing.variable_selection import GrangerSelector

selector = GrangerSelector(max_lag=6, alpha=0.05)

# Retorna lista de variáveis selecionadas
selected_vars = selector.select_variables(
    df,  # DataFrame completo
    target_col='target'
)
# Retorna: ['target', 'var1', 'var2', ...]
```

### Teste Individual
```python
# Testa se x Granger-causa y
causes, pvalue = selector.test_granger_causality(
    x_series,  # np.array ou pd.Series
    y_series
)
```

## 📈 Forecasting - Modelos

### ARIMA / SARIMA
```python
from forecasting.arima import ARIMAForecaster
from forecasting.sarima import SARIMAForecaster

# ARIMA
model = ARIMAForecaster(order=(p, d, q))
model.fit(y_train)  # Apenas y
predictions = model.forecast(n_periods)

# SARIMA
model = SARIMAForecaster(
    order=(p, d, q),
    seasonal_order=(P, D, Q, s)
)
model.fit(y_train)
predictions = model.forecast(n_periods)
```

### Ridge / Lasso / Random Forest
```python
from forecasting.ridge import RidgeForecaster
from forecasting.random_forest import RandomForestForecaster

# Ridge
model = RidgeForecaster(alpha=1.0, lags=3)
model.fit(X_train, y_train)  # X e y
predictions = model.forecast(X_test)

# Random Forest
model = RandomForestForecaster(
    n_estimators=100,
    max_depth=10,
    lags=5,
    random_state=42
)
model.fit(X_train, y_train)
predictions = model.forecast(X_test)

# Feature importance
importance = model.feature_importance(['var1', 'var2', 'var3'])
# Retorna: {'var1': 0.45, 'var2': 0.35, 'var3': 0.20}
```

### Quantile Regression
```python
from forecasting.quantile_reg import QuantileForecaster

# Para cenários (pessimista/base/otimista)
model_q10 = QuantileForecaster(quantile=0.10, lags=3)
model_q50 = QuantileForecaster(quantile=0.50, lags=3)
model_q90 = QuantileForecaster(quantile=0.90, lags=3)

model_q50.fit(X_train, y_train)
predictions = model_q50.forecast(X_test)
```

### Markov-Switching
```python
from forecasting.markov_switching import MarkovSwitchingForecaster

model = MarkovSwitchingForecaster(n_regimes=2, order=2)
model.fit(y_train)
predictions = model.forecast(n_periods)
```

## 🎯 Evaluation - Métricas

### Calcular Métricas
```python
from evaluation.metrics import calculate_metrics

metrics = calculate_metrics(y_true, y_pred)
# Retorna dicionário:
# {
#     'mae': float,
#     'rmse': float,
#     'mape': float,
#     'r2': float
# }
```

### Ensemble
```python
from evaluation.ensemble import EnsembleCombiner

# predictions_matrix: (n_samples, n_models)
combiner = EnsembleCombiner(method='weighted_average')
combiner.fit(predictions_matrix, y_true)
ensemble_pred = combiner.predict(predictions_matrix)

# Métodos disponíveis:
# - 'simple_average'
# - 'weighted_average'
# - 'median'
```

## 🧪 Factor Model

### Dynamic Factor Model
```python
from factor_model.dynamic_factor import DynamicFactorModel

dfm = DynamicFactorModel(n_factors=1)
dfm.fit(X)  # X: (n_samples, n_features)

# Extrair fatores
factors = dfm.factors_  # (n_samples, n_factors)

# Normalizar para escala 0-10 (IDCI-VIX)
factor = factors.flatten()
idci_vix = 10 * (factor - factor.min()) / (factor.max() - factor.min())
```

## 🔧 Exemplo Completo

```python
import pandas as pd
from preprocessing.stationarity import StationarityTester
from preprocessing.variable_selection import GrangerSelector
from factor_model.dynamic_factor import DynamicFactorModel
from forecasting.random_forest import RandomForestForecaster
from evaluation.metrics import calculate_metrics

# 1. Testar estacionaridade
tester = StationarityTester(alpha=0.05)
df_stat = tester.fit_transform(df)

# 2. Selecionar variáveis
selector = GrangerSelector(max_lag=6, alpha=0.05)
selected_vars = selector.select_variables(df_stat, target_col='pib_real')

# 3. Construir fator
dfm = DynamicFactorModel(n_factors=1)
dfm.fit(df_stat[selected_vars].values)
factor = dfm.factors_.flatten()
df_stat['IDCI_VIX'] = 10 * (factor - factor.min()) / (factor.max() - factor.min())

# 4. Previsão
train_size = int(0.8 * len(df_stat))
X_train = df_stat[selected_vars].iloc[:train_size].values
y_train = df_stat['IDCI_VIX'].iloc[:train_size].values
X_test = df_stat[selected_vars].iloc[train_size:].values
y_test = df_stat['IDCI_VIX'].iloc[train_size:].values

model = RandomForestForecaster(n_estimators=100, lags=5, random_state=42)
model.fit(X_train, y_train)
predictions = model.forecast(X_test)

# 5. Avaliar
metrics = calculate_metrics(y_test, predictions)
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"R²: {metrics['r2']:.4f}")
```

## ⚠️ Erros Comuns

### 1. StationarityTester
❌ `tester.test_stationarity(df[col].values, alpha=0.05)`
✅ `tester.test_stationarity(df[col])`

❌ `is_stat, adf, kpss = tester.test_stationarity(...)`
✅ `result = tester.test_stationarity(...)`
✅ `is_stat = result['is_stationary']`

### 2. Modelos de Forecasting
❌ `model.fit(X_train)` para Ridge/RF (falta y)
✅ `model.fit(X_train, y_train)`

❌ `model.fit(X_train, y_train)` para ARIMA (não aceita X)
✅ `model.fit(y_train)`

### 3. Forecast vs Predict
- Use `model.forecast(X_test)` - API padrão do sistema
- Não use `model.predict()` diretamente

### 4. Tipos de Dados
- StationarityTester: aceita `pd.Series`
- Modelos: aceita `np.array` ou `pd.Series`
- DataFrames: use `.values` para converter para array quando necessário

## 📝 Convenções

1. **Variável target**: sempre chamada de `y_train`, `y_test`
2. **Features**: sempre chamadas de `X_train`, `X_test`
3. **Previsões**: sempre chamadas de `predictions` ou `y_pred`
4. **Índice temporal**: sempre use `pd.date_range` com `freq='M'` para mensal
5. **Random state**: sempre use `random_state=42` para reproducibilidade

---

**Última atualização**: 2024
**Versão do sistema**: 1.0.0
