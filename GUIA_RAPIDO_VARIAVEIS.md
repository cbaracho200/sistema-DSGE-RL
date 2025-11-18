# 📋 Guia Rápido: Onde Definir Variáveis

## 🎯 Onde Colocar Suas Variáveis?

### **1. Estrutura dos Seus Dados**

Seus dados devem estar em um DataFrame pandas com **uma coluna para cada variável**:

```python
import pandas as pd

# ← AQUI VOCÊ DEFINE TODAS AS SUAS VARIÁVEIS
df = pd.DataFrame({
    'preco_m2': [100, 105, 110, ...],              # ← Pode ser target
    'lancamentos': [50, 55, 60, ...],              # ← Pode ser feature
    'credito_imob': [1000, 1100, 1200, ...],       # ← Pode ser feature
    'emprego_construcao': [5000, 5200, 5100, ...], # ← Pode ser feature
    'vendas': [30, 35, 40, ...],                   # ← Pode ser feature
    'renda_media': [3000, 3100, 3200, ...],        # ← Pode ser feature
    # ... adicione quantas quiser
}, index=datas_mensais)  # ← Índice = datas (mensais)
```

**Formato do DataFrame:**
- ✅ Índice = datas (datetime)
- ✅ Colunas = variáveis
- ✅ Valores = dados numéricos
- ✅ Frequência = mensal (recomendado)

---

## 🔀 4 Formas de Usar

### **Forma 1: Automático Total (IDCI-VIX)** ⭐ Mais fácil

```python
from src.pipeline import VitoriaForecastPipeline

# Carrega TODAS as variáveis
df = pd.read_csv('seus_dados.csv', index_col=0, parse_dates=True)

# ← O SISTEMA FAZ TUDO AUTOMATICAMENTE:
# 1. Pega todas as colunas
# 2. Seleciona top-5 via Granger
# 3. Cria índice IDCI-VIX
# 4. Prevê o índice

pipeline = VitoriaForecastPipeline(max_vars=5, forecast_horizon=12)
results = pipeline.run_full_pipeline(df)  # ← Só isso!

print(f"Previsto: IDCI-VIX (índice 0-10)")
print(f"Variáveis usadas: {results['selected_vars']}")
print(f"Previsão 12M: {results['ensemble']['forecast'].iloc[0]:.2f}")
```

---

### **Forma 2: Escolher Qual Variável Prever** 🎯

```python
from src.pipeline import VitoriaForecastPipeline

# Seus dados
df = pd.read_csv('seus_dados.csv', index_col=0, parse_dates=True)

# ← AQUI VOCÊ ESCOLHE O QUE PREVER
target_name = 'preco_m2'  # ← SUA VARIÁVEL ALVO

# ← AQUI VOCÊ ESCOLHE QUAIS USAR PARA PREVER
# Opção A: Usar todas exceto o target
exog_columns = [col for col in df.columns if col != target_name]

# Opção B: Escolher manualmente
exog_columns = ['lancamentos', 'credito_imob', 'emprego_construcao']

# Separa
target = df[target_name]
exog = df[exog_columns]

# Pipeline
pipeline = VitoriaForecastPipeline(max_vars=5, forecast_horizon=12)

# Preprocessa features
pipeline.preprocess(exog)
pipeline.select_variables()  # Seleciona top-5 das exógenas

# Treina com SEU target
pipeline.train_models(
    target=target,  # ← SUA VARIÁVEL PARA PREVER
    exog=pipeline.data_stationary[pipeline.selected_vars]
)

# Previsões
forecasts = pipeline.forecast_all(target=target, exog=...)
```

---

### **Forma 3: Controle Total Manual** 🔧

```python
from src.forecasting.arima_models import ARIMAForecaster
from src.preprocessing.stationarity import StationarityTester

# Seus dados
df = pd.read_csv('seus_dados.csv', index_col=0, parse_dates=True)

# ← DEFINE EXATAMENTE QUAIS VARIÁVEIS USAR
target = df['preco_m2']           # ← O QUE PREVER
exog = df[['lancamentos',         # ← USAR PARA PREVER
           'credito_imob',
           'emprego_construcao']]

# Preprocessa (opcional mas recomendado)
tester = StationarityTester()
target_stat = tester.fit_transform(pd.DataFrame({'target': target}))
exog_stat = tester.fit_transform(exog)

# Treina
model = ARIMAForecaster()
model.fit(target_stat['target'], exog=exog_stat, auto=True)

# Prevê
forecast = model.forecast(steps=12, exog=exog_stat.iloc[[-1]])
print(forecast)
```

---

### **Forma 4: Pipeline Flexível** 🚀 Recomendado

```python
from src.pipeline import VitoriaForecastPipeline

df = pd.read_csv('seus_dados.csv', index_col=0, parse_dates=True)

pipeline = VitoriaForecastPipeline(max_vars=5, forecast_horizon=12)

# Passo 1: Preprocessa
pipeline.preprocess(df)

# Passo 2: Seleciona features (automático via Granger)
pipeline.select_variables()

# Passo 3: ← AQUI VOCÊ DEFINE O TARGET
# Opção A: Usa uma variável do DataFrame
target = df['preco_m2']

# Opção B: Cria índice sintético (padrão)
# pipeline.build_index()
# target = pipeline.idci_vix

# Passo 4: Treina
exog = pipeline.data_stationary[pipeline.selected_vars]
pipeline.train_models(target=target, exog=exog)

# Passo 5: Prevê
forecasts = pipeline.forecast_all(target=target, exog=exog)
```

---

## 📊 Exemplos Práticos

### **Exemplo 1: Prever Preço m² usando Crédito e Lançamentos**

```python
# Dados
df = pd.DataFrame({
    'preco_m2': [100, 105, 110, 108, 112, 115],        # ← PREVER ISSO
    'credito_imob': [1000, 1100, 1200, 1150, 1300, 1400],  # ← USAR
    'lancamentos': [50, 55, 60, 58, 65, 70],           # ← USAR
}, index=pd.date_range('2020-01', periods=6, freq='MS'))

# Separa
target = df['preco_m2']  # ← ALVO
exog = df[['credito_imob', 'lancamentos']]  # ← FEATURES

# Modelo simples
from src.forecasting.arima_models import ARIMAForecaster
model = ARIMAForecaster()
model.fit(target, exog=exog, auto=True)
forecast = model.forecast(steps=12, exog=exog.iloc[[-1]])

print(f"Previsão 12M: {forecast['forecast'].iloc[-1]:.2f}")
```

---

### **Exemplo 2: Usar Pipeline com Seleção Automática**

```python
# Dados com MUITAS variáveis
df = pd.DataFrame({
    'preco_m2': [...],
    'lancamentos': [...],
    'credito_imob': [...],
    'emprego_construcao': [...],
    'vendas': [...],
    'renda_media': [...],
    'pib_es': [...],
    'selic': [...],
    # ... 20+ variáveis
}, index=datas)

# Pipeline seleciona automaticamente top-5 mais relevantes
pipeline = VitoriaForecastPipeline(max_vars=5, forecast_horizon=12)

# Modo 1: Criar índice IDCI-VIX
results = pipeline.run_full_pipeline(df)  # ← Tudo automático

# Modo 2: Prever variável específica
target = df['preco_m2']
pipeline.preprocess(df.drop(columns=['preco_m2']))
pipeline.select_variables()  # Seleciona top-5 automaticamente
pipeline.train_models(
    target=target,
    exog=pipeline.data_stationary[pipeline.selected_vars]
)
```

---

## ✅ Checklist Rápido

Antes de rodar, verifique:

- [ ] **DataFrame criado** com todas as variáveis
- [ ] **Índice = datas** (datetime)
- [ ] **Frequência = mensal** (MS)
- [ ] **Dados numéricos** (sem texto)
- [ ] **Sem NaN excessivos** (< 10%)
- [ ] **Deflatado** (valores reais)
- [ ] **Log aplicado** (se apropriado)

---

## 🎯 Regra de Ouro

### **ANTES:**
```python
# Você fornece
df = pd.DataFrame({
    'var1': [...],  # ← Todas as suas variáveis
    'var2': [...],  #    em um único DataFrame
    'var3': [...],
})
```

### **DURANTE:**
```python
# Você escolhe
target = df['var1']              # ← O que prever
exog = df[['var2', 'var3']]      # ← Usar para prever
```

### **DEPOIS:**
```python
# Sistema retorna
forecasts = {
    'arima': previsão_12_meses,
    'ridge': previsão_12_meses,
    'rf': previsão_12_meses,
    ...
}
```

---

## 💡 Dica Final

**Se estiver em dúvida, use a Forma 1 (Automático)**:

```python
from src.pipeline import VitoriaForecastPipeline

df = pd.read_csv('seus_dados.csv', index_col=0, parse_dates=True)
pipeline = VitoriaForecastPipeline()
results = pipeline.run_full_pipeline(df)
```

O sistema vai:
- ✅ Processar todas as variáveis
- ✅ Selecionar as mais importantes
- ✅ Criar índice sintético
- ✅ Gerar previsões

Depois você pode customizar conforme sua necessidade!

---

## 📚 Arquivos de Exemplo

- `exemplos_uso.py` - Exemplo automático completo
- `exemplos_target_custom.py` - Exemplos com targets customizados
- `notebooks/exemplo_com_graficos.ipynb` - Exemplos interativos

Execute qualquer um deles para ver na prática!
