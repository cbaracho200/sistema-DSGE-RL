"""
GUIA: Como Definir Variáveis de Previsão

Este guia mostra como especificar:
1. Qual variável prever (target)
2. Quais variáveis usar para prever (features/exógenas)
"""

import sys
sys.path.append('src')

import numpy as np
import pandas as pd
from pipeline import VitoriaForecastPipeline

# ============================================================================
# CARREGA SEUS DADOS
# ============================================================================

# Exemplo com suas variáveis reais
df = pd.read_csv('seus_dados.csv', index_col=0, parse_dates=True)

# Ou cria exemplo:
# df = pd.DataFrame({
#     'preco_m2': [...],           # Variável que você quer prever
#     'lancamentos': [...],        # Variável explicativa 1
#     'credito_imob': [...],       # Variável explicativa 2
#     'emprego_construcao': [...], # Variável explicativa 3
#     'vendas': [...],             # Variável explicativa 4
#     'renda_media': [...],        # Variável explicativa 5
# }, index=datas_mensais)


# ============================================================================
# OPÇÃO 1: MODO PADRÃO - CRIA ÍNDICE IDCI-VIX
# ============================================================================
# O sistema:
# 1. Pega TODAS as colunas do DataFrame
# 2. Seleciona top-5 via Granger
# 3. Cria IDCI-VIX como índice sintético
# 4. Prevê o IDCI-VIX

print("\n" + "="*80)
print("OPÇÃO 1: MODO PADRÃO (IDCI-VIX)")
print("="*80)

pipeline = VitoriaForecastPipeline(
    max_vars=5,              # Seleciona top-5 variáveis
    forecast_horizon=12,
    verbose=True
)

results = pipeline.run_full_pipeline(df)

# O que foi previsto?
print(f"\nAlvo previsto: IDCI-VIX (índice sintético 0-10)")
print(f"Variáveis usadas: {results['selected_vars']}")
print(f"Previsão 12M: {results['ensemble']['forecast'].iloc[0]:.2f}")


# ============================================================================
# OPÇÃO 2: PREVER UMA VARIÁVEL ESPECÍFICA (SEM ÍNDICE)
# ============================================================================
# Você escolhe qual variável prever diretamente

print("\n" + "="*80)
print("OPÇÃO 2: PREVER VARIÁVEL ESPECÍFICA (ex: preco_m2)")
print("="*80)

# 1. Separa target e features
target_name = 'preco_m2'  # ← VARIÁVEL QUE VOCÊ QUER PREVER
target = df[target_name]

# 2. Escolhe features (pode ser automático ou manual)
# Automático: remove apenas o target
features = df.drop(columns=[target_name])

# Ou manual: escolhe quais usar
# features = df[['lancamentos', 'credito_imob', 'emprego_construcao']]

# 3. Pré-processa
pipeline = VitoriaForecastPipeline(max_vars=5, forecast_horizon=12, verbose=True)

# Preprocessa features
pipeline.preprocess(features)

# Seleciona melhores features via Granger
# (usa o target como referência)
from preprocessing.granger import GrangerSelector

granger = GrangerSelector(max_vars=5)
selected_vars, _ = granger.select_top_k(
    pipeline.data_stationary,
    target=target,  # ← Seu target como referência
    use_pca_factor=False,
    verbose=True
)

# Features selecionadas
exog = pipeline.data_stationary[selected_vars]

# 4. Treina modelos DIRETAMENTE no target
models_dict = {}

# ARIMA
from forecasting.arima_models import ARIMAForecaster

print("\nTreinando ARIMA para", target_name)
arima = ARIMAForecaster()
arima.fit(target, exog=exog, auto=True, verbose=True)
models_dict['arima'] = arima

# Ridge
from forecasting.regularized_models import MultiHorizonRegularized

print("\nTreinando Ridge para", target_name)
ridge = MultiHorizonRegularized(model_type='ridge', max_lag=12, max_horizon=12)
ridge.fit(target, exog=exog, verbose=True)
models_dict['ridge'] = ridge

# Random Forest
from forecasting.tree_models import MultiHorizonTree

print("\nTreinando Random Forest para", target_name)
rf = MultiHorizonTree(model_type='random_forest', max_lag=12, max_horizon=12)
rf.fit(target, exog=exog, verbose=True)
models_dict['random_forest'] = rf

# 5. Faz previsões
print(f"\nPrevisões para {target_name}:")

forecasts_dict = {}

# ARIMA
fc_arima = arima.forecast(steps=12, exog=exog.iloc[[-1]].values.repeat(12, axis=0))
forecasts_dict['arima'] = fc_arima
print(f"  ARIMA 12M: {fc_arima['forecast'].iloc[-1]:.2f}")

# Ridge
fc_ridge = ridge.forecast(target, exog=exog)
forecasts_dict['ridge'] = fc_ridge
print(f"  Ridge 12M: {fc_ridge['h12'].iloc[0]:.2f}")

# Random Forest
fc_rf = rf.forecast(target, exog=exog)
forecasts_dict['random_forest'] = fc_rf
print(f"  RF 12M: {fc_rf['h12'].iloc[0]:.2f}")


# ============================================================================
# OPÇÃO 3: ESCOLHER MANUALMENTE AS VARIÁVEIS (SEM GRANGER)
# ============================================================================
# Você define exatamente quais variáveis usar

print("\n" + "="*80)
print("OPÇÃO 3: VARIÁVEIS MANUAIS (você escolhe)")
print("="*80)

# 1. Define manualmente
target_name = 'preco_m2'
target = df[target_name]

# ← VOCÊ ESCOLHE QUAIS VARIÁVEIS USAR
exog_manual = df[['lancamentos', 'credito_imob', 'emprego_construcao']]

print(f"\nTarget: {target_name}")
print(f"Exógenas escolhidas: {exog_manual.columns.tolist()}")

# 2. Preprocessa (opcional mas recomendado)
from preprocessing.stationarity import StationarityTester

tester = StationarityTester()

# Torna target estacionário
target_stat_df = pd.DataFrame({target_name: target})
target_stat = tester.fit_transform(target_stat_df, verbose=False)

# Torna exógenas estacionárias
exog_stat = tester.fit_transform(exog_manual, verbose=False)

# 3. Treina modelo
print("\nTreinando com suas variáveis...")

arima_manual = ARIMAForecaster()
arima_manual.fit(
    target_stat[target_name],
    exog=exog_stat,
    auto=True,
    verbose=True
)

# 4. Prevê
fc_manual = arima_manual.forecast(steps=12, exog=exog_stat.iloc[[-1]].values.repeat(12, axis=0))
print(f"\nPrevisão 12M para {target_name}: {fc_manual['forecast'].iloc[-1]:.2f}")


# ============================================================================
# OPÇÃO 4: PIPELINE FLEXÍVEL - CUSTOM TARGET
# ============================================================================
# Usa o pipeline mas com target customizado

print("\n" + "="*80)
print("OPÇÃO 4: PIPELINE COM TARGET CUSTOMIZADO")
print("="*80)

# Inicializa pipeline
pipeline_custom = VitoriaForecastPipeline(max_vars=5, forecast_horizon=12, verbose=True)

# Pré-processa
pipeline_custom.preprocess(df)

# Seleciona variáveis (pode pular se quiser escolher manual)
pipeline_custom.select_variables()

# ← DEFINE SEU TARGET (ao invés de criar IDCI-VIX)
target_custom = df['preco_m2']  # Sua variável alvo
exog_custom = pipeline_custom.data_stationary[pipeline_custom.selected_vars]

# Treina modelos
pipeline_custom.train_models(
    target=target_custom,  # ← SEU TARGET AQUI
    exog=exog_custom,
    models_to_train=['arima', 'ridge', 'random_forest']
)

# Previsões
forecasts_custom = pipeline_custom.forecast_all(
    target=target_custom,
    exog=exog_custom
)

print(f"\nModelos treinados: {list(forecasts_custom.keys())}")
print(f"Previsão 12M (ARIMA): {forecasts_custom['arima']['forecast'].iloc[-1]:.2f}")


# ============================================================================
# RESUMO: ONDE DEFINIR O QUE?
# ============================================================================

print("\n" + "#"*80)
print("# RESUMO: ONDE DEFINIR VARIÁVEIS")
print("#"*80)

print("""
1️⃣  VARIÁVEL A PREVER (TARGET):

   # Modo padrão (IDCI-VIX):
   results = pipeline.run_full_pipeline(df)
   # ↑ Cria índice sintético automaticamente

   # Target customizado:
   target = df['preco_m2']  # ← SUA VARIÁVEL AQUI
   pipeline.train_models(target=target, exog=exog)


2️⃣  VARIÁVEIS PARA PREVER (FEATURES/EXÓGENAS):

   # Automático (Granger seleciona top-5):
   pipeline.select_variables()  # ← Seleciona automaticamente
   exog = pipeline.data_stationary[pipeline.selected_vars]

   # Manual (você escolhe):
   exog = df[['lancamentos', 'credito', 'emprego']]  # ← SUAS VARIÁVEIS


3️⃣  CONJUNTO DE DADOS:

   # Todas as variáveis disponíveis:
   df = pd.DataFrame({
       'preco_m2': [...],           # ← Candidata a target
       'lancamentos': [...],        # ← Candidata a feature
       'credito_imob': [...],       # ← Candidata a feature
       'emprego_construcao': [...], # ← Candidata a feature
       # ... quantas quiser
   })

   # O sistema decide quais usar baseado em Granger
   # OU você escolhe manualmente


4️⃣  EXEMPLO COMPLETO TÍPICO:

   import pandas as pd
   from pipeline import VitoriaForecastPipeline

   # Seus dados
   df = pd.read_csv('vitoria_dados.csv', index_col=0, parse_dates=True)

   # Opção A: Automático (IDCI-VIX)
   pipeline = VitoriaForecastPipeline(max_vars=5, forecast_horizon=12)
   results = pipeline.run_full_pipeline(df)

   # Opção B: Target específico
   target = df['preco_m2']  # ← O QUE PREVER
   features = df.drop(columns=['preco_m2'])  # ← USAR PARA PREVER

   pipeline.preprocess(features)
   pipeline.select_variables()
   pipeline.train_models(
       target=target,
       exog=pipeline.data_stationary[pipeline.selected_vars]
   )
""")

print("#"*80)
