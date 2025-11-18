"""
Guia de Resolução de Erros Comuns

Este arquivo explica os erros mais comuns e como corrigi-los.
"""

# ============================================================================
# ERRO 1: 'str' object has no attribute 'name'
# ============================================================================

print("""
ERRO: 'str' object has no attribute 'name'

CAUSA:
Você passou o NOME da coluna (string) ao invés da SÉRIE pandas.

❌ ERRADO:
""")

import pandas as pd
df = pd.DataFrame({'preco_m2': [100, 105, 110]})

# ❌ Isso causa erro:
# target = 'preco_m2'  # String!
# pipeline.train_models(target=target)

print("""
✅ CORRETO:
""")

# ✅ Isso funciona:
target = df['preco_m2']  # Series pandas!
print(f"target tipo: {type(target)}")  # <class 'pandas.core.series.Series'>

# Ou use diretamente:
# pipeline.train_models(target=df['preco_m2'])

print("""

# ============================================================================
# ERRO 2: 'str' object has no attribute 'shift'
# ============================================================================

ERRO: 'str' object has no attribute 'shift'

CAUSA:
Mesmo problema - passou string ao invés de Series.

❌ ERRADO:
target = 'preco_m2'  # String

✅ CORRETO:
target = df['preco_m2']  # Series


# ============================================================================
# ERRO 3: KeyError ou ValueError em run_full_pipeline
# ============================================================================

ERRO: KeyError: 'coluna_x' ou ValueError

CAUSAS POSSÍVEIS:

1. DataFrame vazio ou com poucas linhas
   - Mínimo recomendado: 36 observações
   - Ideal: 60+ observações

2. Muitos valores NaN
   - Séries com >50% NaN são problemáticas
   - Use df.dropna() ou preencha valores

3. Variáveis não numéricas
   - Todas as colunas devem ser numéricas
   - Remova colunas de texto/categorias

4. Índice não é datetime
   - Índice deve ser pd.DatetimeIndex
   - Use: df.index = pd.to_datetime(df.index)

EXEMPLO DE DADOS CORRETOS:
""")

import pandas as pd
import numpy as np

# ✅ Estrutura correta
dates = pd.date_range('2015-01-01', periods=60, freq='MS')  # Mensal

df_correto = pd.DataFrame({
    'preco_m2': np.random.randn(60).cumsum() + 3500,     # Numérico
    'lancamentos': np.random.randn(60).cumsum() + 120,   # Numérico
    'credito': np.random.randn(60).cumsum() + 50000,     # Numérico
}, index=dates)  # ← Índice datetime

print(f"✓ DataFrame válido:")
print(f"  Shape: {df_correto.shape}")
print(f"  Tipo do índice: {type(df_correto.index)}")
print(f"  Frequência: {df_correto.index.freq}")
print(f"  Primeiras datas: {df_correto.index[:3].tolist()}")

print("""

# ============================================================================
# ERRO 4: "Dados insuficientes" ou "Série muito curta"
# ============================================================================

ERRO: ValueError: Dados insuficientes após criar features e remover NaN

CAUSA:
Após criar lags (defasagens), muitas linhas viram NaN e são removidas.

SOLUÇÃO:
- Use mais dados (mínimo 60 observações para 12 lags)
- Reduza max_lag se tiver poucos dados
- Reduza forecast_horizon

EXEMPLO:
""")

from src.pipeline import VitoriaForecastPipeline

# ❌ Poucos dados + muitos lags:
# df_pequeno = DataFrame com 24 linhas
# pipeline = VitoriaForecastPipeline(max_lag=12)  # Perde metade!

# ✅ Ajuste os parâmetros:
# pipeline = VitoriaForecastPipeline(
#     max_vars=3,          # Menos variáveis
#     forecast_horizon=6,  # Horizonte menor
# )

print("""

# ============================================================================
# ERRO 5: Modelo não converge ou demora muito
# ============================================================================

SINTOMA:
Pipeline trava ou demora horas para executar.

CAUSAS E SOLUÇÕES:

1. Muitas variáveis:
   ❌ df com 50+ colunas
   ✅ Reduza para max_vars=5 ou menos

2. Muitas observações:
   ❌ df com 1000+ linhas
   ✅ Use apenas últimos 120 meses (10 anos)

3. Modelos pesados:
   ❌ models_to_train=['arima', 'sarima', 'sarimax', 'markov', ...]
   ✅ Comece com: models_to_train=['arima', 'ridge']

EXEMPLO RÁPIDO PARA TESTE:
""")

# ✅ Configuração rápida para testar:
pipeline_teste = VitoriaForecastPipeline(
    max_vars=3,           # Poucas variáveis
    forecast_horizon=6,   # Horizonte curto
    verbose=False         # Menos output
)

# Use apenas modelos rápidos:
# results = pipeline_teste.run_full_pipeline(
#     df.tail(60),  # Últimas 60 observações
#     models_to_train=['ridge', 'lasso'],  # Modelos rápidos
# )

print("""

# ============================================================================
# CHECKLIST ANTES DE RODAR
# ============================================================================

Antes de executar o pipeline, verifique:

[ ] DataFrame criado corretamente
    - df = pd.DataFrame({...})

[ ] Índice é datetime
    - df.index = pd.to_datetime(df.index)
    - pd.infer_freq(df.index) retorna 'MS' ou similar

[ ] Todas as colunas são numéricas
    - df.dtypes mostra float64 ou int64

[ ] Sem muitos NaN
    - df.isna().sum() / len(df) < 0.5

[ ] Dados suficientes
    - len(df) >= 60 para forecast_horizon=12

[ ] Target é Series (não string)
    - target = df['coluna']  # ✅
    - target = 'coluna'       # ❌

# ============================================================================
# EXEMPLO COMPLETO FUNCIONANDO
# ============================================================================
""")

import sys
sys.path.append('src')
import numpy as np
import pandas as pd
from pipeline import VitoriaForecastPipeline

# 1. Cria dados corretos
np.random.seed(42)
dates = pd.date_range('2015-01-01', periods=60, freq='MS')

df = pd.DataFrame({
    'var1': np.random.randn(60).cumsum() + 100,
    'var2': np.random.randn(60).cumsum() + 50,
    'var3': np.random.randn(60).cumsum() + 75,
    'var4': np.random.randn(60).cumsum() + 30,
}, index=dates)

print(f"\n✓ Dados criados: {df.shape}")

# 2. Pipeline automático
pipeline = VitoriaForecastPipeline(max_vars=3, forecast_horizon=6, verbose=False)

try:
    results = pipeline.run_full_pipeline(
        df,
        models_to_train=['arima', 'ridge'],  # Poucos modelos para teste
        ensemble_method='simple_avg'
    )

    print(f"\n✓ SUCESSO!")
    print(f"   IDCI-VIX: {results['idci_vix'].iloc[-1]:.2f}")
    print(f"   Variáveis: {results['selected_vars']}")
    print(f"   Modelos: {list(results['models'].keys())}")
    print(f"   Previsão 6M: {results['ensemble']['forecast'].iloc[0]:.2f}")

except Exception as e:
    print(f"\n❌ ERRO: {e}")
    print("\nVeja o checklist acima!")

print("""

# ============================================================================
# PRECISA DE AJUDA?
# ============================================================================

1. Leia: GUIA_RAPIDO_VARIAVEIS.md
2. Execute: python test_debug.py
3. Veja exemplos: exemplos_target_custom.py
4. Jupyter: notebooks/exemplo_com_graficos.ipynb

""")
