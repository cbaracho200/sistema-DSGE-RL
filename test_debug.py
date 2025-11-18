"""
Script de teste para diagnosticar erro de tipos.
"""

import sys
sys.path.append('src')

import numpy as np
import pandas as pd

# Teste 1: Dados simples
print("="*80)
print("TESTE 1: Dados sintéticos simples")
print("="*80)

np.random.seed(42)
dates = pd.date_range('2015-01-01', periods=60, freq='MS')

df_test = pd.DataFrame({
    'var1': np.random.randn(60).cumsum() + 10,
    'var2': np.random.randn(60).cumsum() + 5,
    'var3': np.random.randn(60).cumsum() + 3,
    'var4': np.random.randn(60).cumsum() + 7,
    'var5': np.random.randn(60).cumsum() + 4,
}, index=dates)

print(f"DataFrame criado: {df_test.shape}")
print(f"Tipo do DataFrame: {type(df_test)}")
print(f"Tipo da coluna 'var1': {type(df_test['var1'])}")
print(f"\nPrimeiras linhas:")
print(df_test.head())

# Teste 2: Pipeline
print("\n" + "="*80)
print("TESTE 2: Executando pipeline")
print("="*80)

from pipeline import VitoriaForecastPipeline

try:
    pipeline = VitoriaForecastPipeline(max_vars=3, forecast_horizon=6, verbose=True)

    print("\n1. Pré-processamento...")
    pipeline.preprocess(df_test)
    print(f"   ✓ data_stationary tipo: {type(pipeline.data_stationary)}")

    print("\n2. Seleção de variáveis...")
    pipeline.select_variables()
    print(f"   ✓ selected_vars: {pipeline.selected_vars}")

    print("\n3. Construção do índice...")
    idci = pipeline.build_index()
    print(f"   ✓ IDCI-VIX tipo: {type(idci)}")
    print(f"   ✓ IDCI-VIX name: {idci.name if hasattr(idci, 'name') else 'N/A'}")
    print(f"   ✓ self.idci_vix tipo: {type(pipeline.idci_vix)}")
    print(f"   ✓ self.idci_vix name: {pipeline.idci_vix.name if hasattr(pipeline.idci_vix, 'name') else 'N/A'}")

    print("\n4. Treinamento (apenas ARIMA)...")
    pipeline.train_models(models_to_train=['arima'])

    print("\n✓ TESTE PASSOU!")

except Exception as e:
    print(f"\n❌ ERRO: {e}")
    import traceback
    traceback.print_exc()

# Teste 3: Run completo
print("\n" + "="*80)
print("TESTE 3: run_full_pipeline completo")
print("="*80)

try:
    pipeline2 = VitoriaForecastPipeline(max_vars=3, forecast_horizon=6, verbose=False)
    results = pipeline2.run_full_pipeline(
        df_test,
        models_to_train=['arima'],
        ensemble_method='simple_avg'
    )

    print(f"\n✓ Pipeline completo executado!")
    print(f"   IDCI-VIX tipo: {type(results['idci_vix'])}")
    print(f"   Modelos: {list(results['models'].keys())}")

except Exception as e:
    print(f"\n❌ ERRO: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("DIAGNÓSTICO COMPLETO")
print("="*80)
