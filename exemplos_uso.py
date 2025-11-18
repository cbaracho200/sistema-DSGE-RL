"""
Exemplos de uso do sistema de previsão - Vitória/ES

Execute este script para ver as visualizações funcionando.
"""

import sys
sys.path.append('src')

import numpy as np
import pandas as pd
from pipeline import VitoriaForecastPipeline
from utils.plot_results import plot_all_results, print_summary

# ============================================================================
# EXEMPLO COM DADOS SINTÉTICOS
# Substitua por seus dados reais: df = pd.read_csv('seus_dados.csv', ...)
# ============================================================================

print("Gerando dados sintéticos...")

np.random.seed(42)
dates = pd.date_range('2010-01-01', periods=120, freq='MS')

# Simula séries correlacionadas
trend = np.linspace(0, 2, 120)
cycle = 0.5 * np.sin(2 * np.pi * np.arange(120) / 12)

df = pd.DataFrame({
    'preco_m2': trend + cycle + 0.2 * np.random.randn(120).cumsum(),
    'lancamentos': 0.8 * trend + 0.3 * cycle + 0.15 * np.random.randn(120).cumsum(),
    'credito_imob': 0.9 * trend + 0.4 * cycle + 0.1 * np.random.randn(120).cumsum(),
    'emprego_construcao': 0.7 * trend + 0.2 * cycle + 0.2 * np.random.randn(120).cumsum(),
    'massa_salarial': 0.6 * trend + 0.1 * cycle + 0.15 * np.random.randn(120).cumsum(),
    'pib_es': 0.85 * trend + 0.25 * cycle + 0.1 * np.random.randn(120).cumsum(),
    'selic': -0.3 * trend + 0.1 * np.random.randn(120).cumsum(),
}, index=dates)

print(f"Dados: {df.shape[0]} observações, {df.shape[1]} variáveis")
print(f"Período: {df.index[0].strftime('%Y-%m')} a {df.index[-1].strftime('%Y-%m')}")

# ============================================================================
# EXECUTA PIPELINE
# ============================================================================

print("\n" + "="*80)
print("EXECUTANDO PIPELINE")
print("="*80)

pipeline = VitoriaForecastPipeline(
    max_vars=5,
    forecast_horizon=12,
    ar_order=2,
    verbose=True
)

results = pipeline.run_full_pipeline(
    df,
    models_to_train=['arima', 'ridge', 'lasso', 'random_forest', 'quantile'],
    ensemble_method='weighted_avg'
)

# ============================================================================
# GERA TODAS AS VISUALIZAÇÕES
# ============================================================================

print("\n" + "="*80)
print("GERANDO VISUALIZAÇÕES")
print("="*80)

figures = plot_all_results(
    results,
    output_dir='data/processed/',
    show_plots=False,  # Mude para True para exibir gráficos
    save_plots=True
)

# ============================================================================
# RESUMO EXECUTIVO
# ============================================================================

print_summary(results)

# ============================================================================
# SALVA DADOS
# ============================================================================

print("\n" + "="*80)
print("SALVANDO RESULTADOS")
print("="*80)

results['idci_vix'].to_csv('data/processed/idci_vix.csv', header=True)
results['ensemble'].to_csv('data/processed/forecast_ensemble_12m.csv')

if 'quantile_quantiles' in results['forecasts']:
    results['forecasts']['quantile_quantiles'].to_csv('data/processed/forecast_scenarios_12m.csv')

print("\n✓ Dados salvos em data/processed/")
print("  - idci_vix.csv")
print("  - forecast_ensemble_12m.csv")
print("  - forecast_scenarios_12m.csv")
print("  - Múltiplos gráficos PNG")

print("\n" + "="*80)
print("CONCLUÍDO!")
print("="*80)
print("\nPróximos passos:")
print("  1. Substitua dados sintéticos por seus dados reais")
print("  2. Ajuste parâmetros em config/config_example.yaml")
print("  3. Execute notebooks/exemplo_com_graficos.ipynb")
print("  4. Integre com seus modelos DSGE ou RL")
