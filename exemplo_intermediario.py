"""
EXEMPLO INTERMEDIÁRIO - Sistema de Previsão Vitória/ES
======================================================

Nível: Intermediário
Tempo: 15 minutos
Objetivo: Customizar pipeline, diagnosticar dados, analisar modelos

Este exemplo mostra:
- Carregamento de dados de arquivo CSV
- Customização de parâmetros do pipeline
- Diagnóstico antes do treinamento
- Análise individual de modelos
- Visualizações
- Comparação de performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('src')

from pipeline import VitoriaForecastPipeline
from diagnostico_serie import diagnose_series
from utils.visualization import VitoriaVisualizer


# ============================================================================
# 1. CONFIGURAÇÕES
# ============================================================================

print("="*80)
print("EXEMPLO INTERMEDIÁRIO - Sistema de Previsão Vitória/ES")
print("="*80)

# Configurações customizadas
CONFIG = {
    'max_vars': 4,              # Seleciona top-4 variáveis
    'forecast_horizon': 12,     # Horizonte de 12 meses
    'ar_order': 2,              # Ordem AR do fator dinâmico
    'verbose': True
}

print("\n⚙️ Configurações:")
for key, value in CONFIG.items():
    print(f"   {key}: {value}")


# ============================================================================
# 2. CARREGAR DADOS
# ============================================================================

print("\n📂 Carregando dados...")

# Opção 1: Carregar de CSV (se você tiver)
# df = pd.read_csv('dados_vitoria.csv', index_col=0, parse_dates=True)

# Opção 2: Dados sintéticos mais realistas
np.random.seed(42)
dates = pd.date_range('2018-01-01', periods=72, freq='MS')

# Simula variáveis com tendências e sazonalidade
t = np.arange(72)
seasonal = 10 * np.sin(2 * np.pi * t / 12)

df = pd.DataFrame({
    'preco_m2': 3000 + 50*t + seasonal + 100*np.random.randn(72),
    'vendas': 100 + 2*t - seasonal/2 + 10*np.random.randn(72),
    'lancamentos': 50 + t/2 + seasonal/3 + 5*np.random.randn(72),
    'estoque': 200 - t + seasonal + 15*np.random.randn(72),
    'credito': 5000 + 100*t + 200*np.random.randn(72),
    'taxa_juros': 10 - 0.05*t + 0.5*np.random.randn(72),
    'pib_es': 1000 + 20*t + 50*np.random.randn(72),
    'desemprego': 12 - 0.03*t + 0.3*np.random.randn(72),
}, index=dates)

print(f"\n✓ Dados carregados:")
print(f"   Período: {df.index[0].strftime('%Y-%m')} a {df.index[-1].strftime('%Y-%m')}")
print(f"   Observações: {len(df)}")
print(f"   Variáveis: {list(df.columns)}")

# Visualiza dados brutos
print("\n📊 Estatísticas descritivas:")
print(df.describe().round(2))


# ============================================================================
# 3. CRIAR E CONFIGURAR PIPELINE
# ============================================================================

print("\n" + "="*80)
print("CONFIGURANDO PIPELINE")
print("="*80)

pipeline = VitoriaForecastPipeline(**CONFIG)

# Pré-processamento
print("\n1️⃣ Pré-processamento...")
pipeline.preprocess(df)

print(f"\n   Variáveis estacionárias criadas:")
for var, info in pipeline.stationarity_info_.items():
    print(f"   - {var}: {info['transformation']}")

# Seleção de variáveis
print("\n2️⃣ Seleção de variáveis...")
selected_vars, granger_results = pipeline.select_variables()

print(f"\n   ✓ {len(selected_vars)} variáveis selecionadas:")
print("\n   Ranking Granger:")
print(granger_results[['variable', 'f_statistic', 'p_value']].round(4))


# ============================================================================
# 4. CONSTRUIR E DIAGNOSTICAR IDCI-VIX
# ============================================================================

print("\n" + "="*80)
print("CONSTRUÇÃO E DIAGNÓSTICO DO IDCI-VIX")
print("="*80)

print("\n3️⃣ Construindo IDCI-VIX...")
idci = pipeline.build_index()

print(f"\n   ✓ IDCI-VIX criado:")
print(f"   Observações: {len(idci)}")
print(f"   Média: {idci.mean():.2f}")
print(f"   Desvio padrão: {idci.std():.2f}")
print(f"   Mínimo: {idci.min():.2f}")
print(f"   Máximo: {idci.max():.2f}")

# DIAGNÓSTICO
print("\n🔍 Executando diagnóstico...")
is_ok = diagnose_series(idci, name="IDCI-VIX")

if not is_ok:
    print("\n⚠️ ATENÇÃO: Série apresenta problemas!")
    print("   Considere:")
    print("   - Coletar mais dados")
    print("   - Revisar variáveis de entrada")
    print("   - Usar modelos mais simples")

    resposta = input("\n   Continuar mesmo assim? (s/n): ")
    if resposta.lower() != 's':
        print("\n❌ Execução cancelada.")
        sys.exit(0)


# ============================================================================
# 5. TREINAMENTO DE MODELOS
# ============================================================================

print("\n" + "="*80)
print("TREINAMENTO DE MODELOS")
print("="*80)

print("\n4️⃣ Treinando modelos...")
print("   (Aguarde, isso pode levar alguns minutos)\n")

# Treina modelos individuais
pipeline.train_models()

print("\n✅ Modelos treinados:")
for model_name in pipeline.models_.keys():
    print(f"   ✓ {model_name}")


# ============================================================================
# 6. ANÁLISE DE MODELOS INDIVIDUAIS
# ============================================================================

print("\n" + "="*80)
print("ANÁLISE DE MODELOS INDIVIDUAIS")
print("="*80)

# ARIMA
if 'ARIMA' in pipeline.models_:
    arima_model = pipeline.models_['ARIMA']
    print(f"\n📈 ARIMA:")
    print(f"   Ordem: {arima_model.order_}")
    if hasattr(arima_model.model_fit_, 'aic'):
        print(f"   AIC: {arima_model.model_fit_.aic:.2f}")
        print(f"   BIC: {arima_model.model_fit_.bic:.2f}")

# Ridge
if 'Ridge' in pipeline.models_:
    ridge_model = pipeline.models_['Ridge']
    print(f"\n📈 Ridge Regression:")
    print(f"   Alpha: {ridge_model.alpha}")
    print(f"   Lags usados: {ridge_model.lags}")

# Random Forest
if 'RandomForest' in pipeline.models_:
    rf_model = pipeline.models_['RandomForest']
    print(f"\n📈 Random Forest:")
    print(f"   Árvores: {rf_model.n_estimators}")
    print(f"   Lags usados: {rf_model.lags}")


# ============================================================================
# 7. GERAR PREVISÕES
# ============================================================================

print("\n" + "="*80)
print("PREVISÕES")
print("="*80)

print("\n5️⃣ Gerando previsões para 12 meses...")
forecasts_df = pipeline.forecast_all()

print("\n📊 Previsões geradas:")
print(forecasts_df.round(2))

# Estatísticas
print("\n📈 Estatísticas por modelo:")
print(forecasts_df.describe().round(2))

# Variação prevista
print("\n📉 Variação prevista (atual → 12 meses):")
ultimo_valor = idci.iloc[-1]
for col in forecasts_df.columns:
    variacao = forecasts_df[col].iloc[-1] - ultimo_valor
    pct = (variacao / ultimo_valor) * 100
    print(f"   {col:20s}: {variacao:+.2f} ({pct:+.1f}%)")


# ============================================================================
# 8. VISUALIZAÇÕES
# ============================================================================

print("\n" + "="*80)
print("VISUALIZAÇÕES")
print("="*80)

print("\n📊 Gerando gráficos...")

viz = VitoriaVisualizer()

# Gráfico 1: IDCI-VIX histórico
fig1 = viz.plot_idci_vix(
    idci,
    title="IDCI-VIX Histórico - Mercado Imobiliário Vitória/ES"
)
plt.savefig('idci_vix_historico.png', dpi=150, bbox_inches='tight')
print("   ✓ Gráfico salvo: idci_vix_historico.png")
plt.close()

# Gráfico 2: Comparação de previsões
fig2 = viz.plot_forecasts_comparison(
    historical=idci,
    forecasts_df=forecasts_df,
    title="Comparação de Modelos de Previsão"
)
plt.savefig('comparacao_modelos.png', dpi=150, bbox_inches='tight')
print("   ✓ Gráfico salvo: comparacao_modelos.png")
plt.close()

# Gráfico 3: Previsão com intervalos (Ensemble)
if 'Ensemble' in forecasts_df.columns:
    # Simula intervalos de confiança (você pode calcular reais)
    lower = forecasts_df['Ensemble'] - 0.5
    upper = forecasts_df['Ensemble'] + 0.5

    fig3 = viz.plot_forecast_with_intervals(
        historical=idci,
        forecast=forecasts_df['Ensemble'],
        lower=lower,
        upper=upper,
        title="Previsão Ensemble com Intervalos de Confiança"
    )
    plt.savefig('previsao_intervalos.png', dpi=150, bbox_inches='tight')
    print("   ✓ Gráfico salvo: previsao_intervalos.png")
    plt.close()


# ============================================================================
# 9. SALVAR RESULTADOS
# ============================================================================

print("\n💾 Salvando resultados...")

# Previsões
forecasts_df.to_csv('previsoes_intermediario.csv')
print("   ✓ previsoes_intermediario.csv")

# IDCI-VIX
idci.to_csv('idci_vix_intermediario.csv', header=['IDCI_VIX'])
print("   ✓ idci_vix_intermediario.csv")

# Resultados Granger
granger_results.to_csv('granger_results.csv', index=False)
print("   ✓ granger_results.csv")

# Sumário em texto
with open('sumario_intermediario.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("SUMÁRIO - Sistema de Previsão Vitória/ES\n")
    f.write("="*80 + "\n\n")

    f.write(f"Período analisado: {df.index[0]} a {df.index[-1]}\n")
    f.write(f"Observações: {len(df)}\n")
    f.write(f"Variáveis originais: {len(df.columns)}\n")
    f.write(f"Variáveis selecionadas: {len(selected_vars)}\n\n")

    f.write("Variáveis selecionadas (Granger):\n")
    for var in selected_vars:
        f.write(f"  - {var}\n")

    f.write(f"\nIDCI-VIX:\n")
    f.write(f"  Média: {idci.mean():.2f}\n")
    f.write(f"  Desvio: {idci.std():.2f}\n")
    f.write(f"  Valor atual: {idci.iloc[-1]:.2f}\n")

    f.write(f"\nPrevisão 12 meses (Ensemble): {forecasts_df['Ensemble'].iloc[-1]:.2f}\n")

print("   ✓ sumario_intermediario.txt")


# ============================================================================
# 10. RECOMENDAÇÕES
# ============================================================================

print("\n" + "="*80)
print("RECOMENDAÇÕES")
print("="*80)

ultimo = idci.iloc[-1]
previsao = forecasts_df['Ensemble'].iloc[-1]
variacao = previsao - ultimo

print(f"\n📌 Situação atual: IDCI-VIX = {ultimo:.2f}/10")
print(f"🔮 Previsão 12m: IDCI-VIX = {previsao:.2f}/10")
print(f"📊 Variação esperada: {variacao:+.2f} pontos\n")

if variacao > 1.0:
    print("✅ CENÁRIO OTIMISTA")
    print("   → Mercado deve apresentar melhora significativa")
    print("   → Bom momento para investimentos")
elif variacao > 0.3:
    print("🟢 CENÁRIO POSITIVO")
    print("   → Mercado deve apresentar melhora moderada")
    print("   → Cenário favorável para negócios")
elif variacao > -0.3:
    print("🟡 CENÁRIO ESTÁVEL")
    print("   → Mercado deve permanecer estável")
    print("   → Manutenção do status quo")
elif variacao > -1.0:
    print("🟠 CENÁRIO NEGATIVO")
    print("   → Mercado deve apresentar leve deterioração")
    print("   → Cautela recomendada")
else:
    print("🔴 CENÁRIO PESSIMISTA")
    print("   → Mercado deve apresentar deterioração significativa")
    print("   → Atenção e prudência necessárias")


print("\n" + "="*80)
print("✅ EXEMPLO INTERMEDIÁRIO CONCLUÍDO!")
print("="*80)
print("\nArquivos gerados:")
print("  📄 previsoes_intermediario.csv")
print("  📄 idci_vix_intermediario.csv")
print("  📄 granger_results.csv")
print("  📄 sumario_intermediario.txt")
print("  📊 idci_vix_historico.png")
print("  📊 comparacao_modelos.png")
print("  📊 previsao_intervalos.png")
print("\nPróximo passo:")
print("  → Ver exemplo_avancado.py para análise completa com:")
print("     - Diagnóstico ARIMA detalhado")
print("     - Análise de regimes (Markov-Switching)")
print("     - Cenários quantílicos")
print("     - Cross-validation")
print("     - Ensemble customizado")
print()
