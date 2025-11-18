"""
EXEMPLO BÁSICO - Sistema de Previsão Vitória/ES
================================================

Nível: Iniciante
Tempo: 5 minutos
Objetivo: Executar pipeline completo com configuração padrão

Este exemplo mostra o uso mais simples do sistema:
- Carrega dados
- Executa pipeline automático
- Gera previsões 12 meses
- Salva resultados
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('src')

from pipeline import VitoriaForecastPipeline


# ============================================================================
# 1. PREPARAR DADOS DE EXEMPLO
# ============================================================================

print("="*80)
print("EXEMPLO BÁSICO - Sistema de Previsão Vitória/ES")
print("="*80)

# Dados sintéticos simulando 5 anos de dados mensais
np.random.seed(42)
dates = pd.date_range('2019-01-01', periods=60, freq='MS')

# Cria DataFrame com variáveis do mercado imobiliário
df = pd.DataFrame({
    'preco_m2': np.random.randn(60).cumsum() + 3000,
    'vendas': np.random.randn(60).cumsum() + 100,
    'lancamentos': np.random.randn(60).cumsum() + 50,
    'credito_imobiliario': np.random.randn(60).cumsum() + 5000,
    'taxa_juros': np.random.randn(60).cumsum() + 8,
    'pib_es': np.random.randn(60).cumsum() + 1000,
}, index=dates)

print("\n📊 Dados carregados:")
print(f"   Período: {df.index[0].strftime('%Y-%m')} a {df.index[-1].strftime('%Y-%m')}")
print(f"   Observações: {len(df)}")
print(f"   Variáveis: {len(df.columns)}")


# ============================================================================
# 2. CRIAR PIPELINE
# ============================================================================

print("\n🔧 Criando pipeline...")

# Pipeline com configuração padrão
pipeline = VitoriaForecastPipeline(
    max_vars=5,           # Seleciona até 5 variáveis
    forecast_horizon=12,  # Prevê 12 meses à frente
    verbose=True          # Mostra progresso
)

print("   ✓ Pipeline criado")


# ============================================================================
# 3. EXECUTAR PIPELINE COMPLETO
# ============================================================================

print("\n" + "="*80)
print("EXECUTANDO PIPELINE")
print("="*80)

# Etapa 1: Pré-processamento
print("\n1️⃣ Pré-processamento...")
pipeline.preprocess(df)
print("   ✓ Dados tornados estacionários")

# Etapa 2: Seleção de variáveis
print("\n2️⃣ Seleção de variáveis (Granger)...")
pipeline.select_variables()
print(f"   ✓ {len(pipeline.selected_vars_)} variáveis selecionadas")

# Etapa 3: Construção do IDCI-VIX
print("\n3️⃣ Construção do IDCI-VIX...")
idci = pipeline.build_index()
print(f"   ✓ Índice criado: {len(idci)} observações")
print(f"   Média: {idci.mean():.2f}, Desvio: {idci.std():.2f}")

# Etapa 4: Treinamento de modelos
print("\n4️⃣ Treinamento de modelos...")
print("   (Isso pode levar alguns minutos...)")
print()

pipeline.train_models()

print("\n   ✓ Modelos treinados com sucesso!")

# Etapa 5: Previsões
print("\n5️⃣ Gerando previsões...")
forecasts_df = pipeline.forecast_all()

print(f"   ✓ Previsões geradas: {len(forecasts_df)} meses")


# ============================================================================
# 4. VISUALIZAR RESULTADOS
# ============================================================================

print("\n" + "="*80)
print("RESULTADOS")
print("="*80)

print("\n📈 Previsões para os próximos 12 meses:")
print()
print(forecasts_df[['ARIMA', 'Ridge', 'RandomForest', 'Ensemble']].round(2))

print("\n📊 Estatísticas das previsões:")
print(forecasts_df[['ARIMA', 'Ridge', 'RandomForest', 'Ensemble']].describe().round(2))


# ============================================================================
# 5. SALVAR RESULTADOS
# ============================================================================

print("\n💾 Salvando resultados...")

# Salva previsões
forecasts_df.to_csv('previsoes_basico.csv')
print("   ✓ Previsões salvas em: previsoes_basico.csv")

# Salva IDCI-VIX histórico
idci.to_csv('idci_vix_historico.csv', header=['IDCI_VIX'])
print("   ✓ IDCI-VIX salvo em: idci_vix_historico.csv")


# ============================================================================
# 6. INTERPRETAÇÃO RÁPIDA
# ============================================================================

print("\n" + "="*80)
print("INTERPRETAÇÃO")
print("="*80)

ultimo_valor = idci.iloc[-1]
previsao_1m = forecasts_df['Ensemble'].iloc[0]
previsao_12m = forecasts_df['Ensemble'].iloc[-1]

print(f"\n📌 IDCI-VIX atual: {ultimo_valor:.2f}/10")
if ultimo_valor < 3:
    print("   → Mercado em condição FRACA")
elif ultimo_valor < 7:
    print("   → Mercado em condição MODERADA")
else:
    print("   → Mercado em condição FORTE")

print(f"\n🔮 Previsão 1 mês: {previsao_1m:.2f}/10")
print(f"🔮 Previsão 12 meses: {previsao_12m:.2f}/10")

variacao = previsao_12m - ultimo_valor
if variacao > 0.5:
    print(f"\n✅ Tendência de MELHORA (+{variacao:.2f})")
elif variacao < -0.5:
    print(f"\n⚠️ Tendência de PIORA ({variacao:.2f})")
else:
    print(f"\n➡️ Tendência de ESTABILIDADE ({variacao:+.2f})")


print("\n" + "="*80)
print("✅ EXEMPLO BÁSICO CONCLUÍDO!")
print("="*80)
print("\nPróximos passos:")
print("  → Ver exemplo_intermediario.py para customização")
print("  → Ver exemplo_avancado.py para análise completa")
print()
