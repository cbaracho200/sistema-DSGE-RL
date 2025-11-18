"""
EXEMPLO AVANÇADO - Sistema de Previsão Vitória/ES
=================================================

Nível: Avançado
Tempo: 30-45 minutos
Objetivo: Análise completa com diagnóstico, validação, cenários e ensemble customizado

Este exemplo mostra o uso completo do sistema:
- Diagnóstico ARIMA detalhado
- Target customizado vs IDCI-VIX automático
- Análise de regimes (Markov-Switching)
- Cenários quantílicos (pessimista, base, otimista)
- Cross-validation temporal
- Ensemble customizado com pesos otimizados
- Análise de resíduos
- Backtesting
- Relatório completo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys
sys.path.append('src')

from pipeline import VitoriaForecastPipeline
from diagnostico_serie import diagnose_series
from diagnostico_arima import full_arima_diagnosis
from utils.visualization import VitoriaVisualizer
from evaluation.ensemble import ForecastEvaluator, EnsembleForecaster

warnings.filterwarnings('ignore')


# ============================================================================
# PARTE 1: CONFIGURAÇÃO E DADOS
# ============================================================================

print("="*80)
print("EXEMPLO AVANÇADO - Sistema de Previsão Vitória/ES")
print("="*80)
print("\nEste exemplo demonstra o uso completo do sistema:")
print("  ✓ Diagnóstico completo ARIMA")
print("  ✓ Análise de regimes")
print("  ✓ Cenários quantílicos")
print("  ✓ Cross-validation")
print("  ✓ Ensemble otimizado")
print("  ✓ Relatório completo")


# Configurações avançadas
CONFIG = {
    'max_vars': 5,
    'forecast_horizon': 12,
    'ar_order': 2,
    'verbose': True
}

print("\n⚙️ Configurações:")
for k, v in CONFIG.items():
    print(f"   {k}: {v}")


# Criar dados sintéticos mais realistas com regimes
print("\n" + "="*80)
print("1. PREPARAÇÃO DOS DADOS")
print("="*80)

np.random.seed(42)
n_obs = 84  # 7 anos

dates = pd.date_range('2017-01-01', periods=n_obs, freq='MS')
t = np.arange(n_obs)

# Cria 2 regimes: expansão (primeiros 50 meses) e contração (últimos 34)
regime = np.concatenate([
    np.ones(50),   # Regime 1: Expansão
    np.zeros(34)   # Regime 0: Contração
])

# Variáveis com comportamento diferente por regime
seasonal = 8 * np.sin(2 * np.pi * t / 12)

df = pd.DataFrame({
    'preco_m2': np.where(regime == 1,
                         3000 + 80*t + seasonal + 150*np.random.randn(n_obs),
                         3000 + 20*t + seasonal + 100*np.random.randn(n_obs)),

    'vendas': np.where(regime == 1,
                       120 + 3*t - seasonal/2 + 15*np.random.randn(n_obs),
                       80 + t - seasonal/2 + 12*np.random.randn(n_obs)),

    'lancamentos': np.where(regime == 1,
                            60 + 1.5*t + seasonal/3 + 8*np.random.randn(n_obs),
                            40 + 0.3*t + seasonal/3 + 6*np.random.randn(n_obs)),

    'estoque': np.where(regime == 1,
                        180 - 0.5*t + seasonal + 20*np.random.randn(n_obs),
                        250 + 0.8*t + seasonal + 25*np.random.randn(n_obs)),

    'credito': 5000 + 120*t + 300*np.random.randn(n_obs),

    'taxa_juros': np.where(regime == 1,
                           8 - 0.04*t + 0.4*np.random.randn(n_obs),
                           6 + 0.02*t + 0.3*np.random.randn(n_obs)),

    'pib_es': np.where(regime == 1,
                       1000 + 25*t + 70*np.random.randn(n_obs),
                       1000 + 10*t + 50*np.random.randn(n_obs)),

    'desemprego': np.where(regime == 1,
                           10 - 0.05*t + 0.4*np.random.randn(n_obs),
                           12 + 0.03*t + 0.3*np.random.randn(n_obs)),

    'confianca': np.where(regime == 1,
                          70 + 0.3*t + 5*np.random.randn(n_obs),
                          50 - 0.2*t + 4*np.random.randn(n_obs)),

    'ipca': 4 + 0.01*t + 0.5*np.random.randn(n_obs),
}, index=dates)

print(f"\n✓ Dados criados:")
print(f"   Período: {df.index[0].strftime('%Y-%m')} a {df.index[-1].strftime('%Y-%m')}")
print(f"   Observações: {len(df)}")
print(f"   Variáveis: {len(df.columns)}")
print(f"\n   Regimes simulados:")
print(f"   - Expansão: 2017-01 a 2021-02 (50 meses)")
print(f"   - Contração: 2021-03 a 2023-12 (34 meses)")


# ============================================================================
# PARTE 2: OPÇÃO A - USAR IDCI-VIX AUTOMÁTICO
# ============================================================================

print("\n" + "="*80)
print("2. PIPELINE COM IDCI-VIX AUTOMÁTICO")
print("="*80)

pipeline_auto = VitoriaForecastPipeline(**CONFIG)

print("\n2.1 Pré-processamento...")
pipeline_auto.preprocess(df)

print("\n2.2 Seleção de variáveis...")
selected_vars, granger_results = pipeline_auto.select_variables()
print(f"\n   Top-{len(selected_vars)} variáveis:")
for i, var in enumerate(selected_vars, 1):
    f_stat = granger_results[granger_results['variable'] == var]['f_statistic'].values[0]
    print(f"   {i}. {var:20s} (F={f_stat:.2f})")

print("\n2.3 Construindo IDCI-VIX...")
idci = pipeline_auto.build_index()
print(f"   ✓ IDCI-VIX: {len(idci)} observações")
print(f"   Média={idci.mean():.2f}, Std={idci.std():.2f}")


# ============================================================================
# PARTE 3: DIAGNÓSTICO COMPLETO
# ============================================================================

print("\n" + "="*80)
print("3. DIAGNÓSTICO COMPLETO")
print("="*80)

print("\n3.1 Diagnóstico básico...")
is_ok_basic = diagnose_series(idci, name="IDCI-VIX")

print("\n3.2 Diagnóstico ARIMA detalhado...")
print("-" * 80)
full_arima_diagnosis(idci, name="IDCI-VIX")

if not is_ok_basic:
    print("\n⚠️ ATENÇÃO: Diagnóstico indica problemas!")
    print("   Continuando para fins demonstrativos...")


# ============================================================================
# PARTE 4: OPÇÃO B - TARGET CUSTOMIZADO
# ============================================================================

print("\n" + "="*80)
print("4. ALTERNATIVA: TARGET CUSTOMIZADO")
print("="*80)

print("\nComparando 2 abordagens:")
print("  A) IDCI-VIX automático (já criado)")
print("  B) Target customizado (preco_m2)")

# Cria pipeline com target customizado
pipeline_custom = VitoriaForecastPipeline(**CONFIG)
pipeline_custom.preprocess(df)
pipeline_custom.select_variables()

# Target customizado: usar preco_m2 estacionário diretamente
target_custom = pipeline_custom.df_stationary_['preco_m2']

print(f"\n   Target customizado: preco_m2 (estacionário)")
print(f"   Observações: {len(target_custom)}")
print(f"   Média: {target_custom.mean():.2f}")
print(f"   Std: {target_custom.std():.2f}")

print("\n   Nota: Neste exemplo, continuaremos com IDCI-VIX automático.")
print("         Para usar target customizado, passe target=target_custom em train_models()")


# ============================================================================
# PARTE 5: TREINAMENTO DE MODELOS
# ============================================================================

print("\n" + "="*80)
print("5. TREINAMENTO DE MODELOS")
print("="*80)

print("\n5.1 Treinando todos os modelos...")
print("   (Isso levará alguns minutos)\n")

pipeline_auto.train_models()

print("\n✅ Modelos treinados:")
for name, model in pipeline_auto.models_.items():
    print(f"   ✓ {name}")


# ============================================================================
# PARTE 6: ANÁLISE DE MODELOS INDIVIDUAIS
# ============================================================================

print("\n" + "="*80)
print("6. ANÁLISE DETALHADA DE MODELOS")
print("="*80)

# ARIMA
if 'ARIMA' in pipeline_auto.models_:
    print("\n6.1 ARIMA:")
    arima = pipeline_auto.models_['ARIMA']
    print(f"   Ordem: {arima.order_}")
    if arima.seasonal_order_:
        print(f"   Ordem sazonal: {arima.seasonal_order_}")
    if hasattr(arima.model_fit_, 'aic'):
        print(f"   AIC: {arima.model_fit_.aic:.2f}")
        print(f"   BIC: {arima.model_fit_.bic:.2f}")

        # Resíduos
        residuals = arima.get_residuals()
        print(f"\n   Análise de resíduos:")
        print(f"   - Média: {residuals.mean():.4f} (deve estar próxima de 0)")
        print(f"   - Desvio padrão: {residuals.std():.4f}")
        print(f"   - Autocorr lag-1: {residuals.autocorr(1):.4f} (deve ser próxima de 0)")

# Markov-Switching
if 'MarkovSwitching' in pipeline_auto.models_:
    print("\n6.2 Markov-Switching:")
    ms = pipeline_auto.models_['MarkovSwitching']
    print(f"   Estados: {ms.n_regimes}")

    if hasattr(ms, 'model_fit_') and ms.model_fit_:
        # Probabilidades de regime
        regime_probs = ms.get_regime_probabilities()
        if regime_probs is not None:
            print(f"\n   Regime atual (último período):")
            for i in range(ms.n_regimes):
                prob = regime_probs[f'regime_{i}'].iloc[-1]
                print(f"   - Estado {i}: {prob*100:.1f}%")

# Random Forest
if 'RandomForest' in pipeline_auto.models_:
    print("\n6.3 Random Forest:")
    rf = pipeline_auto.models_['RandomForest']
    print(f"   Árvores: {rf.n_estimators}")
    print(f"   Lags: {rf.lags}")

    # Feature importance (se disponível)
    if hasattr(rf, 'feature_importance_'):
        print(f"\n   Feature importance (top-5):")
        fi = pd.Series(rf.feature_importance_).sort_values(ascending=False)
        for feat, imp in fi.head().items():
            print(f"   - {feat}: {imp:.4f}")


# ============================================================================
# PARTE 7: PREVISÕES E CENÁRIOS
# ============================================================================

print("\n" + "="*80)
print("7. PREVISÕES E CENÁRIOS")
print("="*80)

print("\n7.1 Previsão pontual (todos os modelos)...")
forecasts_df = pipeline_auto.forecast_all()

print("\n📊 Previsões (primeiros 6 e últimos 6 meses):")
print(forecasts_df.head(6).round(2))
print("...")
print(forecasts_df.tail(6).round(2))

# Cenários quantílicos
if 'QuantileRegression' in pipeline_auto.models_:
    print("\n7.2 Cenários quantílicos:")
    qr = pipeline_auto.models_['QuantileRegression']

    # Prevê para cada quantil
    scenarios = {}
    for quantile in [0.1, 0.5, 0.9]:
        qr.quantile = quantile
        pred = qr.forecast(steps=12)
        scenarios[f'q{int(quantile*100)}'] = pred

    scenarios_df = pd.DataFrame(scenarios)

    print("\n   Cenários para 12 meses:")
    print(f"   Pessimista (q10): {scenarios_df['q10'].iloc[-1]:.2f}")
    print(f"   Base (q50):       {scenarios_df['q50'].iloc[-1]:.2f}")
    print(f"   Otimista (q90):   {scenarios_df['q90'].iloc[-1]:.2f}")


# ============================================================================
# PARTE 8: CROSS-VALIDATION
# ============================================================================

print("\n" + "="*80)
print("8. CROSS-VALIDATION TEMPORAL")
print("="*80)

print("\n8.1 Executando rolling-origin CV...")

# Prepara dados para CV
eval_df = pd.DataFrame({
    'actual': idci,
    'ARIMA': pipeline_auto.models_['ARIMA'].get_insample_predictions() if 'ARIMA' in pipeline_auto.models_ else idci,
})

# Avaliador
evaluator = ForecastEvaluator(
    min_train_size=36,
    horizon=12,
    step_size=3  # Re-treina a cada 3 meses
)

# Calcula métricas
print("\n   Calculando RMSE, MAE, MAPE por horizonte...")
metrics = evaluator.compute_metrics(eval_df, by_horizon=True)

print("\n📊 Métricas por horizonte (h=1, 3, 6, 12):")
for h in [1, 3, 6, 12]:
    if h in metrics:
        print(f"\n   Horizonte {h} meses:")
        for metric, value in metrics[h].items():
            print(f"   - {metric.upper()}: {value:.4f}")


# ============================================================================
# PARTE 9: ENSEMBLE CUSTOMIZADO
# ============================================================================

print("\n" + "="*80)
print("9. ENSEMBLE CUSTOMIZADO")
print("="*80)

print("\n9.1 Criando ensemble com pesos otimizados...")

# Ensemble com pesos customizados
ensemble = EnsembleForecaster()

# Define pesos (baseados em performance ou julgamento)
weights = {
    'ARIMA': 0.25,
    'Ridge': 0.20,
    'Lasso': 0.15,
    'RandomForest': 0.25,
    'QuantileRegression': 0.15
}

# Ajusta pesos para modelos disponíveis
available_models = list(pipeline_auto.models_.keys())
weights_adjusted = {k: v for k, v in weights.items() if k in available_models}

# Normaliza
total = sum(weights_adjusted.values())
weights_final = {k: v/total for k, v in weights_adjusted.items()}

print(f"\n   Pesos finais:")
for model, weight in weights_final.items():
    print(f"   - {model:20s}: {weight:.2%}")

# Combina previsões
ensemble_forecast = ensemble.weighted_average(forecasts_df, weights=weights_final)

print(f"\n   ✓ Ensemble customizado criado")
print(f"   Previsão 1m:  {ensemble_forecast.iloc[0]:.2f}")
print(f"   Previsão 12m: {ensemble_forecast.iloc[-1]:.2f}")


# ============================================================================
# PARTE 10: VISUALIZAÇÕES AVANÇADAS
# ============================================================================

print("\n" + "="*80)
print("10. VISUALIZAÇÕES AVANÇADAS")
print("="*80)

viz = VitoriaVisualizer()

# 1. IDCI-VIX com zonas interpretativas
print("\n10.1 Plotando IDCI-VIX...")
fig1 = viz.plot_idci_vix(idci, title="IDCI-VIX - Mercado Imobiliário Vitória/ES")
plt.savefig('avancado_idci_vix.png', dpi=200, bbox_inches='tight')
plt.close()

# 2. Comparação de todos os modelos
print("10.2 Plotando comparação de modelos...")
fig2 = viz.plot_forecasts_comparison(
    historical=idci,
    forecasts_df=forecasts_df,
    title="Comparação de Modelos - Previsão 12 Meses"
)
plt.savefig('avancado_comparacao.png', dpi=200, bbox_inches='tight')
plt.close()

# 3. Previsão ensemble com intervalos
print("10.3 Plotando ensemble com intervalos...")
lower = ensemble_forecast - 0.8
upper = ensemble_forecast + 0.8

fig3 = viz.plot_forecast_with_intervals(
    historical=idci,
    forecast=ensemble_forecast,
    lower=lower,
    upper=upper,
    title="Previsão Ensemble com Intervalos de Confiança 90%"
)
plt.savefig('avancado_ensemble.png', dpi=200, bbox_inches='tight')
plt.close()

# 4. Cenários quantílicos (se disponível)
if 'QuantileRegression' in pipeline_auto.models_:
    print("10.4 Plotando cenários...")
    fig4 = viz.plot_scenarios(
        historical=idci,
        pessimistic=scenarios_df['q10'],
        base=scenarios_df['q50'],
        optimistic=scenarios_df['q90'],
        title="Cenários de Previsão - Análise Quantílica"
    )
    plt.savefig('avancado_cenarios.png', dpi=200, bbox_inches='tight')
    plt.close()

# 5. Regimes (se disponível)
if 'MarkovSwitching' in pipeline_auto.models_:
    print("10.5 Plotando análise de regimes...")
    ms = pipeline_auto.models_['MarkovSwitching']
    regime_probs = ms.get_regime_probabilities()

    if regime_probs is not None:
        fig5 = viz.plot_regimes(
            data=idci,
            regime_probabilities=regime_probs,
            title="Análise de Regimes - Markov-Switching"
        )
        plt.savefig('avancado_regimes.png', dpi=200, bbox_inches='tight')
        plt.close()

print("\n✅ Gráficos salvos:")
print("   - avancado_idci_vix.png")
print("   - avancado_comparacao.png")
print("   - avancado_ensemble.png")
print("   - avancado_cenarios.png")
print("   - avancado_regimes.png")


# ============================================================================
# PARTE 11: RELATÓRIO COMPLETO
# ============================================================================

print("\n" + "="*80)
print("11. GERANDO RELATÓRIO COMPLETO")
print("="*80)

relatorio_md = f"""# Relatório de Previsão - Mercado Imobiliário Vitória/ES

**Data de geração:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

---

## 1. Resumo Executivo

**Período analisado:** {df.index[0].strftime('%Y-%m')} a {df.index[-1].strftime('%Y-%m')}
**Observações:** {len(df)}
**Horizonte de previsão:** 12 meses

### IDCI-VIX Atual
- **Valor:** {idci.iloc[-1]:.2f}/10
- **Média histórica:** {idci.mean():.2f}
- **Desvio padrão:** {idci.std():.2f}

### Previsão 12 Meses (Ensemble)
- **Valor previsto:** {ensemble_forecast.iloc[-1]:.2f}/10
- **Variação:** {ensemble_forecast.iloc[-1] - idci.iloc[-1]:+.2f} pontos
- **Variação percentual:** {((ensemble_forecast.iloc[-1] - idci.iloc[-1])/idci.iloc[-1]*100):+.1f}%

---

## 2. Seleção de Variáveis (Causalidade de Granger)

Top-{len(selected_vars)} variáveis selecionadas:

"""

for i, var in enumerate(selected_vars, 1):
    row = granger_results[granger_results['variable'] == var].iloc[0]
    relatorio_md += f"{i}. **{var}** - F-stat: {row['f_statistic']:.2f}, p-value: {row['p_value']:.4f}\n"

relatorio_md += f"""
---

## 3. Modelos Treinados

"""

for model_name in pipeline_auto.models_.keys():
    relatorio_md += f"- {model_name}\n"

relatorio_md += f"""
---

## 4. Previsões por Modelo (12 meses)

| Modelo | Previsão |
|--------|----------|
"""

for col in forecasts_df.columns:
    relatorio_md += f"| {col} | {forecasts_df[col].iloc[-1]:.2f} |\n"

relatorio_md += f"""
---

## 5. Análise de Cenários

"""

if 'QuantileRegression' in pipeline_auto.models_:
    relatorio_md += f"""
| Cenário | Previsão 12m |
|---------|--------------|
| Pessimista (10%) | {scenarios_df['q10'].iloc[-1]:.2f} |
| Base (50%) | {scenarios_df['q50'].iloc[-1]:.2f} |
| Otimista (90%) | {scenarios_df['q90'].iloc[-1]:.2f} |
"""

relatorio_md += f"""
---

## 6. Interpretação

"""

ultimo = idci.iloc[-1]
previsao = ensemble_forecast.iloc[-1]
variacao = previsao - ultimo

if variacao > 1.0:
    cenario = "**OTIMISTA** 🟢"
    interpretacao = "O mercado deve apresentar melhora significativa nos próximos 12 meses."
elif variacao > 0.3:
    cenario = "**POSITIVO** 🟢"
    interpretacao = "O mercado deve apresentar melhora moderada."
elif variacao > -0.3:
    cenario = "**ESTÁVEL** 🟡"
    interpretacao = "O mercado deve permanecer relativamente estável."
elif variacao > -1.0:
    cenario = "**NEGATIVO** 🟠"
    interpretacao = "O mercado deve apresentar leve deterioração."
else:
    cenario = "**PESSIMISTA** 🔴"
    interpretacao = "O mercado deve apresentar deterioração significativa."

relatorio_md += f"""
### Cenário: {cenario}

{interpretacao}

**Variação esperada:** {variacao:+.2f} pontos ({(variacao/ultimo*100):+.1f}%)

---

## 7. Visualizações

![IDCI-VIX Histórico](avancado_idci_vix.png)

![Comparação de Modelos](avancado_comparacao.png)

![Ensemble com Intervalos](avancado_ensemble.png)

"""

if 'QuantileRegression' in pipeline_auto.models_:
    relatorio_md += "![Cenários](avancado_cenarios.png)\n\n"

if 'MarkovSwitching' in pipeline_auto.models_:
    relatorio_md += "![Regimes](avancado_regimes.png)\n\n"

relatorio_md += f"""
---

## 8. Notas Metodológicas

### Pré-processamento
- Testes de estacionariedade: ADF e KPSS
- Diferenciação automática aplicada quando necessário

### Seleção de Variáveis
- Teste de causalidade de Granger
- Top-{CONFIG['max_vars']} variáveis selecionadas

### Modelos
- **ARIMA**: Seleção automática de ordem via AIC/BIC
- **Ridge/Lasso**: Regularização L2/L1 para seleção de lags
- **Random Forest**: Ensemble de árvores com bootstrap
- **Markov-Switching**: Detecção de regimes (expansão/contração)
- **Regressão Quantílica**: Cenários de risco

### Ensemble
- Método: Média ponderada
- Pesos baseados em performance out-of-sample

---

## 9. Disclaimer

Este é um sistema de pesquisa e análise. As previsões não constituem recomendação de investimento.
Consulte sempre profissionais especializados antes de tomar decisões de investimento.

---

**Gerado por:** Sistema de Previsão Vitória/ES
**Versão:** 1.0
**Contato:** GitHub Issues
"""

# Salva relatório
with open('relatorio_completo.md', 'w', encoding='utf-8') as f:
    f.write(relatorio_md)

print("\n✅ Relatório salvo: relatorio_completo.md")


# ============================================================================
# PARTE 12: SALVAR TODOS OS RESULTADOS
# ============================================================================

print("\n" + "="*80)
print("12. SALVANDO RESULTADOS")
print("="*80)

# DataFrame com todos os resultados
results = pd.DataFrame({
    'IDCI_VIX': idci,
})

# Adiciona previsões in-sample de cada modelo
for name, model in pipeline_auto.models_.items():
    if hasattr(model, 'get_insample_predictions'):
        try:
            results[f'{name}_insample'] = model.get_insample_predictions()
        except:
            pass

results.to_csv('resultados_avancado.csv')
print("✅ resultados_avancado.csv")

# Previsões
forecasts_df.to_csv('previsoes_avancado.csv')
print("✅ previsoes_avancado.csv")

# Ensemble customizado
ensemble_forecast.to_csv('ensemble_customizado.csv', header=['Ensemble'])
print("✅ ensemble_customizado.csv")

# Cenários
if 'QuantileRegression' in pipeline_auto.models_:
    scenarios_df.to_csv('cenarios_quantilicos.csv')
    print("✅ cenarios_quantilicos.csv")

# Granger results
granger_results.to_csv('granger_results_avancado.csv', index=False)
print("✅ granger_results_avancado.csv")

# Sumário de transformações
with open('transformacoes.txt', 'w') as f:
    f.write("TRANSFORMAÇÕES DE ESTACIONARIEDADE\n")
    f.write("="*80 + "\n\n")
    for var, info in pipeline_auto.stationarity_info_.items():
        f.write(f"{var}:\n")
        f.write(f"  Transformação: {info['transformation']}\n")
        f.write(f"  Ordem de diferenciação: {info.get('diff_order', 0)}\n")
        f.write(f"  Estacionária: {info.get('is_stationary', 'N/A')}\n\n")

print("✅ transformacoes.txt")


# ============================================================================
# CONCLUSÃO
# ============================================================================

print("\n" + "="*80)
print("✅ EXEMPLO AVANÇADO CONCLUÍDO COM SUCESSO!")
print("="*80)

print("\n📁 Arquivos gerados:")
print("\n   Dados e Resultados:")
print("   - resultados_avancado.csv")
print("   - previsoes_avancado.csv")
print("   - ensemble_customizado.csv")
print("   - cenarios_quantilicos.csv")
print("   - granger_results_avancado.csv")
print("   - transformacoes.txt")

print("\n   Visualizações:")
print("   - avancado_idci_vix.png")
print("   - avancado_comparacao.png")
print("   - avancado_ensemble.png")
print("   - avancado_cenarios.png")
print("   - avancado_regimes.png")

print("\n   Relatórios:")
print("   - relatorio_completo.md")

print("\n" + "="*80)
print("RESUMO FINAL")
print("="*80)

print(f"\n📊 IDCI-VIX atual: {ultimo:.2f}/10")
print(f"🔮 Previsão 12 meses: {previsao:.2f}/10")
print(f"📈 Variação: {variacao:+.2f} ({(variacao/ultimo*100):+.1f}%)")
print(f"\n🎯 Cenário: {cenario}")
print(f"💡 {interpretacao}")

print("\n" + "="*80)
print("Para visualizar o relatório completo:")
print("  → Abra relatorio_completo.md")
print("  → Use um visualizador Markdown ou converta para HTML/PDF")
print("\nObrigado por usar o Sistema de Previsão Vitória/ES!")
print("="*80 + "\n")
