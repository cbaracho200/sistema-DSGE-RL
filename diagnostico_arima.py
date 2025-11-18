"""
Diagnóstico específico para problemas de convergência ARIMA.

Este script ajuda a entender POR QUÊ modelos ARIMA não convergem,
mesmo quando os dados parecem OK.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import warnings


def test_stationarity(series, name="Série"):
    """Testa estacionariedade com ADF e KPSS."""
    print(f"\n{'='*80}")
    print(f"TESTE DE ESTACIONARIEDADE: {name}")
    print(f"{'='*80}")

    # ADF Test
    print("\n1. Teste ADF (H0: série tem raiz unitária = não-estacionária)")
    try:
        adf_result = adfuller(series.dropna(), autolag='AIC')
        print(f"   Estatística ADF: {adf_result[0]:.4f}")
        print(f"   p-valor: {adf_result[1]:.4f}")
        print(f"   Valores críticos:")
        for key, value in adf_result[4].items():
            print(f"      {key}: {value:.4f}")

        if adf_result[1] < 0.05:
            print("   ✓ Série parece ESTACIONÁRIA (p < 0.05)")
        else:
            print("   ⚠ Série parece NÃO-ESTACIONÁRIA (p >= 0.05)")
    except Exception as e:
        print(f"   ❌ Erro no teste ADF: {e}")

    # KPSS Test
    print("\n2. Teste KPSS (H0: série é estacionária)")
    try:
        kpss_result = kpss(series.dropna(), regression='c', nlags='auto')
        print(f"   Estatística KPSS: {kpss_result[0]:.4f}")
        print(f"   p-valor: {kpss_result[1]:.4f}")
        print(f"   Valores críticos:")
        for key, value in kpss_result[3].items():
            print(f"      {key}: {value:.4f}")

        if kpss_result[1] >= 0.05:
            print("   ✓ Série parece ESTACIONÁRIA (p >= 0.05)")
        else:
            print("   ⚠ Série parece NÃO-ESTACIONÁRIA (p < 0.05)")
    except Exception as e:
        print(f"   ❌ Erro no teste KPSS: {e}")


def analyze_acf_pacf(series, name="Série", lags=20):
    """Analisa ACF e PACF para identificar possíveis ordens."""
    print(f"\n{'='*80}")
    print(f"ANÁLISE ACF/PACF: {name}")
    print(f"{'='*80}")

    series_clean = series.dropna()
    max_lags = min(lags, len(series_clean) // 2 - 1)

    try:
        # ACF
        acf_values = acf(series_clean, nlags=max_lags, fft=False)
        print(f"\n1. ACF (primeiros {min(10, max_lags)} lags):")
        for i, val in enumerate(acf_values[:min(10, max_lags)]):
            print(f"   Lag {i}: {val:7.4f} {'|' * int(abs(val) * 20)}")

        # PACF
        pacf_values = pacf(series_clean, nlags=max_lags, method='ywm')
        print(f"\n2. PACF (primeiros {min(10, max_lags)} lags):")
        for i, val in enumerate(pacf_values[:min(10, max_lags)]):
            print(f"   Lag {i}: {val:7.4f} {'|' * int(abs(val) * 20)}")

        # Sugestões
        print("\n3. SUGESTÕES BASEADAS EM ACF/PACF:")

        # ACF corta abruptamente? -> MA
        acf_significant = sum(abs(acf_values[1:6]) > 0.3)
        if acf_significant <= 2:
            print(f"   → ACF corta após {acf_significant} lags → sugere MA({acf_significant})")

        # PACF corta abruptamente? -> AR
        pacf_significant = sum(abs(pacf_values[1:6]) > 0.3)
        if pacf_significant <= 2:
            print(f"   → PACF corta após {pacf_significant} lags → sugere AR({pacf_significant})")

        # Ambos decaem lentamente? -> ARMA
        if acf_significant > 3 and pacf_significant > 3:
            print("   → Ambos decaem lentamente → sugere ARMA")

        # Autocorrelação muito baixa?
        if max(abs(acf_values[1:])) < 0.2:
            print("   ⚠ Autocorrelação muito baixa → série pode ser RUÍDO BRANCO")
            print("     (ARIMA não é apropriado para ruído branco)")

    except Exception as e:
        print(f"   ❌ Erro na análise ACF/PACF: {e}")


def test_specific_arima_models(series, name="Série"):
    """Testa modelos ARIMA específicos e captura erros detalhados."""
    print(f"\n{'='*80}")
    print(f"TESTE DE MODELOS ARIMA ESPECÍFICOS: {name}")
    print(f"{'='*80}")

    # Lista de modelos para testar
    test_models = [
        (0, 0, 0),  # Modelo trivial (baseline)
        (1, 0, 0),  # AR(1)
        (0, 0, 1),  # MA(1)
        (1, 0, 1),  # ARMA(1,1)
        (0, 1, 0),  # Random walk
        (1, 1, 0),  # ARIMA(1,1,0)
        (0, 1, 1),  # ARIMA(0,1,1)
        (1, 1, 1),  # ARIMA(1,1,1) - fallback
        (2, 1, 2),  # ARIMA(2,1,2)
    ]

    results = []

    for order in test_models:
        print(f"\nTestando ARIMA{order}...")

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                model = ARIMA(
                    series,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )

                fit = model.fit()

                # Sucesso!
                print(f"   ✓ Convergiu!")
                print(f"     AIC: {fit.aic:.2f}")
                print(f"     BIC: {fit.bic:.2f}")
                print(f"     Log-Likelihood: {fit.llf:.2f}")

                if len(w) > 0:
                    print(f"     ⚠ Avisos: {len(w)}")
                    for warning in w[:3]:  # Mostra até 3 avisos
                        print(f"       - {warning.message}")

                results.append({
                    'order': order,
                    'status': 'OK',
                    'aic': fit.aic,
                    'bic': fit.bic,
                    'llf': fit.llf
                })

        except Exception as e:
            print(f"   ❌ FALHOU: {type(e).__name__}")
            print(f"      Mensagem: {str(e)[:200]}")

            results.append({
                'order': order,
                'status': 'FALHOU',
                'error': str(e)[:100]
            })

    # Resumo
    print(f"\n{'='*80}")
    print("RESUMO DOS TESTES")
    print(f"{'='*80}")

    success_count = sum(1 for r in results if r['status'] == 'OK')
    print(f"\nModelos que convergiram: {success_count}/{len(results)}")

    if success_count > 0:
        print("\n✓ Modelos bem-sucedidos:")
        for r in results:
            if r['status'] == 'OK':
                print(f"   ARIMA{r['order']}: AIC={r['aic']:.2f}, BIC={r['bic']:.2f}")

    if success_count < len(results):
        print("\n❌ Modelos que falharam:")
        for r in results:
            if r['status'] == 'FALHOU':
                print(f"   ARIMA{r['order']}: {r['error']}")

    return results


def test_differencing(series, name="Série", max_d=2):
    """Testa diferentes níveis de diferenciação."""
    print(f"\n{'='*80}")
    print(f"TESTE DE DIFERENCIAÇÃO: {name}")
    print(f"{'='*80}")

    for d in range(max_d + 1):
        print(f"\nDiferenciação d={d}:")

        if d == 0:
            diff_series = series
        elif d == 1:
            diff_series = series.diff().dropna()
        else:
            diff_series = series.diff().diff().dropna()

        print(f"   Observações: {len(diff_series)}")
        print(f"   Média: {diff_series.mean():.4f}")
        print(f"   Desvio padrão: {diff_series.std():.4f}")
        print(f"   Variância: {diff_series.var():.4f}")

        # Testa estacionariedade
        try:
            adf_result = adfuller(diff_series.dropna(), autolag='AIC')
            print(f"   ADF p-valor: {adf_result[1]:.4f}", end="")
            if adf_result[1] < 0.05:
                print(" ✓ (estacionária)")
            else:
                print(" ⚠ (não-estacionária)")
        except Exception as e:
            print(f"   ❌ Erro no ADF: {e}")


def full_arima_diagnosis(series, name="Série"):
    """Diagnóstico completo para problemas ARIMA."""
    print(f"\n{'#'*80}")
    print(f"# DIAGNÓSTICO COMPLETO ARIMA")
    print(f"# Série: {name}")
    print(f"# Observações: {len(series)}")
    print(f"{'#'*80}")

    # 1. Estatísticas básicas
    print(f"\nESTATÍSTICAS BÁSICAS:")
    print(f"   Média: {series.mean():.4f}")
    print(f"   Mediana: {series.median():.4f}")
    print(f"   Desvio padrão: {series.std():.4f}")
    print(f"   Mínimo: {series.min():.4f}")
    print(f"   Máximo: {series.max():.4f}")

    # 2. Testes de estacionariedade
    test_stationarity(series, name)

    # 3. Análise ACF/PACF
    analyze_acf_pacf(series, name)

    # 4. Teste diferentes níveis de diferenciação
    test_differencing(series, name)

    # 5. Teste modelos específicos
    results = test_specific_arima_models(series, name)

    # 6. Recomendações finais
    print(f"\n{'='*80}")
    print("RECOMENDAÇÕES FINAIS")
    print(f"{'='*80}")

    success_count = sum(1 for r in results if r['status'] == 'OK')

    if success_count == 0:
        print("\n❌ NENHUM modelo ARIMA convergiu!")
        print("\nPossíveis causas:")
        print("   1. Série muito curta (< 50 observações recomendado)")
        print("   2. Série é ruído branco (sem autocorrelação)")
        print("   3. Série tem propriedades numéricas problemáticas")
        print("   4. Dados de entrada têm problemas de qualidade")
        print("\nSugestões:")
        print("   → Use modelos mais simples (Ridge, Lasso)")
        print("   → Colete mais dados")
        print("   → Revise o processo de construção do IDCI-VIX")
    elif success_count < len(results) // 2:
        print("\n⚠ Apenas alguns modelos convergiram.")
        print("\nSugestão:")
        print("   → Use um dos modelos que convergiu")
        print("   → Considere usar ensemble com outros métodos")
    else:
        print("\n✓ Maioria dos modelos convergiu.")
        print("\nSugestão:")
        print("   → Escolha o modelo com menor AIC/BIC")

        # Encontra melhor modelo
        best = min([r for r in results if r['status'] == 'OK'],
                   key=lambda x: x['aic'])
        print(f"   → Melhor modelo: ARIMA{best['order']} (AIC={best['aic']:.2f})")


if __name__ == "__main__":
    print("="*80)
    print("DIAGNÓSTICO ARIMA - Sistema de Previsão Vitória/ES")
    print("="*80)

    # Exemplo de uso
    print("\nPara usar este script:")
    print("\n1. No seu código Python:")
    print("   from diagnostico_arima import full_arima_diagnosis")
    print("   full_arima_diagnosis(sua_serie, name='Nome da Série')")
    print("\n2. Ou modifique este arquivo para carregar seus dados")
    print("\nExemplo:")
    print("""
    import pandas as pd
    from diagnostico_arima import full_arima_diagnosis

    # Carregue seus dados
    df = pd.read_csv('dados.csv', index_col=0, parse_dates=True)

    # Execute diagnóstico
    full_arima_diagnosis(df['sua_variavel'], name='Sua Variável')
    """)
