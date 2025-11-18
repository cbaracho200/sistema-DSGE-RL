"""
Diagnóstico detalhado para problemas com ARIMA/SARIMA.

Este script identifica por que os modelos não convergem.
"""

import sys
sys.path.append('src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def diagnose_series(series: pd.Series, name: str = "Série"):
    """Diagnostica problemas em uma série temporal."""

    print("="*80)
    print(f"DIAGNÓSTICO: {name}")
    print("="*80)

    # 1. Informações básicas
    print(f"\n1. INFORMAÇÕES BÁSICAS")
    print(f"   Tipo: {type(series)}")
    print(f"   Nome: {series.name}")
    print(f"   Tamanho: {len(series)}")
    print(f"   Tipo de dados: {series.dtype}")

    # 2. Valores faltantes
    print(f"\n2. VALORES FALTANTES")
    n_nan = series.isna().sum()
    pct_nan = (n_nan / len(series)) * 100
    print(f"   NaN: {n_nan} ({pct_nan:.1f}%)")

    if n_nan > 0:
        print(f"   ⚠ ATENÇÃO: Série tem {pct_nan:.1f}% de valores faltantes!")
        if pct_nan > 50:
            print(f"   ❌ PROBLEMA GRAVE: Mais de 50% são NaN!")

    # 3. Estatísticas descritivas
    print(f"\n3. ESTATÍSTICAS")
    series_clean = series.dropna()

    if len(series_clean) == 0:
        print(f"   ❌ ERRO FATAL: Série vazia após remover NaN!")
        return False

    print(f"   Mínimo: {series_clean.min():.6f}")
    print(f"   Máximo: {series_clean.max():.6f}")
    print(f"   Média: {series_clean.mean():.6f}")
    print(f"   Mediana: {series_clean.median():.6f}")
    print(f"   Desvio padrão: {series_clean.std():.6f}")
    print(f"   Variância: {series_clean.var():.6f}")

    # 4. Verifica problemas comuns
    print(f"\n4. VERIFICAÇÃO DE PROBLEMAS")

    has_problems = False

    # 4.1. Série constante
    if series_clean.std() < 1e-10:
        print(f"   ❌ PROBLEMA: Série é praticamente CONSTANTE!")
        print(f"      Todos os valores são ~{series_clean.mean():.6f}")
        print(f"      ARIMA não pode ser ajustado em série constante.")
        has_problems = True
    else:
        print(f"   ✓ Variabilidade OK (std={series_clean.std():.6f})")

    # 4.2. Valores infinitos
    n_inf = np.isinf(series_clean).sum()
    if n_inf > 0:
        print(f"   ❌ PROBLEMA: {n_inf} valores infinitos!")
        has_problems = True
    else:
        print(f"   ✓ Sem infinitos")

    # 4.3. Valores únicos
    n_unique = series_clean.nunique()
    pct_unique = (n_unique / len(series_clean)) * 100
    print(f"   Valores únicos: {n_unique} ({pct_unique:.1f}%)")

    if n_unique < 5:
        print(f"   ⚠ ATENÇÃO: Poucos valores únicos ({n_unique})")
        print(f"      Série pode ser muito discretizada")
        has_problems = True

    # 4.4. Autocorrelação
    try:
        acf_lag1 = series_clean.autocorr(lag=1)
        print(f"   Autocorrelação (lag 1): {acf_lag1:.4f}")

        if abs(acf_lag1) < 0.01:
            print(f"   ⚠ ATENÇÃO: Autocorrelação muito baixa - série pode ser ruído branco")
    except:
        print(f"   ⚠ Não foi possível calcular autocorrelação")

    # 4.5. Tamanho
    if len(series_clean) < 30:
        print(f"   ⚠ ATENÇÃO: Série muito curta ({len(series_clean)} obs)")
        print(f"      Recomendado: mínimo 50 observações para ARIMA")
        has_problems = True
    else:
        print(f"   ✓ Tamanho adequado ({len(series_clean)} obs)")

    # 5. Visualização
    print(f"\n5. PRIMEIROS E ÚLTIMOS VALORES")
    print(f"   Primeiros 5:")
    for i, val in enumerate(series_clean.head(5)):
        print(f"     [{i}] {val:.6f}")

    print(f"   Últimos 5:")
    for i, val in enumerate(series_clean.tail(5)):
        idx = len(series_clean) - 5 + i
        print(f"     [{idx}] {val:.6f}")

    # 6. Conclusão
    print(f"\n6. DIAGNÓSTICO FINAL")
    if has_problems:
        print(f"   ❌ PROBLEMAS DETECTADOS - série pode não funcionar com ARIMA")
        return False
    else:
        print(f"   ✓ Série parece OK para modelagem")
        return True


# ============================================================================
# TESTE COM PIPELINE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# TESTE COMPLETO DE DIAGNÓSTICO")
    print("#"*80)

    # Dados sintéticos
    np.random.seed(42)
    dates = pd.date_range('2015-01-01', periods=60, freq='MS')

    df_test = pd.DataFrame({
        'var1': np.random.randn(60).cumsum() + 10,
        'var2': np.random.randn(60).cumsum() + 5,
        'var3': np.random.randn(60).cumsum() + 3,
        'var4': np.random.randn(60).cumsum() + 7,
    }, index=dates)

    print("\n📊 Dados de teste criados")

    # Pipeline
    from pipeline import VitoriaForecastPipeline

    pipeline = VitoriaForecastPipeline(max_vars=3, forecast_horizon=6, verbose=False)

    # Pré-processa
    print("\n1. Pré-processamento...")
    pipeline.preprocess(df_test)

    # Seleciona variáveis
    print("2. Seleção de variáveis...")
    pipeline.select_variables()

    # Constrói índice
    print("3. Construindo IDCI-VIX...\n")
    idci = pipeline.build_index()

    # DIAGNÓSTICO DO IDCI-VIX
    is_ok = diagnose_series(idci, "IDCI-VIX")

    print("\n" + "="*80)
    print("TENTANDO ARIMA")
    print("="*80)

    if is_ok:
        print("\nSérie passou no diagnóstico. Tentando ARIMA...")

        try:
            from forecasting.arima_models import ARIMAForecaster

            model = ARIMAForecaster()
            model.fit(idci, auto=True, verbose=True)

            print("\n✅ ARIMA ajustado com sucesso!")
            print(f"   Ordem: {model.order_}")
            print(f"   AIC: {model.model_fit_.aic:.2f}")

        except Exception as e:
            print(f"\n❌ ERRO ao ajustar ARIMA: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠ Série NÃO passou no diagnóstico.")
        print("   Corrija os problemas antes de usar ARIMA.")

    print("\n" + "#"*80)
    print("# DIAGNÓSTICO COMPLETO")
    print("#"*80)
