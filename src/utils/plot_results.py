"""
Script para gerar todas as visualizações dos resultados do pipeline.

Uso:
    from utils.plot_results import plot_all_results

    plot_all_results(results, output_dir='../data/processed/')
"""

import os
from typing import Dict, Optional
import pandas as pd
import matplotlib.pyplot as plt

from .visualization import VitoriaVisualizer


def plot_all_results(results: Dict,
                     output_dir: str = './output/',
                     show_plots: bool = True,
                     save_plots: bool = True) -> Dict:
    """
    Gera todas as visualizações dos resultados do pipeline.

    Parâmetros:
    -----------
    results : dicionário retornado por pipeline.run_full_pipeline()
    output_dir : diretório para salvar gráficos
    show_plots : se True, exibe gráficos
    save_plots : se True, salva gráficos em PNG

    Retorna:
    --------
    Dict com figuras matplotlib
    """
    # Cria diretório de saída
    if save_plots and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Inicializa visualizador
    viz = VitoriaVisualizer(figsize=(14, 7))

    # Extrai dados
    idci_vix = results['idci_vix']
    forecasts = results['forecasts']
    selected_vars = results['selected_vars']
    models = results['models']

    figures = {}

    print("\n" + "="*80)
    print("GERANDO VISUALIZAÇÕES")
    print("="*80)

    # 1. IDCI-VIX Histórico
    print("\n1. IDCI-VIX histórico...")
    save_path = os.path.join(output_dir, '01_idci_vix_historico.png') if save_plots else None

    fig1 = viz.plot_idci_vix(
        idci_vix,
        show_zones=True,
        save_path=save_path
    )
    figures['idci_vix'] = fig1

    if show_plots:
        plt.show()
    else:
        plt.close(fig1)

    # 2. Comparação de Modelos
    print("2. Comparação de modelos...")
    save_path = os.path.join(output_dir, '02_comparacao_modelos.png') if save_plots else None

    fig2 = viz.plot_forecasts_comparison(
        historical=idci_vix,
        forecasts_dict=forecasts,
        save_path=save_path
    )
    figures['comparison'] = fig2

    if show_plots:
        plt.show()
    else:
        plt.close(fig2)

    # 3. Intervalos de Confiança (se disponível)
    if 'quantile_quantiles' in forecasts:
        print("3. Intervalos de confiança...")
        save_path = os.path.join(output_dir, '03_intervalos_confianca.png') if save_plots else None

        quantiles = forecasts['quantile_quantiles']

        # Prepara índice futuro
        last_date = idci_vix.index[-1]
        if isinstance(last_date, pd.Timestamp):
            freq = pd.infer_freq(idci_vix.index) or 'MS'
            future_dates = pd.date_range(start=last_date, periods=13, freq=freq)[1:]
        else:
            future_dates = range(len(idci_vix), len(idci_vix) + 12)

        forecast_median = pd.Series(quantiles['q0.5'].values[:12], index=future_dates)
        forecast_lower = pd.Series(quantiles['q0.1'].values[:12], index=future_dates)
        forecast_upper = pd.Series(quantiles['q0.9'].values[:12], index=future_dates)

        fig3 = viz.plot_forecast_with_intervals(
            historical=idci_vix,
            forecast_median=forecast_median,
            forecast_lower=forecast_lower,
            forecast_upper=forecast_upper,
            save_path=save_path
        )
        figures['intervals'] = fig3

        if show_plots:
            plt.show()
        else:
            plt.close(fig3)

        # 4. Cenários
        print("4. Análise de cenários...")
        save_path = os.path.join(output_dir, '04_cenarios.png') if save_plots else None

        scenarios_df = quantiles.iloc[:12]

        fig4 = viz.plot_scenarios(
            historical=idci_vix,
            scenarios_df=scenarios_df,
            save_path=save_path
        )
        figures['scenarios'] = fig4

        if show_plots:
            plt.show()
        else:
            plt.close(fig4)

    # 5. Feature Importance (Random Forest)
    if 'random_forest' in models:
        print("5. Feature importance (Random Forest)...")
        save_path = os.path.join(output_dir, '05_feature_importance.png') if save_plots else None

        rf_model = models['random_forest']

        try:
            if hasattr(rf_model, 'models') and 1 in rf_model.models:
                model_h1 = rf_model.models[1]['model']
                feature_names = rf_model.models[1]['feature_names']

                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model_h1.feature_importances_
                }).sort_values('importance', ascending=False)

                fig5 = viz.plot_feature_importance(
                    importance_df,
                    top_k=15,
                    model_name='Random Forest (h=1)',
                    save_path=save_path
                )
                figures['feature_importance'] = fig5

                if show_plots:
                    plt.show()
                else:
                    plt.close(fig5)
        except Exception as e:
            print(f"   ⚠ Erro ao plotar feature importance: {e}")

    # 6. Regimes (Markov)
    if 'markov' in models:
        print("6. Análise de regimes (Markov-switching)...")
        save_path = os.path.join(output_dir, '06_regimes.png') if save_plots else None

        markov_model = models['markov']

        try:
            regime_probs = markov_model.get_regime_probabilities(smoothed=True)

            fig6 = viz.plot_regimes(
                data=idci_vix,
                regime_probs=regime_probs,
                threshold=0.7,
                save_path=save_path
            )
            figures['regimes'] = fig6

            if show_plots:
                plt.show()
            else:
                plt.close(fig6)
        except Exception as e:
            print(f"   ⚠ Erro ao plotar regimes: {e}")

    # 7. Treino vs Teste (ARIMA)
    if 'arima' in models:
        print("7. Validação ARIMA (treino vs teste)...")
        save_path = os.path.join(output_dir, '07_arima_validacao.png') if save_plots else None

        arima_model = models['arima']

        try:
            train_predictions = arima_model.get_insample_predictions()
            train_data = idci_vix.loc[train_predictions.index]

            # Simula split
            split_idx = len(idci_vix) - 12
            test_data = idci_vix.iloc[split_idx:]

            # Previsão (simplificado)
            if 'arima' in forecasts and 'forecast' in forecasts['arima'].columns:
                test_predictions = forecasts['arima']['forecast'].iloc[:len(test_data)]
                test_predictions.index = test_data.index

                fig7 = viz.plot_training_vs_prediction(
                    train_data=train_data,
                    train_predictions=train_predictions,
                    test_data=test_data,
                    test_predictions=test_predictions,
                    model_name='ARIMA',
                    save_path=save_path
                )
                figures['arima_validation'] = fig7

                if show_plots:
                    plt.show()
                else:
                    plt.close(fig7)
        except Exception as e:
            print(f"   ⚠ Erro ao plotar validação ARIMA: {e}")

    print("\n" + "="*80)
    print(f"✓ {len(figures)} gráficos gerados")

    if save_plots:
        print(f"✓ Gráficos salvos em: {output_dir}")

    print("="*80)

    return figures


def print_summary(results: Dict):
    """
    Imprime resumo executivo dos resultados.
    """
    idci_vix = results['idci_vix']
    forecasts = results['forecasts']
    selected_vars = results['selected_vars']
    models = results['models']

    print("\n" + "#"*80)
    print("# RESUMO EXECUTIVO - MERCADO IMOBILIÁRIO VITÓRIA/ES")
    print("#"*80)

    print(f"\n📊 SITUAÇÃO ATUAL")
    current = idci_vix.iloc[-1]
    print(f"  IDCI-VIX: {current:.2f}")
    print(f"  Data: {idci_vix.index[-1]}")

    if current > 7:
        status = "🔴 AQUECIMENTO FORTE"
    elif current > 5:
        status = "🟠 AQUECIMENTO MODERADO"
    elif current > 3:
        status = "🟡 ESTÁVEL"
    else:
        status = "🔵 RESFRIADO"

    print(f"  Status: {status}")

    print(f"\n🔮 PREVISÃO 12 MESES")
    ensemble_12m = results['ensemble']['forecast'].iloc[0]
    print(f"  Ensemble: {ensemble_12m:.2f}")
    print(f"  Variação: {((ensemble_12m/current - 1) * 100):.1f}%")

    if 'quantile_quantiles' in forecasts:
        q10 = forecasts['quantile_quantiles']['q0.1'].iloc[-1]
        q50 = forecasts['quantile_quantiles']['q0.5'].iloc[-1]
        q90 = forecasts['quantile_quantiles']['q0.9'].iloc[-1]

        print(f"\n📈 CENÁRIOS")
        print(f"  Pessimista (10%): {q10:.2f}")
        print(f"  Base (50%): {q50:.2f}")
        print(f"  Otimista (90%): {q90:.2f}")

    print(f"\n🎯 VARIÁVEIS SELECIONADAS")
    for i, var in enumerate(selected_vars, 1):
        print(f"  {i}. {var}")

    print(f"\n🤖 MODELOS TREINADOS ({len(models)})")
    for model in models.keys():
        print(f"  ✓ {model.upper()}")

    print("\n" + "#"*80)
