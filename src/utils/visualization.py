"""
Módulo de visualização para análise e previsão do mercado imobiliário.

Funções para plotar:
- IDCI-VIX histórico
- Previsões vs valores reais
- Comparação entre modelos
- Intervalos de confiança
- Análise de resíduos
- Feature importance
- Regimes (Markov-switching)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# Configuração de estilo
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')


class VitoriaVisualizer:
    """
    Classe para visualização de resultados do pipeline de previsão.
    """

    def __init__(self, figsize: Tuple[int, int] = (14, 7)):
        """
        Parâmetros:
        -----------
        figsize : tupla com (largura, altura) padrão dos gráficos
        """
        self.figsize = figsize
        self.colors = sns.color_palette('husl', 10)

    def plot_idci_vix(self, idci_vix: pd.Series,
                     show_zones: bool = True,
                     title: str = None,
                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Plota evolução histórica do IDCI-VIX.

        Parâmetros:
        -----------
        idci_vix : Series com índice IDCI-VIX
        show_zones : se True, mostra zonas de interpretação
        title : título do gráfico (opcional)
        save_path : caminho para salvar figura (opcional)
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plota série
        ax.plot(idci_vix, linewidth=2.5, color='#2E86AB', label='IDCI-VIX')

        # Zonas de interpretação
        if show_zones:
            ax.axhspan(0, 3, alpha=0.15, color='blue', label='Resfriado')
            ax.axhspan(3, 5, alpha=0.15, color='gray', label='Estável')
            ax.axhspan(5, 7, alpha=0.15, color='orange', label='Aquecimento Moderado')
            ax.axhspan(7, 10, alpha=0.15, color='red', label='Aquecimento Forte')

            # Linhas de referência
            ax.axhline(y=5, color='black', linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(y=3, color='blue', linestyle='--', alpha=0.3, linewidth=0.8)
            ax.axhline(y=7, color='red', linestyle='--', alpha=0.3, linewidth=0.8)

        # Estatísticas
        mean_val = idci_vix.mean()
        current_val = idci_vix.iloc[-1]

        ax.axhline(y=mean_val, color='green', linestyle=':', alpha=0.6,
                  linewidth=1.5, label=f'Média histórica: {mean_val:.2f}')

        # Título e labels
        if title is None:
            title = 'IDCI-VIX - Índice de Condições do Mercado Imobiliário de Vitória/ES'

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('IDCI-VIX (0-10)', fontsize=12)
        ax.set_xlabel('Data', fontsize=12)

        # Grid e legenda
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', framealpha=0.9)

        # Anotação do valor atual
        ax.annotate(f'Atual: {current_val:.2f}',
                   xy=(idci_vix.index[-1], current_val),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_forecasts_comparison(self,
                                 historical: pd.Series,
                                 forecasts_dict: Dict[str, pd.DataFrame],
                                 actual_future: Optional[pd.Series] = None,
                                 title: str = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Compara previsões de múltiplos modelos.

        Parâmetros:
        -----------
        historical : série histórica
        forecasts_dict : {model_name: DataFrame com 'forecast'}
        actual_future : valores reais futuros (se disponível, para validação)
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Histórico
        ax.plot(historical, linewidth=2.5, color='black',
               label='Histórico', alpha=0.8)

        # Marca divisão entre histórico e previsão
        last_date = historical.index[-1]
        ax.axvline(x=last_date, color='red', linestyle='--',
                  alpha=0.5, linewidth=2, label='Início das previsões')

        # Previsões de cada modelo
        for i, (model_name, forecast_df) in enumerate(forecasts_dict.items()):
            if 'forecast' in forecast_df.columns:
                # Cria índice de datas futuras
                if isinstance(last_date, pd.Timestamp):
                    freq = pd.infer_freq(historical.index) or 'MS'
                    future_dates = pd.date_range(
                        start=last_date,
                        periods=len(forecast_df) + 1,
                        freq=freq
                    )[1:]
                else:
                    future_dates = range(
                        len(historical),
                        len(historical) + len(forecast_df)
                    )

                ax.plot(future_dates, forecast_df['forecast'].values,
                       marker='o', linewidth=2, alpha=0.7,
                       label=model_name.upper(), color=self.colors[i % len(self.colors)])

        # Valores reais futuros (se disponível)
        if actual_future is not None:
            ax.plot(actual_future, linewidth=2.5, color='green',
                   marker='s', label='Real (futuro)', alpha=0.8)

        # Título e labels
        if title is None:
            title = 'Comparação de Previsões - Múltiplos Modelos'

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('IDCI-VIX (0-10)', fontsize=12)
        ax.set_xlabel('Data', fontsize=12)

        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', framealpha=0.9, ncol=2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_forecast_with_intervals(self,
                                    historical: pd.Series,
                                    forecast_median: pd.Series,
                                    forecast_lower: pd.Series,
                                    forecast_upper: pd.Series,
                                    title: str = None,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Plota previsão com intervalos de confiança.

        Parâmetros:
        -----------
        historical : série histórica
        forecast_median : previsão mediana
        forecast_lower : limite inferior do IC
        forecast_upper : limite superior do IC
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Histórico
        ax.plot(historical.iloc[-24:], linewidth=2.5, color='black',
               label='Histórico', alpha=0.8)

        # Marca início das previsões
        last_date = historical.index[-1]
        ax.axvline(x=last_date, color='red', linestyle='--',
                  alpha=0.5, linewidth=2)

        # Previsão mediana
        ax.plot(forecast_median.index, forecast_median.values,
               linewidth=2.5, color='blue', marker='o',
               label='Previsão (mediana)', alpha=0.9)

        # Intervalo de confiança
        ax.fill_between(forecast_median.index,
                       forecast_lower.values,
                       forecast_upper.values,
                       alpha=0.3, color='blue',
                       label='Intervalo 80% (q0.1-q0.9)')

        # Título e labels
        if title is None:
            title = 'Previsão com Intervalos de Confiança'

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('IDCI-VIX (0-10)', fontsize=12)
        ax.set_xlabel('Data', fontsize=12)

        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', framealpha=0.9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_training_vs_prediction(self,
                                    train_data: pd.Series,
                                    train_predictions: pd.Series,
                                    test_data: pd.Series,
                                    test_predictions: pd.Series,
                                    model_name: str = 'Modelo',
                                    title: str = None,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Plota dados de treino e teste com previsões.

        Parâmetros:
        -----------
        train_data : dados reais de treino
        train_predictions : previsões no conjunto de treino
        test_data : dados reais de teste
        test_predictions : previsões no conjunto de teste
        model_name : nome do modelo
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1]*1.3))

        # Painel 1: Treino e Teste
        ax1.plot(train_data, linewidth=2, color='black', label='Treino (real)', alpha=0.7)
        ax1.plot(train_predictions, linewidth=1.5, color='blue',
                linestyle='--', label='Treino (predito)', alpha=0.7)

        ax1.plot(test_data, linewidth=2, color='green', label='Teste (real)', alpha=0.7)
        ax1.plot(test_predictions, linewidth=1.5, color='red',
                marker='o', label='Teste (predito)', alpha=0.7)

        # Marca divisão treino/teste
        if len(train_data) > 0:
            split_date = train_data.index[-1]
            ax1.axvline(x=split_date, color='orange', linestyle='--',
                       alpha=0.5, linewidth=2, label='Split treino/teste')

        if title is None:
            title = f'{model_name} - Treino vs Teste'

        ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax1.set_ylabel('IDCI-VIX', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', framealpha=0.9)

        # Painel 2: Resíduos
        train_residuals = train_data - train_predictions
        test_residuals = test_data - test_predictions

        ax2.scatter(train_data.index, train_residuals, alpha=0.5,
                   color='blue', label='Resíduos (treino)', s=30)
        ax2.scatter(test_data.index, test_residuals, alpha=0.7,
                   color='red', label='Resíduos (teste)', s=50, marker='D')

        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.axhline(y=train_residuals.std(), color='gray', linestyle='--', alpha=0.5)
        ax2.axhline(y=-train_residuals.std(), color='gray', linestyle='--', alpha=0.5)

        ax2.set_title('Análise de Resíduos', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Resíduo', fontsize=12)
        ax2.set_xlabel('Data', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', framealpha=0.9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_feature_importance(self,
                               importance_df: pd.DataFrame,
                               top_k: int = 20,
                               model_name: str = 'Modelo',
                               title: str = None,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plota importância das features.

        Parâmetros:
        -----------
        importance_df : DataFrame com colunas 'feature' e 'importance'
        top_k : número de features a mostrar
        model_name : nome do modelo
        """
        fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.3)))

        # Ordena e pega top-K
        importance_sorted = importance_df.nlargest(top_k, 'importance')

        # Plota barras horizontais
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_sorted)))

        ax.barh(range(len(importance_sorted)),
               importance_sorted['importance'].values,
               color=colors)

        ax.set_yticks(range(len(importance_sorted)))
        ax.set_yticklabels(importance_sorted['feature'].values)
        ax.invert_yaxis()

        if title is None:
            title = f'{model_name} - Importância das Features (Top {top_k})'

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Importância', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_regimes(self,
                    data: pd.Series,
                    regime_probs: pd.DataFrame,
                    threshold: float = 0.7,
                    title: str = None,
                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Plota séries com regimes identificados (Markov-switching).

        Parâmetros:
        -----------
        data : série temporal
        regime_probs : DataFrame com probabilidades de regime
        threshold : probabilidade mínima para considerar "em regime"
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1]*1.3),
                                       sharex=True)

        # Painel 1: Dados com regimes destacados
        ax1.plot(data, linewidth=2, color='black', label='IDCI-VIX', alpha=0.8)

        # Identifica períodos em cada regime
        n_regimes = regime_probs.shape[1]

        for regime in range(n_regimes):
            col = f'regime_{regime}'
            in_regime = regime_probs[col] > threshold

            # Pinta áreas
            if regime == 0:
                # Regime 0 = contração/crise
                color = 'red'
                label = 'Regime Baixo'
            else:
                # Regime 1 = expansão
                color = 'green'
                label = 'Regime Alto'

            # Pinta regiões
            for idx in range(len(in_regime)):
                if in_regime.iloc[idx]:
                    ax1.axvspan(regime_probs.index[idx],
                               regime_probs.index[min(idx+1, len(in_regime)-1)],
                               alpha=0.2, color=color)

        # Adiciona labels manualmente
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.2, label='Regime Baixo'),
            Patch(facecolor='green', alpha=0.2, label='Regime Alto')
        ]
        ax1.legend(handles=legend_elements, loc='best', framealpha=0.9)

        if title is None:
            title = 'Análise de Regimes (Markov-Switching)'

        ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax1.set_ylabel('IDCI-VIX', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Painel 2: Probabilidades de regime
        for regime in range(n_regimes):
            col = f'regime_{regime}'
            color = 'red' if regime == 0 else 'green'
            label = f'P(Regime {regime})'

            ax2.plot(regime_probs[col], linewidth=2, color=color,
                    label=label, alpha=0.7)

        ax2.axhline(y=threshold, color='black', linestyle='--',
                   alpha=0.5, label=f'Threshold ({threshold})')

        ax2.set_ylabel('Probabilidade', fontsize=12)
        ax2.set_xlabel('Data', fontsize=12)
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', framealpha=0.9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_scenarios(self,
                      historical: pd.Series,
                      scenarios_df: pd.DataFrame,
                      title: str = None,
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plota cenários de previsão (pessimista/base/otimista).

        Parâmetros:
        -----------
        historical : série histórica
        scenarios_df : DataFrame com colunas para cada quantil
                      Ex: 'q0.1', 'q0.5', 'q0.9'
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Histórico (últimos 24 períodos)
        ax.plot(historical.iloc[-24:], linewidth=2.5, color='black',
               label='Histórico', alpha=0.8)

        # Marca início das previsões
        last_date = historical.index[-1]
        ax.axvline(x=last_date, color='red', linestyle='--',
                  alpha=0.5, linewidth=2)

        # Cria índice futuro
        if isinstance(last_date, pd.Timestamp):
            freq = pd.infer_freq(historical.index) or 'MS'
            future_dates = pd.date_range(
                start=last_date,
                periods=len(scenarios_df) + 1,
                freq=freq
            )[1:]
        else:
            future_dates = range(
                len(historical),
                len(historical) + len(scenarios_df)
            )

        # Cenário base (mediana)
        if 'q0.5' in scenarios_df.columns:
            ax.plot(future_dates, scenarios_df['q0.5'].values,
                   linewidth=2.5, color='blue', marker='o',
                   label='Cenário Base (q0.5)', alpha=0.9)

        # Intervalo pessimista-otimista
        if 'q0.1' in scenarios_df.columns and 'q0.9' in scenarios_df.columns:
            ax.fill_between(future_dates,
                           scenarios_df['q0.1'].values,
                           scenarios_df['q0.9'].values,
                           alpha=0.3, color='blue',
                           label='IC 80% (q0.1-q0.9)')

            # Linhas dos extremos
            ax.plot(future_dates, scenarios_df['q0.1'].values,
                   linewidth=1.5, color='red', linestyle='--',
                   label='Cenário Pessimista (q0.1)', alpha=0.7)

            ax.plot(future_dates, scenarios_df['q0.9'].values,
                   linewidth=1.5, color='green', linestyle='--',
                   label='Cenário Otimista (q0.9)', alpha=0.7)

        if title is None:
            title = 'Cenários de Previsão - Análise de Risco'

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('IDCI-VIX (0-10)', fontsize=12)
        ax.set_xlabel('Data', fontsize=12)

        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', framealpha=0.9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_model_comparison_metrics(self,
                                     metrics_dict: Dict[str, pd.DataFrame],
                                     metric: str = 'rmse',
                                     title: str = None,
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Compara métricas de erro entre modelos por horizonte.

        Parâmetros:
        -----------
        metrics_dict : {model_name: DataFrame de métricas com 'horizon' e metric}
        metric : nome da métrica ('rmse', 'mae', 'mape')
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plota cada modelo
        for i, (model_name, metrics_df) in enumerate(metrics_dict.items()):
            if 'horizon' in metrics_df.columns and metric in metrics_df.columns:
                ax.plot(metrics_df['horizon'], metrics_df[metric],
                       marker='o', linewidth=2, label=model_name.upper(),
                       color=self.colors[i % len(self.colors)])

        if title is None:
            title = f'Comparação de Modelos - {metric.upper()} por Horizonte'

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_xlabel('Horizonte (meses)', fontsize=12)

        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', framealpha=0.9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig
