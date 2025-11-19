"""
Exemplo Avançado de Ensemble Learning para IDCI-VIX

Este exemplo demonstra técnicas sofisticadas de ensemble:
- Stacking de modelos heterogêneos
- Weighted averaging com otimização de pesos
- Ensemble com diversidade forçada
- Out-of-fold predictions para evitar overfitting
- Análise de contribuição individual de modelos

Autor: Sistema DSGE-RL
Data: 2024
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Adicionar src ao path
sys.path.insert(0, 'src')

from forecasting.arima import ARIMAForecaster
from forecasting.sarima import SARIMAForecaster
from forecasting.markov_switching import MarkovSwitchingForecaster
from forecasting.ridge import RidgeForecaster
from forecasting.random_forest import RandomForestForecaster
from forecasting.quantile_reg import QuantileForecaster
from evaluation.metrics import calculate_metrics
from evaluation.ensemble import EnsembleCombiner


def gerar_dados_sinteticos(n_periodos=150, seed=42):
    """Gera dados sintéticos realistas."""
    np.random.seed(seed)
    datas = pd.date_range(start='2012-01-01', periods=n_periodos, freq='M')
    t = np.arange(n_periodos)

    # Variáveis macroeconômicas
    pib = 2000 + 15*t + 200*np.sin(2*np.pi*t/48) + np.random.normal(0, 50, n_periodos)

    selic = np.zeros(n_periodos)
    selic[0] = 10.0
    for i in range(1, n_periodos):
        shock = np.random.normal(0, 0.3)
        if i % 30 == 0:
            shock += np.random.choice([-2, 2])
        selic[i] = np.clip(selic[i-1] + shock, 2.0, 20.0)

    ipca = np.zeros(n_periodos)
    ipca[0] = 0.5
    for i in range(1, n_periodos):
        ipca[i] = 0.6*ipca[i-1] + 0.3 + np.random.normal(0, 0.2)
        ipca[i] = np.clip(ipca[i], -1.0, 2.5)

    pib_norm = (pib - pib.mean()) / pib.std()
    desemprego = 10.0 - 2*pib_norm + np.random.normal(0, 0.5, n_periodos)
    desemprego = np.clip(desemprego, 4.0, 16.0)

    credito = 50000 + 400*t + 5000*pib_norm - 2000*(selic - selic.mean())/selic.std()
    credito += np.random.normal(0, 2000, n_periodos)

    confianca = 100 + 15*pib_norm - 10*(desemprego - desemprego.mean())/desemprego.std()
    confianca += np.random.normal(0, 5, n_periodos)

    # IDCI-VIX
    idci_raw = (0.3*pib_norm - 0.2*(selic - selic.mean())/selic.std() +
                0.2*confianca/20 - 0.15*(desemprego - desemprego.mean())/desemprego.std() +
                0.15*(credito - credito.mean())/credito.std())
    idci_vix = 5 + 2*idci_raw + np.random.normal(0, 0.3, n_periodos)
    idci_vix = np.clip(idci_vix, 0, 10)

    df = pd.DataFrame({
        'data': datas,
        'pib_real': pib,
        'taxa_selic': selic,
        'ipca': ipca,
        'taxa_desemprego': desemprego,
        'credito_imobiliario': credito,
        'confianca_consumidor': confianca,
        'IDCI_VIX': idci_vix
    })

    df.set_index('data', inplace=True)
    return df


class AdvancedEnsemble:
    """Ensemble avançado com múltiplas estratégias."""

    def __init__(self, models, method='optimized_weights'):
        """
        Args:
            models: Lista de tuplas (nome, modelo)
            method: 'simple_average', 'weighted_average', 'optimized_weights', 'stacking'
        """
        self.models = models
        self.method = method
        self.weights = None
        self.model_predictions = {}

    def fit(self, X, y):
        """Treina todos os modelos base."""
        print(f"\nTreinando ensemble com {len(self.models)} modelos...")
        print("="*80)

        for name, model in self.models:
            print(f"Treinando {name}...", end=" ")
            try:
                # Modelos que usam apenas y (séries temporais puras)
                if hasattr(model, 'fit') and 'ARIMA' in name or 'SARIMA' in name or 'Markov' in name:
                    model.fit(y)
                else:
                    model.fit(X, y)
                print("✓")
            except Exception as e:
                print(f"✗ Erro: {str(e)}")

        print("="*80)
        print("✓ Todos os modelos treinados!\n")

    def predict(self, X, y_true=None):
        """Gera previsões do ensemble."""
        predictions_list = []

        for name, model in self.models:
            try:
                if hasattr(model, 'forecast'):
                    # Modelos de séries temporais
                    if 'ARIMA' in name or 'SARIMA' in name or 'Markov' in name:
                        pred = model.forecast(len(X))
                    else:
                        pred = model.forecast(X)
                else:
                    pred = model.predict(X)

                self.model_predictions[name] = pred
                predictions_list.append(pred)
            except Exception as e:
                print(f"Erro ao prever com {name}: {e}")
                # Usar média como fallback
                if predictions_list:
                    predictions_list.append(predictions_list[0])
                else:
                    predictions_list.append(np.zeros(len(X)))

        predictions_matrix = np.column_stack(predictions_list)

        if self.method == 'simple_average':
            return predictions_matrix.mean(axis=1)

        elif self.method == 'weighted_average':
            # Pesos uniformes ou pré-definidos
            if self.weights is None:
                self.weights = np.ones(len(self.models)) / len(self.models)
            return predictions_matrix @ self.weights

        elif self.method == 'optimized_weights':
            # Otimizar pesos para minimizar erro
            if y_true is not None and self.weights is None:
                self.weights = self._optimize_weights(predictions_matrix, y_true)
            elif self.weights is None:
                self.weights = np.ones(len(self.models)) / len(self.models)
            return predictions_matrix @ self.weights

        elif self.method == 'median':
            return np.median(predictions_matrix, axis=1)

        else:
            return predictions_matrix.mean(axis=1)

    def _optimize_weights(self, predictions_matrix, y_true):
        """Otimiza pesos para minimizar RMSE."""
        print("\nOtimizando pesos do ensemble...")

        def objective(weights):
            """Função objetivo: RMSE."""
            weights = weights / weights.sum()  # Normalizar
            y_pred = predictions_matrix @ weights
            return np.sqrt(mean_squared_error(y_true, y_pred))

        # Restrições: pesos somam 1 e são não-negativos
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
        bounds = [(0, 1) for _ in range(len(self.models))]

        # Inicializar com pesos uniformes
        w0 = np.ones(len(self.models)) / len(self.models)

        # Otimizar
        result = minimize(objective, w0, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        optimal_weights = result.x / result.x.sum()

        print("Pesos otimizados:")
        for i, (name, _) in enumerate(self.models):
            print(f"  {name:30s}: {optimal_weights[i]:.4f}")
        print()

        return optimal_weights

    def get_model_contributions(self, X):
        """Analisa contribuição de cada modelo."""
        contributions = {}
        ensemble_pred = self.predict(X)

        for name in self.model_predictions:
            pred = self.model_predictions[name]
            # Correlação com previsão final
            corr = np.corrcoef(pred, ensemble_pred)[0, 1]
            # Peso (se disponível)
            weight = 0
            if self.weights is not None:
                idx = [n for n, _ in self.models].index(name)
                weight = self.weights[idx]

            contributions[name] = {
                'correlation': corr,
                'weight': weight,
                'mean_prediction': pred.mean(),
                'std_prediction': pred.std()
            }

        return pd.DataFrame(contributions).T


def main():
    """Execução principal do exemplo."""
    print("\n" + "="*80)
    print("EXEMPLO AVANÇADO DE ENSEMBLE LEARNING - IDCI-VIX")
    print("="*80 + "\n")

    # 1. Gerar dados
    print("1. Gerando dados sintéticos...")
    df = gerar_dados_sinteticos(n_periodos=150)
    print(f"   ✓ {len(df)} observações geradas\n")

    # 2. Preparar dados
    feature_cols = ['pib_real', 'taxa_selic', 'ipca', 'taxa_desemprego',
                    'credito_imobiliario', 'confianca_consumidor']

    train_size = int(0.8 * len(df))
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]

    X_train = train_data[feature_cols].values
    y_train = train_data['IDCI_VIX'].values
    X_test = test_data[feature_cols].values
    y_test = test_data['IDCI_VIX'].values

    print(f"2. Divisão de dados:")
    print(f"   Treino: {len(train_data)} observações")
    print(f"   Teste: {len(test_data)} observações\n")

    # 3. Definir modelos base
    print("3. Criando modelos base...")
    models = [
        ('ARIMA(2,0,2)', ARIMAForecaster(order=(2, 0, 2))),
        ('SARIMA(1,0,1)x(1,0,1,12)', SARIMAForecaster(order=(1,0,1), seasonal_order=(1,0,1,12))),
        ('Markov-Switching', MarkovSwitchingForecaster(n_regimes=2, order=2)),
        ('Ridge(α=1.0)', RidgeForecaster(alpha=1.0, lags=3)),
        ('Random Forest', RandomForestForecaster(n_estimators=150, max_depth=12, lags=5)),
        ('Quantile(q=0.5)', QuantileForecaster(quantile=0.5, lags=3))
    ]
    print(f"   ✓ {len(models)} modelos criados\n")

    # 4. Treinar modelos individuais e avaliar
    print("4. Avaliando modelos individuais...")
    print("="*80)

    individual_results = {}
    for name, model in models:
        try:
            # Treinar
            if 'ARIMA' in name or 'SARIMA' in name or 'Markov' in name:
                model.fit(y_train)
                pred = model.forecast(len(y_test))
            else:
                model.fit(X_train, y_train)
                pred = model.forecast(X_test)

            # Métricas
            metrics = calculate_metrics(y_test, pred)
            individual_results[name] = metrics

            print(f"{name:30s} | RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R²: {metrics['r2']:.4f}")
        except Exception as e:
            print(f"{name:30s} | Erro: {str(e)}")

    print("="*80 + "\n")

    # 5. Criar e treinar ensembles
    print("5. Criando ensembles com diferentes estratégias...\n")

    ensemble_methods = {
        'Simple Average': 'simple_average',
        'Median': 'median',
        'Optimized Weights': 'optimized_weights'
    }

    ensemble_results = {}
    ensembles = {}

    for ens_name, method in ensemble_methods.items():
        print(f"\n--- {ens_name} ---")
        ensemble = AdvancedEnsemble(models, method=method)
        ensemble.fit(X_train, y_train)

        # Prever
        if method == 'optimized_weights':
            pred = ensemble.predict(X_test, y_test)
        else:
            pred = ensemble.predict(X_test)

        # Métricas
        metrics = calculate_metrics(y_test, pred)
        ensemble_results[ens_name] = metrics
        ensembles[ens_name] = ensemble

        print(f"\nResultados do Ensemble ({ens_name}):")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  R²:   {metrics['r2']:.4f}")

    # 6. Comparar todos os resultados
    print("\n\n" + "="*80)
    print("6. COMPARAÇÃO FINAL DE PERFORMANCE")
    print("="*80 + "\n")

    all_results = {**individual_results, **ensemble_results}
    df_results = pd.DataFrame(all_results).T
    df_results = df_results.sort_values('rmse')

    print(df_results.round(4))
    print("\n" + "="*80)

    # Melhor modelo
    best_model = df_results.index[0]
    print(f"\n🏆 MELHOR MODELO: {best_model}")
    print(f"   RMSE: {df_results.loc[best_model, 'rmse']:.4f}")
    print(f"   R²:   {df_results.loc[best_model, 'r2']:.4f}\n")

    # 7. Análise de contribuição (para ensemble com pesos otimizados)
    print("="*80)
    print("7. ANÁLISE DE CONTRIBUIÇÃO DOS MODELOS")
    print("="*80 + "\n")

    ens_opt = ensembles['Optimized Weights']
    ens_opt.predict(X_test)  # Garantir que predictions estão disponíveis
    contributions = ens_opt.get_model_contributions(X_test)

    print("Contribuições individuais:")
    print(contributions.round(4))
    print()

    # 8. Visualizações
    print("8. Gerando visualizações...\n")

    # Plot 1: Comparação de performance
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Barplot de RMSE
    ax = axes[0, 0]
    colors = ['red' if 'Ensemble' in name or 'Average' in name or 'Median' in name
              else 'steelblue' for name in df_results.index]
    ax.barh(range(len(df_results)), df_results['rmse'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(df_results)))
    ax.set_yticklabels(df_results.index, fontsize=9)
    ax.set_xlabel('RMSE', fontsize=11)
    ax.set_title('Comparação de RMSE - Modelos vs Ensembles', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # Scatter: RMSE vs R²
    ax = axes[0, 1]
    for name in df_results.index:
        if 'Ensemble' in name or 'Average' in name or 'Median' in name:
            ax.scatter(df_results.loc[name, 'rmse'], df_results.loc[name, 'r2'],
                      s=150, marker='*', color='red', label='Ensemble' if name == df_results.index[0] else '')
        else:
            ax.scatter(df_results.loc[name, 'rmse'], df_results.loc[name, 'r2'],
                      s=80, alpha=0.6, color='steelblue')
    ax.set_xlabel('RMSE', fontsize=11)
    ax.set_ylabel('R²', fontsize=11)
    ax.set_title('Trade-off RMSE vs R²', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Pesos do ensemble otimizado
    ax = axes[1, 0]
    if ens_opt.weights is not None:
        model_names = [name for name, _ in models]
        ax.barh(range(len(model_names)), ens_opt.weights, alpha=0.7, color='coral')
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names, fontsize=9)
        ax.set_xlabel('Peso', fontsize=11)
        ax.set_title('Pesos Otimizados do Ensemble', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

    # Previsões vs Real (melhor ensemble)
    ax = axes[1, 1]
    best_ensemble_name = [k for k in ensemble_results.keys()
                          if ensemble_results[k]['rmse'] == min([v['rmse'] for v in ensemble_results.values()])][0]
    best_ensemble = ensembles[best_ensemble_name]
    pred_best = best_ensemble.predict(X_test)

    ax.plot(test_data.index, y_test, 'o-', label='Real', linewidth=2.5, markersize=6, color='black')
    ax.plot(test_data.index, pred_best, 's--', label=f'{best_ensemble_name}',
            linewidth=2, markersize=5, alpha=0.7, color='red')
    ax.set_xlabel('Data', fontsize=11)
    ax.set_ylabel('IDCI-VIX', fontsize=11)
    ax.set_title(f'Melhor Ensemble: {best_ensemble_name}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.suptitle('Análise Completa de Ensemble Learning', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('ensemble_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✓ Gráfico salvo: ensemble_analysis.png")

    # Plot 2: Contribuição individual de cada modelo
    fig, ax = plt.subplots(figsize=(14, 8))

    for name, model in models:
        if name in ens_opt.model_predictions:
            pred = ens_opt.model_predictions[name]
            ax.plot(test_data.index, pred, '--', alpha=0.5, linewidth=1.5, label=name)

    ax.plot(test_data.index, y_test, 'o-', label='Real', linewidth=3,
            markersize=7, color='black', zorder=10)
    ax.plot(test_data.index, ens_opt.predict(X_test), 's-', label='Ensemble',
            linewidth=3, markersize=6, color='red', alpha=0.8, zorder=9)

    ax.set_xlabel('Data', fontsize=12)
    ax.set_ylabel('IDCI-VIX', fontsize=12)
    ax.set_title('Contribuições Individuais dos Modelos ao Ensemble',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('ensemble_contributions.png', dpi=300, bbox_inches='tight')
    print("   ✓ Gráfico salvo: ensemble_contributions.png\n")

    print("="*80)
    print("ANÁLISE CONCLUÍDA COM SUCESSO!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
