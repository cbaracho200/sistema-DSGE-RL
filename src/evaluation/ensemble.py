"""
Sistema de combinação e ensemble de modelos de previsão.

Implementa:
- Combinação por média simples
- Combinação por média ponderada (inverso do erro)
- Rolling-origin evaluation
- Seleção de melhores modelos por horizonte
- Stacking (meta-modelo)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings


class ForecastEvaluator:
    """
    Avalia modelos de previsão usando rolling-origin.
    """

    def __init__(self, min_train_size: int = 36, horizon: int = 12, step: int = 1):
        """
        Parâmetros:
        -----------
        min_train_size : tamanho mínimo da janela de treino
        horizon : horizonte máximo de previsão
        step : passo entre janelas (1 = rolling, >1 = expanding)
        """
        self.min_train_size = min_train_size
        self.horizon = horizon
        self.step = step

    def rolling_origin_eval(self, data: pd.Series,
                           model_fit_func: Callable,
                           model_predict_func: Callable,
                           exog: Optional[pd.DataFrame] = None,
                           verbose: bool = False) -> pd.DataFrame:
        """
        Avaliação rolling-origin.

        Parâmetros:
        -----------
        data : série temporal completa
        model_fit_func : função(train_data, train_exog) -> modelo ajustado
        model_predict_func : função(modelo, h, test_exog) -> previsão h-passos
        exog : variáveis exógenas (alinhadas com data)

        Retorna:
        --------
        DataFrame com colunas: ['origin', 'horizon', 'actual', 'forecast', 'error']
        """
        results = []

        n = len(data)
        origins = range(self.min_train_size, n - self.horizon, self.step)

        if verbose:
            print(f"Rolling-origin evaluation: {len(origins)} janelas")

        for i, origin in enumerate(origins):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Janela {i+1}/{len(origins)}")

            # Divide dados
            train_data = data.iloc[:origin]
            train_exog = exog.iloc[:origin] if exog is not None else None

            # Treina modelo
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = model_fit_func(train_data, train_exog)

                # Prevê para cada horizonte
                for h in range(1, min(self.horizon + 1, n - origin + 1)):
                    if origin + h - 1 >= n:
                        break

                    actual = data.iloc[origin + h - 1]

                    # Exógenas para previsão (se necessário)
                    if exog is not None and h <= len(exog) - origin:
                        test_exog = exog.iloc[origin:origin + h]
                    else:
                        test_exog = None

                    # Prevê
                    forecast = model_predict_func(model, h, test_exog)

                    error = actual - forecast

                    results.append({
                        'origin': origin,
                        'horizon': h,
                        'actual': actual,
                        'forecast': forecast,
                        'error': error
                    })

            except Exception as e:
                if verbose:
                    print(f"  ⚠ Erro na janela {origin}: {e}")
                continue

        return pd.DataFrame(results)

    def compute_metrics(self, eval_df: pd.DataFrame, by_horizon: bool = True) -> pd.DataFrame:
        """
        Calcula métricas de erro.

        Retorna:
        --------
        DataFrame com RMSE, MAE, MAPE por horizonte (se by_horizon=True)
        """
        if by_horizon:
            metrics = []

            for h in eval_df['horizon'].unique():
                h_data = eval_df[eval_df['horizon'] == h]

                rmse = np.sqrt(mean_squared_error(h_data['actual'], h_data['forecast']))
                mae = mean_absolute_error(h_data['actual'], h_data['forecast'])

                # MAPE (evita divisão por zero)
                mape_vals = []
                for actual, forecast in zip(h_data['actual'], h_data['forecast']):
                    if abs(actual) > 1e-6:
                        mape_vals.append(abs((actual - forecast) / actual))

                mape = np.mean(mape_vals) * 100 if mape_vals else np.nan

                metrics.append({
                    'horizon': h,
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape,
                    'n_forecasts': len(h_data)
                })

            return pd.DataFrame(metrics)

        else:
            # Métricas globais
            rmse = np.sqrt(mean_squared_error(eval_df['actual'], eval_df['forecast']))
            mae = mean_absolute_error(eval_df['actual'], eval_df['forecast'])

            mape_vals = []
            for actual, forecast in zip(eval_df['actual'], eval_df['forecast']):
                if abs(actual) > 1e-6:
                    mape_vals.append(abs((actual - forecast) / actual))

            mape = np.mean(mape_vals) * 100 if mape_vals else np.nan

            return pd.DataFrame([{
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'n_forecasts': len(eval_df)
            }])


class EnsembleForecaster:
    """
    Combina previsões de múltiplos modelos.
    """

    def __init__(self, combination_method: str = 'weighted_avg'):
        """
        Parâmetros:
        -----------
        combination_method : 'simple_avg', 'weighted_avg', 'median', 'stacking'
        """
        self.combination_method = combination_method
        self.weights_ = {}
        self.meta_models_ = {}  # Para stacking

    def fit_weights(self, forecasts_dict: Dict[str, pd.DataFrame],
                   actuals: pd.Series,
                   by_horizon: bool = True,
                   verbose: bool = False) -> 'EnsembleForecaster':
        """
        Calcula pesos ótimos baseados em erros históricos.

        Parâmetros:
        -----------
        forecasts_dict : {model_name: DataFrame com coluna 'forecast'}
        actuals : valores reais observados
        by_horizon : se True, calcula pesos separados por horizonte
        """
        if self.combination_method == 'simple_avg':
            # Pesos iguais
            n_models = len(forecasts_dict)
            self.weights_ = {model: 1.0 / n_models for model in forecasts_dict.keys()}

            if verbose:
                print("Pesos (média simples):")
                for model, w in self.weights_.items():
                    print(f"  {model}: {w:.4f}")

        elif self.combination_method == 'weighted_avg':
            # Pesos inversamente proporcionais ao RMSE

            if by_horizon:
                # Calcula por horizonte
                horizons = forecasts_dict[list(forecasts_dict.keys())[0]]['horizon'].unique()

                for h in horizons:
                    weights_h = {}
                    errors = {}

                    for model_name, forecast_df in forecasts_dict.items():
                        h_data = forecast_df[forecast_df['horizon'] == h]
                        h_actuals = actuals.loc[h_data.index]

                        rmse = np.sqrt(mean_squared_error(h_actuals, h_data['forecast']))
                        errors[model_name] = rmse

                    # Inverte erros
                    total_inv_error = sum(1.0 / (e + 1e-6) for e in errors.values())

                    for model_name, rmse in errors.items():
                        weights_h[model_name] = (1.0 / (rmse + 1e-6)) / total_inv_error

                    self.weights_[h] = weights_h

                if verbose:
                    print("Pesos por horizonte (inverso RMSE):")
                    for h in sorted(self.weights_.keys()):
                        print(f"  Horizonte {h}:")
                        for model, w in self.weights_[h].items():
                            print(f"    {model}: {w:.4f}")

            else:
                # Pesos globais
                weights = {}
                errors = {}

                for model_name, forecast_df in forecasts_dict.items():
                    actuals_aligned = actuals.loc[forecast_df.index]
                    rmse = np.sqrt(mean_squared_error(actuals_aligned, forecast_df['forecast']))
                    errors[model_name] = rmse

                total_inv_error = sum(1.0 / (e + 1e-6) for e in errors.values())

                for model_name, rmse in errors.items():
                    weights[model_name] = (1.0 / (rmse + 1e-6)) / total_inv_error

                self.weights_ = weights

                if verbose:
                    print("Pesos globais (inverso RMSE):")
                    for model, w in self.weights_.items():
                        print(f"  {model}: {w:.4f}")

        elif self.combination_method == 'stacking':
            # Meta-modelo (regressão linear)
            # Por horizonte

            horizons = forecasts_dict[list(forecasts_dict.keys())[0]]['horizon'].unique()

            for h in horizons:
                # Monta matriz de previsões
                X_meta = []
                y_meta = []

                for model_name, forecast_df in forecasts_dict.items():
                    h_data = forecast_df[forecast_df['horizon'] == h]
                    X_meta.append(h_data['forecast'].values)

                X_meta = np.column_stack(X_meta)

                # Pega valores reais
                h_actuals = actuals.loc[h_data.index]
                y_meta = h_actuals.values

                # Treina meta-modelo
                meta_model = LinearRegression(positive=True)  # Pesos não-negativos
                meta_model.fit(X_meta, y_meta)

                self.meta_models_[h] = {
                    'model': meta_model,
                    'model_names': list(forecasts_dict.keys())
                }

            if verbose:
                print("Meta-modelos (stacking) treinados:")
                for h, meta_dict in self.meta_models_.items():
                    print(f"  Horizonte {h}:")
                    for name, coef in zip(meta_dict['model_names'], meta_dict['model'].coef_):
                        print(f"    {name}: {coef:.4f}")

        return self

    def combine(self, forecasts_dict: Dict[str, pd.DataFrame],
               horizon: Optional[int] = None) -> pd.Series:
        """
        Combina previsões de múltiplos modelos.

        Parâmetros:
        -----------
        forecasts_dict : {model_name: DataFrame ou valor escalar}
        horizon : horizonte (se pesos por horizonte)

        Retorna:
        --------
        Series com previsão combinada
        """
        if self.combination_method == 'simple_avg':
            # Média simples
            forecasts = [df['forecast'] if isinstance(df, pd.DataFrame) else df
                        for df in forecasts_dict.values()]
            return pd.concat(forecasts, axis=1).mean(axis=1)

        elif self.combination_method == 'weighted_avg':
            # Média ponderada
            if isinstance(self.weights_, dict) and horizon is not None and horizon in self.weights_:
                weights = self.weights_[horizon]
            else:
                weights = self.weights_

            combined = None
            for model_name, forecast in forecasts_dict.items():
                w = weights.get(model_name, 0.0)

                if isinstance(forecast, pd.DataFrame):
                    fc = forecast['forecast']
                else:
                    fc = forecast

                if combined is None:
                    combined = w * fc
                else:
                    combined += w * fc

            return combined

        elif self.combination_method == 'median':
            # Mediana
            forecasts = [df['forecast'] if isinstance(df, pd.DataFrame) else df
                        for df in forecasts_dict.values()]
            return pd.concat(forecasts, axis=1).median(axis=1)

        elif self.combination_method == 'stacking':
            # Stacking
            if horizon is None or horizon not in self.meta_models_:
                raise ValueError("Stacking requer horizonte específico e meta-modelo treinado")

            meta_dict = self.meta_models_[horizon]
            meta_model = meta_dict['model']
            model_names = meta_dict['model_names']

            # Monta matriz de previsões
            X_meta = []
            for name in model_names:
                forecast = forecasts_dict[name]
                fc = forecast['forecast'].values if isinstance(forecast, pd.DataFrame) else forecast
                X_meta.append(fc)

            X_meta = np.column_stack(X_meta)

            # Prediz com meta-modelo
            combined = meta_model.predict(X_meta)

            return pd.Series(combined, index=forecasts_dict[model_names[0]].index)

        else:
            raise ValueError(f"Método inválido: {self.combination_method}")


class ModelSelector:
    """
    Seleciona melhores modelos baseado em performance histórica.
    """

    def __init__(self, metric: str = 'rmse', top_k: Optional[int] = None):
        """
        Parâmetros:
        -----------
        metric : métrica para seleção ('rmse', 'mae', 'mape')
        top_k : seleciona top-K modelos (se None, usa todos)
        """
        self.metric = metric
        self.top_k = top_k
        self.rankings_ = {}

    def rank_models(self, metrics_dict: Dict[str, pd.DataFrame],
                   by_horizon: bool = True) -> pd.DataFrame:
        """
        Rankeia modelos por performance.

        Parâmetros:
        -----------
        metrics_dict : {model_name: DataFrame de métricas}
        by_horizon : se True, rankeia por horizonte

        Retorna:
        --------
        DataFrame com ranking
        """
        if by_horizon:
            rankings = []

            # Pega horizontes
            first_model = list(metrics_dict.keys())[0]
            horizons = metrics_dict[first_model]['horizon'].unique()

            for h in horizons:
                scores = []

                for model_name, metrics_df in metrics_dict.items():
                    h_metrics = metrics_df[metrics_df['horizon'] == h]

                    if len(h_metrics) == 0:
                        continue

                    score = h_metrics[self.metric].iloc[0]
                    scores.append({
                        'horizon': h,
                        'model': model_name,
                        'score': score
                    })

                # Ordena por score (menor = melhor)
                scores_df = pd.DataFrame(scores).sort_values('score')

                # Adiciona rank
                scores_df['rank'] = range(1, len(scores_df) + 1)

                rankings.append(scores_df)

            rankings_df = pd.concat(rankings, ignore_index=True)
            self.rankings_ = rankings_df

            return rankings_df

        else:
            # Ranking global
            scores = []

            for model_name, metrics_df in metrics_dict.items():
                # Média do metric em todos os horizontes
                score = metrics_df[self.metric].mean()

                scores.append({
                    'model': model_name,
                    'score': score
                })

            rankings_df = pd.DataFrame(scores).sort_values('score')
            rankings_df['rank'] = range(1, len(rankings_df) + 1)

            self.rankings_ = rankings_df

            return rankings_df

    def select_best(self, horizon: Optional[int] = None) -> List[str]:
        """
        Retorna nomes dos melhores modelos.

        Parâmetros:
        -----------
        horizon : se especificado, retorna melhores para esse horizonte
        """
        if self.rankings_ is None or len(self.rankings_) == 0:
            raise ValueError("Deve chamar rank_models() primeiro")

        if horizon is not None:
            # Filtra por horizonte
            h_rankings = self.rankings_[self.rankings_['horizon'] == horizon]
        else:
            h_rankings = self.rankings_

        # Ordena e pega top-K
        h_rankings = h_rankings.sort_values('score')

        if self.top_k is not None:
            h_rankings = h_rankings.head(self.top_k)

        return h_rankings['model'].tolist()
