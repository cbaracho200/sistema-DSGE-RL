"""
Regressão Quantílica para séries temporais.

Implementa:
- Regressão Quantílica Linear
- Múltiplos quantis (cenários pessimista/base/otimista)
- Previsão com intervalos de confiança
- Multi-horizonte
"""

import numpy as np
import pandas as pd
from statsmodels.regression.quantile_regression import QuantReg
from typing import Dict, Optional, List, Tuple
import warnings


class QuantileRegressionForecaster:
    """
    Regressão Quantílica para previsão de séries temporais.

    Estima diferentes quantis condicionais:
    Q_y(τ | X) para τ ∈ [0, 1]

    Ex.: τ = 0.1 (pessimista), 0.5 (mediana), 0.9 (otimista)
    """

    def __init__(self, quantiles: List[float] = None, max_lag: int = 12):
        """
        Parâmetros:
        -----------
        quantiles : lista de quantis a estimar (ex: [0.1, 0.5, 0.9])
        max_lag : máximo de lags para features
        """
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

        self.quantiles = sorted(quantiles)
        self.max_lag = max_lag

        self.models_ = {}  # Um modelo para cada quantil
        self.is_fitted = False

    def _build_features(self, target: pd.Series, exog: Optional[pd.DataFrame] = None,
                       target_lags: Optional[List[int]] = None,
                       exog_lags: Optional[List[int]] = None) -> pd.DataFrame:
        """Constrói matriz de features."""
        features = {}

        # Lags do alvo
        if target_lags is None:
            target_lags = list(range(1, self.max_lag + 1))

        for lag in target_lags:
            features[f'y_lag_{lag}'] = target.shift(lag)

        # Médias móveis
        for window in [3, 6, 12]:
            if window <= self.max_lag:
                features[f'y_ma_{window}'] = target.rolling(window=window).mean()

        # Exógenas
        if exog is not None:
            if exog_lags is None:
                exog_lags = list(range(1, min(self.max_lag, 6) + 1))

            for col in exog.columns:
                for lag in exog_lags:
                    features[f'{col}_lag_{lag}'] = exog[col].shift(lag)

        X = pd.DataFrame(features, index=target.index)
        return X

    def fit(self, target: pd.Series, exog: Optional[pd.DataFrame] = None,
           target_lags: Optional[List[int]] = None,
           exog_lags: Optional[List[int]] = None,
           verbose: bool = False) -> 'QuantileRegressionForecaster':
        """
        Ajusta modelos de regressão quantílica para cada quantil.

        Parâmetros:
        -----------
        target : série temporal alvo
        exog : variáveis exógenas
        """
        self.target_name = target.name or 'target'
        self.data_index = target.index

        if verbose:
            print(f"Construindo features (max_lag={self.max_lag})...")

        # Constrói features
        X = self._build_features(target, exog=exog, target_lags=target_lags, exog_lags=exog_lags)

        # Remove NaN
        y = target.loc[X.index]
        valid_idx = X.notna().all(axis=1) & y.notna()
        X = X[valid_idx]
        y = y[valid_idx]

        if len(X) < 20:
            raise ValueError("Dados insuficientes após criar features e remover NaN")

        self.feature_names_ = X.columns.tolist()

        # Adiciona constante
        X = X.copy()
        X.insert(0, 'const', 1.0)

        if verbose:
            print(f"Features criadas: {len(self.feature_names_)}")
            print(f"Amostras de treino: {len(X)}")
            print(f"\nAjustando modelos para {len(self.quantiles)} quantis...")

        # Treina um modelo para cada quantil
        for q in self.quantiles:
            if verbose:
                print(f"  Quantil τ={q:.2f}...", end=' ')

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model = QuantReg(y, X)
                fit = model.fit(q=q, max_iter=1000)

                self.models_[q] = {
                    'model': model,
                    'fit': fit,
                    'params': fit.params
                }

                if verbose:
                    print(f"✓ (pseudo R²: {fit.prsquared:.4f})")

        self.is_fitted = True

        if verbose:
            print(f"\n✓ Todos os quantis ajustados")

        return self

    def predict(self, target: pd.Series, exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prediz quantis para o próximo passo.

        Retorna:
        --------
        DataFrame com uma coluna para cada quantil (q0.1, q0.5, q0.9, etc.)
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi ajustado. Chame fit() primeiro.")

        # Constrói features
        X = self._build_features(target, exog=exog)
        X = X.iloc[[-1]][self.feature_names_]

        if X.isna().any().any():
            return pd.DataFrame({f'q{q}': [np.nan] for q in self.quantiles})

        # Adiciona constante
        X = X.copy()
        X.insert(0, 'const', 1.0)

        # Prediz cada quantil
        predictions = {}
        for q in self.quantiles:
            fit = self.models_[q]['fit']
            y_pred = fit.predict(X)[0]
            predictions[f'q{q}'] = y_pred

        return pd.DataFrame([predictions])

    def forecast(self, target: pd.Series, steps: int = 12,
                exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Previsão recursiva multi-horizonte para cada quantil.

        ATENÇÃO: Implementação simplificada - usa apenas mediana para recursão.

        Retorna:
        --------
        DataFrame com índice = horizonte e colunas = quantis
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi ajustado.")

        # Para cada horizonte, faz previsão
        # Simplificação: usa mediana (0.5) para atualizar histórico
        median_q = 0.5
        if median_q not in self.quantiles:
            # Usa quantil mais próximo
            median_q = min(self.quantiles, key=lambda x: abs(x - 0.5))

        target_extended = target.copy()
        forecasts = {f'q{q}': [] for q in self.quantiles}

        # Índice futuro
        last_date = target.index[-1]
        if isinstance(last_date, pd.Timestamp):
            freq = pd.infer_freq(target.index) or 'MS'
            future_index = pd.date_range(start=last_date, periods=steps + 1, freq=freq)[1:]
        else:
            future_index = range(len(target), len(target) + steps)

        for h in range(steps):
            # Prediz todos os quantis
            preds = self.predict(target_extended, exog=exog)

            for q in self.quantiles:
                forecasts[f'q{q}'].append(preds[f'q{q}'].iloc[0])

            # Usa mediana para atualizar histórico
            median_pred = preds[f'q{median_q}'].iloc[0]

            if isinstance(future_index[h], pd.Timestamp):
                target_extended.loc[future_index[h]] = median_pred
            else:
                target_extended = pd.concat([
                    target_extended,
                    pd.Series([median_pred], index=[future_index[h]])
                ])

        result_df = pd.DataFrame(forecasts, index=future_index)
        return result_df

    def get_prediction_intervals(self, target: pd.Series, exog: Optional[pd.DataFrame] = None,
                                alpha: float = 0.1) -> pd.DataFrame:
        """
        Retorna intervalos de previsão baseados em quantis.

        Parâmetros:
        -----------
        alpha : nível de significância (default: 0.1 para IC de 90%)

        Retorna:
        --------
        DataFrame com colunas ['median', 'lower', 'upper']
        """
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2
        median_q = 0.5

        # Garante que quantis necessários estão ajustados
        needed_q = [lower_q, median_q, upper_q]
        if not all(q in self.quantiles for q in needed_q):
            raise ValueError(f"Modelo deve ter quantis {needed_q} para calcular IC com α={alpha}")

        # Prediz
        preds = self.predict(target, exog=exog)

        result = pd.DataFrame({
            'median': [preds[f'q{median_q}'].iloc[0]],
            'lower': [preds[f'q{lower_q}'].iloc[0]],
            'upper': [preds[f'q{upper_q}'].iloc[0]]
        })

        return result

    def get_params(self, quantile: float) -> pd.Series:
        """Retorna parâmetros estimados para um quantil específico."""
        if not self.is_fitted:
            raise ValueError("Modelo não foi ajustado.")

        if quantile not in self.models_:
            raise ValueError(f"Quantil {quantile} não foi ajustado.")

        return self.models_[quantile]['params']

    def summary(self, quantile: float) -> str:
        """Retorna sumário estatístico para um quantil."""
        if not self.is_fitted:
            raise ValueError("Modelo não foi ajustado.")

        if quantile not in self.models_:
            raise ValueError(f"Quantil {quantile} não foi ajustado.")

        return self.models_[quantile]['fit'].summary()


class MultiHorizonQuantile:
    """
    Treina um modelo de regressão quantílica para cada horizonte (Direct Multi-Step).
    """

    def __init__(self, quantiles: List[float] = None, max_lag: int = 12,
                max_horizon: int = 12):
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]

        self.quantiles = quantiles
        self.max_lag = max_lag
        self.max_horizon = max_horizon

        self.models = {}  # {horizon: QuantileRegressionForecaster}

    def fit(self, target: pd.Series, exog: Optional[pd.DataFrame] = None,
           verbose: bool = False):
        """
        Treina um modelo para cada horizonte h = 1, ..., max_horizon.
        """
        for h in range(1, self.max_horizon + 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Treinando modelos para horizonte h={h}")
                print(f"{'='*60}")

            # Cria target deslocado
            target_h = target.shift(-h)

            # Treina modelo de quantis
            model_h = QuantileRegressionForecaster(
                quantiles=self.quantiles,
                max_lag=self.max_lag
            )

            # Constrói features
            X = model_h._build_features(target, exog=exog)
            y_h = target_h.loc[X.index]

            valid_idx = X.notna().all(axis=1) & y_h.notna()
            X_valid = X[valid_idx]
            y_valid = y_h[valid_idx]

            # Adiciona constante
            X_valid = X_valid.copy()
            X_valid.insert(0, 'const', 1.0)

            # Treina cada quantil
            models_q = {}

            for q in self.quantiles:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    qr_model = QuantReg(y_valid, X_valid)
                    fit = qr_model.fit(q=q, max_iter=1000)

                    models_q[q] = {
                        'model': qr_model,
                        'fit': fit,
                        'params': fit.params
                    }

            self.models[h] = {
                'models': models_q,
                'feature_names': model_h.feature_names_
            }

            if verbose:
                median_q = 0.5 if 0.5 in self.quantiles else self.quantiles[len(self.quantiles)//2]
                print(f"  ✓ {len(self.quantiles)} quantis ajustados "
                      f"(mediana pseudo-R²: {models_q[median_q]['fit'].prsquared:.4f})")

        if verbose:
            print(f"\n✓ {len(self.models)} horizontes treinados (h=1 a {self.max_horizon})")

    def forecast(self, target: pd.Series, exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Gera previsões de quantis para todos os horizontes.

        Retorna:
        --------
        DataFrame com índice = horizonte e colunas multi-nível = quantis
        """
        forecasts = {f'h{h}': {f'q{q}': None for q in self.quantiles}
                    for h in range(1, self.max_horizon + 1)}

        # Constrói features com dados mais recentes
        model_h1 = QuantileRegressionForecaster(quantiles=self.quantiles, max_lag=self.max_lag)
        X_current = model_h1._build_features(target, exog=exog)
        X_current = X_current.iloc[[-1]]

        for h, model_dict in self.models.items():
            feature_names = model_dict['feature_names']
            models_q = model_dict['models']

            X_h = X_current[feature_names]

            if X_h.isna().any().any():
                for q in self.quantiles:
                    forecasts[f'h{h}'][f'q{q}'] = np.nan
                continue

            # Adiciona constante
            X_h = X_h.copy()
            X_h.insert(0, 'const', 1.0)

            # Prediz cada quantil
            for q in self.quantiles:
                fit = models_q[q]['fit']
                y_pred = fit.predict(X_h)[0]
                forecasts[f'h{h}'][f'q{q}'] = y_pred

        # Converte para DataFrame
        result_list = []
        for h in range(1, self.max_horizon + 1):
            row = {'horizon': h}
            for q in self.quantiles:
                row[f'q{q}'] = forecasts[f'h{h}'][f'q{q}']
            result_list.append(row)

        return pd.DataFrame(result_list).set_index('horizon')

    def get_prediction_intervals(self, target: pd.Series, exog: Optional[pd.DataFrame] = None,
                                alpha: float = 0.1) -> pd.DataFrame:
        """
        Retorna intervalos de previsão para todos os horizontes.
        """
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2
        median_q = 0.5

        forecasts = self.forecast(target, exog=exog)

        intervals = pd.DataFrame({
            'horizon': range(1, self.max_horizon + 1),
            'median': forecasts[f'q{median_q}'].values,
            'lower': forecasts[f'q{lower_q}'].values,
            'upper': forecasts[f'q{upper_q}'].values
        }).set_index('horizon')

        return intervals
