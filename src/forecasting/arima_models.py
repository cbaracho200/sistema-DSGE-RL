"""
Modelos ARIMA, SARIMA e SARIMAX para previsão de séries temporais.

Implementa:
- ARIMA(p,d,q)
- SARIMA(p,d,q)(P,D,Q,s)
- SARIMAX com variáveis exógenas
- Seleção automática de ordens via AIC/BIC
- Previsão multi-horizonte (h passos à frente)
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from itertools import product
from typing import Dict, Tuple, Optional, List
import warnings


class ARIMAForecaster:
    """
    Wrapper para modelos ARIMA/SARIMA/SARIMAX com seleção automática.
    """

    def __init__(self):
        self.model_ = None
        self.model_fit_ = None
        self.order_ = None
        self.seasonal_order_ = None
        self.is_fitted = False

    def _try_fit(self, endog, order, seasonal_order=None, exog=None):
        """Tenta ajustar modelo e retorna AIC/BIC."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if seasonal_order is not None:
                    model = SARIMAX(
                        endog,
                        order=order,
                        seasonal_order=seasonal_order,
                        exog=exog,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                else:
                    model = ARIMA(
                        endog,
                        order=order,
                        exog=exog,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )

                fit = model.fit(disp=False)
                return fit, fit.aic, fit.bic

        except Exception:
            return None, np.inf, np.inf

    def auto_arima(self, endog: pd.Series,
                   p_range: Tuple[int, int] = (0, 3),
                   d_range: Tuple[int, int] = (0, 2),
                   q_range: Tuple[int, int] = (0, 3),
                   criterion: str = 'aic',
                   verbose: bool = False) -> Tuple[int, int, int]:
        """
        Seleção automática de ordem ARIMA via grid search.

        Parâmetros:
        -----------
        p_range, d_range, q_range : tuplas (min, max)
        criterion : 'aic' ou 'bic'

        Retorna:
        --------
        (p, d, q) ótimos
        """
        best_order = None
        best_score = np.inf

        p_values = range(p_range[0], p_range[1] + 1)
        d_values = range(d_range[0], d_range[1] + 1)
        q_values = range(q_range[0], q_range[1] + 1)

        total = len(list(product(p_values, d_values, q_values)))
        count = 0

        if verbose:
            print(f"Testando {total} combinações de (p,d,q)...")

        for p, d, q in product(p_values, d_values, q_values):
            count += 1

            fit, aic, bic = self._try_fit(endog, order=(p, d, q))

            score = aic if criterion == 'aic' else bic

            if score < best_score:
                best_score = score
                best_order = (p, d, q)

            if verbose and count % 10 == 0:
                print(f"  Progresso: {count}/{total}, Melhor até agora: {best_order} ({criterion.upper()}={best_score:.2f})")

        if verbose:
            print(f"\n✓ Melhor ordem: ARIMA{best_order}, {criterion.upper()}={best_score:.2f}")

        return best_order

    def auto_sarima(self, endog: pd.Series,
                    p_range: Tuple[int, int] = (0, 2),
                    d_range: Tuple[int, int] = (0, 1),
                    q_range: Tuple[int, int] = (0, 2),
                    P_range: Tuple[int, int] = (0, 2),
                    D_range: Tuple[int, int] = (0, 1),
                    Q_range: Tuple[int, int] = (0, 2),
                    s: int = 12,
                    criterion: str = 'aic',
                    verbose: bool = False) -> Tuple[Tuple, Tuple]:
        """
        Seleção automática de ordem SARIMA.

        Retorna:
        --------
        ((p,d,q), (P,D,Q,s))
        """
        best_order = None
        best_seasonal = None
        best_score = np.inf

        p_values = range(p_range[0], p_range[1] + 1)
        d_values = range(d_range[0], d_range[1] + 1)
        q_values = range(q_range[0], q_range[1] + 1)

        P_values = range(P_range[0], P_range[1] + 1)
        D_values = range(D_range[0], D_range[1] + 1)
        Q_values = range(Q_range[0], Q_range[1] + 1)

        total = len(list(product(p_values, d_values, q_values, P_values, D_values, Q_values)))
        count = 0

        if verbose:
            print(f"Testando {total} combinações de SARIMA(p,d,q)(P,D,Q,{s})...")

        for p, d, q, P, D, Q in product(p_values, d_values, q_values, P_values, D_values, Q_values):
            count += 1

            order = (p, d, q)
            seasonal_order = (P, D, Q, s)

            fit, aic, bic = self._try_fit(endog, order=order, seasonal_order=seasonal_order)

            score = aic if criterion == 'aic' else bic

            if score < best_score:
                best_score = score
                best_order = order
                best_seasonal = seasonal_order

            if verbose and count % 20 == 0:
                print(f"  Progresso: {count}/{total}, Melhor: SARIMA{best_order}{best_seasonal}, {criterion.upper()}={best_score:.2f}")

        if verbose:
            print(f"\n✓ Melhor ordem: SARIMA{best_order}{best_seasonal}, {criterion.upper()}={best_score:.2f}")

        return best_order, best_seasonal

    def fit(self, endog: pd.Series,
           order: Optional[Tuple] = None,
           seasonal_order: Optional[Tuple] = None,
           exog: Optional[pd.DataFrame] = None,
           auto: bool = False,
           auto_seasonal: bool = False,
           s: int = 12,
           criterion: str = 'aic',
           verbose: bool = False) -> 'ARIMAForecaster':
        """
        Ajusta modelo ARIMA/SARIMA/SARIMAX.

        Parâmetros:
        -----------
        endog : série temporal alvo
        order : (p,d,q) - se None e auto=True, seleciona automaticamente
        seasonal_order : (P,D,Q,s) - se None e auto_seasonal=True, seleciona
        exog : variáveis exógenas (para SARIMAX)
        auto : se True, usa auto_arima
        auto_seasonal : se True, usa auto_sarima
        """
        self.endog_name = endog.name or 'target'
        self.data_index = endog.index

        # Seleção automática
        if auto and order is None:
            if auto_seasonal:
                order, seasonal_order = self.auto_sarima(
                    endog, s=s, criterion=criterion, verbose=verbose
                )
            else:
                order = self.auto_arima(endog, criterion=criterion, verbose=verbose)

        if order is None:
            raise ValueError("Deve fornecer 'order' ou usar auto=True")

        self.order_ = order
        self.seasonal_order_ = seasonal_order

        # Ajusta modelo final
        if verbose:
            if seasonal_order is not None:
                print(f"\nAjustando SARIMA{order}{seasonal_order}...")
            else:
                print(f"\nAjustando ARIMA{order}...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if seasonal_order is not None:
                self.model_ = SARIMAX(
                    endog,
                    order=order,
                    seasonal_order=seasonal_order,
                    exog=exog,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                self.model_ = ARIMA(
                    endog,
                    order=order,
                    exog=exog,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )

            self.model_fit_ = self.model_.fit(disp=False)

        self.is_fitted = True
        self.has_exog = exog is not None

        if verbose:
            print(f"✓ Modelo ajustado:")
            print(f"  AIC: {self.model_fit_.aic:.2f}")
            print(f"  BIC: {self.model_fit_.bic:.2f}")
            print(f"  Log-Likelihood: {self.model_fit_.llf:.2f}")

        return self

    def forecast(self, steps: int = 12, exog: Optional[pd.DataFrame] = None,
                return_conf_int: bool = True, alpha: float = 0.05) -> pd.DataFrame:
        """
        Previsão multi-horizonte.

        Parâmetros:
        -----------
        steps : número de passos à frente
        exog : valores futuros das exógenas (se modelo SARIMAX)
        return_conf_int : se True, retorna intervalos de confiança
        alpha : nível de significância (default: 5% -> IC de 95%)

        Retorna:
        --------
        DataFrame com colunas: ['forecast', 'lower', 'upper'] (se return_conf_int)
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi ajustado. Chame fit() primeiro.")

        if self.has_exog and exog is None:
            raise ValueError("Modelo usa exógenas. Deve fornecer 'exog' para previsão.")

        # Previsão
        forecast_result = self.model_fit_.get_forecast(steps=steps, exog=exog)

        forecast = forecast_result.predicted_mean

        # Monta DataFrame de resultados
        result_df = pd.DataFrame({
            'forecast': forecast.values
        }, index=forecast.index)

        if return_conf_int:
            conf_int = forecast_result.conf_int(alpha=alpha)
            result_df['lower'] = conf_int.iloc[:, 0].values
            result_df['upper'] = conf_int.iloc[:, 1].values

        return result_df

    def get_insample_predictions(self) -> pd.Series:
        """Retorna previsões dentro da amostra (fitted values)."""
        if not self.is_fitted:
            raise ValueError("Modelo não foi ajustado.")

        return self.model_fit_.fittedvalues

    def get_residuals(self) -> pd.Series:
        """Retorna resíduos do modelo."""
        if not self.is_fitted:
            raise ValueError("Modelo não foi ajustado.")

        return self.model_fit_.resid

    def summary(self) -> str:
        """Retorna sumário estatístico do modelo."""
        if not self.is_fitted:
            raise ValueError("Modelo não foi ajustado.")

        return self.model_fit_.summary()

    def get_params(self) -> pd.Series:
        """Retorna parâmetros estimados."""
        if not self.is_fitted:
            raise ValueError("Modelo não foi ajustado.")

        return self.model_fit_.params


class MultiHorizonARIMA:
    """
    Wrapper para treinar múltiplos modelos ARIMA, um para cada horizonte.

    Estratégia "Direct Multi-Step".
    """

    def __init__(self, max_horizon: int = 12):
        self.max_horizon = max_horizon
        self.models = {}

    def fit(self, endog: pd.Series, exog: Optional[pd.DataFrame] = None,
           order: Optional[Tuple] = None, seasonal_order: Optional[Tuple] = None,
           auto: bool = True, verbose: bool = False):
        """
        Treina um modelo para cada horizonte h = 1, ..., max_horizon.
        """
        for h in range(1, self.max_horizon + 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Treinando modelo para horizonte h={h}")
                print(f"{'='*60}")

            # Cria target deslocado
            target_h = endog.shift(-h).dropna()

            # Alinha exógenas
            if exog is not None:
                exog_h = exog.loc[target_h.index]
            else:
                exog_h = None

            # Treina modelo
            model_h = ARIMAForecaster()
            model_h.fit(
                target_h,
                order=order,
                seasonal_order=seasonal_order,
                exog=exog_h,
                auto=auto,
                auto_seasonal=(seasonal_order is not None),
                verbose=verbose
            )

            self.models[h] = model_h

        if verbose:
            print(f"\n✓ {len(self.models)} modelos treinados (h=1 a {self.max_horizon})")

    def forecast(self, exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Gera previsões para todos os horizontes.

        Retorna:
        --------
        DataFrame com uma coluna para cada horizonte
        """
        forecasts = {}

        for h, model in self.models.items():
            # Cada modelo prevê apenas 1 passo (que corresponde ao horizonte h)
            exog_h = exog.iloc[[0]] if exog is not None else None
            fc = model.forecast(steps=1, exog=exog_h, return_conf_int=False)
            forecasts[f'h{h}'] = fc['forecast'].iloc[0]

        return pd.DataFrame([forecasts])
