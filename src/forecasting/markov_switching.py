"""
Modelos de Markov-Switching para captura de regimes (expansão/contração).

Implementa:
- Markov-Switching AR
- Estimação via algoritmo de Hamilton (EM)
- Probabilidades de regime filtradas e suavizadas
- Previsão condicional a regimes
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from typing import Dict, Optional, List
import warnings


class MarkovSwitchingForecaster:
    """
    Wrapper para modelos de Markov-Switching.

    Modelo:
        y_t = μ_{s_t} + Σ φ_{s_t,j} * y_{t-j} + ε_t,  ε_t ~ N(0, σ²_{s_t})

    onde s_t ∈ {0, 1, ..., K-1} é o regime oculto.
    """

    def __init__(self, k_regimes: int = 2, order: int = 2, switching_variance: bool = True):
        """
        Parâmetros:
        -----------
        k_regimes : número de regimes (default: 2 = expansão/contração)
        order : ordem do AR
        switching_variance : se True, variância muda por regime
        """
        self.k_regimes = k_regimes
        self.order = order
        self.switching_variance = switching_variance
        self.model_ = None
        self.model_fit_ = None
        self.is_fitted = False

    def fit(self, endog: pd.Series, exog: Optional[pd.DataFrame] = None,
           verbose: bool = False, maxiter: int = 1000) -> 'MarkovSwitchingForecaster':
        """
        Ajusta modelo de Markov-Switching.

        Parâmetros:
        -----------
        endog : série temporal alvo
        exog : variáveis exógenas (opcional)
        """
        self.endog_name = endog.name or 'target'
        self.data_index = endog.index

        if verbose:
            print(f"Ajustando Markov-Switching AR({self.order}) com {self.k_regimes} regimes...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if exog is None:
                # Modelo MS-AR puro
                self.model_ = MarkovAutoregression(
                    endog,
                    k_regimes=self.k_regimes,
                    order=self.order,
                    switching_ar=True,
                    switching_variance=self.switching_variance
                )
            else:
                # Modelo MS com exógenas
                self.model_ = MarkovRegression(
                    endog,
                    k_regimes=self.k_regimes,
                    exog=exog,
                    switching_variance=self.switching_variance
                )

            try:
                self.model_fit_ = self.model_.fit(maxiter=maxiter, disp=False)
                self.is_fitted = True

                if verbose:
                    print(f"✓ Modelo ajustado:")
                    print(f"  AIC: {self.model_fit_.aic:.2f}")
                    print(f"  BIC: {self.model_fit_.bic:.2f}")
                    print(f"  Log-Likelihood: {self.model_fit_.llf:.2f}")
                    print(f"\nParâmetros por regime:")
                    for regime in range(self.k_regimes):
                        print(f"  Regime {regime}:")
                        if hasattr(self.model_fit_.params, f'const[{regime}]'):
                            print(f"    μ: {self.model_fit_.params[f'const[{regime}]']:.4f}")
                        if self.switching_variance:
                            print(f"    σ: {self.model_fit_.params[f'sigma2[{regime}]']:.4f}")

            except Exception as e:
                if verbose:
                    print(f"⚠ Erro ao ajustar modelo: {e}")
                raise

        return self

    def get_regime_probabilities(self, smoothed: bool = True) -> pd.DataFrame:
        """
        Retorna probabilidades de regime.

        Parâmetros:
        -----------
        smoothed : se True, usa probabilidades suavizadas (mais precisas)
                   se False, usa probabilidades filtradas (tempo real)

        Retorna:
        --------
        DataFrame com P(s_t = k | dados) para cada regime k
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi ajustado.")

        if smoothed:
            probs = self.model_fit_.smoothed_marginal_probabilities
        else:
            probs = self.model_fit_.filtered_marginal_probabilities

        # Renomeia colunas
        probs.columns = [f'regime_{i}' for i in range(self.k_regimes)]

        return probs

    def get_expected_regime(self, smoothed: bool = True) -> pd.Series:
        """
        Retorna regime mais provável em cada período.
        """
        probs = self.get_regime_probabilities(smoothed=smoothed)
        return probs.idxmax(axis=1)

    def forecast(self, steps: int = 12, exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Previsão multi-horizonte.

        Retorna previsão média ponderada pelas probabilidades de regime:
            E[y_{T+h} | dados] = Σ_k P(s_{T+h}=k | dados) * E[y_{T+h} | s_{T+h}=k]

        Retorna:
        --------
        DataFrame com colunas:
            - 'forecast': previsão média
            - 'regime_0', 'regime_1', ...: previsões condicionais por regime
            - 'prob_0', 'prob_1', ...: probabilidades de cada regime
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi ajustado.")

        # Previsão do statsmodels (já faz a ponderação)
        forecast_result = self.model_fit_.forecast(steps=steps, exog=exog)

        result_df = pd.DataFrame({
            'forecast': forecast_result
        })

        # Probabilidades de regime no último período observado
        last_probs = self.get_regime_probabilities(smoothed=False).iloc[-1]

        # Adiciona probabilidades (assumindo que se mantêm constantes - simplificação)
        for regime in range(self.k_regimes):
            result_df[f'prob_{regime}'] = last_probs[f'regime_{regime}']

        return result_df

    def forecast_by_regime(self, steps: int = 12, exog: Optional[pd.DataFrame] = None) -> Dict:
        """
        Previsão separada por regime.

        Retorna:
        --------
        Dict com chaves 'regime_0', 'regime_1', etc.
        Cada valor é um array com a previsão condicional.
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi ajustado.")

        # Para simplificação, vamos usar simulação
        # Statsmodels não fornece API direta para previsão por regime

        # Alternativa: extrai parâmetros e simula manualmente
        forecasts_by_regime = {}

        # Simplificação: assume que último valor observado é y_T
        y_last = self.model_fit_.model.endog[-self.order:]

        for regime in range(self.k_regimes):
            # Extrai parâmetros do regime
            # (Isso depende da estrutura interna do statsmodels, pode variar)
            try:
                # Pega constante
                if f'const[{regime}]' in self.model_fit_.params:
                    const = self.model_fit_.params[f'const[{regime}]']
                else:
                    const = 0.0

                # Pega coeficientes AR
                ar_coeffs = []
                for lag in range(1, self.order + 1):
                    param_name = f'ar.L{lag}[{regime}]'
                    if param_name in self.model_fit_.params:
                        ar_coeffs.append(self.model_fit_.params[param_name])
                    else:
                        ar_coeffs.append(0.0)

                # Simula previsão (determinística, sem ruído)
                forecast_regime = []
                y_hist = list(y_last)

                for h in range(steps):
                    y_pred = const
                    for lag, coeff in enumerate(ar_coeffs, start=1):
                        if lag <= len(y_hist):
                            y_pred += coeff * y_hist[-lag]

                    forecast_regime.append(y_pred)
                    y_hist.append(y_pred)

                forecasts_by_regime[f'regime_{regime}'] = np.array(forecast_regime)

            except Exception as e:
                # Se falhar, retorna NaN
                forecasts_by_regime[f'regime_{regime}'] = np.full(steps, np.nan)

        return forecasts_by_regime

    def get_transition_matrix(self) -> np.ndarray:
        """
        Retorna matriz de transição de regimes.

        P[i,j] = P(s_t = j | s_{t-1} = i)
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi ajustado.")

        # Extrai da estimação
        # Statsmodels armazena em regime_transition
        P = np.zeros((self.k_regimes, self.k_regimes))

        for i in range(self.k_regimes):
            for j in range(self.k_regimes):
                param_name = f'p[{i}->{j}]'
                if param_name in self.model_fit_.params:
                    P[i, j] = self.model_fit_.params[param_name]

        # Normaliza linhas (garantir soma = 1)
        P = P / P.sum(axis=1, keepdims=True)

        return P

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


class RegimeAnalyzer:
    """
    Classe auxiliar para análise de regimes.
    """

    def __init__(self, model_fit):
        self.model_fit = model_fit
        self.k_regimes = model_fit.k_regimes

    def regime_durations(self, probabilities: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
        """
        Calcula duração dos períodos em cada regime.

        Parâmetros:
        -----------
        probabilities : DataFrame de probabilidades de regime
        threshold : probabilidade mínima para considerar "em regime"

        Retorna:
        --------
        DataFrame com início, fim e duração de cada período
        """
        results = []

        for regime in range(self.k_regimes):
            col = f'regime_{regime}'
            in_regime = probabilities[col] > threshold

            # Identifica blocos consecutivos
            blocks = (in_regime != in_regime.shift()).cumsum()
            regime_blocks = blocks[in_regime]

            for block_id in regime_blocks.unique():
                block_data = probabilities[regime_blocks == block_id]
                results.append({
                    'regime': regime,
                    'start': block_data.index[0],
                    'end': block_data.index[-1],
                    'duration': len(block_data),
                    'avg_probability': block_data[col].mean()
                })

        return pd.DataFrame(results)

    def regime_statistics(self, data: pd.Series, probabilities: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula estatísticas da variável em cada regime.

        Retorna:
        --------
        DataFrame com média, desvio, min, max por regime
        """
        stats = []

        for regime in range(self.k_regimes):
            col = f'regime_{regime}'
            # Média ponderada pelas probabilidades
            weights = probabilities[col]
            weighted_mean = (data * weights).sum() / weights.sum()
            weighted_std = np.sqrt(((data - weighted_mean) ** 2 * weights).sum() / weights.sum())

            stats.append({
                'regime': regime,
                'mean': weighted_mean,
                'std': weighted_std,
                'freq': (probabilities[col] > 0.5).sum() / len(probabilities)
            })

        return pd.DataFrame(stats)
