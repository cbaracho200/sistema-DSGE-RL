"""
Módulo para testes de estacionaridade e diferenciação de séries temporais.

Implementa:
- Testes ADF (Augmented Dickey-Fuller)
- Testes KPSS (Kwiatkowski-Phillips-Schmidt-Shin)
- Diferenciação automática (ordem mínima)
- Diferenciação sazonal
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from typing import Dict, Tuple, Optional, List
import warnings


class StationarityTester:
    """
    Classe para testar estacionaridade e transformar séries temporais.
    """

    def __init__(self, alpha: float = 0.05, max_diff: int = 2):
        """
        Parâmetros:
        -----------
        alpha : float
            Nível de significância para os testes (padrão: 5%)
        max_diff : int
            Máximo de diferenciações a tentar (padrão: 2)
        """
        self.alpha = alpha
        self.max_diff = max_diff
        self.transformations_ = {}

    def adf_test(self, series: pd.Series, verbose: bool = False) -> Dict:
        """
        Teste ADF para raiz unitária.

        H0: série tem raiz unitária (não estacionária)
        H1: série é estacionária

        Retorna:
        --------
        dict com 'statistic', 'pvalue', 'is_stationary'
        """
        # Remove NaN
        series_clean = series.dropna()

        if len(series_clean) < 12:
            raise ValueError("Série muito curta para teste ADF (mín. 12 obs)")

        result = adfuller(series_clean, autolag='AIC')

        is_stationary = result[1] < self.alpha  # Rejeita H0

        if verbose:
            print(f"ADF Test: statistic={result[0]:.4f}, p-value={result[1]:.4f}")
            print(f"Estacionária: {is_stationary}")

        return {
            'statistic': result[0],
            'pvalue': result[1],
            'usedlag': result[2],
            'nobs': result[3],
            'critical_values': result[4],
            'is_stationary': is_stationary
        }

    def kpss_test(self, series: pd.Series, regression: str = 'c', verbose: bool = False) -> Dict:
        """
        Teste KPSS para estacionaridade.

        H0: série é estacionária
        H1: série tem raiz unitária

        Parâmetros:
        -----------
        regression : str
            'c' para constante, 'ct' para constante e tendência
        """
        series_clean = series.dropna()

        if len(series_clean) < 12:
            raise ValueError("Série muito curta para teste KPSS (mín. 12 obs)")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = kpss(series_clean, regression=regression, nlags='auto')

        is_stationary = result[1] >= self.alpha  # Não rejeita H0

        if verbose:
            print(f"KPSS Test: statistic={result[0]:.4f}, p-value={result[1]:.4f}")
            print(f"Estacionária: {is_stationary}")

        return {
            'statistic': result[0],
            'pvalue': result[1],
            'lags': result[2],
            'critical_values': result[3],
            'is_stationary': is_stationary
        }

    def test_stationarity(self, series: pd.Series, use_kpss: bool = True,
                         verbose: bool = False) -> Dict:
        """
        Testa estacionaridade combinando ADF e opcionalmente KPSS.

        Critério conservador:
        - ADF: rejeita H0 (é estacionária)
        - KPSS: não rejeita H0 (é estacionária)
        - Conclusão: ambos concordam
        """
        adf_result = self.adf_test(series, verbose=verbose)

        if not use_kpss:
            return {
                'adf': adf_result,
                'is_stationary': adf_result['is_stationary']
            }

        kpss_result = self.kpss_test(series, verbose=verbose)

        # Ambos concordam?
        is_stationary = adf_result['is_stationary'] and kpss_result['is_stationary']

        return {
            'adf': adf_result,
            'kpss': kpss_result,
            'is_stationary': is_stationary,
            'agreement': adf_result['is_stationary'] == kpss_result['is_stationary']
        }

    def difference(self, series: pd.Series, order: int = 1) -> pd.Series:
        """Diferenciação de ordem d."""
        result = series.copy()
        for _ in range(order):
            result = result.diff()
        return result

    def seasonal_difference(self, series: pd.Series, period: int = 12,
                           order: int = 1) -> pd.Series:
        """Diferenciação sazonal."""
        result = series.copy()
        for _ in range(order):
            result = result.diff(period)
        return result

    def find_min_diff_order(self, series: pd.Series, max_order: Optional[int] = None,
                           seasonal: bool = False, period: int = 12,
                           verbose: bool = False) -> Tuple[int, pd.Series]:
        """
        Encontra a ordem mínima de diferenciação para tornar série estacionária.

        Retorna:
        --------
        (ordem_diff, série_transformada)
        """
        if max_order is None:
            max_order = self.max_diff

        # Testa série original
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testando série original (ordem 0)")
            print(f"{'='*60}")

        test_result = self.test_stationarity(series, verbose=verbose)
        if test_result['is_stationary']:
            if verbose:
                print("✓ Série já é estacionária!")
            return 0, series

        # Tenta diferenciações
        for d in range(1, max_order + 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Testando diferença de ordem {d}")
                print(f"{'='*60}")

            if seasonal:
                diff_series = self.seasonal_difference(series, period=period, order=d)
            else:
                diff_series = self.difference(series, order=d)

            # Remove NaN iniciais
            diff_series = diff_series.dropna()

            if len(diff_series) < 12:
                if verbose:
                    print(f"⚠ Série muito curta após {d} diferenciações")
                continue

            test_result = self.test_stationarity(diff_series, verbose=verbose)

            if test_result['is_stationary']:
                if verbose:
                    print(f"✓ Estacionária com {d} diferenciação(ões)!")
                return d, diff_series

        # Se não conseguiu, retorna última tentativa com aviso
        if verbose:
            print(f"\n⚠ Não conseguiu estacionaridade até ordem {max_order}")
            print(f"Retornando diferença de ordem {max_order}")

        if seasonal:
            final_series = self.seasonal_difference(series, period=period, order=max_order)
        else:
            final_series = self.difference(series, order=max_order)

        return max_order, final_series.dropna()

    def make_stationary(self, series: pd.Series, name: str = None,
                       try_seasonal: bool = True, period: int = 12,
                       verbose: bool = False) -> Tuple[pd.Series, Dict]:
        """
        Torna série estacionária escolhendo melhor transformação.

        Estratégia:
        1. Testa série original
        2. Tenta diferença regular
        3. Tenta diferença sazonal (se habilitado)
        4. Tenta diferença composta (regular + sazonal)

        Retorna:
        --------
        (série_estacionária, info_transformação)
        """
        if name is None:
            name = series.name or "unnamed"

        if verbose:
            print(f"\n{'#'*60}")
            print(f"# Processando série: {name}")
            print(f"{'#'*60}")

        # 1. Testa original
        test_orig = self.test_stationarity(series, verbose=verbose)
        if test_orig['is_stationary']:
            transform_info = {
                'type': 'none',
                'order': 0,
                'seasonal_order': 0,
                'period': None
            }
            self.transformations_[name] = transform_info
            return series, transform_info

        # 2. Diferença regular
        if verbose:
            print("\n" + "="*60)
            print("Tentando diferenciação regular...")
            print("="*60)

        d_reg, series_reg = self.find_min_diff_order(series, verbose=verbose)

        if d_reg > 0:
            test_reg = self.test_stationarity(series_reg, verbose=False)
            if test_reg['is_stationary']:
                transform_info = {
                    'type': 'regular',
                    'order': d_reg,
                    'seasonal_order': 0,
                    'period': None
                }
                self.transformations_[name] = transform_info
                return series_reg, transform_info

        if not try_seasonal:
            transform_info = {
                'type': 'regular',
                'order': d_reg,
                'seasonal_order': 0,
                'period': None,
                'warning': 'not_fully_stationary'
            }
            self.transformations_[name] = transform_info
            return series_reg, transform_info

        # 3. Diferença sazonal
        if verbose:
            print("\n" + "="*60)
            print(f"Tentando diferenciação sazonal (período={period})...")
            print("="*60)

        D_seas, series_seas = self.find_min_diff_order(
            series, seasonal=True, period=period, verbose=verbose
        )

        if D_seas > 0:
            test_seas = self.test_stationarity(series_seas, verbose=False)
            if test_seas['is_stationary']:
                transform_info = {
                    'type': 'seasonal',
                    'order': 0,
                    'seasonal_order': D_seas,
                    'period': period
                }
                self.transformations_[name] = transform_info
                return series_seas, transform_info

        # 4. Diferença composta: (1-L)(1-L^s)
        if verbose:
            print("\n" + "="*60)
            print(f"Tentando diferenciação composta...")
            print("="*60)

        # Aplica diferença regular primeiro
        series_temp = self.difference(series, order=1)
        # Depois sazonal
        series_comp = self.seasonal_difference(series_temp, period=period, order=1)
        series_comp = series_comp.dropna()

        test_comp = self.test_stationarity(series_comp, verbose=verbose)

        transform_info = {
            'type': 'composite',
            'order': 1,
            'seasonal_order': 1,
            'period': period,
            'is_stationary': test_comp['is_stationary']
        }
        self.transformations_[name] = transform_info

        return series_comp, transform_info

    def fit_transform(self, df: pd.DataFrame, try_seasonal: bool = True,
                     period: int = 12, verbose: bool = False) -> pd.DataFrame:
        """
        Transforma todas as colunas de um DataFrame para estacionaridade.

        Retorna:
        --------
        DataFrame com séries estacionárias
        """
        result_dict = {}

        for col in df.columns:
            if verbose:
                print(f"\n{'#'*80}")
                print(f"Processando coluna: {col}")
                print(f"{'#'*80}")

            series_stat, info = self.make_stationary(
                df[col],
                name=col,
                try_seasonal=try_seasonal,
                period=period,
                verbose=verbose
            )

            result_dict[col] = series_stat

        # Alinha índices (remove NaN iniciais)
        result_df = pd.DataFrame(result_dict)
        result_df = result_df.dropna()

        return result_df

    def get_transformation_info(self, name: str) -> Dict:
        """Retorna informação sobre transformação aplicada."""
        return self.transformations_.get(name, None)

    def get_all_transformations(self) -> Dict:
        """Retorna todas as transformações aplicadas."""
        return self.transformations_.copy()
