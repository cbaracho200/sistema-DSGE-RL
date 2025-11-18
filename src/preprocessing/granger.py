"""
Módulo para testes de causalidade de Granger e seleção de variáveis.

Implementa:
- Teste de causalidade de Granger
- Seleção das top-K variáveis por F-statistic
- Cálculo de fator preliminar via PCA
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional
import warnings


class GrangerSelector:
    """
    Classe para seleção de variáveis via teste de causalidade de Granger.
    """

    def __init__(self, max_lag: int = 4, alpha: float = 0.05, max_vars: int = 5):
        """
        Parâmetros:
        -----------
        max_lag : int
            Máximo de lags a considerar no teste de Granger
        alpha : float
            Nível de significância
        max_vars : int
            Número máximo de variáveis a selecionar
        """
        self.max_lag = max_lag
        self.alpha = alpha
        self.max_vars = max_vars
        self.granger_results_ = {}
        self.selected_vars_ = []
        self.factor_ = None
        self.scaler_ = StandardScaler()

    def compute_preliminary_factor(self, df: pd.DataFrame,
                                   n_components: int = 1,
                                   verbose: bool = False) -> pd.Series:
        """
        Calcula fator preliminar via PCA.

        Parâmetros:
        -----------
        df : DataFrame com séries estacionárias (já padronizadas)
        n_components : número de componentes (default: 1 = primeiro fator)

        Retorna:
        --------
        Series com o fator f_t^(0)
        """
        # Padroniza
        X_scaled = self.scaler_.fit_transform(df)

        # PCA
        pca = PCA(n_components=n_components)
        factor = pca.fit_transform(X_scaled)

        if verbose:
            print(f"Variância explicada pelo 1º componente: {pca.explained_variance_ratio_[0]:.2%}")

        # Retorna como Series
        factor_series = pd.Series(
            factor[:, 0],
            index=df.index,
            name='preliminary_factor'
        )

        self.factor_ = factor_series
        self.pca_ = pca

        return factor_series

    def granger_test(self, effect_series: pd.Series, cause_series: pd.Series,
                    max_lag: Optional[int] = None,
                    verbose: bool = False) -> Dict:
        """
        Teste de causalidade de Granger: cause_series -> effect_series.

        H0: cause_series NÃO Granger-causa effect_series
        H1: cause_series Granger-causa effect_series

        Retorna:
        --------
        dict com estatísticas do teste
        """
        if max_lag is None:
            max_lag = self.max_lag

        # Monta DataFrame com ambas séries
        data = pd.DataFrame({
            'effect': effect_series,
            'cause': cause_series
        }).dropna()

        if len(data) < max_lag + 10:
            raise ValueError(f"Dados insuficientes para teste com {max_lag} lags")

        # Executa teste
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gc_result = grangercausalitytests(
                data[['effect', 'cause']],
                maxlag=max_lag,
                verbose=False
            )

        # Extrai resultados de cada lag
        results_by_lag = {}
        min_pvalue = 1.0
        best_lag = 1
        best_fstat = 0.0

        for lag in range(1, max_lag + 1):
            # F-test é o mais usado: gc_result[lag][0]['ssr_ftest']
            # Formato: (F-statistic, p-value, df_denom, df_num)
            ftest = gc_result[lag][0]['ssr_ftest']

            f_stat = ftest[0]
            p_value = ftest[1]

            results_by_lag[lag] = {
                'f_statistic': f_stat,
                'pvalue': p_value,
                'df_denom': ftest[2],
                'df_num': ftest[3]
            }

            if p_value < min_pvalue:
                min_pvalue = p_value
                best_lag = lag
                best_fstat = f_stat

        granger_causes = min_pvalue < self.alpha

        if verbose:
            print(f"Granger Test: {cause_series.name} -> {effect_series.name}")
            print(f"  Best lag: {best_lag}")
            print(f"  F-statistic: {best_fstat:.4f}")
            print(f"  p-value: {min_pvalue:.4f}")
            print(f"  Granger-causes: {granger_causes}")

        return {
            'granger_causes': granger_causes,
            'min_pvalue': min_pvalue,
            'best_lag': best_lag,
            'best_fstat': best_fstat,
            'results_by_lag': results_by_lag
        }

    def test_all_variables(self, df: pd.DataFrame, target: Optional[pd.Series] = None,
                          use_pca_factor: bool = True,
                          verbose: bool = False) -> pd.DataFrame:
        """
        Testa causalidade de Granger de todas as variáveis para um alvo.

        Se target=None e use_pca_factor=True, usa fator preliminar via PCA.

        Retorna:
        --------
        DataFrame com resultados ordenados por F-statistic
        """
        # Define alvo
        if target is None:
            if use_pca_factor:
                if verbose:
                    print("Calculando fator preliminar via PCA...")
                target = self.compute_preliminary_factor(df, verbose=verbose)
            else:
                raise ValueError("Deve fornecer target ou usar use_pca_factor=True")

        # Testa cada variável
        results_list = []

        for col in df.columns:
            if verbose:
                print(f"\nTestando {col}...")

            try:
                result = self.granger_test(
                    effect_series=target,
                    cause_series=df[col],
                    verbose=verbose
                )

                results_list.append({
                    'variable': col,
                    'granger_causes': result['granger_causes'],
                    'pvalue': result['min_pvalue'],
                    'f_statistic': result['best_fstat'],
                    'best_lag': result['best_lag']
                })

                # Guarda resultado completo
                self.granger_results_[col] = result

            except Exception as e:
                if verbose:
                    print(f"  ⚠ Erro ao testar {col}: {e}")
                continue

        # DataFrame de resultados
        results_df = pd.DataFrame(results_list)

        # Ordena por F-statistic (maior = mais importante)
        results_df = results_df.sort_values('f_statistic', ascending=False)

        return results_df

    def select_top_k(self, df: pd.DataFrame, k: Optional[int] = None,
                    target: Optional[pd.Series] = None,
                    use_pca_factor: bool = True,
                    only_significant: bool = True,
                    verbose: bool = False) -> Tuple[List[str], pd.DataFrame]:
        """
        Seleciona as top-K variáveis que Granger-causam o alvo.

        Parâmetros:
        -----------
        k : int
            Número de variáveis (default: self.max_vars)
        only_significant : bool
            Se True, considera apenas variáveis com p < alpha

        Retorna:
        --------
        (lista_de_nomes, dataframe_de_resultados)
        """
        if k is None:
            k = self.max_vars

        # Testa todas
        results_df = self.test_all_variables(
            df, target=target, use_pca_factor=use_pca_factor, verbose=verbose
        )

        # Filtra apenas significantes (se solicitado)
        if only_significant:
            results_df = results_df[results_df['granger_causes'] == True]

        # Pega top-K
        top_k = results_df.head(k)

        selected_vars = top_k['variable'].tolist()
        self.selected_vars_ = selected_vars

        if verbose:
            print(f"\n{'='*60}")
            print(f"Variáveis selecionadas (top-{k}):")
            print(f"{'='*60}")
            for i, row in top_k.iterrows():
                print(f"{row['variable']:30s} | F={row['f_statistic']:8.2f} | p={row['pvalue']:.4f}")

        return selected_vars, top_k

    def fit(self, df: pd.DataFrame, k: Optional[int] = None,
           target: Optional[pd.Series] = None,
           use_pca_factor: bool = True,
           verbose: bool = False) -> 'GrangerSelector':
        """
        Ajusta o seletor: calcula fator preliminar e seleciona variáveis.
        """
        self.select_top_k(
            df, k=k, target=target, use_pca_factor=use_pca_factor, verbose=verbose
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Retorna DataFrame apenas com variáveis selecionadas.
        """
        if not self.selected_vars_:
            raise ValueError("Deve chamar fit() antes de transform()")

        return df[self.selected_vars_].copy()

    def fit_transform(self, df: pd.DataFrame, k: Optional[int] = None,
                     target: Optional[pd.Series] = None,
                     use_pca_factor: bool = True,
                     verbose: bool = False) -> pd.DataFrame:
        """
        Ajusta e transforma de uma vez.
        """
        self.fit(df, k=k, target=target, use_pca_factor=use_pca_factor, verbose=verbose)
        return self.transform(df)

    def get_selected_variables(self) -> List[str]:
        """Retorna lista de variáveis selecionadas."""
        return self.selected_vars_.copy()

    def get_granger_results(self, variable: Optional[str] = None) -> Dict:
        """
        Retorna resultados de Granger.

        Se variable=None, retorna todos.
        """
        if variable is None:
            return self.granger_results_.copy()
        return self.granger_results_.get(variable, {})
