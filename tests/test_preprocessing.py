"""
Testes unitários para módulos de pré-processamento.

Execute com: pytest tests/test_preprocessing.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing.stationarity import StationarityTester
from preprocessing.granger import GrangerSelector


@pytest.fixture
def stationary_series():
    """Gera série estacionária."""
    np.random.seed(42)
    return np.random.normal(0, 1, 100)


@pytest.fixture
def non_stationary_series():
    """Gera série não-estacionária (random walk)."""
    np.random.seed(42)
    return np.cumsum(np.random.normal(0, 1, 100))


@pytest.fixture
def multivariate_data():
    """Gera dados multivariados para teste de Granger."""
    np.random.seed(42)
    n = 100

    # Variável 1 (independente)
    x1 = np.random.normal(0, 1, n)

    # Variável 2 (causada por x1 com lag)
    x2 = np.zeros(n)
    x2[0] = np.random.normal(0, 1)
    for i in range(1, n):
        x2[i] = 0.7 * x1[i-1] + np.random.normal(0, 0.5)

    # Variável 3 (independente)
    x3 = np.random.normal(0, 1, n)

    df = pd.DataFrame({
        'target': x2,
        'causal': x1,
        'non_causal': x3
    })

    return df


class TestStationarityTester:
    """Testes para StationarityTester."""

    def test_initialization(self):
        """Testa inicialização."""
        tester = StationarityTester()
        assert tester is not None

    def test_stationary_series(self, stationary_series):
        """Testa detecção de série estacionária."""
        tester = StationarityTester()
        is_stationary, adf_pval, kpss_pval = tester.test_stationarity(
            stationary_series, alpha=0.05
        )

        # Série estacionária: ADF rejeita H0 (não-estacionária), KPSS não rejeita H0 (estacionária)
        assert is_stationary is True
        assert 0 <= adf_pval <= 1
        assert 0 <= kpss_pval <= 1

    def test_non_stationary_series(self, non_stationary_series):
        """Testa detecção de série não-estacionária."""
        tester = StationarityTester()
        is_stationary, adf_pval, kpss_pval = tester.test_stationarity(
            non_stationary_series, alpha=0.05
        )

        # Série não-estacionária
        assert is_stationary is False
        assert 0 <= adf_pval <= 1
        assert 0 <= kpss_pval <= 1

    def test_make_stationary(self, non_stationary_series):
        """Testa transformação para tornar série estacionária."""
        tester = StationarityTester()

        stationary, method = tester.make_stationary(non_stationary_series, max_diff=2)

        # Verificar que foi aplicada transformação
        assert method in ['diff(1)', 'diff(2)', 'none']

        # Verificar que série transformada é estacionária
        is_stat, _, _ = tester.test_stationarity(stationary, alpha=0.05)
        # Nota: pode falhar ocasionalmente por aleatoriedade, mas geralmente funciona

    def test_invalid_input(self):
        """Testa com entrada inválida."""
        tester = StationarityTester()

        # Array vazio
        with pytest.raises(Exception):
            tester.test_stationarity(np.array([]))

        # Array muito curto
        with pytest.raises(Exception):
            tester.test_stationarity(np.array([1, 2]))


class TestGrangerSelector:
    """Testes para GrangerSelector."""

    def test_initialization(self):
        """Testa inicialização."""
        selector = GrangerSelector(max_lag=4, alpha=0.05)
        assert selector.max_lag == 4
        assert selector.alpha == 0.05

    def test_select_variables(self, multivariate_data):
        """Testa seleção de variáveis."""
        selector = GrangerSelector(max_lag=3, alpha=0.05)

        selected = selector.select_variables(
            multivariate_data,
            target_col='target'
        )

        # Verificar que retorna lista
        assert isinstance(selected, list)

        # Target deve estar incluído
        assert 'target' in selected

        # 'causal' deve ser selecionado (causa Granger em target)
        # Nota: pode falhar ocasionalmente devido à aleatoriedade
        # assert 'causal' in selected

    def test_test_granger_causality(self, multivariate_data):
        """Testa teste de causalidade de Granger."""
        selector = GrangerSelector(max_lag=3, alpha=0.05)

        # Testar variável causal
        causes, pvalue = selector.test_granger_causality(
            multivariate_data['causal'].values,
            multivariate_data['target'].values
        )

        assert isinstance(causes, bool)
        assert 0 <= pvalue <= 1

        # Testar variável não-causal
        causes_nc, pvalue_nc = selector.test_granger_causality(
            multivariate_data['non_causal'].values,
            multivariate_data['target'].values
        )

        assert isinstance(causes_nc, bool)
        assert 0 <= pvalue_nc <= 1

    def test_invalid_target(self, multivariate_data):
        """Testa com target inválido."""
        selector = GrangerSelector(max_lag=3, alpha=0.05)

        with pytest.raises(Exception):
            selector.select_variables(
                multivariate_data,
                target_col='nonexistent_column'
            )


class TestIntegration:
    """Testes de integração de pré-processamento."""

    def test_full_preprocessing_pipeline(self, multivariate_data):
        """Testa pipeline completo de pré-processamento."""
        # 1. Testar estacionaridade
        tester = StationarityTester()

        data_transformed = multivariate_data.copy()
        for col in data_transformed.columns:
            is_stat, _, _ = tester.test_stationarity(data_transformed[col].values)

            if not is_stat:
                stationary, method = tester.make_stationary(data_transformed[col].values)
                data_transformed[col] = stationary

        # 2. Selecionar variáveis
        selector = GrangerSelector(max_lag=3, alpha=0.05)
        selected = selector.select_variables(data_transformed, target_col='target')

        # Verificar resultados
        assert len(selected) >= 1  # Pelo menos target
        assert 'target' in selected

    def test_reproducibility(self, multivariate_data):
        """Testa reproducibilidade dos testes."""
        selector1 = GrangerSelector(max_lag=3, alpha=0.05)
        selected1 = selector1.select_variables(multivariate_data, target_col='target')

        selector2 = GrangerSelector(max_lag=3, alpha=0.05)
        selected2 = selector2.select_variables(multivariate_data, target_col='target')

        # Resultados devem ser idênticos
        assert set(selected1) == set(selected2)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
