"""
Testes unitários para módulos de forecasting.

Execute com: pytest tests/test_forecasting.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from forecasting.arima import ARIMAForecaster
from forecasting.sarima import SARIMAForecaster
from forecasting.ridge import RidgeForecaster
from forecasting.random_forest import RandomForestForecaster
from evaluation.metrics import calculate_metrics


@pytest.fixture
def sample_data():
    """Gera dados sintéticos para testes."""
    np.random.seed(42)
    n = 100

    # Série temporal simples
    t = np.arange(n)
    y = 10 + 0.5*t + 2*np.sin(2*np.pi*t/12) + np.random.normal(0, 1, n)

    # Features
    X = np.column_stack([
        np.random.normal(0, 1, n),
        np.random.normal(0, 1, n),
        np.random.normal(0, 1, n)
    ])

    # Split
    train_size = 80
    y_train = y[:train_size]
    y_test = y[train_size:]
    X_train = X[:train_size]
    X_test = X[train_size:]

    return {
        'y_train': y_train,
        'y_test': y_test,
        'X_train': X_train,
        'X_test': X_test,
        'n_test': len(y_test)
    }


class TestARIMAForecaster:
    """Testes para ARIMAForecaster."""

    def test_initialization(self):
        """Testa inicialização do modelo."""
        model = ARIMAForecaster(order=(1, 0, 1))
        assert model.order == (1, 0, 1)
        assert model.model_ is None

    def test_fit_and_forecast(self, sample_data):
        """Testa fit e forecast."""
        model = ARIMAForecaster(order=(1, 0, 1))
        model.fit(sample_data['y_train'])

        # Verificar que o modelo foi ajustado
        assert model.model_ is not None

        # Fazer previsão
        predictions = model.forecast(sample_data['n_test'])

        # Verificar formato
        assert len(predictions) == sample_data['n_test']
        assert isinstance(predictions, np.ndarray)
        assert not np.isnan(predictions).any()

    def test_forecast_without_fit(self, sample_data):
        """Testa que forecast falha sem fit."""
        model = ARIMAForecaster(order=(1, 0, 1))

        with pytest.raises(ValueError):
            model.forecast(10)


class TestSARIMAForecaster:
    """Testes para SARIMAForecaster."""

    def test_initialization(self):
        """Testa inicialização."""
        model = SARIMAForecaster(order=(1, 0, 1), seasonal_order=(1, 0, 1, 12))
        assert model.order == (1, 0, 1)
        assert model.seasonal_order == (1, 0, 1, 12)

    def test_fit_and_forecast(self, sample_data):
        """Testa fit e forecast."""
        model = SARIMAForecaster(order=(1, 0, 0), seasonal_order=(1, 0, 0, 12))
        model.fit(sample_data['y_train'])

        predictions = model.forecast(sample_data['n_test'])

        assert len(predictions) == sample_data['n_test']
        assert not np.isnan(predictions).any()


class TestRidgeForecaster:
    """Testes para RidgeForecaster."""

    def test_initialization(self):
        """Testa inicialização."""
        model = RidgeForecaster(alpha=1.0, lags=3)
        assert model.alpha == 1.0
        assert model.lags == 3

    def test_fit_and_forecast(self, sample_data):
        """Testa fit e forecast."""
        model = RidgeForecaster(alpha=1.0, lags=3)
        model.fit(sample_data['X_train'], sample_data['y_train'])

        predictions = model.forecast(sample_data['X_test'])

        assert len(predictions) == len(sample_data['X_test'])
        assert not np.isnan(predictions).any()

    def test_different_alpha_values(self, sample_data):
        """Testa diferentes valores de alpha."""
        alphas = [0.1, 1.0, 10.0]
        predictions = []

        for alpha in alphas:
            model = RidgeForecaster(alpha=alpha, lags=3)
            model.fit(sample_data['X_train'], sample_data['y_train'])
            pred = model.forecast(sample_data['X_test'])
            predictions.append(pred)

        # Previsões devem ser diferentes para diferentes alphas
        assert not np.allclose(predictions[0], predictions[1])
        assert not np.allclose(predictions[1], predictions[2])


class TestRandomForestForecaster:
    """Testes para RandomForestForecaster."""

    def test_initialization(self):
        """Testa inicialização."""
        model = RandomForestForecaster(n_estimators=100, max_depth=10, lags=5)
        assert model.n_estimators == 100
        assert model.max_depth == 10
        assert model.lags == 5

    def test_fit_and_forecast(self, sample_data):
        """Testa fit e forecast."""
        model = RandomForestForecaster(n_estimators=50, max_depth=5, lags=3, random_state=42)
        model.fit(sample_data['X_train'], sample_data['y_train'])

        predictions = model.forecast(sample_data['X_test'])

        assert len(predictions) == len(sample_data['X_test'])
        assert not np.isnan(predictions).any()

    def test_feature_importance(self, sample_data):
        """Testa cálculo de feature importance."""
        model = RandomForestForecaster(n_estimators=50, lags=3, random_state=42)
        model.fit(sample_data['X_train'], sample_data['y_train'])

        feature_names = ['feature1', 'feature2', 'feature3']
        importance = model.feature_importance(feature_names)

        assert isinstance(importance, dict)
        assert len(importance) == len(feature_names)
        assert all(0 <= v <= 1 for v in importance.values())
        assert abs(sum(importance.values()) - 1.0) < 0.01  # Soma próxima a 1


class TestMetrics:
    """Testes para módulo de métricas."""

    def test_calculate_metrics_perfect_prediction(self):
        """Testa métricas com previsão perfeita."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        metrics = calculate_metrics(y_true, y_pred)

        assert metrics['mae'] == pytest.approx(0, abs=1e-10)
        assert metrics['rmse'] == pytest.approx(0, abs=1e-10)
        assert metrics['mape'] == pytest.approx(0, abs=1e-10)
        assert metrics['r2'] == pytest.approx(1.0, abs=1e-10)

    def test_calculate_metrics_random_prediction(self):
        """Testa métricas com previsão aleatória."""
        np.random.seed(42)
        y_true = np.random.normal(10, 2, 100)
        y_pred = np.random.normal(10, 2, 100)

        metrics = calculate_metrics(y_true, y_pred)

        # Verificar que todas as métricas foram calculadas
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'mape' in metrics
        assert 'r2' in metrics

        # Verificar ranges razoáveis
        assert metrics['mae'] > 0
        assert metrics['rmse'] > 0
        assert metrics['rmse'] >= metrics['mae']  # RMSE sempre >= MAE
        assert -1 <= metrics['r2'] <= 1

    def test_calculate_metrics_with_zeros(self):
        """Testa métricas com zeros (pode causar divisão por zero em MAPE)."""
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1.1, 2.1, 3.1, 4.1])

        metrics = calculate_metrics(y_true, y_pred)

        # Verificar que não dá erro
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics


class TestIntegration:
    """Testes de integração."""

    def test_multiple_models_same_data(self, sample_data):
        """Testa múltiplos modelos nos mesmos dados."""
        models = [
            ARIMAForecaster(order=(1, 0, 1)),
            RidgeForecaster(alpha=1.0, lags=3),
            RandomForestForecaster(n_estimators=50, lags=3, random_state=42)
        ]

        results = {}

        for model in models:
            model_name = model.__class__.__name__

            if model_name == 'ARIMAForecaster':
                model.fit(sample_data['y_train'])
                pred = model.forecast(sample_data['n_test'])
            else:
                model.fit(sample_data['X_train'], sample_data['y_train'])
                pred = model.forecast(sample_data['X_test'])

            metrics = calculate_metrics(sample_data['y_test'], pred)
            results[model_name] = metrics

        # Verificar que todos os modelos produziram resultados
        assert len(results) == len(models)

        # Verificar que todos têm métricas válidas
        for model_name, metrics in results.items():
            assert metrics['rmse'] > 0
            assert metrics['mae'] > 0

    def test_reproducibility(self, sample_data):
        """Testa reproducibilidade com random_state."""
        model1 = RandomForestForecaster(n_estimators=50, lags=3, random_state=42)
        model1.fit(sample_data['X_train'], sample_data['y_train'])
        pred1 = model1.forecast(sample_data['X_test'])

        model2 = RandomForestForecaster(n_estimators=50, lags=3, random_state=42)
        model2.fit(sample_data['X_train'], sample_data['y_train'])
        pred2 = model2.forecast(sample_data['X_test'])

        # Previsões devem ser idênticas
        assert np.allclose(pred1, pred2)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
