"""
Modelos baseados em árvores para séries temporais.

Implementa:
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting (opcional)
- Feature engineering para séries temporais
- Previsão multi-horizonte (direta e recursiva)
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from typing import Dict, Optional, List, Tuple
import warnings


class TreeForecaster:
    """
    Modelos de árvores para previsão de séries temporais.
    """

    def __init__(self, model_type: str = 'random_forest', max_lag: int = 12,
                **model_params):
        """
        Parâmetros:
        -----------
        model_type : 'decision_tree', 'random_forest', ou 'gradient_boosting'
        max_lag : máximo de lags a incluir como features
        **model_params : parâmetros específicos do modelo (n_estimators, max_depth, etc.)
        """
        self.model_type = model_type.lower()
        self.max_lag = max_lag
        self.model_params = model_params

        self.model_ = None
        self.is_fitted = False

    def _build_features(self, target: pd.Series, exog: Optional[pd.DataFrame] = None,
                       target_lags: Optional[List[int]] = None,
                       exog_lags: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Constrói matriz de features para árvores.

        Features:
        - Lags da variável alvo
        - Lags de variáveis exógenas
        - Médias móveis
        - Diferenças
        """
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

        # Diferenças
        features['y_diff_1'] = target.diff()
        if self.max_lag >= 12:
            features['y_diff_12'] = target.diff(12)

        # Estatísticas móveis
        for window in [6, 12]:
            if window <= self.max_lag:
                features[f'y_std_{window}'] = target.rolling(window=window).std()
                features[f'y_min_{window}'] = target.rolling(window=window).min()
                features[f'y_max_{window}'] = target.rolling(window=window).max()

        # Exógenas e seus lags
        if exog is not None:
            if exog_lags is None:
                exog_lags = list(range(1, min(self.max_lag, 6) + 1))

            for col in exog.columns:
                for lag in exog_lags:
                    features[f'{col}_lag_{lag}'] = exog[col].shift(lag)

                # Médias móveis das exógenas
                for window in [3, 6]:
                    features[f'{col}_ma_{window}'] = exog[col].rolling(window=window).mean()

        X = pd.DataFrame(features, index=target.index)
        return X

    def _get_model(self):
        """Retorna modelo sklearn apropriado."""
        if self.model_type == 'decision_tree':
            default_params = {'max_depth': 10, 'min_samples_split': 20, 'min_samples_leaf': 10}
            params = {**default_params, **self.model_params}
            return DecisionTreeRegressor(**params, random_state=42)

        elif self.model_type == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'max_features': 'sqrt',
                'n_jobs': -1
            }
            params = {**default_params, **self.model_params}
            return RandomForestRegressor(**params, random_state=42)

        elif self.model_type == 'gradient_boosting':
            default_params = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8
            }
            params = {**default_params, **self.model_params}
            return GradientBoostingRegressor(**params, random_state=42)

        else:
            raise ValueError(f"model_type inválido: {self.model_type}")

    def fit(self, target: pd.Series, exog: Optional[pd.DataFrame] = None,
           target_lags: Optional[List[int]] = None,
           exog_lags: Optional[List[int]] = None,
           verbose: bool = False) -> 'TreeForecaster':
        """
        Ajusta modelo de árvore.

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

        if verbose:
            print(f"Features criadas: {len(self.feature_names_)}")
            print(f"Amostras de treino: {len(X)}")

        # Treina modelo
        if verbose:
            print(f"\nTreinando modelo {self.model_type.upper()}...")

        self.model_ = self._get_model()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(X, y)

        self.is_fitted = True

        # Score in-sample
        self.train_score_ = self.model_.score(X, y)

        if verbose:
            print(f"  R² in-sample: {self.train_score_:.4f}")

        return self

    def forecast_recursive(self, target: pd.Series, steps: int = 12,
                          exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Previsão recursiva multi-horizonte.

        A cada passo, usa a previsão anterior como input.
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi ajustado.")

        forecasts = []
        target_extended = target.copy()

        # Índice futuro
        last_date = target.index[-1]
        if isinstance(last_date, pd.Timestamp):
            freq = pd.infer_freq(target.index) or 'MS'
            future_index = pd.date_range(start=last_date, periods=steps + 1, freq=freq)[1:]
        else:
            future_index = range(len(target), len(target) + steps)

        for h in range(steps):
            # Constrói features com dados atualizados
            X_current = self._build_features(target_extended, exog=exog)
            X_current = X_current.iloc[[-1]][self.feature_names_]

            if X_current.isna().any().any():
                forecasts.append(np.nan)
                continue

            # Prediz
            y_pred = self.model_.predict(X_current)[0]
            forecasts.append(y_pred)

            # Adiciona previsão ao histórico
            if isinstance(future_index[h], pd.Timestamp):
                target_extended.loc[future_index[h]] = y_pred
            else:
                target_extended = pd.concat([
                    target_extended,
                    pd.Series([y_pred], index=[future_index[h]])
                ])

        result_df = pd.DataFrame({
            'forecast': forecasts
        }, index=future_index)

        return result_df

    def get_feature_importance(self, top_k: Optional[int] = None) -> pd.DataFrame:
        """
        Retorna importância das features.

        Para Random Forest: importância baseada em Gini/MSE reduction.
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi ajustado.")

        if not hasattr(self.model_, 'feature_importances_'):
            raise ValueError("Modelo não possui feature_importances_")

        importance = self.model_.feature_importances_

        feature_importance = pd.DataFrame({
            'feature': self.feature_names_,
            'importance': importance
        }).sort_values('importance', ascending=False)

        if top_k is not None:
            feature_importance = feature_importance.head(top_k)

        return feature_importance


class MultiHorizonTree:
    """
    Treina um modelo de árvore separado para cada horizonte (Direct Multi-Step).
    """

    def __init__(self, model_type: str = 'random_forest', max_lag: int = 12,
                max_horizon: int = 12, **model_params):
        self.model_type = model_type
        self.max_lag = max_lag
        self.max_horizon = max_horizon
        self.model_params = model_params

        self.models = {}

    def fit(self, target: pd.Series, exog: Optional[pd.DataFrame] = None,
           verbose: bool = False):
        """
        Treina um modelo para cada horizonte h = 1, ..., max_horizon.
        """
        for h in range(1, self.max_horizon + 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Treinando modelo para horizonte h={h}")
                print(f"{'='*60}")

            # Cria target deslocado
            target_h = target.shift(-h)

            # Treina modelo
            model_h = TreeForecaster(
                model_type=self.model_type,
                max_lag=self.max_lag,
                **self.model_params
            )

            # Constrói features
            X = model_h._build_features(target, exog=exog)
            y_h = target_h.loc[X.index]

            valid_idx = X.notna().all(axis=1) & y_h.notna()
            X_valid = X[valid_idx]
            y_valid = y_h[valid_idx]

            # Treina
            model_sklearn = model_h._get_model()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_sklearn.fit(X_valid, y_valid)

            self.models[h] = {
                'model': model_sklearn,
                'feature_names': X_valid.columns.tolist(),
                'score': model_sklearn.score(X_valid, y_valid)
            }

            if verbose:
                print(f"  R² in-sample: {self.models[h]['score']:.4f}")

        if verbose:
            print(f"\n✓ {len(self.models)} modelos treinados (h=1 a {self.max_horizon})")

    def forecast(self, target: pd.Series, exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Gera previsões para todos os horizontes.
        """
        forecasts = {}

        # Constrói features com dados mais recentes
        # Usa primeiro modelo para construir (todos têm mesma estrutura)
        model_h1 = TreeForecaster(model_type=self.model_type, max_lag=self.max_lag)
        X_current = model_h1._build_features(target, exog=exog)
        X_current = X_current.iloc[[-1]]  # Última linha

        for h, model_dict in self.models.items():
            model = model_dict['model']
            feature_names = model_dict['feature_names']

            X_h = X_current[feature_names]

            if X_h.isna().any().any():
                forecasts[f'h{h}'] = np.nan
                continue

            y_pred = model.predict(X_h)[0]
            forecasts[f'h{h}'] = y_pred

        return pd.DataFrame([forecasts])

    def get_feature_importance(self, horizon: int, top_k: Optional[int] = None) -> pd.DataFrame:
        """
        Retorna importância das features para um horizonte específico.
        """
        if horizon not in self.models:
            raise ValueError(f"Horizonte {horizon} não foi treinado.")

        model_dict = self.models[horizon]
        model = model_dict['model']
        feature_names = model_dict['feature_names']

        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Modelo não possui feature_importances_")

        importance = model.feature_importances_

        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        if top_k is not None:
            feature_importance = feature_importance.head(top_k)

        return feature_importance


class QuantileRandomForest:
    """
    Random Forest para Regressão Quantílica.

    Implementação simplificada que usa as árvores individuais para estimar quantis.
    """

    def __init__(self, max_lag: int = 12, n_estimators: int = 100, **rf_params):
        self.max_lag = max_lag
        self.n_estimators = n_estimators
        self.rf_params = rf_params

        self.model_ = None
        self.is_fitted = False

    def _build_features(self, target: pd.Series, exog: Optional[pd.DataFrame] = None):
        """Mesma lógica de TreeForecaster."""
        forecaster = TreeForecaster(max_lag=self.max_lag)
        return forecaster._build_features(target, exog=exog)

    def fit(self, target: pd.Series, exog: Optional[pd.DataFrame] = None,
           verbose: bool = False):
        """Treina Random Forest."""
        if verbose:
            print(f"Construindo features para Quantile RF...")

        X = self._build_features(target, exog=exog)
        y = target.loc[X.index]

        valid_idx = X.notna().all(axis=1) & y.notna()
        X = X[valid_idx]
        y = y[valid_idx]

        self.feature_names_ = X.columns.tolist()

        if verbose:
            print(f"Treinando Quantile Random Forest ({self.n_estimators} árvores)...")

        default_params = {
            'n_estimators': self.n_estimators,
            'max_depth': 15,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'n_jobs': -1
        }
        params = {**default_params, **self.rf_params}

        self.model_ = RandomForestRegressor(**params, random_state=42)
        self.model_.fit(X, y)

        self.is_fitted = True

        if verbose:
            print(f"  R² in-sample: {self.model_.score(X, y):.4f}")

    def predict_quantiles(self, target: pd.Series, quantiles: List[float] = [0.1, 0.5, 0.9],
                         exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prediz quantis usando distribuição das previsões das árvores.

        Parâmetros:
        -----------
        quantiles : lista de quantis (ex: [0.1, 0.5, 0.9])

        Retorna:
        --------
        DataFrame com uma coluna para cada quantil
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi ajustado.")

        # Constrói features
        X = self._build_features(target, exog=exog)
        X = X.iloc[[-1]][self.feature_names_]

        if X.isna().any().any():
            return pd.DataFrame({f'q{q}': [np.nan] for q in quantiles})

        # Predições de cada árvore
        tree_predictions = np.array([
            tree.predict(X)[0] for tree in self.model_.estimators_
        ])

        # Calcula quantis
        quantile_predictions = {}
        for q in quantiles:
            quantile_predictions[f'q{q}'] = [np.quantile(tree_predictions, q)]

        return pd.DataFrame(quantile_predictions)
