"""
Modelos lineares regularizados (Ridge/Lasso) para séries temporais.

Implementa:
- Ridge Regression (L2)
- Lasso Regression (L1)
- Elastic Net (combinação L1 + L2)
- Seleção automática de lags
- Time-Series Cross-Validation
- Previsão multi-horizonte
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Optional, List, Tuple
import warnings


class TimeSeriesFeatureBuilder:
    """
    Constrói features para modelos de séries temporais.

    Features:
    - Lags da variável alvo
    - Lags de variáveis exógenas
    - Médias móveis
    - Tendência temporal
    """

    def __init__(self, max_lag: int = 12, include_ma: bool = False, ma_windows: List[int] = None):
        """
        Parâmetros:
        -----------
        max_lag : máximo de lags a incluir
        include_ma : se True, inclui médias móveis
        ma_windows : lista de janelas para médias móveis (ex: [3, 6, 12])
        """
        self.max_lag = max_lag
        self.include_ma = include_ma
        self.ma_windows = ma_windows or [3, 6, 12]

    def build_lag_features(self, series: pd.Series, lags: Optional[List[int]] = None,
                          prefix: str = '') -> pd.DataFrame:
        """
        Cria features de lags.

        Parâmetros:
        -----------
        lags : lista de lags (ex: [1,2,3,12]). Se None, usa 1 até max_lag.
        prefix : prefixo para nomes das colunas
        """
        if lags is None:
            lags = list(range(1, self.max_lag + 1))

        features = {}
        for lag in lags:
            col_name = f'{prefix}lag_{lag}' if prefix else f'lag_{lag}'
            features[col_name] = series.shift(lag)

        return pd.DataFrame(features, index=series.index)

    def build_ma_features(self, series: pd.Series, windows: Optional[List[int]] = None,
                         prefix: str = '') -> pd.DataFrame:
        """Cria features de médias móveis."""
        if windows is None:
            windows = self.ma_windows

        features = {}
        for window in windows:
            col_name = f'{prefix}ma_{window}' if prefix else f'ma_{window}'
            features[col_name] = series.rolling(window=window).mean()

        return pd.DataFrame(features, index=series.index)

    def build_features(self, target: pd.Series, exog: Optional[pd.DataFrame] = None,
                      target_lags: Optional[List[int]] = None,
                      exog_lags: Optional[List[int]] = None,
                      include_trend: bool = False) -> pd.DataFrame:
        """
        Constrói matriz completa de features.

        Retorna:
        --------
        DataFrame com todas as features
        """
        feature_dfs = []

        # Lags do alvo
        if target_lags is None:
            target_lags = list(range(1, self.max_lag + 1))

        lag_features = self.build_lag_features(target, lags=target_lags, prefix='y_')
        feature_dfs.append(lag_features)

        # Médias móveis do alvo
        if self.include_ma:
            ma_features = self.build_ma_features(target, prefix='y_')
            feature_dfs.append(ma_features)

        # Exógenas e seus lags
        if exog is not None:
            if exog_lags is None:
                exog_lags = list(range(1, min(self.max_lag, 6) + 1))  # Menos lags para exógenas

            for col in exog.columns:
                exog_lag_features = self.build_lag_features(
                    exog[col], lags=exog_lags, prefix=f'{col}_'
                )
                feature_dfs.append(exog_lag_features)

                if self.include_ma:
                    exog_ma_features = self.build_ma_features(
                        exog[col], prefix=f'{col}_'
                    )
                    feature_dfs.append(exog_ma_features)

        # Tendência temporal
        if include_trend:
            trend = pd.DataFrame({
                'trend': np.arange(len(target))
            }, index=target.index)
            feature_dfs.append(trend)

        # Combina tudo
        X = pd.concat(feature_dfs, axis=1)

        return X


class RegularizedForecaster:
    """
    Modelo de previsão com regularização L1/L2.
    """

    def __init__(self, model_type: str = 'lasso', max_lag: int = 12,
                alpha: Optional[float] = None, cv_folds: int = 5):
        """
        Parâmetros:
        -----------
        model_type : 'ridge', 'lasso', ou 'elasticnet'
        max_lag : máximo de lags a considerar
        alpha : parâmetro de regularização (se None, usa CV)
        cv_folds : número de folds para cross-validation
        """
        self.model_type = model_type.lower()
        self.max_lag = max_lag
        self.alpha = alpha
        self.cv_folds = cv_folds

        self.feature_builder = TimeSeriesFeatureBuilder(max_lag=max_lag)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        self.model_ = None
        self.is_fitted = False
        self.selected_features_ = None

    def _get_model(self):
        """Retorna modelo sklearn apropriado."""
        if self.alpha is not None:
            # Alpha fixo
            if self.model_type == 'ridge':
                return Ridge(alpha=self.alpha)
            elif self.model_type == 'lasso':
                return Lasso(alpha=self.alpha, max_iter=10000)
            elif self.model_type == 'elasticnet':
                return ElasticNet(alpha=self.alpha, max_iter=10000)
            else:
                raise ValueError(f"model_type inválido: {self.model_type}")
        else:
            # Cross-validation para selecionar alpha
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)

            if self.model_type == 'ridge':
                return RidgeCV(alphas=np.logspace(-3, 3, 50), cv=tscv)
            elif self.model_type == 'lasso':
                return LassoCV(cv=tscv, max_iter=10000, n_jobs=-1)
            elif self.model_type == 'elasticnet':
                return ElasticNetCV(cv=tscv, max_iter=10000, n_jobs=-1)
            else:
                raise ValueError(f"model_type inválido: {self.model_type}")

    def fit(self, target: pd.Series, exog: Optional[pd.DataFrame] = None,
           target_lags: Optional[List[int]] = None,
           exog_lags: Optional[List[int]] = None,
           verbose: bool = False) -> 'RegularizedForecaster':
        """
        Ajusta modelo regularizado.

        Parâmetros:
        -----------
        target : série temporal alvo
        exog : variáveis exógenas
        target_lags : lags do alvo a incluir (se None, usa 1 até max_lag)
        exog_lags : lags das exógenas
        """
        self.target_name = target.name or 'target'
        self.data_index = target.index

        if verbose:
            print(f"Construindo features (max_lag={self.max_lag})...")

        # Constrói features
        X = self.feature_builder.build_features(
            target, exog=exog,
            target_lags=target_lags,
            exog_lags=exog_lags,
            include_trend=False
        )

        # Remove NaN (primeiras linhas devido aos lags)
        y = target.loc[X.index]
        valid_idx = X.notna().all(axis=1) & y.notna()
        X = X[valid_idx]
        y = y[valid_idx]

        if len(X) < 20:
            raise ValueError("Dados insuficientes após criar features e remover NaN")

        self.feature_names_ = X.columns.tolist()

        if verbose:
            print(f"Features criadas: {len(self.feature_names_)} (após {len(target) - len(X)} NaN removidos)")
            print(f"Amostras de treino: {len(X)}")

        # Padroniza
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

        # Treina modelo
        if verbose:
            print(f"\nTreinando modelo {self.model_type.upper()}...")

        self.model_ = self._get_model()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(X_scaled, y_scaled)

        self.is_fitted = True

        # Alpha selecionado (se CV)
        if hasattr(self.model_, 'alpha_'):
            self.alpha_ = self.model_.alpha_
            if verbose:
                print(f"  Alpha selecionado via CV: {self.alpha_:.6f}")
        else:
            self.alpha_ = self.alpha

        # Identifica features selecionadas (para Lasso)
        if self.model_type in ['lasso', 'elasticnet']:
            coefs = self.model_.coef_
            selected_mask = np.abs(coefs) > 1e-6
            self.selected_features_ = [
                feat for feat, sel in zip(self.feature_names_, selected_mask) if sel
            ]

            if verbose:
                print(f"  Features selecionadas: {len(self.selected_features_)} / {len(self.feature_names_)}")
                if len(self.selected_features_) <= 20:
                    print(f"  → {self.selected_features_}")
        else:
            self.selected_features_ = self.feature_names_

        # Score in-sample
        self.train_score_ = self.model_.score(X_scaled, y_scaled)

        if verbose:
            print(f"  R² in-sample: {self.train_score_:.4f}")

        return self

    def forecast(self, steps: int = 12, exog: Optional[pd.DataFrame] = None,
                return_std: bool = False) -> pd.DataFrame:
        """
        Previsão multi-horizonte (recursiva).

        Parâmetros:
        -----------
        steps : número de passos à frente
        exog : valores futuros das exógenas (deve ter 'steps' linhas)
        return_std : se True, estima desvio-padrão via bootstrap (simplificado)

        Retorna:
        --------
        DataFrame com coluna 'forecast' (e 'std' se return_std=True)
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi ajustado. Chame fit() primeiro.")

        # Preparação: últimas observações para construir lags
        # (Simplificação: assume que temos histórico suficiente)

        forecasts = []

        # Para previsão recursiva, precisamos manter histórico atualizado
        # Aqui faremos uma versão simplificada

        # Idealmente, deveria reconstruir features em cada passo
        # Por simplicidade, vamos retornar apenas previsão 1-passo

        warnings.warn("Previsão multi-horizonte recursiva ainda não totalmente implementada. "
                     "Use MultiHorizonRegularized para previsão direta.", UserWarning)

        # Placeholder
        result_df = pd.DataFrame({
            'forecast': np.full(steps, np.nan)
        })

        return result_df

    def get_feature_importance(self, top_k: Optional[int] = None) -> pd.DataFrame:
        """
        Retorna importância das features (coeficientes).

        Para Ridge/Lasso: abs(coeficiente) padronizado.
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi ajustado.")

        coefs = self.model_.coef_
        importance = np.abs(coefs)

        feature_importance = pd.DataFrame({
            'feature': self.feature_names_,
            'coefficient': coefs,
            'importance': importance
        }).sort_values('importance', ascending=False)

        if top_k is not None:
            feature_importance = feature_importance.head(top_k)

        return feature_importance

    def get_selected_features(self) -> List[str]:
        """Retorna features selecionadas (com coef != 0 para Lasso)."""
        if not self.is_fitted:
            raise ValueError("Modelo não foi ajustado.")

        return self.selected_features_


class MultiHorizonRegularized:
    """
    Treina um modelo separado para cada horizonte (Direct Multi-Step).
    """

    def __init__(self, model_type: str = 'lasso', max_lag: int = 12,
                max_horizon: int = 12, alpha: Optional[float] = None):
        self.model_type = model_type
        self.max_lag = max_lag
        self.max_horizon = max_horizon
        self.alpha = alpha

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
            model_h = RegularizedForecaster(
                model_type=self.model_type,
                max_lag=self.max_lag,
                alpha=self.alpha
            )

            # Alinha dados
            feature_builder = TimeSeriesFeatureBuilder(max_lag=self.max_lag)
            X = feature_builder.build_features(target, exog=exog)

            y_h = target_h.loc[X.index]
            valid_idx = X.notna().all(axis=1) & y_h.notna()
            X_valid = X[valid_idx]
            y_valid = y_h[valid_idx]

            # Fit direto em X e y (sem usar fit padrão que reconstrói features)
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            X_scaled = scaler_X.fit_transform(X_valid)
            y_scaled = scaler_y.fit_transform(y_valid.values.reshape(-1, 1)).ravel()

            if model_h.alpha is not None:
                if model_h.model_type == 'ridge':
                    model = Ridge(alpha=model_h.alpha)
                elif model_h.model_type == 'lasso':
                    model = Lasso(alpha=model_h.alpha, max_iter=10000)
                else:
                    model = ElasticNet(alpha=model_h.alpha, max_iter=10000)
            else:
                tscv = TimeSeriesSplit(n_splits=5)
                if model_h.model_type == 'ridge':
                    model = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=tscv)
                elif model_h.model_type == 'lasso':
                    model = LassoCV(cv=tscv, max_iter=10000, n_jobs=-1)
                else:
                    model = ElasticNetCV(cv=tscv, max_iter=10000, n_jobs=-1)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_scaled, y_scaled)

            # Guarda modelo e scalers
            self.models[h] = {
                'model': model,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'feature_names': X_valid.columns.tolist()
            }

            if verbose:
                alpha_used = model.alpha_ if hasattr(model, 'alpha_') else model_h.alpha
                print(f"  Alpha: {alpha_used:.6f}")
                print(f"  R² in-sample: {model.score(X_scaled, y_scaled):.4f}")

        if verbose:
            print(f"\n✓ {len(self.models)} modelos treinados (h=1 a {self.max_horizon})")

    def forecast(self, target: pd.Series, exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Gera previsões para todos os horizontes.

        Usa os valores mais recentes de target/exog para construir features.
        """
        forecasts = {}

        # Constrói features com dados mais recentes
        feature_builder = TimeSeriesFeatureBuilder(max_lag=self.max_lag)
        X_current = feature_builder.build_features(target, exog=exog)
        X_current = X_current.iloc[[-1]]  # Última linha (mais recente)

        for h, model_dict in self.models.items():
            model = model_dict['model']
            scaler_X = model_dict['scaler_X']
            scaler_y = model_dict['scaler_y']

            # Garante que features estão alinhadas
            X_h = X_current[model_dict['feature_names']]

            if X_h.isna().any().any():
                forecasts[f'h{h}'] = np.nan
                continue

            # Prediz
            X_scaled = scaler_X.transform(X_h)
            y_scaled = model.predict(X_scaled)
            y_pred = scaler_y.inverse_transform(y_scaled.reshape(-1, 1))[0, 0]

            forecasts[f'h{h}'] = y_pred

        return pd.DataFrame([forecasts])
