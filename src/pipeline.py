"""
Pipeline principal para previsão do mercado imobiliário de Vitória/ES.

Integra todos os componentes:
1. Pré-processamento (estacionaridade, Granger, seleção de variáveis)
2. Construção do IDCI-VIX (fator dinâmico)
3. Modelos de previsão (ARIMA, Markov, Ridge/Lasso, Trees, Quantílica)
4. Ensemble e combinação
5. Avaliação e validação
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings

from preprocessing.stationarity import StationarityTester
from preprocessing.granger import GrangerSelector
from factor_model.dynamic_factor import DynamicFactorModel
from forecasting.arima_models import ARIMAForecaster, MultiHorizonARIMA
from forecasting.markov_switching import MarkovSwitchingForecaster
from forecasting.regularized_models import RegularizedForecaster, MultiHorizonRegularized
from forecasting.tree_models import TreeForecaster, MultiHorizonTree, QuantileRandomForest
from forecasting.quantile_regression import QuantileRegressionForecaster, MultiHorizonQuantile
from evaluation.ensemble import ForecastEvaluator, EnsembleForecaster, ModelSelector


class VitoriaForecastPipeline:
    """
    Pipeline completo para previsão do mercado imobiliário de Vitória/ES.
    """

    def __init__(self,
                max_vars: int = 5,
                forecast_horizon: int = 12,
                ar_order: int = 2,
                verbose: bool = True):
        """
        Parâmetros:
        -----------
        max_vars : número máximo de variáveis a selecionar
        forecast_horizon : horizonte de previsão em meses
        ar_order : ordem do processo AR para o fator dinâmico
        verbose : exibir mensagens de progresso
        """
        self.max_vars = max_vars
        self.forecast_horizon = forecast_horizon
        self.ar_order = ar_order
        self.verbose = verbose

        # Componentes
        self.stationarity_tester = StationarityTester()
        self.granger_selector = GrangerSelector(max_vars=max_vars)
        self.dfm = DynamicFactorModel(ar_order=ar_order)

        # Modelos de previsão
        self.models = {}

        # Dados processados
        self.data_stationary = None
        self.selected_vars = None
        self.idci_vix = None

        self.is_fitted = False

    def preprocess(self, df: pd.DataFrame, try_seasonal: bool = True) -> pd.DataFrame:
        """
        Passo 1: Torna séries estacionárias.

        Parâmetros:
        -----------
        df : DataFrame com séries originais (já log-deflacionadas)

        Retorna:
        --------
        DataFrame com séries estacionárias
        """
        if self.verbose:
            print("\n" + "="*80)
            print("PASSO 1: PRÉ-PROCESSAMENTO - ESTACIONARIDADE")
            print("="*80)

        self.data_stationary = self.stationarity_tester.fit_transform(
            df,
            try_seasonal=try_seasonal,
            verbose=self.verbose
        )

        if self.verbose:
            print(f"\n✓ {len(self.data_stationary.columns)} séries tornadas estacionárias")
            print(f"  Observações: {len(self.data_stationary)}")

        return self.data_stationary

    def select_variables(self, df: Optional[pd.DataFrame] = None) -> List[str]:
        """
        Passo 2: Seleciona top-K variáveis via Granger.

        Retorna:
        --------
        Lista com nomes das variáveis selecionadas
        """
        if df is None:
            if self.data_stationary is None:
                raise ValueError("Deve chamar preprocess() primeiro")
            df = self.data_stationary

        if self.verbose:
            print("\n" + "="*80)
            print("PASSO 2: SELEÇÃO DE VARIÁVEIS VIA GRANGER")
            print("="*80)

        self.selected_vars, granger_results = self.granger_selector.select_top_k(
            df,
            k=self.max_vars,
            use_pca_factor=True,
            verbose=self.verbose
        )

        if self.verbose:
            print(f"\n✓ {len(self.selected_vars)} variáveis selecionadas:")
            for var in self.selected_vars:
                print(f"  - {var}")

        return self.selected_vars

    def build_index(self, df: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Passo 3: Constrói índice IDCI-VIX via modelo de fator dinâmico.

        Retorna:
        --------
        Series com IDCI-VIX (escala 0-10)
        """
        if df is None:
            if self.selected_vars is None:
                raise ValueError("Deve chamar select_variables() primeiro")
            df = self.data_stationary[self.selected_vars]

        if self.verbose:
            print("\n" + "="*80)
            print("PASSO 3: CONSTRUÇÃO DO IDCI-VIX (FATOR DINÂMICO)")
            print("="*80)

        # Padroniza
        df_scaled = (df - df.mean()) / df.std()

        # Estima modelo
        self.dfm.fit(df_scaled, verbose=self.verbose)

        # Extrai fator e escala
        factor = self.dfm.get_factor(smoothed=True)
        self.idci_vix = self.dfm.scale_to_index(
            factor,
            min_val=0.0,
            max_val=10.0,
            use_normal_cdf=True
        )

        if self.verbose:
            print(f"\n✓ IDCI-VIX construído:")
            print(f"  Média: {self.idci_vix.mean():.2f}")
            print(f"  Desvio: {self.idci_vix.std():.2f}")
            print(f"  Min: {self.idci_vix.min():.2f}, Max: {self.idci_vix.max():.2f}")

        return self.idci_vix

    def train_models(self, target: Optional[pd.Series] = None,
                    exog: Optional[pd.DataFrame] = None,
                    models_to_train: Optional[List[str]] = None) -> Dict:
        """
        Passo 4: Treina modelos de previsão.

        Parâmetros:
        -----------
        target : série alvo (se None, usa IDCI-VIX)
        exog : variáveis exógenas (se None, usa variáveis selecionadas)
        models_to_train : lista de modelos a treinar (se None, treina todos)

        Retorna:
        --------
        Dicionário com modelos treinados
        """
        if target is None:
            if self.idci_vix is None:
                raise ValueError("Deve chamar build_index() primeiro")
            target = self.idci_vix

        # Valida que target é Series (não string)
        if isinstance(target, str):
            raise ValueError(f"target deve ser pd.Series, não string '{target}'. "
                           f"Use: target = df['{target}'] ao invés de target = '{target}'")

        if not isinstance(target, pd.Series):
            raise ValueError(f"target deve ser pd.Series, recebeu {type(target)}")

        if exog is None and self.selected_vars is not None:
            exog = self.data_stationary[self.selected_vars]

        if models_to_train is None:
            models_to_train = [
                'arima', 'sarima', 'sarimax',
                'markov',
                'ridge', 'lasso',
                'random_forest',
                'quantile'
            ]

        if self.verbose:
            print("\n" + "="*80)
            print("PASSO 4: TREINAMENTO DE MODELOS")
            print("="*80)

        # ARIMA
        if 'arima' in models_to_train:
            if self.verbose:
                print("\n--- ARIMA ---")
            try:
                model_arima = ARIMAForecaster()
                model_arima.fit(target, auto=True, verbose=self.verbose)
                self.models['arima'] = model_arima
            except Exception as e:
                if self.verbose:
                    print(f"⚠ Erro ao treinar ARIMA: {e}")
                    print(f"   Tipo de target: {type(target)}")
                    if hasattr(target, 'name'):
                        print(f"   Nome do target: {target.name}")
                    import traceback
                    traceback.print_exc()

        # SARIMA
        if 'sarima' in models_to_train:
            if self.verbose:
                print("\n--- SARIMA ---")
            try:
                model_sarima = ARIMAForecaster()
                model_sarima.fit(target, auto=True, auto_seasonal=True, s=12, verbose=self.verbose)
                self.models['sarima'] = model_sarima
            except Exception as e:
                if self.verbose:
                    print(f"⚠ Erro ao treinar SARIMA: {e}")

        # SARIMAX
        if 'sarimax' in models_to_train and exog is not None:
            if self.verbose:
                print("\n--- SARIMAX ---")
            try:
                model_sarimax = ARIMAForecaster()
                model_sarimax.fit(target, exog=exog, auto=True, auto_seasonal=True, s=12, verbose=self.verbose)
                self.models['sarimax'] = model_sarimax
            except Exception as e:
                if self.verbose:
                    print(f"⚠ Erro ao treinar SARIMAX: {e}")

        # Markov-Switching
        if 'markov' in models_to_train:
            if self.verbose:
                print("\n--- MARKOV-SWITCHING ---")
            try:
                model_markov = MarkovSwitchingForecaster(k_regimes=2, order=2)
                model_markov.fit(target, verbose=self.verbose)
                self.models['markov'] = model_markov
            except Exception as e:
                if self.verbose:
                    print(f"⚠ Erro ao treinar Markov: {e}")

        # Ridge
        if 'ridge' in models_to_train:
            if self.verbose:
                print("\n--- RIDGE ---")
            try:
                model_ridge = MultiHorizonRegularized(
                    model_type='ridge',
                    max_lag=12,
                    max_horizon=self.forecast_horizon
                )
                model_ridge.fit(target, exog=exog, verbose=self.verbose)
                self.models['ridge'] = model_ridge
            except Exception as e:
                if self.verbose:
                    print(f"⚠ Erro ao treinar Ridge: {e}")

        # Lasso
        if 'lasso' in models_to_train:
            if self.verbose:
                print("\n--- LASSO ---")
            try:
                model_lasso = MultiHorizonRegularized(
                    model_type='lasso',
                    max_lag=12,
                    max_horizon=self.forecast_horizon
                )
                model_lasso.fit(target, exog=exog, verbose=self.verbose)
                self.models['lasso'] = model_lasso
            except Exception as e:
                if self.verbose:
                    print(f"⚠ Erro ao treinar Lasso: {e}")

        # Random Forest
        if 'random_forest' in models_to_train:
            if self.verbose:
                print("\n--- RANDOM FOREST ---")
            try:
                model_rf = MultiHorizonTree(
                    model_type='random_forest',
                    max_lag=12,
                    max_horizon=self.forecast_horizon,
                    n_estimators=100
                )
                model_rf.fit(target, exog=exog, verbose=self.verbose)
                self.models['random_forest'] = model_rf
            except Exception as e:
                if self.verbose:
                    print(f"⚠ Erro ao treinar Random Forest: {e}")

        # Quantile Regression
        if 'quantile' in models_to_train:
            if self.verbose:
                print("\n--- QUANTILE REGRESSION ---")
            try:
                model_quantile = MultiHorizonQuantile(
                    quantiles=[0.1, 0.5, 0.9],
                    max_lag=12,
                    max_horizon=self.forecast_horizon
                )
                model_quantile.fit(target, exog=exog, verbose=self.verbose)
                self.models['quantile'] = model_quantile
            except Exception as e:
                if self.verbose:
                    print(f"⚠ Erro ao treinar Quantile: {e}")

        if self.verbose:
            print(f"\n✓ {len(self.models)} modelos treinados com sucesso")

        self.is_fitted = True
        return self.models

    def forecast_all(self, target: Optional[pd.Series] = None,
                    exog: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        """
        Passo 5: Gera previsões de todos os modelos.

        Retorna:
        --------
        Dicionário {model_name: DataFrame de previsões}
        """
        if not self.is_fitted:
            raise ValueError("Deve chamar train_models() primeiro")

        if target is None:
            target = self.idci_vix

        # Valida que target é Series (não string)
        if isinstance(target, str):
            raise ValueError(f"target deve ser pd.Series, não string '{target}'. "
                           f"Use: target = df['{target}'] ao invés de target = '{target}'")

        if not isinstance(target, pd.Series):
            raise ValueError(f"target deve ser pd.Series, recebeu {type(target)}")

        if exog is None and self.selected_vars is not None:
            exog = self.data_stationary[self.selected_vars]

        if self.verbose:
            print("\n" + "="*80)
            print("PASSO 5: GERAÇÃO DE PREVISÕES")
            print("="*80)

        forecasts = {}

        for model_name, model in self.models.items():
            if self.verbose:
                print(f"\nPrevendo com {model_name.upper()}...")

            try:
                if model_name in ['arima', 'sarima']:
                    fc = model.forecast(steps=self.forecast_horizon, return_conf_int=False)
                    forecasts[model_name] = fc

                elif model_name == 'sarimax':
                    # Precisa de valores futuros de exog (aqui, usa últimos valores - simplificação)
                    if exog is not None:
                        exog_future = pd.DataFrame(
                            np.tile(exog.iloc[-1].values, (self.forecast_horizon, 1)),
                            columns=exog.columns
                        )
                        fc = model.forecast(steps=self.forecast_horizon, exog=exog_future, return_conf_int=False)
                        forecasts[model_name] = fc

                elif model_name == 'markov':
                    fc = model.forecast(steps=self.forecast_horizon)
                    forecasts[model_name] = fc

                elif model_name in ['ridge', 'lasso', 'random_forest']:
                    fc = model.forecast(target, exog=exog)
                    # Reformata para ter coluna 'forecast'
                    fc_reformed = pd.DataFrame({
                        'forecast': [fc[f'h{h}'].iloc[0] for h in range(1, self.forecast_horizon + 1)]
                    })
                    forecasts[model_name] = fc_reformed

                elif model_name == 'quantile':
                    fc = model.forecast(target, exog=exog)
                    # Usa mediana
                    fc_median = pd.DataFrame({
                        'forecast': fc['q0.5'].values
                    })
                    forecasts[model_name] = fc_median
                    # Guarda também os quantis
                    forecasts[f'{model_name}_quantiles'] = fc

            except Exception as e:
                if self.verbose:
                    print(f"  ⚠ Erro: {e}")

        if self.verbose:
            print(f"\n✓ Previsões geradas para {len(forecasts)} modelos")

        return forecasts

    def create_ensemble(self, forecasts: Dict[str, pd.DataFrame],
                       method: str = 'weighted_avg') -> pd.DataFrame:
        """
        Passo 6: Combina previsões via ensemble.

        Parâmetros:
        -----------
        method : 'simple_avg', 'weighted_avg', 'median'

        Retorna:
        --------
        DataFrame com previsão combinada
        """
        if self.verbose:
            print("\n" + "="*80)
            print(f"PASSO 6: ENSEMBLE ({method.upper()})")
            print("="*80)

        ensemble = EnsembleForecaster(combination_method=method)

        # Para weighted_avg, precisamos de pesos (aqui, usa pesos iguais como simplificação)
        if method == 'weighted_avg':
            # Simplificação: pesos iguais
            n = len(forecasts)
            ensemble.weights_ = {name: 1.0/n for name in forecasts.keys()}

        # Combina
        combined = ensemble.combine(forecasts)

        if self.verbose:
            print(f"\n✓ Ensemble criado com {len(forecasts)} modelos")

        return pd.DataFrame({'forecast': combined})

    def run_full_pipeline(self, df: pd.DataFrame,
                         models_to_train: Optional[List[str]] = None,
                         ensemble_method: str = 'weighted_avg') -> Dict:
        """
        Executa pipeline completo.

        Retorna:
        --------
        Dicionário com todos os resultados
        """
        if self.verbose:
            print("\n" + "#"*80)
            print("# PIPELINE COMPLETO - PREVISÃO MERCADO IMOBILIÁRIO VITÓRIA/ES")
            print("#"*80)

        # 1. Pré-processamento
        self.preprocess(df)

        # 2. Seleção de variáveis
        self.select_variables()

        # 3. Construção do índice
        self.build_index()

        # 4. Treinamento
        self.train_models(models_to_train=models_to_train)

        # 5. Previsões
        forecasts = self.forecast_all()

        # 6. Ensemble
        ensemble_forecast = self.create_ensemble(forecasts, method=ensemble_method)

        if self.verbose:
            print("\n" + "#"*80)
            print("# PIPELINE CONCLUÍDO")
            print("#"*80)
            print(f"\n📊 Resultados:")
            print(f"  - Variáveis selecionadas: {len(self.selected_vars)}")
            print(f"  - IDCI-VIX atual: {self.idci_vix.iloc[-1]:.2f}")
            print(f"  - Modelos treinados: {len(self.models)}")
            print(f"  - Previsão 12M (ensemble): {ensemble_forecast['forecast'].iloc[0]:.2f}")

        return {
            'data_stationary': self.data_stationary,
            'selected_vars': self.selected_vars,
            'idci_vix': self.idci_vix,
            'models': self.models,
            'forecasts': forecasts,
            'ensemble': ensemble_forecast
        }
