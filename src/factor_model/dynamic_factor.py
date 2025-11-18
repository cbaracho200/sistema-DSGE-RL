"""
Modelo de Fator Dinâmico com Filtro de Kalman.

Implementa:
- Modelo de fator dinâmico (DFM)
- Estimação via Filtro de Kalman e Máxima Verossimilhança
- Construção do índice IDCI-VIX (0-10)
- Suavização via Kalman Smoother
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DFMResults:
    """Resultados do modelo de fator dinâmico."""
    factor: pd.Series
    factor_filtered: pd.Series
    factor_smoothed: pd.Series
    loadings: np.ndarray
    phi: np.ndarray
    sigma_eta: float
    R: np.ndarray
    loglikelihood: float
    aic: float
    bic: float


class DynamicFactorModel:
    """
    Modelo de Fator Dinâmico com AR(p) para o fator latente.

    Equação de medição:
        Z_t = λ * f_t + ε_t,  ε_t ~ N(0, R)

    Equação de estado (AR(p)):
        f_t = φ_1*f_{t-1} + φ_2*f_{t-2} + ... + φ_p*f_{t-p} + η_t,  η_t ~ N(0, σ²_η)

    Forma em espaço de estados:
        x_t = F * x_{t-1} + v_t    (estado)
        y_t = H * x_t + w_t         (observação)

    onde x_t = [f_t, f_{t-1}, ..., f_{t-p+1}]'
    """

    def __init__(self, ar_order: int = 2):
        """
        Parâmetros:
        -----------
        ar_order : int
            Ordem do processo AR para o fator (default: 2 para ciclos)
        """
        self.ar_order = ar_order
        self.is_fitted = False

    def _kalman_filter(self, y: np.ndarray, H: np.ndarray, F: np.ndarray,
                       Q: np.ndarray, R: np.ndarray,
                       x0: np.ndarray, P0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Filtro de Kalman.

        Parâmetros:
        -----------
        y : (T, n) observações
        H : (n, m) matriz de medição
        F : (m, m) matriz de transição de estado
        Q : (m, m) covariância do ruído de estado
        R : (n, n) covariância do ruído de medição
        x0 : (m,) estado inicial
        P0 : (m, m) covariância inicial

        Retorna:
        --------
        x_filtered : (T, m) estados filtrados
        P_filtered : (T, m, m) covariâncias filtradas
        loglik : log-verossimilhança
        """
        T, n = y.shape
        m = F.shape[0]

        # Arrays para armazenar resultados
        x_pred = np.zeros((T, m))
        P_pred = np.zeros((T, m, m))
        x_filt = np.zeros((T, m))
        P_filt = np.zeros((T, m, m))

        loglik = 0.0

        # Inicialização
        x_t = x0
        P_t = P0

        for t in range(T):
            # Predição
            x_pred[t] = F @ x_t
            P_pred[t] = F @ P_t @ F.T + Q

            # Inovação
            y_pred = H @ x_pred[t]
            innov = y[t] - y_pred
            S = H @ P_pred[t] @ H.T + R

            # Ganho de Kalman
            K = P_pred[t] @ H.T @ np.linalg.inv(S)

            # Atualização
            x_filt[t] = x_pred[t] + K @ innov
            P_filt[t] = (np.eye(m) - K @ H) @ P_pred[t]

            # Log-verossimilhança
            sign, logdet = np.linalg.slogdet(S)
            if sign > 0:
                loglik += -0.5 * (n * np.log(2 * np.pi) + logdet + innov.T @ np.linalg.inv(S) @ innov)

            # Próximo passo
            x_t = x_filt[t]
            P_t = P_filt[t]

        return x_filt, P_filt, loglik

    def _kalman_smoother(self, x_filt: np.ndarray, P_filt: np.ndarray,
                        F: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """
        Suavizador de Kalman (Rauch-Tung-Striebel).

        Retorna:
        --------
        x_smooth : (T, m) estados suavizados
        """
        T, m = x_filt.shape
        x_smooth = np.zeros((T, m))
        x_smooth[-1] = x_filt[-1]

        for t in range(T - 2, -1, -1):
            # Predição um passo à frente
            x_pred = F @ x_filt[t]
            P_pred = F @ P_filt[t] @ F.T + Q

            # Ganho do smoother
            J = P_filt[t] @ F.T @ np.linalg.inv(P_pred)

            # Suavização
            x_smooth[t] = x_filt[t] + J @ (x_smooth[t + 1] - x_pred)

        return x_smooth

    def _build_state_space(self, lambdas: np.ndarray, phi: np.ndarray,
                          sigma_eta: float, R: np.ndarray) -> Tuple:
        """
        Constrói matrizes do espaço de estados.

        Retorna:
        --------
        H, F, Q
        """
        n = len(lambdas)  # número de variáveis observadas
        p = len(phi)      # ordem AR

        # Matriz de observação: [λ, 0, 0, ..., 0]
        H = np.zeros((n, p))
        H[:, 0] = lambdas

        # Matriz de transição: forma companheira
        F = np.zeros((p, p))
        F[0, :] = phi
        if p > 1:
            F[1:, :-1] = np.eye(p - 1)

        # Covariância do ruído de estado
        Q = np.zeros((p, p))
        Q[0, 0] = sigma_eta ** 2

        return H, F, Q

    def _params_to_arrays(self, params: np.ndarray, n: int, p: int) -> Tuple:
        """
        Converte vetor de parâmetros para arrays individuais.

        params = [λ_1, ..., λ_n, φ_1, ..., φ_p, log(σ_η), log(diag(R))]
        """
        idx = 0

        # Loadings
        lambdas = params[idx:idx + n]
        idx += n

        # Coeficientes AR
        phi = params[idx:idx + p]
        idx += p

        # Desvio-padrão do fator
        sigma_eta = np.exp(params[idx])
        idx += 1

        # Covariância das observações (diagonal)
        log_diag_R = params[idx:idx + n]
        R = np.diag(np.exp(log_diag_R))

        return lambdas, phi, sigma_eta, R

    def _arrays_to_params(self, lambdas: np.ndarray, phi: np.ndarray,
                         sigma_eta: float, R: np.ndarray) -> np.ndarray:
        """Converte arrays para vetor de parâmetros."""
        params = np.concatenate([
            lambdas,
            phi,
            [np.log(sigma_eta)],
            np.log(np.diag(R))
        ])
        return params

    def _negative_loglik(self, params: np.ndarray, y: np.ndarray) -> float:
        """
        Função objetivo: negativo da log-verossimilhança.
        """
        T, n = y.shape
        p = self.ar_order

        try:
            # Extrai parâmetros
            lambdas, phi, sigma_eta, R = self._params_to_arrays(params, n, p)

            # Constrói espaço de estados
            H, F, Q = self._build_state_space(lambdas, phi, sigma_eta, R)

            # Estado inicial
            x0 = np.zeros(p)
            P0 = np.eye(p) * 10.0  # Incerteza inicial grande

            # Filtro de Kalman
            _, _, loglik = self._kalman_filter(y, H, F, Q, R, x0, P0)

            return -loglik

        except (np.linalg.LinAlgError, ValueError):
            return 1e10  # Penalidade para parâmetros inválidos

    def fit(self, df: pd.DataFrame, verbose: bool = False,
           maxiter: int = 1000) -> 'DynamicFactorModel':
        """
        Estima o modelo via Máxima Verossimilhança.

        Parâmetros:
        -----------
        df : DataFrame com variáveis observadas (já estacionárias e padronizadas)
        """
        # Converte para array
        self.data_index = df.index
        self.var_names = df.columns.tolist()
        y = df.values
        T, n = y.shape
        p = self.ar_order

        if verbose:
            print(f"Ajustando DFM: {n} variáveis, {T} observações, AR({p})")

        # Inicialização via PCA + AR
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        f_init = pca.fit_transform(y)[:, 0]

        # Loadings iniciais
        lambdas_init = pca.components_[0]

        # Coeficientes AR via OLS
        from statsmodels.tsa.ar_model import AutoReg
        ar_model = AutoReg(f_init, lags=p, old_names=False)
        ar_fit = ar_model.fit()
        phi_init = ar_fit.params[1:]  # Remove constante

        # Variâncias iniciais
        sigma_eta_init = np.std(ar_fit.resid)
        R_init = np.diag(np.var(y - f_init[:, None] * lambdas_init, axis=0))

        # Vetor de parâmetros inicial
        params_init = self._arrays_to_params(lambdas_init, phi_init, sigma_eta_init, R_init)

        # Otimização
        if verbose:
            print("Otimizando via Máxima Verossimilhança...")

        result = minimize(
            self._negative_loglik,
            params_init,
            args=(y,),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'disp': verbose}
        )

        if not result.success and verbose:
            print(f"⚠ Otimização não convergiu: {result.message}")

        # Extrai parâmetros ótimos
        self.params_opt = result.x
        self.lambdas_, self.phi_, self.sigma_eta_, self.R_ = self._params_to_arrays(
            result.x, n, p
        )

        # Reconstrói espaço de estados
        self.H_, self.F_, self.Q_ = self._build_state_space(
            self.lambdas_, self.phi_, self.sigma_eta_, self.R_
        )

        # Filtro final
        x0 = np.zeros(p)
        P0 = np.eye(p) * 10.0
        x_filt, P_filt, loglik = self._kalman_filter(
            y, self.H_, self.F_, self.Q_, self.R_, x0, P0
        )

        # Suavizador
        x_smooth = self._kalman_smoother(x_filt, P_filt, self.F_, self.Q_)

        # Extrai fator (primeira componente do estado)
        self.factor_filtered_ = pd.Series(x_filt[:, 0], index=df.index, name='factor_filtered')
        self.factor_smoothed_ = pd.Series(x_smooth[:, 0], index=df.index, name='factor_smoothed')

        # Informação
        n_params = len(result.x)
        self.loglikelihood_ = loglik
        self.aic_ = -2 * loglik + 2 * n_params
        self.bic_ = -2 * loglik + n_params * np.log(T)

        self.is_fitted = True

        if verbose:
            print(f"\n{'='*60}")
            print("Resultados:")
            print(f"Log-Likelihood: {loglik:.2f}")
            print(f"AIC: {self.aic_:.2f}")
            print(f"BIC: {self.bic_:.2f}")
            print(f"Loadings: {self.lambdas_}")
            print(f"AR coefficients (φ): {self.phi_}")
            print(f"σ_η: {self.sigma_eta_:.4f}")
            print(f"{'='*60}")

        return self

    def get_factor(self, smoothed: bool = True) -> pd.Series:
        """Retorna série do fator estimado."""
        if not self.is_fitted:
            raise ValueError("Modelo não foi ajustado. Chame fit() primeiro.")

        return self.factor_smoothed_ if smoothed else self.factor_filtered_

    def scale_to_index(self, factor: Optional[pd.Series] = None,
                      min_val: float = 0.0, max_val: float = 10.0,
                      use_normal_cdf: bool = True) -> pd.Series:
        """
        Escala fator para índice [min_val, max_val].

        Se use_normal_cdf=True, usa CDF normal:
            u_t = (f_t - μ) / σ
            q_t = Φ(u_t)
            índice_t = min_val + (max_val - min_val) * q_t

        Caso contrário, usa min-max scaling.
        """
        if factor is None:
            factor = self.get_factor(smoothed=True)

        if use_normal_cdf:
            # Padroniza
            mu = factor.mean()
            sigma = factor.std()
            u = (factor - mu) / sigma

            # CDF normal
            q = norm.cdf(u)

            # Escala
            index_values = min_val + (max_val - min_val) * q

            # Garante que é Series
            index = pd.Series(index_values, index=factor.index, name=f'IDCI_VIX_{min_val}-{max_val}')

        else:
            # Min-max
            f_min = factor.min()
            f_max = factor.max()
            index_values = min_val + (max_val - min_val) * (factor - f_min) / (f_max - f_min)

            # Garante que é Series
            index = pd.Series(index_values, index=factor.index, name=f'IDCI_VIX_{min_val}-{max_val}')

        return index

    def get_results(self) -> DFMResults:
        """Retorna objeto com todos os resultados."""
        if not self.is_fitted:
            raise ValueError("Modelo não foi ajustado.")

        return DFMResults(
            factor=self.get_factor(smoothed=False),
            factor_filtered=self.factor_filtered_,
            factor_smoothed=self.factor_smoothed_,
            loadings=self.lambdas_,
            phi=self.phi_,
            sigma_eta=self.sigma_eta_,
            R=self.R_,
            loglikelihood=self.loglikelihood_,
            aic=self.aic_,
            bic=self.bic_
        )
