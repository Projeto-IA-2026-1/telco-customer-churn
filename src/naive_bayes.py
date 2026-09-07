"""Módulo do classificador Naive Bayes Híbrido (Gaussiano-Bernoulli).

Implementa:
- HybridNaiveBayesClassifier: Classificador probabilístico ajustado de raiz (from scratch),
  combinando as distribuições analíticas univariadas de máxima verossimilhança (ddof=0 para
  Gaussianas e alpha=0.0 para Bernoulli com suporte a Laplace).
- Inferência numericamente estável em espaço logarítmico:
  L_c(x) = log(pi_c) + log p(tenure|c) + log p(MonthlyCharges|c) + log P(gender|c)
- Normalização rigorosa via logsumexp e desempate canônico determinístico (empate exato -> 0).
- Inspeção de componentes condicionais e log-verossimilhanças individuais.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.distributions import BernoulliMLE, ClassPriors, DistributionError, GaussianMLE


class ClassifierNotFittedError(RuntimeError):
    """Exceção lançada quando inferência é solicitada antes do ajuste do modelo."""
    pass


class InvalidInputError(ValueError):
    """Exceção lançada para validações de dados de entrada na inferência ou treino."""
    pass


def logsumexp_pair(a: float, b: float) -> float:
    """Calcula log(exp(a) + exp(b)) de forma numericamente estável para um par escalar."""
    m = max(a, b)
    if not math.isfinite(m):
        return m
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def logsumexp_matrix(log_matrix: np.ndarray) -> np.ndarray:
    """Calcula logsumexp ao longo das colunas (axis=1) para matriz 2D (N, C)."""
    max_vals = np.max(log_matrix, axis=1, keepdims=True)
    sum_exp = np.sum(np.exp(log_matrix - max_vals), axis=1, keepdims=True)
    return max_vals + np.log(sum_exp)


class HybridNaiveBayesClassifier:
    """Classificador Naive Bayes Híbrido manual para duas classes {0, 1}.

    Características:
    - tenure (X1): Gaussiana MLE (ddof=0)
    - MonthlyCharges (X2): Gaussiana MLE (ddof=0)
    - gender (X3): Bernoulli MLE (alpha=0.0 ou 1.0)
    """

    FEATURE_NAMES: List[str] = ["tenure", "MonthlyCharges", "gender"]
    CLASSES: List[int] = [0, 1]

    def __init__(self) -> None:
        self.is_fitted: bool = False
        self.priors: Optional[ClassPriors] = None
        self.model_tenure_0: Optional[GaussianMLE] = None
        self.model_tenure_1: Optional[GaussianMLE] = None
        self.model_monthly_0: Optional[GaussianMLE] = None
        self.model_monthly_1: Optional[GaussianMLE] = None
        self.model_gender_0: Optional[BernoulliMLE] = None
        self.model_gender_1: Optional[BernoulliMLE] = None
        self.smoothing_alpha: float = 0.0

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> HybridNaiveBayesClassifier:
        """Ajusta os estimadores de verossimilhança e os priors exclusivamente no treino.

        Args:
            X: Matriz de features contendo ['tenure', 'MonthlyCharges', 'gender'].
            y: Vetor de rótulos binários de classe {0, 1}.

        Returns:
            Instância ajustada do classificador (self).
        """
        # Conversão e validação das features
        X_df = self._validate_and_convert_X(X)
        y_raw = np.asarray(y)
        if y_raw.ndim != 1:
            if y_raw.ndim == 2 and y_raw.shape[1] == 1:
                y_raw = y_raw.ravel()
            else:
                raise InvalidInputError("Vetor alvo (y) deve ser unidimensional.")
        
        if not np.issubdtype(y_raw.dtype, np.number):
            raise InvalidInputError("Vetor alvo (y) deve conter apenas números.")
        if not np.all(np.isfinite(y_raw)):
            raise InvalidInputError("Vetor alvo (y) contém valores não finitos.")
            
        y_arr_int = y_raw.astype(int)
        if not np.array_equal(y_raw, y_arr_int):
            raise InvalidInputError("Vetor alvo (y) deve conter apenas binários exatos (0 ou 1).")
        y_arr = y_arr_int

        if len(X_df) != len(y_arr):
            raise InvalidInputError(
                f"Tamanho incompatível entre X ({len(X_df)}) e y ({len(y_arr)})."
            )

        if len(y_arr) == 0:
            raise InvalidInputError("Conjunto de treinamento vazio.")

        unique_classes = set(np.unique(y_arr))
        if not unique_classes.issubset({0, 1}):
            raise InvalidInputError(f"Classes em y devem ser {{0, 1}}, recebido: {unique_classes}")

        # 1. Estimação das probabilidades a priori
        self.priors = ClassPriors.fit(y_arr)

        # 2. Separação das partições por classe
        mask_0 = y_arr == 0
        mask_1 = y_arr == 1

        df_0 = X_df[mask_0]
        df_1 = X_df[mask_1]

        # 3. Ajuste Gaussiano MLE para tenure (X1)
        self.model_tenure_0 = GaussianMLE.fit(df_0["tenure"].values)
        self.model_tenure_1 = GaussianMLE.fit(df_1["tenure"].values)

        # 4. Ajuste Gaussiano MLE para MonthlyCharges (X2)
        self.model_monthly_0 = GaussianMLE.fit(df_0["MonthlyCharges"].values)
        self.model_monthly_1 = GaussianMLE.fit(df_1["MonthlyCharges"].values)

        # 5. Ajuste Bernoulli para gender (X3) com política canônica de suavização
        z0 = df_0["gender"].values
        z1 = df_1["gender"].values

        n0_ones, n0_zeros = int(np.sum(z0 == 1)), int(np.sum(z0 == 0))
        n1_ones, n1_zeros = int(np.sum(z1 == 1)), int(np.sum(z1 == 0))

        # Se todas as 4 contagens são > 0, alpha=0.0; caso contrário, alpha=1.0 para ambas as classes
        if n0_ones > 0 and n0_zeros > 0 and n1_ones > 0 and n1_zeros > 0:
            self.smoothing_alpha = 0.0
        else:
            self.smoothing_alpha = 1.0

        self.model_gender_0 = BernoulliMLE.fit(z0, alpha=self.smoothing_alpha)
        self.model_gender_1 = BernoulliMLE.fit(z1, alpha=self.smoothing_alpha)

        self.is_fitted = True
        return self

    def _validate_and_convert_X(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Valida a conformidade dimensional e tipológica da matriz de features."""
        if isinstance(X, pd.DataFrame):
            missing_cols = [c for c in self.FEATURE_NAMES if c not in X.columns]
            if missing_cols:
                raise InvalidInputError(f"Colunas obrigatórias ausentes em X: {missing_cols}")
            X_df = X[self.FEATURE_NAMES].copy()
        elif isinstance(X, np.ndarray):
            if X.ndim != 2 or X.shape[1] != 3:
                raise InvalidInputError(
                    f"Array NumPy deve ter dimensão (N, 3), recebido: {X.shape}"
                )
            X_df = pd.DataFrame(X, columns=self.FEATURE_NAMES)
        else:
            raise InvalidInputError(f"Tipo não suportado para X: {type(X)}")

        # Validação de não-finitos
        if not np.all(np.isfinite(X_df.values)):
            raise InvalidInputError("Matriz X contém valores infinitos ou NaN.")

        # Validação do domínio de gender: deve ser binário 0 ou 1
        gender_vals = set(np.unique(X_df["gender"].values))
        if not gender_vals.issubset({0, 1, 0.0, 1.0}):
            raise InvalidInputError(f"Valores fora do domínio {{0, 1}} em gender: {gender_vals}")

        return X_df

    def predict_joint_log_likelihood(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """Calcula os escores logarítmicos não normalizados L_c(x) para cada classe.

        L_c(x) = log(pi_c) + log p(tenure|c) + log p(MonthlyCharges|c) + log P(gender|c)
        O prior é somado uma única vez por classe.

        Returns:
            Array 2D de forma (N, 2) contendo [L_0(x), L_1(x)].
        """
        if not self.is_fitted:
            raise ClassifierNotFittedError("Classificador precisa ser ajustado com fit() antes de inferir.")

        assert self.priors is not None
        assert self.model_tenure_0 is not None and self.model_tenure_1 is not None
        assert self.model_monthly_0 is not None and self.model_monthly_1 is not None
        assert self.model_gender_0 is not None and self.model_gender_1 is not None

        X_df = self._validate_and_convert_X(X)
        N = len(X_df)

        tenure = X_df["tenure"].values.astype(float)
        monthly = X_df["MonthlyCharges"].values.astype(float)
        gender = X_df["gender"].values.astype(int)

        # Log-verossimilhanças Classe 0
        ll_tenure_0 = self.model_tenure_0.log_pdf(tenure)
        ll_monthly_0 = self.model_monthly_0.log_pdf(monthly)
        ll_gender_0 = self.model_gender_0.log_pmf(gender)
        L_0 = self.priors.log_pi_0 + ll_tenure_0 + ll_monthly_0 + ll_gender_0

        # Log-verossimilhanças Classe 1
        ll_tenure_1 = self.model_tenure_1.log_pdf(tenure)
        ll_monthly_1 = self.model_monthly_1.log_pdf(monthly)
        ll_gender_1 = self.model_gender_1.log_pmf(gender)
        L_1 = self.priors.log_pi_1 + ll_tenure_1 + ll_monthly_1 + ll_gender_1

        return np.column_stack([L_0, L_1])

    def predict_log_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """Calcula as probabilidades a posteriori normalizadas em escala logarítmica.

        log P(Y = c | x) = L_c(x) - logsumexp(L_0(x), L_1(x))

        Returns:
            Array 2D de forma (N, 2) contendo [log P(Y=0|x), log P(Y=1|x)].
        """
        log_joint = self.predict_joint_log_likelihood(X)
        self._validate_log_joint(log_joint)
        log_marginal = logsumexp_matrix(log_joint)
        return log_joint - log_marginal

    def _validate_log_joint(self, log_joint: np.ndarray) -> None:
        """Valida que os escores não sejam NaN, +inf, e nem ambos -inf (cenário impossível)."""
        if np.any(np.isnan(log_joint)) or np.any(log_joint == np.inf):
            raise InvalidInputError("Escores log-conjuntos contêm NaN ou +inf.")
        if np.any(np.all(log_joint == -np.inf, axis=1)):
            raise InvalidInputError("Instância impossível detectada: log-verossimilhança é -inf para ambas as classes simultaneamente.")

    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """Calcula as probabilidades a posteriori normalizadas P(Y = c | x).

        Garante soma estritamente igual a 1.0 ao longo das classes.

        Returns:
            Array 2D de forma (N, 2) contendo [P(Y=0|x), P(Y=1|x)].
        """
        log_proba = self.predict_log_proba(X)
        proba = np.exp(log_proba)
        # Normalização de segurança contra resíduos infinitesimais de arredondamento
        return proba / np.sum(proba, axis=1, keepdims=True)

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """Classifica instâncias segundo a regra de decisão de Bayes (MAP).

        Regra de desempate:
        Se L_1(x) > L_0(x) -> 1
        Se L_1(x) <= L_0(x) -> 0 (empate exato favorece a classe 0).

        Returns:
            Array 1D de inteiros {0, 1}.
        """
        log_joint = self.predict_joint_log_likelihood(X)
        self._validate_log_joint(log_joint)
        # L_1 > L_0 prediz 1, caso contrário prediz 0 (inclui empate L_1 == L_0 -> 0)
        return np.where(log_joint[:, 1] > log_joint[:, 0], 1, 0)

    def inspect_sample(
        self,
        sample: Union[pd.Series, Dict[str, Any], np.ndarray],
    ) -> Dict[str, Any]:
        """Inspeciona a decomposição matemática completa para uma única instância."""
        if not self.is_fitted:
            raise ClassifierNotFittedError("Classificador precisa ser ajustado com fit() antes de inspecionar.")

        assert self.priors is not None
        assert self.model_tenure_0 is not None and self.model_tenure_1 is not None
        assert self.model_monthly_0 is not None and self.model_monthly_1 is not None
        assert self.model_gender_0 is not None and self.model_gender_1 is not None

        if isinstance(sample, dict):
            sample_df = pd.DataFrame([sample])
        elif isinstance(sample, pd.Series):
            sample_df = pd.DataFrame([sample])
        elif isinstance(sample, np.ndarray):
            sample_df = pd.DataFrame(sample.reshape(1, -1), columns=self.FEATURE_NAMES)
        else:
            raise InvalidInputError(f"Tipo não suportado para amostra: {type(sample)}")

        sample_df = self._validate_and_convert_X(sample_df)

        t_val = float(sample_df["tenure"].iloc[0])
        m_val = float(sample_df["MonthlyCharges"].iloc[0])
        g_val = int(sample_df["gender"].iloc[0])

        # Decomposição por classe
        p_c0 = self.priors.pi_0
        p_c1 = self.priors.pi_1
        log_p_c0 = self.priors.log_pi_0
        log_p_c1 = self.priors.log_pi_1

        ll_t_0 = float(self.model_tenure_0.log_pdf(t_val))
        ll_t_1 = float(self.model_tenure_1.log_pdf(t_val))

        ll_m_0 = float(self.model_monthly_0.log_pdf(m_val))
        ll_m_1 = float(self.model_monthly_1.log_pdf(m_val))

        ll_g_0 = float(self.model_gender_0.log_pmf(g_val))
        ll_g_1 = float(self.model_gender_1.log_pmf(g_val))

        L_0 = log_p_c0 + ll_t_0 + ll_m_0 + ll_g_0
        L_1 = log_p_c1 + ll_t_1 + ll_m_1 + ll_g_1

        if math.isnan(L_0) or math.isnan(L_1) or L_0 == float('inf') or L_1 == float('inf'):
            raise InvalidInputError("Escores log-conjuntos contêm NaN ou +inf.")
            
        if L_0 == -float('inf') and L_1 == -float('inf'):
            raise InvalidInputError("Instância impossível detectada: log-verossimilhança é -inf para ambas as classes simultaneamente.")

        log_marginal = logsumexp_pair(L_0, L_1)
        post_0 = math.exp(L_0 - log_marginal) if log_marginal != -float('inf') else 0.0
        post_1 = math.exp(L_1 - log_marginal) if log_marginal != -float('inf') else 0.0
        decision = 1 if L_1 > L_0 else 0

        return {
            "inputs": {"tenure": t_val, "MonthlyCharges": m_val, "gender": g_val},
            "priors": {
                "pi_0": p_c0, "pi_1": p_c1,
                "log_pi_0": log_p_c0, "log_pi_1": log_p_c1,
            },
            "log_likelihoods": {
                "tenure": {"class_0": ll_t_0, "class_1": ll_t_1},
                "MonthlyCharges": {"class_0": ll_m_0, "class_1": ll_m_1},
                "gender": {"class_0": ll_g_0, "class_1": ll_g_1},
            },
            "log_joint_scores": {"L_0": L_0, "L_1": L_1},
            "log_marginal_px": log_marginal,
            "posteriors": {"P(Churn=0|x)": post_0, "P(Churn=1|x)": post_1},
            "predicted_class": decision,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialização completa dos parâmetros e metadados do classificador."""
        if not self.is_fitted:
            raise ClassifierNotFittedError("Classificador não foi ajustado.")

        assert self.priors is not None
        assert self.model_tenure_0 is not None and self.model_tenure_1 is not None
        assert self.model_monthly_0 is not None and self.model_monthly_1 is not None
        assert self.model_gender_0 is not None and self.model_gender_1 is not None

        return {
            "model_type": "HybridNaiveBayesClassifier",
            "features": self.FEATURE_NAMES,
            "classes": self.CLASSES,
            "smoothing_alpha": self.smoothing_alpha,
            "priors": {
                "total_count": self.priors.total_count,
                "count_0": self.priors.count_0,
                "count_1": self.priors.count_1,
                "pi_0": self.priors.pi_0,
                "pi_1": self.priors.pi_1,
                "log_pi_0": self.priors.log_pi_0,
                "log_pi_1": self.priors.log_pi_1,
            },
            "parameters": {
                "tenure": {
                    "class_0": self.model_tenure_0.to_dict(),
                    "class_1": self.model_tenure_1.to_dict(),
                },
                "MonthlyCharges": {
                    "class_0": self.model_monthly_0.to_dict(),
                    "class_1": self.model_monthly_1.to_dict(),
                },
                "gender": {
                    "class_0": self.model_gender_0.to_dict(),
                    "class_1": self.model_gender_1.to_dict(),
                },
            },
        }
