"""Módulo de distribuições analíticas e estimadores de máxima verossimilhança (MLE).

Implementa:
- GaussianMLE: Ajuste e inferência analítica Normal univariada (ddof=0, sem regularização arbitrária).
- BernoulliMLE: Ajuste e inferência analítica de Bernoulli com suporte a suavização de Laplace controlada.
- ClassPriors: Estimativa e inferência das probabilidades a priori das classes a partir do conjunto de treinamento.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class DistributionError(ValueError):
    """Exceção levantada para falhas na estimação ou inferência das distribuições."""
    pass


@dataclass(frozen=True)
class ClassPriors:
    """Probabilidades a priori das classes estimadas exclusivamente no treinamento."""

    total_count: int
    count_0: int
    count_1: int
    pi_0: float
    pi_1: float
    log_pi_0: float
    log_pi_1: float

    @classmethod
    def fit(cls, y_train: np.ndarray) -> ClassPriors:
        """Estima as probabilidades a priori a partir do vetor de classes de treino."""
        if len(y_train) == 0:
            raise DistributionError("Vetor de classes de treino está vazio.")

        unique_classes = set(np.unique(y_train))
        if not unique_classes.issubset({0, 1}):
            raise DistributionError(f"Classes inválidas encontradas em y_train: {unique_classes}")

        n_total = len(y_train)
        n_0 = int(np.sum(y_train == 0))
        n_1 = int(np.sum(y_train == 1))

        if n_0 == 0 or n_1 == 0:
            raise DistributionError(
                f"Classe ausente no conjunto de treino: N0={n_0}, N1={n_1}. "
                "Ambas as classes devem estar presentes."
            )

        pi_0 = n_0 / n_total
        pi_1 = n_1 / n_total

        return cls(
            total_count=n_total,
            count_0=n_0,
            count_1=n_1,
            pi_0=pi_0,
            pi_1=pi_1,
            log_pi_0=math.log(pi_0),
            log_pi_1=math.log(pi_1),
        )

    def prior(self, c: int) -> float:
        """Retorna P(Y = c)."""
        if c == 0:
            return self.pi_0
        if c == 1:
            return self.pi_1
        raise DistributionError(f"Classe inválida: {c}. Esperado 0 ou 1.")

    def log_prior(self, c: int) -> float:
        """Retorna log P(Y = c)."""
        if c == 0:
            return self.log_pi_0
        if c == 1:
            return self.log_pi_1
        raise DistributionError(f"Classe inválida: {c}. Esperado 0 ou 1.")


@dataclass(frozen=True)
class GaussianMLE:
    """Distribuição Gaussiana univariada ajustada por Máxima Verossimilhança (MLE, ddof=0)."""

    mu: float
    variance: float
    sigma: float
    n_samples: int

    @classmethod
    def fit(cls, x: np.ndarray) -> GaussianMLE:
        """Ajusta parâmetros Gaussianos via MLE (divisor N, ddof=0).

        Args:
            x: Array 1D com observações contínuas da classe.

        Returns:
            Instância de GaussianMLE com mu, variance e sigma estimados.

        Raises:
            DistributionError: Se x estiver vazio, contiver não-finitos ou variância não-positiva.
        """
        if len(x) == 0:
            raise DistributionError("Não é possível ajustar Gaussiana em array vazio.")

        x_arr = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x_arr)):
            raise DistributionError("Array contém valores infinitos ou NaN.")

        n_samples = len(x_arr)
        mu = float(np.mean(x_arr))
        # ddof=0 é obrigatório para o estimador MLE (divisor N_c)
        variance = float(np.var(x_arr, ddof=0))

        if variance <= 0.0 or not math.isfinite(variance):
            raise DistributionError(
                f"Variância não positiva ou não finita encontrada no ajuste Gaussiano: {variance}. "
                "Interrompendo ajuste conforme seção 4.1 da síntese operacional."
            )

        sigma = math.sqrt(variance)
        return cls(mu=mu, variance=variance, sigma=sigma, n_samples=n_samples)

    def log_pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calcula a log-densidade analítica: -0.5*log(2*pi*var) - (x-mu)^2 / (2*var)."""
        x_arr = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x_arr)):
            raise DistributionError("Entrada para log_pdf contém valores não-finitos.")

        const_term = -0.5 * math.log(2.0 * math.pi * self.variance)
        quad_term = -((x_arr - self.mu) ** 2) / (2.0 * self.variance)
        result = const_term + quad_term

        if np.isscalar(x) or (isinstance(x, (float, int))):
            return float(result.item() if hasattr(result, "item") else result)
        return result

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calcula a densidade analítica: exp(log_pdf(x))."""
        l_pdf = self.log_pdf(x)
        return np.exp(l_pdf) if isinstance(l_pdf, np.ndarray) else math.exp(l_pdf)

    def to_dict(self) -> Dict[str, Any]:
        """Serialização dos parâmetros estimados."""
        return {
            "mu": self.mu,
            "variance": self.variance,
            "sigma": self.sigma,
            "n_samples": self.n_samples,
        }


@dataclass(frozen=True)
class BernoulliMLE:
    """Distribuição de Bernoulli para variável categórica binária."""

    p: float  # P(X = 1 | Y = c), ex: P(Male | Y = c)
    n_samples: int
    n_ones: int
    n_zeros: int
    alpha: float  # 0.0 para MLE puro; 1.0 para Laplace

    @classmethod
    def fit(cls, z: np.ndarray, alpha: float = 0.0) -> BernoulliMLE:
        """Estima o parâmetro p da Bernoulli.

        p = (n_ones + alpha) / (n_samples + 2 * alpha)

        Args:
            z: Array binário contendo apenas valores 0 e 1.
            alpha: Fator de suavização (0.0 para MLE de frequências; 1.0 para Laplace).

        Returns:
            Instância de BernoulliMLE.

        Raises:
            DistributionError: Se houver valores fora de {0, 1} ou array vazio.
        """
        if not math.isfinite(alpha) or alpha < 0.0:
            raise DistributionError(f"alpha deve ser finito e não negativo: {alpha}")

        if len(z) == 0:
            raise DistributionError("Não é possível ajustar Bernoulli em array vazio.")

        z_arr = np.asarray(z)
        if z_arr.ndim != 1:
            raise DistributionError(f"Ajuste da Bernoulli exige array 1D. Obtido: {z_arr.ndim}D")
        
        if not np.all(np.isfinite(z_arr)):
            raise DistributionError("Valores não finitos encontrados na Bernoulli.")
        
        # Check if they are exactly 0 or 1 before coercing to int
        if not np.array_equal(z_arr, np.round(z_arr)) or not set(np.unique(z_arr)).issubset({0, 1}):
            raise DistributionError(f"Valores não-binários exatos encontrados na Bernoulli: {np.unique(z_arr)}")
        
        z_arr = z_arr.astype(int)


        n_samples = len(z_arr)
        n_ones = int(np.sum(z_arr == 1))
        n_zeros = n_samples - n_ones

        denom = n_samples + 2.0 * alpha
        if denom <= 0.0:
            raise DistributionError(f"Denominador inválido na Bernoulli: {denom}")

        p = (n_ones + alpha) / denom

        if p < 0.0 or p > 1.0:
            raise DistributionError(f"Probabilidade estimada fora do intervalo [0, 1]: {p}")

        return cls(p=p, n_samples=n_samples, n_ones=n_ones, n_zeros=n_zeros, alpha=alpha)

    def pmf(self, z: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calcula P(X = z | Y = c).

        Para z = 1: p
        Para z = 0: 1 - p
        """
        z_arr = np.asarray(z)
        if not np.all(np.isfinite(z_arr)):
            raise DistributionError("Valores não finitos encontrados.")
        if not np.array_equal(z_arr, np.round(z_arr)) or not set(np.unique(z_arr)).issubset({0, 1}):
            raise DistributionError(f"Valores fora do domínio binário {0, 1} na pmf: {np.unique(z_arr)}")
        
        z_arr = z_arr.astype(int)

        probs = np.where(z_arr == 1, self.p, 1.0 - self.p)
        if np.isscalar(z) or isinstance(z, (int, float, np.integer)):
            return float(probs.item() if hasattr(probs, "item") else probs)
        return probs

    def log_pmf(self, z: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calcula log P(X = z | Y = c). Trata explicitamente o caso de probabilidade zero."""
        z_arr = np.asarray(z)
        if not np.all(np.isfinite(z_arr)):
            raise DistributionError("Valores não finitos encontrados.")
        if not np.array_equal(z_arr, np.round(z_arr)) or not set(np.unique(z_arr)).issubset({0, 1}):
            raise DistributionError(f"Valores fora do domínio binário {0, 1} no log_pmf: {np.unique(z_arr)}")
        
        z_arr = z_arr.astype(int)

        # Verificação explícita de probabilidade zero
        if self.p == 0.0 and np.any(z_arr == 1):
            raise DistributionError("log_pmf indefinido: P(X = 1) = 0 observado sem suavização.")
        if self.p == 1.0 and np.any(z_arr == 0):
            raise DistributionError("log_pmf indefinido: P(X = 0) = 0 observado sem suavização.")

        log_p1 = math.log(self.p) if self.p > 0.0 else -float("inf")
        log_p0 = math.log(1.0 - self.p) if self.p < 1.0 else -float("inf")

        log_probs = np.where(z_arr == 1, log_p1, log_p0)
        if np.isscalar(z) or isinstance(z, (int, np.integer)):
            return float(log_probs.item() if hasattr(log_probs, "item") else log_probs)
        return log_probs

    def to_dict(self) -> Dict[str, Any]:
        """Serialização dos parâmetros estimados."""
        return {
            "p_1": self.p,
            "p_0": 1.0 - self.p,
            "n_ones": self.n_ones,
            "n_zeros": self.n_zeros,
            "n_samples": self.n_samples,
            "alpha": self.alpha,
        }
