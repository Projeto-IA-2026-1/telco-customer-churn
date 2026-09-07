"""Testes unitários rigorosos para o módulo src/distributions.py.

Verifica:
- Estimativa analítica de priors (ClassPriors) e validações.
- Estimativa Gaussiana MLE com ddof=0 (divisor N) e cálculo de log_pdf/pdf.
- Estimativa Bernoulli com e sem suavização de Laplace (alpha=0 vs alpha=1).
- Casos degenerados, não-finitos, variância nula e probabilidades extremas.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
from scipy import stats

from src.distributions import BernoulliMLE, ClassPriors, DistributionError, GaussianMLE


class TestClassPriors:
    """Testes para a classe ClassPriors."""

    def test_priors_synthetic_known_proportions(self) -> None:
        """Verifica cálculo analítico com contagens sintéticas conhecidas."""
        # 3 instâncias de classe 0 e 1 de classe 1 -> pi0 = 0.75, pi1 = 0.25
        y = np.array([0, 0, 1, 0])
        priors = ClassPriors.fit(y)

        assert priors.total_count == 4
        assert priors.count_0 == 3
        assert priors.count_1 == 1
        assert priors.pi_0 == pytest.approx(0.75, abs=1e-12)
        assert priors.pi_1 == pytest.approx(0.25, abs=1e-12)
        assert priors.log_pi_0 == pytest.approx(math.log(0.75), abs=1e-12)
        assert priors.log_pi_1 == pytest.approx(math.log(0.25), abs=1e-12)
        assert priors.prior(0) == priors.pi_0
        assert priors.prior(1) == priors.pi_1
        assert priors.log_prior(0) == priors.log_pi_0
        assert priors.log_prior(1) == priors.log_pi_1

    def test_priors_empty_array_raises(self) -> None:
        """Array vazio deve levantar DistributionError."""
        with pytest.raises(DistributionError, match="vazio"):
            ClassPriors.fit(np.array([]))

    def test_priors_invalid_classes_raises(self) -> None:
        """Classes fora de {0, 1} devem levantar DistributionError."""
        with pytest.raises(DistributionError, match="Classes inválidas"):
            ClassPriors.fit(np.array([0, 1, 2]))

    def test_priors_missing_class_raises(self) -> None:
        """Ausência de uma das classes deve levantar DistributionError."""
        with pytest.raises(DistributionError, match="Classe ausente"):
            ClassPriors.fit(np.array([0, 0, 0]))
        with pytest.raises(DistributionError, match="Classe ausente"):
            ClassPriors.fit(np.array([1, 1, 1]))

    def test_priors_invalid_access_raises(self) -> None:
        """Acesso a classe inválida em prior() ou log_prior()."""
        priors = ClassPriors.fit(np.array([0, 1]))
        with pytest.raises(DistributionError, match="Classe inválida"):
            priors.prior(2)
        with pytest.raises(DistributionError, match="Classe inválida"):
            priors.log_prior(-1)


class TestGaussianMLE:
    """Testes para a classe GaussianMLE."""

    def test_gaussian_mle_ddof0_strictly_enforced(self) -> None:
        """Garante que a variância utiliza ddof=0 (divisor N), e não ddof=1 (divisor N-1)."""
        # Conjunto clássico: [2, 4, 4, 4, 5, 5, 7, 9]
        # N = 8, Soma = 40, Media = 5.0
        # Quadrados das diferenças: [9, 1, 1, 1, 0, 0, 4, 16] = 32
        # Variância populacional (ddof=0, MLE): 32 / 8 = 4.0, Sigma = 2.0
        # Variância amostral (ddof=1, Bessel): 32 / 7 = 4.5714...
        data = np.array([2, 4, 4, 4, 5, 5, 7, 9], dtype=float)
        model = GaussianMLE.fit(data)

        assert model.mu == pytest.approx(5.0, abs=1e-12)
        assert model.variance == pytest.approx(4.0, abs=1e-12)
        assert model.sigma == pytest.approx(2.0, abs=1e-12)
        assert model.n_samples == 8

        # Confirma que é estritamente diferente de ddof=1
        sample_var = float(np.var(data, ddof=1))
        assert model.variance != pytest.approx(sample_var, abs=1e-3)

    def test_gaussian_pdf_log_pdf_parity_with_scipy(self) -> None:
        """Verifica a precisão numérica de log_pdf e pdf contra scipy.stats.norm."""
        data = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
        model = GaussianMLE.fit(data)

        test_points = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        expected_log_pdf = stats.norm.logpdf(test_points, loc=model.mu, scale=model.sigma)
        expected_pdf = stats.norm.pdf(test_points, loc=model.mu, scale=model.sigma)

        actual_log_pdf = model.log_pdf(test_points)
        actual_pdf = model.pdf(test_points)

        np.testing.assert_allclose(actual_log_pdf, expected_log_pdf, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(actual_pdf, expected_pdf, rtol=1e-12, atol=1e-14)

        # Teste com escalar
        assert model.log_pdf(15.0) == pytest.approx(float(expected_log_pdf[2]), abs=1e-12)
        assert model.pdf(15.0) == pytest.approx(float(expected_pdf[2]), abs=1e-12)

    def test_gaussian_zero_variance_raises(self) -> None:
        """Dados com variância zero devem levantar DistributionError conforme política operacional."""
        constant_data = np.array([5.0, 5.0, 5.0, 5.0])
        with pytest.raises(DistributionError, match="Variância não positiva"):
            GaussianMLE.fit(constant_data)

    def test_gaussian_non_finite_raises(self) -> None:
        """Valores NaN ou infinitos devem levantar DistributionError."""
        with pytest.raises(DistributionError, match="infinitos ou NaN"):
            GaussianMLE.fit(np.array([1.0, np.nan, 3.0]))
        with pytest.raises(DistributionError, match="infinitos ou NaN"):
            GaussianMLE.fit(np.array([1.0, np.inf, 3.0]))

    def test_gaussian_log_pdf_non_finite_input_raises(self) -> None:
        """Entrada não-finita para inferência deve levantar DistributionError."""
        model = GaussianMLE.fit(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(DistributionError, match="não-finitos"):
            model.log_pdf(np.nan)


class TestBernoulliMLE:
    """Testes para a classe BernoulliMLE."""

    def test_bernoulli_mle_without_smoothing_alpha_0(self) -> None:
        """Verifica estimação MLE pura (alpha=0)."""
        # 3 uns e 7 zeros -> p = 3 / 10 = 0.3
        z = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        model = BernoulliMLE.fit(z, alpha=0.0)

        assert model.p == pytest.approx(0.3, abs=1e-12)
        assert model.alpha == 0.0
        assert model.n_samples == 10
        assert model.n_ones == 3
        assert model.n_zeros == 7

        assert model.pmf(1) == pytest.approx(0.3, abs=1e-12)
        assert model.pmf(0) == pytest.approx(0.7, abs=1e-12)
        assert model.log_pmf(1) == pytest.approx(math.log(0.3), abs=1e-12)
        assert model.log_pmf(0) == pytest.approx(math.log(0.7), abs=1e-12)

    def test_bernoulli_with_laplace_smoothing_alpha_1(self) -> None:
        """Verifica suavização de Laplace (alpha=1)."""
        # 3 uns e 7 zeros -> p = (3 + 1) / (10 + 2) = 4 / 12 = 1/3
        z = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        model = BernoulliMLE.fit(z, alpha=1.0)

        assert model.p == pytest.approx(4.0 / 12.0, abs=1e-12)
        assert model.alpha == 1.0

    def test_bernoulli_non_binary_raises(self) -> None:
        """Valores não binários devem levantar DistributionError."""
        with pytest.raises(DistributionError, match="não-binários"):
            BernoulliMLE.fit(np.array([0, 1, 2]))

    def test_bernoulli_zero_probability_log_pmf_policy(self) -> None:
        """Garante tratamento explícito de probabilidade zero sem gerar 0*log(0)."""
        # Todos zeros: p = 0.0 com alpha=0
        z_zeros = np.array([0, 0, 0, 0])
        model_zeros = BernoulliMLE.fit(z_zeros, alpha=0.0)
        assert model_zeros.p == 0.0

        # P(X=0) = 1.0 -> log_pmf(0) = 0.0
        assert model_zeros.log_pmf(0) == pytest.approx(0.0, abs=1e-12)

        # Avaliar P(X=1) quando p=0 deve levantar DistributionError explícito
        with pytest.raises(DistributionError, match="P\\(X = 1\\) = 0"):
            model_zeros.log_pmf(1)

        # Todos uns: p = 1.0 com alpha=0
        z_ones = np.array([1, 1, 1, 1])
        model_ones = BernoulliMLE.fit(z_ones, alpha=0.0)
        assert model_ones.p == 1.0

        assert model_ones.log_pmf(1) == pytest.approx(0.0, abs=1e-12)
        with pytest.raises(DistributionError, match="P\\(X = 0\\) = 0"):
            model_ones.log_pmf(0)
