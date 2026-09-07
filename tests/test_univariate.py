"""Testes unitários e analíticos para src/univariate.py.

Verifica:
- Resolução analítica das fronteiras Gaussianas (casos linear, quadrático, delta < 0, raízes fora do domínio).
- Consistência das probabilidades a posteriori e regra de decisão MAP (empate exato -> 0).
- Estabilidade numérica do logsumexp.
- Invariância estrita do ajuste frente a alterações no conjunto reservado para teste.
- Parâmetros canônicos do dataset de treino.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_and_prepare_data
from src.distributions import BernoulliMLE, ClassPriors, GaussianMLE
from src.univariate import (
    BoundaryAnalysis,
    NumericalExample,
    UnivariateFeatureResult,
    compute_univariate_example,
    logsumexp,
    run_all_univariate_analyses,
    solve_gaussian_decision_boundary,
)


class TestBoundarySolver:
    """Testes para resolução analítica de fronteiras Gaussianas."""

    def test_solve_boundary_linear_case_equal_variances(self) -> None:
        """Quando as variâncias são iguais, A = 0 e a fronteira é linear (reta)."""
        # mu0 = 10, var0 = 25, mu1 = 20, var1 = 25
        # pi0 = 0.5, pi1 = 0.5 (priors balanceadas)
        # Fronteira deve ser exatamente o ponto médio: x* = 15.0
        m0 = GaussianMLE(mu=10.0, variance=25.0, sigma=5.0, n_samples=100)
        m1 = GaussianMLE(mu=20.0, variance=25.0, sigma=5.0, n_samples=100)
        priors = ClassPriors(
            total_count=200, count_0=100, count_1=100, pi_0=0.5, pi_1=0.5,
            log_pi_0=math.log(0.5), log_pi_1=math.log(0.5)
        )

        domain = (0.0, 30.0)
        boundary = solve_gaussian_decision_boundary(m0, m1, priors, domain)

        assert abs(boundary.A) < 1e-12
        assert len(boundary.all_roots) == 1
        assert boundary.all_roots[0] == pytest.approx(15.0, abs=1e-10)
        assert len(boundary.in_domain_roots) == 1
        assert boundary.in_domain_roots[0] == pytest.approx(15.0, abs=1e-10)

        # Regiões: [0, 15] deve predizer Classe 0, [15, 30] deve predizer Classe 1
        assert len(boundary.decision_regions) == 2
        assert boundary.decision_regions[0]["predicted_class"] == 0
        assert boundary.decision_regions[1]["predicted_class"] == 1

    def test_solve_boundary_negative_discriminant_no_real_roots(self) -> None:
        """Quando o discriminante é estritamente negativo, não há raízes reais."""
        # Configuração que gera parábola estritamente negativa (ex: MonthlyCharges)
        m0 = GaussianMLE(mu=60.0, variance=900.0, sigma=30.0, n_samples=1000)
        m1 = GaussianMLE(mu=75.0, variance=600.0, sigma=24.494897, n_samples=500)
        # Prior desbalanceada que empurra a curva para baixo
        priors = ClassPriors(
            total_count=1500, count_0=1100, count_1=400, pi_0=1100/1500, pi_1=400/1500,
            log_pi_0=math.log(1100/1500), log_pi_1=math.log(400/1500)
        )

        domain = (10.0, 120.0)
        boundary = solve_gaussian_decision_boundary(m0, m1, priors, domain)

        assert boundary.discriminant < 0
        assert len(boundary.all_roots) == 0
        assert len(boundary.in_domain_roots) == 0
        assert len(boundary.decision_regions) == 1
        assert boundary.decision_regions[0]["predicted_class"] == 0

    def test_solve_boundary_roots_outside_display_domain(self) -> None:
        """Raízes reais que se encontram fora do domínio de suporte da variável."""
        # Se raízes são negativas (ex: -10 e -2) e domínio é [0, 50]
        m0 = GaussianMLE(mu=35.0, variance=550.0, sigma=23.45, n_samples=4000)
        m1 = GaussianMLE(mu=18.0, variance=380.0, sigma=19.49, n_samples=1500)
        priors = ClassPriors(
            total_count=5500, count_0=4000, count_1=1500, pi_0=4000/5500, pi_1=1500/5500,
            log_pi_0=math.log(4000/5500), log_pi_1=math.log(1500/5500)
        )

        domain = (0.0, 72.0)
        boundary = solve_gaussian_decision_boundary(m0, m1, priors, domain)

        assert len(boundary.all_roots) == 2
        for r in boundary.all_roots:
            assert r < 0.0  # Todas as raízes são negativas
        assert len(boundary.in_domain_roots) == 0
        assert len(boundary.decision_regions) == 1
        assert boundary.decision_regions[0]["predicted_class"] == 0


class TestLogSumExpAndBayesExamples:
    """Testes para estabilidade numérica e cálculo de exemplos Bayesianos."""

    def test_logsumexp_numerical_stability(self) -> None:
        """Verifica robustez contra overflow e underflow."""
        # Valores muito grandes
        res_large = logsumexp(1000.0, 1000.0)
        assert res_large == pytest.approx(1000.0 + math.log(2.0), abs=1e-12)

        # Valores muito negativos
        res_small = logsumexp(-1000.0, -1000.0)
        assert res_small == pytest.approx(-1000.0 + math.log(2.0), abs=1e-12)

        # Diferença extrema (deve retornar o maior)
        res_extreme = logsumexp(500.0, -500.0)
        assert res_extreme == pytest.approx(500.0, abs=1e-12)

    def test_compute_univariate_example_bayes_theorem(self) -> None:
        """Verifica cálculo completo do exemplo Bayesiano passo a passo."""
        m0 = GaussianMLE(mu=10.0, variance=4.0, sigma=2.0, n_samples=100)
        m1 = GaussianMLE(mu=20.0, variance=4.0, sigma=2.0, n_samples=100)
        priors = ClassPriors(
            total_count=200, count_0=150, count_1=50, pi_0=0.75, pi_1=0.25,
            log_pi_0=math.log(0.75), log_pi_1=math.log(0.25)
        )

        ex = compute_univariate_example("feature", 12.0, "Valor de Teste 12", m0, m1, priors)

        # P(Y=0|x) + P(Y=1|x) == 1.0
        assert ex.posterior_0 + ex.posterior_1 == pytest.approx(1.0, abs=1e-12)

        # Como x=12 está mais perto de mu0=10 e pi0=0.75, deve favorecer fortemente classe 0
        assert ex.posterior_0 > ex.posterior_1
        assert ex.decision_map == 0

        # Verificação do Teorema de Bayes:
        # post_0 = p(x|0) * pi0 / p(x)
        expected_post_0 = (ex.p_x_given_0 * ex.prior_0) / ex.marginal_evidence_px
        assert ex.posterior_0 == pytest.approx(expected_post_0, abs=1e-10)

    def test_tie_breaking_policy_resolves_to_zero(self) -> None:
        """Garante que empate exato em posteriors (0.5 vs 0.5) decide Classe 0."""
        m0 = GaussianMLE(mu=10.0, variance=4.0, sigma=2.0, n_samples=100)
        m1 = GaussianMLE(mu=20.0, variance=4.0, sigma=2.0, n_samples=100)
        priors = ClassPriors(
            total_count=200, count_0=100, count_1=100, pi_0=0.5, pi_1=0.5,
            log_pi_0=math.log(0.5), log_pi_1=math.log(0.5)
        )

        # Em x = 15.0, as verossimilhanças e os priors são rigorosamente idênticos
        ex = compute_univariate_example("feature", 15.0, "Ponto de Empate Exato", m0, m1, priors)
        assert ex.posterior_0 == pytest.approx(0.5, abs=1e-10)
        assert ex.posterior_1 == pytest.approx(0.5, abs=1e-10)
        # Política canônica de desempate: ordem [0, 1] -> classe 0
        assert ex.decision_map == 0


class TestInvarianceAndCanonicalData:
    """Testes de invariância frente ao conjunto de teste e parâmetros canônicos."""

    def test_fit_invariance_to_test_alteration_synthetic(self) -> None:
        """Garante que o ajuste é estritamente invariante a alterações nos dados de teste no pipeline."""
        df_full = pd.DataFrame({
            "tenure": [10, 20, 30, 40, 50, 60, 70, 80],
            "MonthlyCharges": [25.0, 50.0, 75.0, 30.0, 60.0, 90.0, 100.0, 110.0],
            "gender": [0, 1, 0, 1, 0, 1, 0, 1],
            "Churn": [0, 0, 0, 1, 1, 1, 0, 1],
        })
        
        # Simulamos um particionamento estrito baseado em índices pré-definidos
        train_idx = [0, 1, 2, 3, 4, 5]
        test_idx = [6, 7]
        
        df_train_1 = df_full.iloc[train_idx].copy()
        res_1 = run_all_univariate_analyses(df_train_1)
        
        # Alteramos DRASTICAMENTE o dataframe apenas na partição de teste
        df_full.loc[test_idx, "tenure"] = [9999, 8888]
        df_full.loc[test_idx, "MonthlyCharges"] = [999.0, 888.0]
        # O split continua nos mesmos índices
        df_train_2 = df_full.iloc[train_idx].copy()
        res_2 = run_all_univariate_analyses(df_train_2)
        
        # Conferimos priors
        assert res_1["tenure"].priors.pi_0 == res_2["tenure"].priors.pi_0
        assert res_1["tenure"].priors.pi_1 == res_2["tenure"].priors.pi_1

        # Conferimos ambas as classes para as features
        for feat in ["tenure", "MonthlyCharges", "gender"]:
            m1_0, m1_1 = res_1[feat].model_class_0, res_1[feat].model_class_1
            m2_0, m2_1 = res_2[feat].model_class_0, res_2[feat].model_class_1
            if isinstance(m1_0, GaussianMLE) and isinstance(m2_0, GaussianMLE):
                assert m1_0.mu == pytest.approx(m2_0.mu, abs=1e-15)
                assert m1_0.variance == pytest.approx(m2_0.variance, abs=1e-15)
                assert m1_1.mu == pytest.approx(m2_1.mu, abs=1e-15)
                assert m1_1.variance == pytest.approx(m2_1.variance, abs=1e-15)
            elif isinstance(m1_0, BernoulliMLE) and isinstance(m2_0, BernoulliMLE):
                assert m1_0.p == pytest.approx(m2_0.p, abs=1e-15)
                assert m1_1.p == pytest.approx(m2_1.p, abs=1e-15)

    def test_bernoulli_invalid_inputs(self) -> None:
        """L2-01: Regressão para garantir que Bernoulli rejeite frações e alpha negativo."""
        from src.distributions import BernoulliMLE, DistributionError
        
        with pytest.raises(DistributionError, match="binários exatos"):
            BernoulliMLE.fit(np.array([0.9, 1.9]))
            
        with pytest.raises(DistributionError, match="domínio binário"):
            b = BernoulliMLE.fit(np.array([0, 1]))
            b.pmf(0.9)
            
        with pytest.raises(DistributionError, match="finito e não negativo"):
            BernoulliMLE.fit(np.array([0, 1]), alpha=-1.0)

    def test_global_equality_and_decision_change(self) -> None:
        """L2-02 e L2-03: Testa igualdade global (A=B=C=0) e mudança de decisão."""
        from src.distributions import GaussianMLE, ClassPriors
        from src.univariate import solve_gaussian_decision_boundary
        
        # 1. Identical distributions and equal priors (Global Equality)
        model_0 = GaussianMLE(mu=10.0, variance=5.0, sigma=math.sqrt(5.0), n_samples=100)
        model_1 = GaussianMLE(mu=10.0, variance=5.0, sigma=math.sqrt(5.0), n_samples=100)
        priors_eq = ClassPriors(total_count=200, count_0=100, count_1=100, 
                                pi_0=0.5, pi_1=0.5, log_pi_0=math.log(0.5), log_pi_1=math.log(0.5))
                                
        boundary_eq = solve_gaussian_decision_boundary(model_0, model_1, priors_eq, (0, 20))
        assert boundary_eq.is_globally_equal is True
        assert boundary_eq.is_lambda_globally_equal is True
        assert len(boundary_eq.decision_regions) == 1
        assert boundary_eq.decision_regions[0]["predicted_class"] == 0 # Tie-breaker is 0
        
        # 2. Synthetic decision change
        # Strong prior for class 1
        priors_c1 = ClassPriors(total_count=100, count_0=10, count_1=90, 
                                pi_0=0.1, pi_1=0.9, log_pi_0=math.log(0.1), log_pi_1=math.log(0.9))
        boundary_c1 = solve_gaussian_decision_boundary(model_0, model_1, priors_c1, (0, 20))
        # Likelihoods are same, but prior favors 1, so decision should be 1
        assert boundary_c1.decision_regions[0]["predicted_class"] == 1

    def test_canonical_training_parameters(self) -> None:
        """Verifica os parâmetros estimados sobre o conjunto de treino canônico real."""
        split = load_and_prepare_data(test_size=0.2, random_state=42)
        results = run_all_univariate_analyses(split.X_train.assign(Churn=split.y_train))

        priors = results["tenure"].priors
        assert priors.count_0 == 4139
        assert priors.count_1 == 1495
        assert priors.total_count == 5634
        assert priors.pi_0 == pytest.approx(4139 / 5634, abs=1e-12)
        assert priors.pi_1 == pytest.approx(1495 / 5634, abs=1e-12)

        # tenure
        t_res = results["tenure"]
        m0_t: GaussianMLE = t_res.model_class_0  # type: ignore
        m1_t: GaussianMLE = t_res.model_class_1  # type: ignore
        assert m0_t.mu == pytest.approx(37.5876, abs=1e-2)
        assert m1_t.mu == pytest.approx(18.3585, abs=1e-2)
        assert m0_t.sigma == pytest.approx(24.1405, abs=1e-2)
        assert m1_t.sigma == pytest.approx(19.7313, abs=1e-2)

        # MonthlyCharges
        mc_res = results["MonthlyCharges"]
        m0_mc: GaussianMLE = mc_res.model_class_0  # type: ignore
        m1_mc: GaussianMLE = mc_res.model_class_1  # type: ignore
        assert m0_mc.mu == pytest.approx(61.3432, abs=1e-2)
        assert m1_mc.mu == pytest.approx(74.8602, abs=1e-2)
        assert m0_mc.sigma == pytest.approx(31.1316, abs=1e-2)
        assert m1_mc.sigma == pytest.approx(24.5955, abs=1e-2)

        # gender
        g_res = results["gender"]
        m0_g: BernoulliMLE = g_res.model_class_0  # type: ignore
        m1_g: BernoulliMLE = g_res.model_class_1  # type: ignore
        assert m0_g.alpha == 0.0
        assert m1_g.alpha == 0.0
        assert m0_g.p == pytest.approx(2084 / 4139, abs=1e-4)
        assert m1_g.p == pytest.approx(749 / 1495, abs=1e-4)

    def test_impossible_scores_raise_distribution_error(self) -> None:
        """L2-02: Garantir que um escore impossível (ambas likelihoods = 0) levante erro."""
        from src.distributions import GaussianMLE, DistributionError
        from src.univariate import compute_univariate_example, ClassPriors
        
        import numpy as np
        
        # Se x está muito longe para as duas distribuições:
        m0 = GaussianMLE(mu=0.0, variance=1.0, sigma=1.0, n_samples=100)
        m1 = GaussianMLE(mu=0.0, variance=1.0, sigma=1.0, n_samples=100)
        priors = ClassPriors.fit(np.array([0]*100 + [1]*100))
        
        with pytest.raises(DistributionError, match="Cálculo Bayesiano impossível: Evidência nula"):
            compute_univariate_example("feature", 1e200, "Impossible", m0, m1, priors)

    def test_single_inf_score(self) -> None:
        """L2-02: Garantir que se apenas uma likelihood for 0, o normalizador funciona."""
        from src.distributions import BernoulliMLE
        import numpy as np

        class MockModel(BernoulliMLE):
            def __init__(self, is_inf: bool):
                self.is_inf = is_inf
                # Call parent with dummy values so it passes any internal checks if needed
                super().__init__(p=1.0 if not is_inf else 0.0, n_samples=100, n_ones=50, n_zeros=50, alpha=0.0)
            def pmf(self, x):
                return 0.0 if self.is_inf else 0.5
            def log_pmf(self, x):
                return -float('inf') if self.is_inf else np.log(0.5)

        m0 = MockModel(is_inf=True)
        m1 = MockModel(is_inf=False)
        priors = ClassPriors.fit(np.array([0]*100 + [1]*100))
        
        ex = compute_univariate_example("feature", 0, "Zero", m0, m1, priors) # type: ignore
        
        assert ex.p_x_given_0 == 0.0
        assert ex.posterior_0 == 0.0
        assert ex.posterior_1 == 1.0
        assert ex.likelihood_ratio_lambda == float('inf')

