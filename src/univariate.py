"""Módulo de modelagem probabilística univariada das três características.

Executa as 5 etapas analíticas do GUIA.md para cada uma das características:
- X1 (tenure) - Gaussiana
- X2 (MonthlyCharges) - Gaussiana
- X3 (gender) - Bernoulli

Inclui:
- Estimação dos parâmetros condicionais à classe (exclusivamente no treino, MLE com ddof=0).
- Cálculo e interpretação de verossimilhanças e da razão de verossimilhanças Lambda(x).
- Aplicação do Teorema de Bayes com cálculo de posteriors via logsumexp.
- Dedução e determinação exata das fronteiras e regiões de decisão Bayesianas (MAP).
- Exemplos numéricos didáticos detalhados.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.distributions import BernoulliMLE, ClassPriors, DistributionError, GaussianMLE


def logsumexp(a: float, b: float) -> float:
    """Calcula log(exp(a) + exp(b)) de forma numericamente estável."""
    m = max(a, b)
    if not math.isfinite(m):
        return m
    return m + math.log(math.exp(a - m) + math.exp(b - m))


@dataclass(frozen=True)
class NumericalExample:
    """Exemplo numérico completo de aplicação do Teorema de Bayes."""

    feature_name: str
    feature_value: Union[float, int]
    value_label: str
    p_x_given_0: float  # Densidade contínua ou probabilidade discreta
    p_x_given_1: float
    log_likelihood_0: float
    log_likelihood_1: float
    prior_0: float
    prior_1: float
    likelihood_ratio_lambda: float
    log_likelihood_ratio: float
    evidence_favors_class: int  # 0, 1 ou -1 para indiferença
    marginal_evidence_px: float  # Denominador p(x)
    posterior_0: float  # P(Y = 0 | x)
    posterior_1: float  # P(Y = 1 | x)
    decision_map: int  # Decisão MAP univariada (0 ou 1)


@dataclass(frozen=True)
class BoundaryAnalysis:
    """Análise analítica de fronteiras e regiões de decisão Bayesianas."""

    A: float
    B: float
    C: float
    discriminant: float
    all_roots: List[float]  # Todas as raízes reais da equação Ax^2 + Bx + C = 0
    in_domain_roots: List[float]  # Raízes contidas no domínio de interesse da feature
    is_globally_equal: bool # Se A=0, B=0, C=0 (prior+likelihood empatam)
    display_domain: Tuple[float, float]
    decision_regions: List[Dict[str, Any]]  # Intervalos e a classe predita em cada um
    likelihood_equality_points: List[float]  # Valores onde Lambda(x) = 1
    likelihood_equality_in_domain: List[float]
    is_lambda_globally_equal: bool # Se distribuições são idênticas (A=0, B=0, C_lambda=0)


@dataclass
class UnivariateFeatureResult:
    """Resultado completo da análise univariada de uma característica."""

    feature_name: str
    feature_type: str  # "continuous" ou "categorical_binary"
    distribution_family: str  # "gaussian" ou "bernoulli"
    display_domain: Tuple[float, float]
    domain_justification: str
    model_class_0: Union[GaussianMLE, BernoulliMLE]
    model_class_1: Union[GaussianMLE, BernoulliMLE]
    priors: ClassPriors
    boundary_analysis: Optional[BoundaryAnalysis]
    categorical_rules: Optional[Dict[str, int]]  # Para variáveis categóricas
    examples: List[NumericalExample]
    lambda_1_points: List[float]  # Onde Lambda = 1
    discrimination_commentary: str


def solve_gaussian_decision_boundary(
    model_0: GaussianMLE,
    model_1: GaussianMLE,
    priors: ClassPriors,
    domain: Tuple[float, float],
) -> BoundaryAnalysis:
    """Resolve analiticamente a fronteira MAP: P(Y=0|x) = P(Y=1|x).

    Condição de igualdade das posteriores:
    log p(x|1) + log pi_1 = log p(x|0) + log pi_0
    <=> (1/(2v0) - 1/(2v1)) x^2 + (mu1/v1 - mu0/v0) x + (mu0^2/(2v0) - mu1^2/(2v1) + log(sig0/sig1) + log(pi1/pi0)) = 0
    Ax^2 + Bx + C = 0
    """
    v0, v1 = model_0.variance, model_1.variance
    mu0, mu1 = model_0.mu, model_1.mu
    sig0, sig1 = model_0.sigma, model_1.sigma
    pi0, pi1 = priors.pi_0, priors.pi_1

    A = 1.0 / (2.0 * v0) - 1.0 / (2.0 * v1)
    B = mu1 / v1 - mu0 / v0
    C = (
        (mu0 ** 2) / (2.0 * v0)
        - (mu1 ** 2) / (2.0 * v1)
        + math.log(sig0 / sig1)
        + math.log(pi1 / pi0)
    )

    discriminant = (B ** 2) - (4.0 * A * C)
    roots: List[float] = []
    is_globally_equal = False

    # Tolerância numérica
    if abs(A) < 1e-12:
        if abs(B) > 1e-12:
            r = -C / B
            roots.append(float(r))
        elif abs(C) < 1e-12:
            is_globally_equal = True
    else:
        if discriminant >= -1e-14:
            if discriminant < 1e-14: # raiz dupla
                r = -B / (2.0 * A)
                roots.append(float(r))
            else:
                sqrt_d = math.sqrt(discriminant)
                r1 = (-B - sqrt_d) / (2.0 * A)
                r2 = (-B + sqrt_d) / (2.0 * A)
                roots = sorted([float(r1), float(r2)])

    # Filtragem das raízes no domínio de exibição [d_min, d_max]
    d_min, d_max = domain
    in_domain_roots = [r for r in roots if d_min <= r <= d_max]

    # Pontos onde Lambda(x) = 1 (onde log p(x|1) = log p(x|0), ou seja, sem termo de prior)
    C_lambda = (mu0 ** 2) / (2.0 * v0) - (mu1 ** 2) / (2.0 * v1) + math.log(sig0 / sig1)
    disc_lambda = (B ** 2) - (4.0 * A * C_lambda)
    lambda_roots: List[float] = []
    is_lambda_globally_equal = False
    
    if abs(A) < 1e-12:
        if abs(B) > 1e-12:
            lambda_roots.append(float(-C_lambda / B))
        elif abs(C_lambda) < 1e-12:
            is_lambda_globally_equal = True
    else:
        if disc_lambda >= -1e-14:
            if disc_lambda < 1e-14:
                lambda_roots.append(float(-B / (2.0 * A)))
            else:
                sqrt_d_lambda = math.sqrt(disc_lambda)
                r1_l = (-B - sqrt_d_lambda) / (2.0 * A)
                r2_l = (-B + sqrt_d_lambda) / (2.0 * A)
                lambda_roots = sorted([float(r1_l), float(r2_l)])

    lambda_in_domain = [r for r in lambda_roots if d_min <= r <= d_max]

    # Determinação das regiões de decisão no domínio
    # Cria pontos de partição: [d_min] + in_domain_roots + [d_max]
    boundary_points = sorted(list(set([d_min] + in_domain_roots + [d_max])))
    decision_regions: List[Dict[str, Any]] = []

    for i in range(len(boundary_points) - 1):
        left = boundary_points[i]
        right = boundary_points[i + 1]
        mid = (left + right) / 2.0

        # Avaliação do escore L1(mid) - L0(mid)
        l0 = model_0.log_pdf(mid) + priors.log_prior(0)
        l1 = model_1.log_pdf(mid) + priors.log_prior(1)
        pred_class = 1 if l1 > l0 else 0  # Empate exato vai para classe 0

        decision_regions.append(
            {
                "interval": [float(left), float(right)],
                "predicted_class": int(pred_class),
                "label": "Retained (0)" if pred_class == 0 else "Churn (1)",
            }
        )

    return BoundaryAnalysis(
        A=float(A),
        B=float(B),
        C=float(C),
        discriminant=float(discriminant),
        all_roots=roots,
        in_domain_roots=in_domain_roots,
        is_globally_equal=is_globally_equal,
        display_domain=domain,
        decision_regions=decision_regions,
        likelihood_equality_points=lambda_roots,
        likelihood_equality_in_domain=lambda_in_domain,
        is_lambda_globally_equal=is_lambda_globally_equal,
    )


def compute_univariate_example(
    feature_name: str,
    x_val: Union[float, int],
    label: str,
    model_0: Union[GaussianMLE, BernoulliMLE],
    model_1: Union[GaussianMLE, BernoulliMLE],
    priors: ClassPriors,
) -> NumericalExample:
    """Calcula um exemplo numérico completo de verossimilhança, razão Lambda e Bayes."""
    if isinstance(model_0, GaussianMLE) and isinstance(model_1, GaussianMLE):
        p0 = float(model_0.pdf(float(x_val)))
        p1 = float(model_1.pdf(float(x_val)))
        log_l0 = float(model_0.log_pdf(float(x_val)))
        log_l1 = float(model_1.log_pdf(float(x_val)))
    elif isinstance(model_0, BernoulliMLE) and isinstance(model_1, BernoulliMLE):
        p0 = float(model_0.pmf(x_val))
        p1 = float(model_1.pmf(x_val))
        log_l0 = float(model_0.log_pmf(x_val))
        log_l1 = float(model_1.log_pmf(x_val))
    else:
        raise DistributionError("Modelos incompatíveis para cálculo de exemplo.")

    if log_l0 == -float("inf") and log_l1 == -float("inf"):
        raise DistributionError(f"Cálculo Bayesiano impossível: Evidência nula para ambas as classes (x={x_val}).")
    else:
        if log_l0 == -float("inf"):
            log_lambda = float("inf")
        elif log_l1 == -float("inf"):
            log_lambda = -float("inf")
        else:
            log_lambda = log_l1 - log_l0
            
        try:
            lambda_val = math.exp(log_lambda)
        except OverflowError:
            lambda_val = float("inf")
            
        favors = 1 if log_lambda > 0 else (0 if log_lambda < 0 else -1)

        L0 = log_l0 + priors.log_prior(0)
        L1 = log_l1 + priors.log_prior(1)
        log_marginal = logsumexp(L0, L1)
        marginal_px = math.exp(log_marginal) if log_marginal != -float("inf") else 0.0

        post_0 = math.exp(L0 - log_marginal) if log_marginal != -float("inf") else priors.pi_0
        post_1 = math.exp(L1 - log_marginal) if log_marginal != -float("inf") else priors.pi_1
        decision = 1 if post_1 > post_0 else 0

    return NumericalExample(
        feature_name=feature_name,
        feature_value=x_val,
        value_label=label,
        p_x_given_0=p0,
        p_x_given_1=p1,
        log_likelihood_0=log_l0,
        log_likelihood_1=log_l1,
        prior_0=priors.pi_0,
        prior_1=priors.pi_1,
        likelihood_ratio_lambda=lambda_val,
        log_likelihood_ratio=log_lambda,
        evidence_favors_class=favors,
        marginal_evidence_px=marginal_px,
        posterior_0=post_0,
        posterior_1=post_1,
        decision_map=decision,
    )


def analyze_univariate_tenure(
    df_train: pd.DataFrame,
    priors: ClassPriors,
) -> UnivariateFeatureResult:
    """Executa a análise Bayesiana completa para X1: tenure."""
    domain = (0.0, 72.0)
    x0 = df_train[df_train["Churn"] == 0]["tenure"].values
    x1 = df_train[df_train["Churn"] == 1]["tenure"].values

    model_0 = GaussianMLE.fit(x0)
    model_1 = GaussianMLE.fit(x1)

    boundary = solve_gaussian_decision_boundary(model_0, model_1, priors, domain)

    # Exemplos didáticos conforme Seção 4.1 da síntese: x1 = 3 meses e x1 = 60 meses
    ex1 = compute_univariate_example(
        "tenure", 3.0, "Cliente recente (3 meses)", model_0, model_1, priors
    )
    ex2 = compute_univariate_example(
        "tenure", 60.0, "Cliente antigo / fiel (60 meses)", model_0, model_1, priors
    )

    # Text generation dynamically
    prior0_pct = priors.pi_0 * 100
    
    roots_text = f"as raízes analíticas ({', '.join(f'{r:.2f}' for r in boundary.in_domain_roots)}) encontram-se dentro do domínio" if boundary.in_domain_roots else "as raízes analíticas encontram-se fora do domínio (ou não há raízes físicas)"
    lambda_roots_text = f"{', '.join(f'{r:.2f}' for r in boundary.likelihood_equality_points)} meses" if boundary.likelihood_equality_points else "(sem ponto de igualdade real)"
    if boundary.is_globally_equal:
        roots_text = "as distribuições e priors são idênticas (igualdade global)"
        
    commentary = (
        "A característica tenure (tempo de contrato em meses) apresenta médias distintas "
        f"(mu0 = {model_0.mu:.2f} meses para não-churn vs mu1 = {model_1.mu:.2f} meses para churn). "
        f"A razão de verossimilhanças Lambda(x) = 1 apresenta cruzamentos nas abscissas: {lambda_roots_text}. "
        f"Considerando o desbalanceamento a priori (P(Y=0) ~ {prior0_pct:.1f}%), {roots_text}, "
        f"o que se reflete nas regiões de decisão identificadas."
    )

    return UnivariateFeatureResult(
        feature_name="tenure",
        feature_type="discrete_numeric_continuous_approximation",
        distribution_family="gaussian",
        display_domain=domain,
        domain_justification=(
            "Tempo de permanência do cliente na operadora em meses inteiros [0, 72]. "
            "Aproximado por distribuição contínua Gaussiana condicional à classe. "
            "Essa premissa simplificadora ignora o pico expressivo em 0-5 meses e a cauda pesada de "
            "clientes muito antigos, além de alocar suporte teórico em valores negativos, o que representa "
            "uma ressalva estrutural importante na avaliação contínua do atributo empírico."
        ),
        model_class_0=model_0,
        model_class_1=model_1,
        priors=priors,
        boundary_analysis=boundary,
        categorical_rules=None,
        examples=[ex1, ex2],
        lambda_1_points=boundary.likelihood_equality_points,
        discrimination_commentary=commentary,
    )


def analyze_univariate_monthly_charges(
    df_train: pd.DataFrame,
    priors: ClassPriors,
) -> UnivariateFeatureResult:
    """Executa a análise Bayesiana completa para X2: MonthlyCharges."""
    domain = (18.25, 118.75)
    x0 = df_train[df_train["Churn"] == 0]["MonthlyCharges"].values
    x1 = df_train[df_train["Churn"] == 1]["MonthlyCharges"].values

    model_0 = GaussianMLE.fit(x0)
    model_1 = GaussianMLE.fit(x1)

    boundary = solve_gaussian_decision_boundary(model_0, model_1, priors, domain)

    # Exemplos didáticos: mensalidade baixa ($25.00) vs mensalidade alta ($85.00)
    ex1 = compute_univariate_example(
        "MonthlyCharges", 25.0, "Plano básico / econômico ($25.00)", model_0, model_1, priors
    )
    ex2 = compute_univariate_example(
        "MonthlyCharges", 85.0, "Plano avançado / fibra ($85.00)", model_0, model_1, priors
    )

    prior0_pct = priors.pi_0 * 100
    
    lambda_roots_text = f"{', '.join(f'{r:.2f}' for r in boundary.likelihood_equality_points)} dólares" if boundary.likelihood_equality_points else "(sem ponto de igualdade real)"
    if boundary.is_globally_equal:
        roots_text = "as distribuições e priors são idênticas (igualdade global)"
    else:
        roots_text = f"há raízes reais na fronteira de decisão MAP ({[f'{r:.2f}' for r in boundary.in_domain_roots]})" if boundary.all_roots else "inexistem raízes reais para a fronteira MAP"

    commentary = (
        "A característica MonthlyCharges (ticket médio em dólares) também difere entre as classes "
        f"(mu0 = {model_0.mu:.2f} vs mu1 = {model_1.mu:.2f}). "
        f"A razão de verossimilhanças Lambda(x) = 1 apresenta cruzamentos nas abscissas: {lambda_roots_text}. "
        f"Na avaliação Bayesiana MAP (Delta = {boundary.discriminant:.2e}), {roots_text}. "
        f"A decisão final acompanha o modelo prior (P(Y=0) ~ {prior0_pct:.1f}%) "
        f"e as densidades ponderadas das classes observadas nas regiões calculadas."
    )

    return UnivariateFeatureResult(
        feature_name="MonthlyCharges",
        feature_type="continuous",
        distribution_family="gaussian",
        display_domain=domain,
        domain_justification=(
            "Valor contínuo cobrado mensalmente em dólares [18.25, 118.75]. "
            "A modelagem inicial assume distribuição paramétrica Gaussiana, o que representa "
            "uma limitação estrutural frente à bimodalidade empírica observada no histograma do treino (clientes com planos "
            "básicos vs pacotes premium), já que a Gaussiana impõe um formato de sino simétrico centrado na média."
        ),
        model_class_0=model_0,
        model_class_1=model_1,
        priors=priors,
        boundary_analysis=boundary,
        categorical_rules=None,
        examples=[ex1, ex2],
        lambda_1_points=boundary.likelihood_equality_points,
        discrimination_commentary=commentary,
    )


def analyze_univariate_gender(
    df_train: pd.DataFrame,
    priors: ClassPriors,
) -> UnivariateFeatureResult:
    """Executa a análise Bayesiana completa para X3: gender."""
    domain = (0.0, 1.0)
    z0 = df_train[df_train["Churn"] == 0]["gender"].values
    z1 = df_train[df_train["Churn"] == 1]["gender"].values

    # Regra de suavização da seção 4.1: alpha=0 se ambas as categorias ocorrerem em ambas as classes
    n0_ones, n0_zeros = int(np.sum(z0 == 1)), int(np.sum(z0 == 0))
    n1_ones, n1_zeros = int(np.sum(z1 == 1)), int(np.sum(z1 == 0))

    if n0_ones > 0 and n0_zeros > 0 and n1_ones > 0 and n1_zeros > 0:
        alpha = 0.0
    else:
        alpha = 1.0

    model_0 = BernoulliMLE.fit(z0, alpha=alpha)
    model_1 = BernoulliMLE.fit(z1, alpha=alpha)

    # Exemplos didáticos para ambas as categorias
    ex_female = compute_univariate_example(
        "gender", 0, "Female (0)", model_0, model_1, priors
    )
    ex_male = compute_univariate_example(
        "gender", 1, "Male (1)", model_0, model_1, priors
    )

    categorical_rules = {
        "Female (0)": ex_female.decision_map,
        "Male (1)": ex_male.decision_map,
    }

    alpha_text = f"Suavização Laplace (alpha={alpha})" if alpha > 0.0 else "Sem suavização (alpha=0.0)"
    
    if ex_female.decision_map == ex_male.decision_map:
        decision_text = f"prevê Classe {ex_female.decision_map} para ambas as categorias"
    else:
        decision_text = f"prevê Classe {ex_female.decision_map} para Female e Classe {ex_male.decision_map} para Male"

    commentary = (
        "A característica categórica binária gender exibe as proporções: "
        f"P(Male|0) = {model_0.p:.4f} vs P(Male|1) = {model_1.p:.4f}; "
        f"P(Female|0) = {1.0 - model_0.p:.4f} vs P(Female|1) = {1.0 - model_1.p:.4f}. "
        f"As razões de verossimilhanças são (Lambda_Female = {ex_female.likelihood_ratio_lambda:.4f}, "
        f"Lambda_Male = {ex_male.likelihood_ratio_lambda:.4f}). "
        f"Com base nestas probabilidades e no prior forte, a regra MAP {decision_text} ({alpha_text})."
    )

    return UnivariateFeatureResult(
        feature_name="gender",
        feature_type="categorical_binary",
        distribution_family="bernoulli",
        display_domain=domain,
        domain_justification=(
            "Variável demográfica categórica binária (Female=0, Male=1). "
            "Modelada pela distribuição de Bernoulli com estimação MLE de frequências diretas."
        ),
        model_class_0=model_0,
        model_class_1=model_1,
        priors=priors,
        boundary_analysis=None,
        categorical_rules=categorical_rules,
        examples=[ex_female, ex_male],
        lambda_1_points=[],
        discrimination_commentary=commentary,
    )


def run_all_univariate_analyses(df_train: pd.DataFrame) -> Dict[str, UnivariateFeatureResult]:
    """Executa as três análises univariadas a partir do conjunto de treinamento."""
    priors = ClassPriors.fit(df_train["Churn"].values)

    res_tenure = analyze_univariate_tenure(df_train, priors)
    res_monthly = analyze_univariate_monthly_charges(df_train, priors)
    res_gender = analyze_univariate_gender(df_train, priors)

    return {
        "tenure": res_tenure,
        "MonthlyCharges": res_monthly,
        "gender": res_gender,
    }
