"""Módulo de visualização e geração de figuras dos experimentos.

Gera gráficos de alta qualidade visual para os resultados univariados e multivariados:
- Histogramas de treino por classe com densidades analíticas Gaussianas sobrepostas.
- Marcas dos pontos de igualdade de verossimilhança (Lambda=1).
- Curvas de probabilidades a posteriori P(Y=c|x) com regiões e fronteiras MAP.
- Gráficos de barras para variáveis categóricas (gender).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.univariate import UnivariateFeatureResult

# Configuração de estilo para publicação acadêmica
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


def plot_univariate_continuous(
    result: UnivariateFeatureResult,
    df_train: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Gera figura com densidades ajustadas, razão Lambda e posteriori para feature contínua."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    feat = result.feature_name
    d_min, d_max = result.display_domain
    x_grid = np.linspace(d_min, d_max, 500)

    # Dados de treino por classe
    x0 = df_train[df_train["Churn"] == 0][feat].values
    x1 = df_train[df_train["Churn"] == 1][feat].values

    m0 = result.model_class_0
    m1 = result.model_class_1
    priors = result.priors

    pdf0 = m0.pdf(x_grid)
    pdf1 = m1.pdf(x_grid)

    # Subplot 1: Histograma e Densidades Condicionais p(x|Y)
    ax1.hist(
        x0,
        bins=30,
        density=True,
        alpha=0.35,
        color="#1f77b4",
        label=f"Histórico Y=0 (Retido, N={len(x0)})",
    )
    ax1.hist(
        x1,
        bins=30,
        density=True,
        alpha=0.35,
        color="#d62728",
        label=f"Histórico Y=1 (Churn, N={len(x1)})",
    )

    ax1.plot(
        x_grid,
        pdf0,
        color="#1f77b4",
        lw=2.2,
        label=rf"Gaussiana Y=0 ($\mu$={m0.mu:.1f}, $\sigma$={m0.sigma:.1f})",
    )
    ax1.plot(
        x_grid,
        pdf1,
        color="#d62728",
        lw=2.2,
        label=rf"Gaussiana Y=1 ($\mu$={m1.mu:.1f}, $\sigma$={m1.sigma:.1f})",
    )

    # Pontos de igualdade de verossimilhança (Lambda = 1)
    if result.boundary_analysis:
        for pt in result.boundary_analysis.likelihood_equality_in_domain:
            ax1.axvline(
                pt,
                color="#2ca02c",
                linestyle="--",
                lw=1.8,
                label=rf"$\Lambda(x)=1$ ($x={pt:.1f}$)",
            )

    ax1.set_ylabel(r"Densidade de Probabilidade $p(x \mid Y)$", fontsize=11)
    ax1.set_title(
        f"Etapas 1 e 2: Ajuste Gaussiano e Verossimilhanças — {feat}",
        fontsize=13,
        fontweight="bold",
    )
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Razão de Verossimilhanças e Posteriori MAP
    log_l0 = m0.log_pdf(x_grid) + priors.log_prior(0)
    log_l1 = m1.log_pdf(x_grid) + priors.log_prior(1)
    m_max = np.maximum(log_l0, log_l1)
    log_marginal = m_max + np.log(np.exp(log_l0 - m_max) + np.exp(log_l1 - m_max))
    post_1 = np.exp(log_l1 - log_marginal)

    ax2.plot(
        x_grid,
        post_1,
        color="#9467bd",
        lw=2.5,
        label=r"Posteriori $P(Y=1 \mid x)$ (Bayes)",
    )
    ax2.axhline(
        0.5,
        color="#555555",
        linestyle=":",
        lw=1.5,
        label="Limiar de Decisão MAP ($P=0.5$)",
    )
    ax2.axhline(
        priors.pi_1,
        color="#d62728",
        linestyle="-.",
        alpha=0.7,
        label=f"Prior Churn $P(Y=1)={priors.pi_1:.3f}$",
    )

    # Indicação das raízes MAP no domínio (se existirem)
    if result.boundary_analysis and result.boundary_analysis.in_domain_roots:
        for r in result.boundary_analysis.in_domain_roots:
            ax2.axvline(
                r,
                color="#ff7f0e",
                linestyle="-",
                lw=2.0,
                label=rf"Fronteira MAP ($x^*={r:.1f}$)",
            )
    else:
        # Se não há fronteiras, todo o domínio pertence a uma única região
        if result.boundary_analysis and result.boundary_analysis.decision_regions:
            predicted = result.boundary_analysis.decision_regions[0]["predicted_class"]
        else:
            predicted = 0 # Fallback se decision_regions estiver vazio por algum motivo
            
        ax2.text(
            0.03,
            0.85,
            "Sem fronteira MAP no domínio físico (raízes ausentes/fora do domínio).\n"
            f"Decisão univariada MAP prevê Y={predicted} em todo o suporte [{d_min:.1f}, {d_max:.1f}].",
            transform=ax2.transAxes,
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3cd", edgecolor="#ffeeba"),
        )

    ax2.set_xlabel(f"Valor de {feat} (Domínio de Suporte)", fontsize=11)
    ax2.set_ylabel(r"Probabilidade a Posteriori $P(Y=1 \mid x)$", fontsize=11)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    output_path = output_dir / f"univariate_{feat}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_univariate_gender(
    result: UnivariateFeatureResult,
    output_dir: Path,
) -> Path:
    """Gera figura com distribuição de Bernoulli e comparação condicional para gender."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    m0: Any = result.model_class_0
    m1: Any = result.model_class_1
    priors = result.priors

    categories = ["Female (0)", "Male (1)"]
    p_given_0 = [1.0 - m0.p, m0.p]
    p_given_1 = [1.0 - m1.p, m1.p]

    x = np.arange(len(categories))
    width = 0.35

    # Subplot 1: Verossimilhanças P(X=z | Y=c)
    rects1 = ax1.bar(
        x - width / 2,
        p_given_0,
        width,
        label=f"Y=0 (Retido, N={m0.n_samples})",
        color="#1f77b4",
        alpha=0.85,
    )
    rects2 = ax1.bar(
        x + width / 2,
        p_given_1,
        width,
        label=f"Y=1 (Churn, N={m1.n_samples})",
        color="#d62728",
        alpha=0.85,
    )

    ax1.set_ylabel(r"Probabilidade Condicional $P(X_3=z \mid Y)$", fontsize=11)
    ax1.set_title("Etapa 2: Verossimilhanças Categóricas (gender)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=10)
    ax1.set_ylim(0, 0.65)
    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.6, axis="y")

    # Rótulos nas barras
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(
            f"{h:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(
            f"{h:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Subplot 2: Razão Lambda e Posteriori Bayesiana
    lambdas = [result.examples[0].likelihood_ratio_lambda, result.examples[1].likelihood_ratio_lambda]
    posteriors_1 = [result.examples[0].posterior_1, result.examples[1].posterior_1]

    rects_post = ax2.bar(
        x - width / 2,
        posteriors_1,
        width,
        label=r"Posteriori $P(Y=1 \mid gender)$",
        color="#9467bd",
        alpha=0.85,
    )
    ax2.axhline(
        0.5,
        color="#555555",
        linestyle=":",
        lw=1.5,
        label="Limiar MAP ($P=0.5$)",
    )
    ax2.axhline(
        priors.pi_1,
        color="#d62728",
        linestyle="-.",
        alpha=0.7,
        label=f"Prior Churn ($P={priors.pi_1:.3f}$)",
    )

    ax2.set_ylabel("Probabilidade / Razão", fontsize=11)
    ax2.set_title(r"Etapas 3 e 4: Razão $\Lambda$ e Posteriori MAP", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    cat_rules = result.categorical_rules if result.categorical_rules else {"Female (0)": 0, "Male (1)": 0}
    dec_f = cat_rules.get("Female (0)", 0)
    dec_m = cat_rules.get("Male (1)", 0)
    
    ax2.set_xticklabels(
        [
            rf"Female" + "\n" + rf"$\Lambda={lambdas[0]:.3f}$" + "\n" + rf"Regra $\hat{{Y}}={dec_f}$",
            rf"Male" + "\n" + rf"$\Lambda={lambdas[1]:.3f}$" + "\n" + rf"Regra $\hat{{Y}}={dec_m}$",
        ],
        fontsize=10,
    )
    ax2.set_ylim(0, 0.65)
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True, linestyle=":", alpha=0.6, axis="y")

    for rect in rects_post:
        h = rect.get_height()
        ax2.annotate(
            f"{h:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    output_path = output_dir / "univariate_gender.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def generate_all_univariate_plots(
    results: Dict[str, UnivariateFeatureResult],
    df_train: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Path]:
    """Gera e salva todas as figuras univariadas em output/figures/."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}
    paths["tenure"] = plot_univariate_continuous(results["tenure"], df_train, figures_dir)
    paths["MonthlyCharges"] = plot_univariate_continuous(results["MonthlyCharges"], df_train, figures_dir)
    paths["gender"] = plot_univariate_gender(results["gender"], figures_dir)

    return paths


def plot_confusion_matrix(
    metrics: Any,
    output_dir: Path,
) -> Path:
    """Gera visualização acadêmica da Matriz de Confusão e resumo das quatro métricas."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))

    arr = np.array(metrics.confusion_matrix.matrix, dtype=int)
    tn, fp = arr[0, 0], arr[0, 1]
    fn, tp = arr[1, 0], arr[1, 1]
    total = int(np.sum(arr))

    # Heatmap
    cax = ax.imshow(arr, cmap="Blues", interpolation="nearest", aspect="auto")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    # Rótulos das classes
    class_labels = ["Não-Churn (0)", "Churn (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_labels, fontsize=11)
    ax.set_yticklabels(class_labels, fontsize=11)

    ax.set_xlabel("Classe Predita", fontsize=12, labelpad=10)
    ax.set_ylabel("Classe Real", fontsize=12, labelpad=10)
    ax.set_title(
        f"Matriz de Confusão (N = {total})",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    sum_0 = tn + fp
    sum_1 = fn + tp

    pct_tn = f"{tn/sum_0*100:.1f}% classe" if sum_0 > 0 else "N/A"
    pct_fp = f"{fp/sum_0*100:.1f}% classe" if sum_0 > 0 else "N/A"
    pct_fn = f"{fn/sum_1*100:.1f}% classe" if sum_1 > 0 else "N/A"
    pct_tp = f"{tp/sum_1*100:.1f}% classe" if sum_1 > 0 else "N/A"

    # Textos reduzidos para caber nas células
    cell_info = [
        [
            f"VN\n{tn}\n({tn/total*100:.1f}% total\n{pct_tn})",
            f"FP\n{fp}\n({fp/total*100:.1f}% total\n{pct_fp})",
        ],
        [
            f"FN\n{fn}\n({fn/total*100:.1f}% total\n{pct_fn})",
            f"VP\n{tp}\n({tp/total*100:.1f}% total\n{pct_tp})",
        ],
    ]

    thresh = arr.max() / 2.0
    for i in range(2):
        for j in range(2):
            color = "white" if arr[i, j] > thresh else "#111111"
            ax.text(
                j,
                i,
                cell_info[i][j],
                ha="center",
                va="center",
                color=color,
                fontsize=9.5,
                fontweight="normal",
            )
            
    prec_str = f"Prec: {metrics.precision:.2f}" + (" (Indef)" if metrics.precision_undefined_reason else "")
    rec_str = f"Rec: {metrics.recall:.2f}" + (" (Indef)" if metrics.recall_undefined_reason else "")
    f1_str = f"F1: {metrics.f1_score:.2f}" + (" (Indef)" if metrics.f1_undefined_reason else "")

    metrics_text = (
        f"Acurácia: {metrics.accuracy:.2f} | "
        f"{prec_str} | "
        f"{rec_str} | "
        f"{f1_str}"
    )

    fig.text(
        0.5,
        0.02,
        metrics_text,
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da"),
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    output_path = figures_dir / "confusion_matrix.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
