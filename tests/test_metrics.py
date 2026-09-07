"""Testes unitários rigorosos para o módulo src/metrics.py.

Verifica:
- Cálculo analítico manual de matriz de confusão e métricas em vetores sintéticos independentes.
- Paridade contra as funções de referência do scikit-learn (confusion_matrix, accuracy_score,
  precision_score, recall_score, f1_score).
- Tratamento explícito de indefinição matemática quando o denominador é zero (divisão por zero).
- Validações de dimensão, vetores vazios e rótulos fora do domínio binário {0, 1}.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn import metrics as sk_metrics

from src.metrics import (
    ClassificationMetrics,
    ConfusionMatrix,
    MetricsError,
    compute_confusion_matrix,
    compute_metrics,
)


class TestConfusionMatrixAndMetricsCalculations:
    """Testes com vetores sintéticos calculados analiticamente à mão."""

    def test_synthetic_hand_calculated_case(self) -> None:
        """Verifica caso analítico com contagens exatas:

        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        y_pred = [0, 0, 1, 1, 0, 1, 1, 1]

        VN = 2, FP = 2, FN = 1, VP = 3
        N = 8
        Acurácia = 5 / 8 = 0.625
        Precisão = 3 / (3 + 2) = 0.60
        Recall   = 3 / (3 + 1) = 0.75
        F1       = 2*3 / (2*3 + 2 + 1) = 6 / 9 = 2/3
        """
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 1])

        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm.tn == 2
        assert cm.fp == 2
        assert cm.fn == 1
        assert cm.tp == 3
        assert cm.total == 8
        assert cm.matrix == [[2, 2], [1, 3]]

        res = compute_metrics(y_true, y_pred)
        assert res.accuracy == pytest.approx(5.0 / 8.0, abs=1e-12)
        assert res.precision == pytest.approx(3.0 / 5.0, abs=1e-12)
        assert res.recall == pytest.approx(3.0 / 4.0, abs=1e-12)
        assert res.f1_score == pytest.approx(6.0 / 9.0, abs=1e-12)
        assert res.precision_undefined_reason is None
        assert res.recall_undefined_reason is None
        assert res.f1_undefined_reason is None

    def test_parity_against_scikit_learn_metrics(self) -> None:
        """Compara a matriz de confusão e todas as 4 métricas contra scikit-learn."""
        np.random.seed(42)
        y_true = np.random.binomial(n=1, p=0.3, size=200)
        y_pred = np.random.binomial(n=1, p=0.4, size=200)

        custom_cm = compute_confusion_matrix(y_true, y_pred)
        sklearn_cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])

        # Matriz canônica: [[VN, FP], [FN, VP]]
        np.testing.assert_array_equal(custom_cm.to_numpy(), sklearn_cm)

        custom_metrics = compute_metrics(y_true, y_pred)
        acc_sk = sk_metrics.accuracy_score(y_true, y_pred)
        prec_sk = sk_metrics.precision_score(y_true, y_pred, zero_division=0)
        rec_sk = sk_metrics.recall_score(y_true, y_pred, zero_division=0)
        f1_sk = sk_metrics.f1_score(y_true, y_pred, zero_division=0)

        assert custom_metrics.accuracy == pytest.approx(acc_sk, abs=1e-12)
        assert custom_metrics.precision == pytest.approx(prec_sk, abs=1e-12)
        assert custom_metrics.recall == pytest.approx(rec_sk, abs=1e-12)
        assert custom_metrics.f1_score == pytest.approx(f1_sk, abs=1e-12)

    def test_zero_division_handling_all_negative_predictions(self) -> None:
        """Garante tratamento explícito quando VP + FP = 0 (nenhum positivo predito)."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 0])  # Todos preditos como 0

        res = compute_metrics(y_true, y_pred)

        assert res.confusion_matrix.tp == 0
        assert res.confusion_matrix.fp == 0
        assert res.confusion_matrix.fn == 2
        assert res.confusion_matrix.tn == 2

        # Precisão deve ser 0.0 com anotação explícita
        assert res.precision == 0.0
        assert res.precision_undefined_reason is not None
        assert "Denominador zero" in res.precision_undefined_reason

        # Recall deve ser 0 / (0 + 2) = 0.0
        assert res.recall == 0.0
        assert res.recall_undefined_reason is None

        # F1 deve ser 0.0 com anotação
        assert res.f1_score == 0.0

    def test_zero_division_handling_no_positive_instances(self) -> None:
        """Garante tratamento quando não há instâncias reais da classe positiva."""
        y_true = np.array([0, 0, 0, 0])  # Nenhum positivo real
        y_pred = np.array([0, 0, 0, 0])

        res = compute_metrics(y_true, y_pred)
        assert res.recall == 0.0
        assert res.recall_undefined_reason is not None
        assert "Denominador zero" in res.recall_undefined_reason


class TestMetricsInputValidation:
    """Testes para tratamento de erros e integridade das entradas."""

    def test_mismatched_length_raises_error(self) -> None:
        """Vetores de tamanhos diferentes devem levantar MetricsError."""
        with pytest.raises(MetricsError, match="Comprimentos divergentes"):
            compute_confusion_matrix([0, 1], [0])

    def test_empty_vectors_raise_error(self) -> None:
        """Vetores vazios devem levantar MetricsError."""
        with pytest.raises(MetricsError, match="vazios"):
            compute_confusion_matrix([], [])

    def test_invalid_labels_raise_error(self) -> None:
        """Rótulos fora de {0, 1} devem levantar MetricsError."""
        with pytest.raises(MetricsError, match="Rótulos inválidos"):
            compute_confusion_matrix([0, 2], [0, 1])
        with pytest.raises(MetricsError, match="Rótulos inválidos"):
            compute_confusion_matrix([0, 1], [0, -1])

    def test_fractional_and_multidimensional_arrays_raise_error(self) -> None:
        """L4-01: Arrays multidimensionais, valores fracionários ou NaN devem ser rejeitados rigorosamente."""
        with pytest.raises(MetricsError, match="fracionários"):
            compute_confusion_matrix([0.9, 1.9], [0, 1])
            
        with pytest.raises(MetricsError, match="não finitos"):
            compute_confusion_matrix([0, float('nan')], [0, 1])
            
        with pytest.raises(MetricsError, match="unidimensional"):
            compute_confusion_matrix(np.array([[0], [1], [0]]), np.array([[0, 1, 0]]))
