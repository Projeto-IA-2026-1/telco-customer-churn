"""Módulo de cálculo de métricas de avaliação e matriz de confusão.

Implementa:
- ConfusionMatrix: Estrutura canônica de matriz de confusão binária [[VN, FP], [FN, VP]],
  onde linhas representam a classe real (0 e 1) e colunas a classe predita (0 e 1).
- ClassificationMetrics: Cálculo rigoroso de Acurácia, Precisão, Revocação (Recall) e F1-Score.
- Tratamento explícito de divisão por zero com anotações matemáticas documentadas
  (ao invés de silêncio ou valores arbitrários).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class MetricsError(ValueError):
    """Exceção levantada para dados inválidos no cálculo de métricas."""
    pass


@dataclass(frozen=True)
class ConfusionMatrix:
    """Matriz de confusão binária canônica.

    Estrutura:
    [[VN, FP],
     [FN, VP]]
    Linhas: Classe Real (0 = Não-Churn, 1 = Churn)
    Colunas: Classe Predita (0 = Não-Churn, 1 = Churn)
    """

    tn: int  # Verdadeiros Negativos (VN): Real 0, Predito 0
    fp: int  # Falsos Positivos (FP): Real 0, Predito 1
    fn: int  # Falsos Negativos (FN): Real 1, Predito 0
    tp: int  # Verdadeiros Positivos (VP): Real 1, Predito 1
    total: int

    @property
    def matrix(self) -> List[List[int]]:
        """Retorna a matriz 2x2 em formato de lista aninhada [[VN, FP], [FN, VP]]."""
        return [
            [self.tn, self.fp],
            [self.fn, self.tp],
        ]

    def to_numpy(self) -> np.ndarray:
        """Retorna a matriz como array NumPy 2D de formato (2, 2)."""
        return np.array(self.matrix, dtype=int)

    def to_dict(self) -> Dict[str, Any]:
        """Serialização dos valores da matriz de confusão."""
        return {
            "true_negative_vn": self.tn,
            "false_positive_fp": self.fp,
            "false_negative_fn": self.fn,
            "true_positive_vp": self.tp,
            "total_samples": self.total,
            "matrix_2x2": self.matrix,
            "format_description": "Linhas: Classe Real [0, 1] x Colunas: Classe Predita [0, 1]",
        }


@dataclass(frozen=True)
class ClassificationMetrics:
    """Métricas de desempenho em classificação binária (Classe positiva = 1)."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: ConfusionMatrix
    precision_undefined_reason: Optional[str] = None
    recall_undefined_reason: Optional[str] = None
    f1_undefined_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialização das métricas calculadas."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "precision_undefined_reason": self.precision_undefined_reason,
            "recall_undefined_reason": self.recall_undefined_reason,
            "f1_undefined_reason": self.f1_undefined_reason,
            "confusion_matrix": self.confusion_matrix.to_dict(),
        }


def _validate_binary_labels(y: Union[List[Any], np.ndarray], name: str) -> np.ndarray:
    """Valida se o array contém estritamente valores binários {0, 1} em formato 1D."""
    y_raw = np.asarray(y)
    
    if y_raw.ndim != 1:
        if y_raw.ndim == 2 and y_raw.shape[1] == 1:
            y_raw = y_raw.ravel()
        else:
            raise MetricsError(f"O vetor {name} deve ser estritamente unidimensional.")
            
    if not np.issubdtype(y_raw.dtype, np.number):
        raise MetricsError(f"O vetor {name} deve conter apenas números.")
        
    if not np.all(np.isfinite(y_raw)):
        raise MetricsError(f"O vetor {name} contém valores não finitos (NaN ou Inf).")
        
    y_int = y_raw.astype(int)
    if not np.array_equal(y_raw, y_int):
        raise MetricsError(f"O vetor {name} possui valores fracionários que não são rótulos inteiros exatos.")
        
    valid_labels = {0, 1}
    if not set(np.unique(y_int)).issubset(valid_labels):
        raise MetricsError(f"Rótulos inválidos em {name}. Deve conter apenas {{0, 1}}, encontrado: {set(np.unique(y_int))}")
        
    return y_int


def compute_confusion_matrix(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
) -> ConfusionMatrix:
    """Calcula os quatro quadrantes da matriz de confusão binária.

    Args:
        y_true: Vetor com classes reais {0, 1}.
        y_pred: Vetor com classes preditas {0, 1}.

    Returns:
        Instância de ConfusionMatrix.

    Raises:
        MetricsError: Se comprimentos divergirem, estiverem vazios ou contiverem rótulos fora de {0, 1}.
    """
    y_t = _validate_binary_labels(y_true, "y_true")
    y_p = _validate_binary_labels(y_pred, "y_pred")

    if len(y_t) != len(y_p):
        raise MetricsError(f"Comprimentos divergentes: y_true={len(y_t)} vs y_pred={len(y_p)}.")

    if len(y_t) == 0:
        raise MetricsError("Vetores de entrada estão vazios.")

    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    total = len(y_t)

    assert tn + fp + fn + tp == total

    return ConfusionMatrix(tn=tn, fp=fp, fn=fn, tp=tp, total=total)


def compute_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
) -> ClassificationMetrics:
    """Calcula acurácia, precisão, recall e F1 a partir dos vetores de classe.

    Fórmulas:
    - Acurácia = (VP + VN) / N
    - Precisão = VP / (VP + FP)
    - Revocação = VP / (VP + FN)
    - F1 = 2*VP / (2*VP + FP + FN)

    Quando o denominador é zero, o valor retornado é 0.0 acompanhado de explicação explícita
    da indefinição matemática.
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp, total = cm.tn, cm.fp, cm.fn, cm.tp, cm.total

    # Acurácia
    accuracy = (tp + tn) / total

    # Precisão: VP / (VP + FP)
    prec_denom = tp + fp
    if prec_denom > 0:
        precision = tp / prec_denom
        prec_reason = None
    else:
        precision = 0.0
        prec_reason = "Denominador zero (VP + FP = 0): Nenhuma instância foi predita como Classe 1 (positivo)."

    # Recall: VP / (VP + FN)
    rec_denom = tp + fn
    if rec_denom > 0:
        recall = tp / rec_denom
        rec_reason = None
    else:
        recall = 0.0
        rec_reason = "Denominador zero (VP + FN = 0): Não há instâncias reais da Classe 1 (positivo) no conjunto."

    # F1-Score: 2*VP / (2*VP + FP + FN)
    f1_denom = 2 * tp + fp + fn
    if f1_denom > 0:
        f1_score = (2.0 * tp) / f1_denom
        f1_reason = None
    else:
        f1_score = 0.0
        f1_reason = "Denominador zero (2*VP + FP + FN = 0): Não há positivos reais nem preditos."

    return ClassificationMetrics(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1_score),
        confusion_matrix=cm,
        precision_undefined_reason=prec_reason,
        recall_undefined_reason=rec_reason,
        f1_undefined_reason=f1_reason,
    )
