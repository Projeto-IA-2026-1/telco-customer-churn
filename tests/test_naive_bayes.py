"""Testes unitários rigorosos e benchmarks de paridade para src/naive_bayes.py.

Verifica:
- Ajuste do classificador Naive Bayes Híbrido manual a partir do zero.
- Validações de entrada, casos não ajustados e tratamento de erros.
- Desempate canônico determinístico (empate exato -> 0).
- Adição única da log-prior por classe (sem dupla contagem).
- Paridade estrita (rtol=1e-9, atol=1e-12) contra decomposição do scikit-learn
  (GaussianNB sem regularização + BernoulliNB/CategoricalNB) em dados sintéticos e no treino.
- ZERO acesso ao conjunto de teste congelado.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import logsumexp
from sklearn.naive_bayes import BernoulliNB, GaussianNB

from src.data_loader import load_and_prepare_data
from src.naive_bayes import (
    ClassifierNotFittedError,
    HybridNaiveBayesClassifier,
    InvalidInputError,
)


class TestNaiveBayesInputValidationAndEdgeCases:
    """Testes de robustez, validação de entradas e políticas de borda."""

    def test_predict_before_fit_raises_error(self) -> None:
        """Tentativa de inferência sem ajuste prévio deve levantar ClassifierNotFittedError."""
        clf = HybridNaiveBayesClassifier()
        X = pd.DataFrame({"tenure": [10], "MonthlyCharges": [50.0], "gender": [1]})

        with pytest.raises(ClassifierNotFittedError):
            clf.predict(X)
        with pytest.raises(ClassifierNotFittedError):
            clf.predict_proba(X)
        with pytest.raises(ClassifierNotFittedError):
            clf.predict_joint_log_likelihood(X)
        with pytest.raises(ClassifierNotFittedError):
            clf.inspect_sample({"tenure": 10, "MonthlyCharges": 50.0, "gender": 1})

    def test_missing_required_column_raises_error(self) -> None:
        """Dataframe com colunas incompletas deve levantar InvalidInputError."""
        clf = HybridNaiveBayesClassifier()
        # Falta a coluna 'gender'
        X_bad = pd.DataFrame({"tenure": [10, 20], "MonthlyCharges": [50.0, 60.0]})
        y = np.array([0, 1])

        with pytest.raises(InvalidInputError, match="Colunas obrigatórias ausentes"):
            clf.fit(X_bad, y)

    def test_invalid_dimension_numpy_raises_error(self) -> None:
        """Array com número incorreto de colunas deve ser rejeitado."""
        clf = HybridNaiveBayesClassifier()
        X_bad = np.array([[10, 50], [20, 60]])  # apenas 2 colunas
        y = np.array([0, 1])

        with pytest.raises(InvalidInputError, match="dimensão \\(N, 3\\)"):
            clf.fit(X_bad, y)

    def test_invalid_gender_value_raises_error(self) -> None:
        """Valores fora do domínio binário {0, 1} em gender devem ser rejeitados."""
        clf = HybridNaiveBayesClassifier()
        X_bad = pd.DataFrame({
            "tenure": [10, 20],
            "MonthlyCharges": [50.0, 60.0],
            "gender": [0, 2],  # categoria inválida 2
        })
        y = np.array([0, 1])

        with pytest.raises(InvalidInputError, match="domínio \\{0, 1\\}"):
            clf.fit(X_bad, y)

    def test_non_finite_input_raises_error(self) -> None:
        """Valores NaN ou infinitos devem ser rejeitados na inferência."""
        clf = HybridNaiveBayesClassifier()
        X_train = pd.DataFrame({
            "tenure": [10, 20, 30, 40],
            "MonthlyCharges": [25.0, 50.0, 75.0, 100.0],
            "gender": [0, 1, 0, 1],
        })
        y_train = np.array([0, 0, 1, 1])
        clf.fit(X_train, y_train)

        X_nan = pd.DataFrame({
            "tenure": [10],
            "MonthlyCharges": [np.nan],
            "gender": [0],
        })
        with pytest.raises(InvalidInputError, match="infinitos ou NaN"):
            clf.predict(X_nan)

    def test_tie_breaking_policy_resolves_to_zero(self) -> None:
        """Verifica se empate exato em L_1 == L_0 decide determinística e canonicamente por 0."""
        # Criação sintética simétrica com variância > 0 onde L_0 e L_1 coincidem
        clf = HybridNaiveBayesClassifier()
        X_train = pd.DataFrame({
            "tenure": [10, 20, 10, 20],
            "MonthlyCharges": [40.0, 60.0, 40.0, 60.0],
            "gender": [0, 1, 0, 1],
        })
        # Mesma distribuição condicional e priors balanceadas
        y_train = np.array([0, 0, 1, 1])
        clf.fit(X_train, y_train)

        # Amostra onde L_0 == L_1 exatamente
        sample = pd.DataFrame({"tenure": [15], "MonthlyCharges": [50.0], "gender": [0]})
        log_joint = clf.predict_joint_log_likelihood(sample)

        # Ambas as classes devem ter o mesmo escore
        assert log_joint[0, 0] == pytest.approx(log_joint[0, 1], abs=1e-12)

        pred = clf.predict(sample)
        assert pred[0] == 0  # Desempate canônico favorece classe 0

    def test_impossible_scores(self) -> None:
        """L3-01: Garante que amostras com probabilidade nula sob ambas as classes levantem erro."""
        clf = HybridNaiveBayesClassifier()
        X_train = pd.DataFrame({
            "tenure": [10, 20, 15, 25], 
            "MonthlyCharges": [40.0, 60.0, 45.0, 55.0], 
            "gender": [0, 1, 0, 1]
        })
        y_train = np.array([0, 1, 0, 1])
        clf.fit(X_train, y_train)
        
        # Gera uma amostra que cause -inf no log_pdf (valor fora do escopo matemático, por ex: 1e200)
        sample = pd.DataFrame({"tenure": [1e200], "MonthlyCharges": [50.0], "gender": [0]})
        
        with pytest.raises(InvalidInputError, match="Instância impossível detectada"):
            clf.predict_proba(sample)
            
        with pytest.raises(InvalidInputError, match="Instância impossível detectada"):
            clf.predict(sample)
            
        with pytest.raises(InvalidInputError, match="Instância impossível detectada"):
            clf.inspect_sample(sample.iloc[0])
            
        # Unilateral -inf test (valid):
        # Substitute a model with a dummy that returns -inf for a specific class
        class DummyModel:
            def log_pmf(self, x):
                return -float('inf')
        
        clf.model_gender_0 = DummyModel()
        
        sample2 = pd.DataFrame({"tenure": [15], "MonthlyCharges": [45.0], "gender": [0]})
        probs = clf.predict_proba(sample2)
        assert probs[0, 0] == 0.0
        assert probs[0, 1] == 1.0
        
    def test_target_validation(self) -> None:
        """L3-02: Garante validação estrita do vetor alvo antes do ajuste numérico."""
        clf = HybridNaiveBayesClassifier()
        X_train = pd.DataFrame({
            "tenure": [10, 20, 15, 25], 
            "MonthlyCharges": [40.0, 60.0, 45.0, 55.0], 
            "gender": [0, 1, 0, 1]
        })
        
        with pytest.raises(InvalidInputError, match="binários exatos"):
            clf.fit(X_train, np.array([0.9, 1.9, 0.9, 1.9]))
            
        with pytest.raises(InvalidInputError, match="finitos"):
            clf.fit(X_train, np.array([0, float('nan'), 0, 1]))
            
        with pytest.raises(InvalidInputError, match="unidimensional"):
            clf.fit(X_train, np.array([[0, 1], [1, 0], [0, 0], [1, 1]]))


class TestScikitLearnParityAndDecomposition:
    """Testes de benchmark e paridade matemática contra referências do scikit-learn."""

    @pytest.fixture
    def synthetic_dataset(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Gera um conjunto de dados sintético bem condicionado para teste analítico."""
        np.random.seed(123)
        N = 100
        # Classe 0
        t0 = np.random.normal(loc=40.0, scale=15.0, size=70)
        m0 = np.random.normal(loc=55.0, scale=20.0, size=70)
        g0 = np.random.binomial(n=1, p=0.45, size=70)

        # Classe 1
        t1 = np.random.normal(loc=15.0, scale=10.0, size=30)
        m1 = np.random.normal(loc=80.0, scale=18.0, size=30)
        g1 = np.random.binomial(n=1, p=0.55, size=30)

        df = pd.DataFrame({
            "tenure": np.clip(np.concatenate([t0, t1]), 1, 72),
            "MonthlyCharges": np.clip(np.concatenate([m0, m1]), 18.0, 118.0),
            "gender": np.concatenate([g0, g1]),
        })
        y = np.array([0] * 70 + [1] * 30)
        return df, y

    def test_parity_against_sklearn_on_synthetic_data(
        self,
        synthetic_dataset: Tuple[pd.DataFrame, np.ndarray],
    ) -> None:
        """Compara componentes, log-verossimilhanças, escores e posteriores contra scikit-learn."""
        X_df, y = synthetic_dataset

        # 1. Ajuste do modelo manual
        clf_manual = HybridNaiveBayesClassifier()
        clf_manual.fit(X_df, y)

        # 2. Ajuste das referências scikit-learn
        X_cont = X_df[["tenure", "MonthlyCharges"]].values
        X_cat = X_df[["gender"]].values

        gnb = GaussianNB(var_smoothing=0.0)
        gnb.fit(X_cont, y)

        bnb = BernoulliNB(alpha=0.0, force_alpha=True)
        bnb.fit(X_cat, y)

        # Parâmetros contínuos: médias e variâncias (MLE, divisor N)
        # gnb.theta_ tem shape (2, 2) -> classes x features
        # gnb.var_ tem shape (2, 2) -> classes x features
        assert clf_manual.model_tenure_0.mu == pytest.approx(gnb.theta_[0, 0], abs=1e-12)
        assert clf_manual.model_tenure_0.variance == pytest.approx(gnb.var_[0, 0], abs=1e-12)
        assert clf_manual.model_tenure_1.mu == pytest.approx(gnb.theta_[1, 0], abs=1e-12)
        assert clf_manual.model_tenure_1.variance == pytest.approx(gnb.var_[1, 0], abs=1e-12)

        assert clf_manual.model_monthly_0.mu == pytest.approx(gnb.theta_[0, 1], abs=1e-12)
        assert clf_manual.model_monthly_0.variance == pytest.approx(gnb.var_[0, 1], abs=1e-12)
        assert clf_manual.model_monthly_1.mu == pytest.approx(gnb.theta_[1, 1], abs=1e-12)
        assert clf_manual.model_monthly_1.variance == pytest.approx(gnb.var_[1, 1], abs=1e-12)

        # Parâmetros Bernoulli: P(gender=1 | c)
        assert clf_manual.model_gender_0.p == pytest.approx(np.exp(bnb.feature_log_prob_[0, 0]), abs=1e-12)
        assert clf_manual.model_gender_1.p == pytest.approx(np.exp(bnb.feature_log_prob_[1, 0]), abs=1e-12)

        # Reconstrução dos escores conjuntos do scikit-learn:
        # log_joint = log_prior + ll_cont + ll_cat
        log_prior_sklearn = np.log(gnb.class_prior_)
        ll_cont = gnb._joint_log_likelihood(X_cont) - log_prior_sklearn
        ll_cat = bnb._joint_log_likelihood(X_cat) - log_prior_sklearn
        log_joint_sklearn = log_prior_sklearn + ll_cont + ll_cat

        # Inferência manual
        log_joint_manual = clf_manual.predict_joint_log_likelihood(X_df)

        # Comparação com tolerância rigorosa de precisão de ponto flutuante: rtol=1e-9, atol=1e-12
        np.testing.assert_allclose(log_joint_manual, log_joint_sklearn, rtol=1e-9, atol=1e-12)

        # Comparação das probabilidades a posteriori normalizadas
        proba_sklearn = np.exp(log_joint_sklearn - logsumexp(log_joint_sklearn, axis=1, keepdims=True))
        proba_manual = clf_manual.predict_proba(X_df)

        np.testing.assert_allclose(proba_manual, proba_sklearn, rtol=1e-9, atol=1e-12)

        # Comparação das predições
        pred_sklearn = np.where(log_joint_sklearn[:, 1] > log_joint_sklearn[:, 0], 1, 0)
        pred_manual = clf_manual.predict(X_df)

        np.testing.assert_array_equal(pred_manual, pred_sklearn)

    def test_parity_against_sklearn_on_canonical_training_set(self) -> None:
        """Verifica a paridade matemática rigorosa sobre o conjunto de treinamento canônico real (N=5.634)."""
        split = load_and_prepare_data(test_size=0.2, random_state=42)
        X_train = split.X_train
        y_train = split.y_train

        clf_manual = HybridNaiveBayesClassifier()
        clf_manual.fit(X_train, y_train)

        X_cont = X_train[["tenure", "MonthlyCharges"]].values
        X_cat = X_train[["gender"]].values

        gnb = GaussianNB(var_smoothing=0.0)
        gnb.fit(X_cont, y_train)

        bnb = BernoulliNB(alpha=0.0, force_alpha=True)
        bnb.fit(X_cat, y_train)

        log_prior_sklearn = np.log(gnb.class_prior_)
        ll_cont = gnb._joint_log_likelihood(X_cont) - log_prior_sklearn
        ll_cat = bnb._joint_log_likelihood(X_cat) - log_prior_sklearn
        log_joint_sklearn = log_prior_sklearn + ll_cont + ll_cat

        log_joint_manual = clf_manual.predict_joint_log_likelihood(X_train)
        np.testing.assert_allclose(log_joint_manual, log_joint_sklearn, rtol=1e-9, atol=1e-12)

        proba_sklearn = np.exp(log_joint_sklearn - logsumexp(log_joint_sklearn, axis=1, keepdims=True))
        proba_manual = clf_manual.predict_proba(X_train)
        np.testing.assert_allclose(proba_manual, proba_sklearn, rtol=1e-9, atol=1e-12)

        pred_sklearn = np.where(log_joint_sklearn[:, 1] > log_joint_sklearn[:, 0], 1, 0)
        pred_manual = clf_manual.predict(X_train)
        np.testing.assert_array_equal(pred_manual, pred_sklearn)

    def test_single_prior_addition_and_inspection(self) -> None:
        """Verifica que o prior é somado exatamente uma vez na inspeção detalhada de amostra."""
        split = load_and_prepare_data(test_size=0.2, random_state=42)
        clf = HybridNaiveBayesClassifier().fit(split.X_train, split.y_train)

        sample = split.X_train.iloc[0]
        details = clf.inspect_sample(sample)

        # Verifica consistência da soma L_c:
        # L_0 = log_pi_0 + ll_t_0 + ll_m_0 + ll_g_0
        expected_L0 = (
            details["priors"]["log_pi_0"]
            + details["log_likelihoods"]["tenure"]["class_0"]
            + details["log_likelihoods"]["MonthlyCharges"]["class_0"]
            + details["log_likelihoods"]["gender"]["class_0"]
        )
        assert details["log_joint_scores"]["L_0"] == pytest.approx(expected_L0, abs=1e-12)

        expected_L1 = (
            details["priors"]["log_pi_1"]
            + details["log_likelihoods"]["tenure"]["class_1"]
            + details["log_likelihoods"]["MonthlyCharges"]["class_1"]
            + details["log_likelihoods"]["gender"]["class_1"]
        )
        assert details["log_joint_scores"]["L_1"] == pytest.approx(expected_L1, abs=1e-12)

        # Verifica normalização das probabilidades a posteriori
        assert details["posteriors"]["P(Churn=0|x)"] + details["posteriors"]["P(Churn=1|x)"] == pytest.approx(1.0, abs=1e-12)
