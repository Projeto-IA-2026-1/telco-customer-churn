"""Testes unitários e de integração para carga, validação e particionamento (Lote 1).

Cobre:
- Integridade do hash SHA-256 do dataset.
- Mapeamentos canônicos (gender: Female=0, Male=1; Churn: No=0, Yes=1).
- Diagnóstico claro e rejeição de dados inválidos (sem descarte silencioso).
- Preservação estrita das linhas com espaços em TotalCharges (sem imputação/exclusão).
- Ausência de sobreposição (disjunção) e cobertura completa entre as partições.
- Reprodutibilidade estrita da divisão com a mesma semente (random_state=42).
- Isolamento estrito de customerID fora da matriz de features.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    CHURN_MAP,
    EXPECTED_NO_COUNT,
    EXPECTED_SHA256,
    EXPECTED_TOTAL_ROWS,
    EXPECTED_YES_COUNT,
    FEATURE_COLS,
    GENDER_MAP,
    ID_COL,
    REQUIRED_COLS,
    TARGET_COL,
    DataValidationError,
    compute_sha256,
    create_stratified_split,
    encode_features_and_target,
    load_and_prepare_data,
    load_raw_data,
    resolve_dataset_path,
    validate_required_fields,
)


@pytest.fixture
def dataset_path() -> Path:
    """Fixture que localiza o dataset do projeto."""
    return resolve_dataset_path()


@pytest.fixture
def raw_df(dataset_path: Path) -> pd.DataFrame:
    """Fixture que carrega o DataFrame bruto."""
    return load_raw_data(dataset_path)


@pytest.fixture
def sample_valid_df() -> pd.DataFrame:
    """Fixture com pequeno conjunto de dados válidos para testes controlados."""
    return pd.DataFrame(
        {
            "customerID": ["C01", "C02", "C03", "C04", "C05"],
            "tenure": [1, 12, 24, 48, 70],
            "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 89.10],
            "gender": ["Female", "Male", "Female", "Male", "Female"],
            "Churn": ["No", "No", "Yes", "No", "Yes"],
            "TotalCharges": ["29.85", " ", "1292.4", "2030.4", "6237.0"],  # Espaço intencional
        }
    )


# -----------------------------------------------------------------------------
# 1. Testes de Integridade e Carga do CSV Real
# -----------------------------------------------------------------------------


def test_sha256_integrity(dataset_path: Path) -> None:
    """Verifica se o hash SHA-256 do arquivo CSV corresponde exatamente ao auditado."""
    actual_hash = compute_sha256(dataset_path)
    assert actual_hash == EXPECTED_SHA256, (
        f"Hash SHA-256 divergente: esperado {EXPECTED_SHA256}, obtido {actual_hash}."
    )


def test_load_and_validate_full_dataset(raw_df: pd.DataFrame) -> None:
    """Verifica se o dataset completo possui os 7.043 registros e contagens esperadas."""
    validate_required_fields(raw_df)

    assert len(raw_df) == EXPECTED_TOTAL_ROWS
    churn_counts = raw_df[TARGET_COL].value_counts()
    assert churn_counts["No"] == EXPECTED_NO_COUNT
    assert churn_counts["Yes"] == EXPECTED_YES_COUNT

    gender_counts = raw_df["gender"].value_counts()
    assert gender_counts["Female"] == 3488
    assert gender_counts["Male"] == 3555


def test_total_charges_preserved_without_imputation_or_dropping(raw_df: pd.DataFrame) -> None:
    """Verifica que as 11 linhas com espaços em TotalCharges são preservadas no dataset."""
    blank_total_charges = raw_df["TotalCharges"].astype(str).str.strip() == ""
    assert blank_total_charges.sum() == 11, "Deveria haver exatamente 11 linhas com TotalCharges em branco."

    # Processamento completo não deve descartar essas 11 linhas
    split = load_and_prepare_data()
    all_split_ids = set(split.train_ids).union(set(split.test_ids))
    blank_ids = set(raw_df.loc[blank_total_charges, ID_COL])

    assert blank_ids.issubset(all_split_ids), (
        "Registros com TotalCharges em branco foram indevidamente excluídos!"
    )
    assert split.metadata.total_samples == EXPECTED_TOTAL_ROWS


# -----------------------------------------------------------------------------
# 2. Testes de Mapeamentos Canônicos
# -----------------------------------------------------------------------------


def test_gender_mapping() -> None:
    """Verifica o mapeamento oficial de gender: Female=0, Male=1."""
    assert GENDER_MAP["Female"] == 0
    assert GENDER_MAP["Male"] == 1


def test_churn_mapping() -> None:
    """Verifica o mapeamento oficial de Churn: No=0, Yes=1."""
    assert CHURN_MAP["No"] == 0
    assert CHURN_MAP["Yes"] == 1


def test_encoding_transformation(sample_valid_df: pd.DataFrame) -> None:
    """Verifica que encode_features_and_target aplica os mapeamentos numéricos corretos."""
    encoded = encode_features_and_target(sample_valid_df)

    assert set(encoded.columns) == {ID_COL, "tenure", "MonthlyCharges", "gender", TARGET_COL}
    assert list(encoded["gender"]) == [0, 1, 0, 1, 0]
    assert list(encoded[TARGET_COL]) == [0, 0, 1, 0, 1]
    assert np.issubdtype(encoded["tenure"].dtype, np.integer)
    assert np.issubdtype(encoded["MonthlyCharges"].dtype, np.floating)


# -----------------------------------------------------------------------------
# 3. Testes de Diagnóstico e Rejeição de Dados Inválidos (Sem Descarte Silencioso)
# -----------------------------------------------------------------------------


def test_missing_required_column_rejected(sample_valid_df: pd.DataFrame) -> None:
    """Verifica se ausência de coluna obrigatória gera DataValidationError claro."""
    bad_df = sample_valid_df.drop(columns=["MonthlyCharges"])
    with pytest.raises(DataValidationError, match="Colunas obrigatórias ausentes"):
        validate_required_fields(bad_df)


def test_duplicate_customer_id_rejected(sample_valid_df: pd.DataFrame) -> None:
    """Verifica se duplicidade em customerID gera diagnóstico claro."""
    bad_df = sample_valid_df.copy()
    bad_df.loc[1, "customerID"] = bad_df.loc[0, "customerID"]
    with pytest.raises(DataValidationError, match="IDs de cliente duplicados"):
        validate_required_fields(bad_df)


def test_null_value_in_required_field_rejected(sample_valid_df: pd.DataFrame) -> None:
    """Verifica se valor nulo em campo obrigatório é rejeitado."""
    bad_df = sample_valid_df.copy()
    bad_df.loc[0, "tenure"] = np.nan
    with pytest.raises(DataValidationError, match="valores nulos"):
        validate_required_fields(bad_df)


def test_blank_string_in_required_field_rejected(sample_valid_df: pd.DataFrame) -> None:
    """Verifica se string vazia/espaço em campo obrigatório é rejeitada."""
    bad_df = sample_valid_df.copy()
    bad_df.loc[0, "gender"] = "   "
    with pytest.raises(DataValidationError, match="valores em branco"):
        validate_required_fields(bad_df)


def test_unrecognized_gender_rejected(sample_valid_df: pd.DataFrame) -> None:
    """Verifica se categoria não mapeada em gender é rejeitada com diagnóstico claro."""
    bad_df = sample_valid_df.copy()
    bad_df.loc[0, "gender"] = "Non-Binary"
    with pytest.raises(DataValidationError, match="Valores inválidos encontrados na coluna 'gender'"):
        validate_required_fields(bad_df)


def test_unrecognized_churn_rejected(sample_valid_df: pd.DataFrame) -> None:
    """Verifica se categoria não mapeada em Churn é rejeitada."""
    bad_df = sample_valid_df.copy()
    bad_df.loc[0, TARGET_COL] = "Maybe"
    with pytest.raises(DataValidationError, match="Valores inválidos encontrados na coluna 'Churn'"):
        validate_required_fields(bad_df)


def test_negative_tenure_rejected(sample_valid_df: pd.DataFrame) -> None:
    """Verifica se valores negativos em tenure são rejeitados."""
    bad_df = sample_valid_df.copy()
    bad_df.loc[0, "tenure"] = -5
    with pytest.raises(DataValidationError, match="valores negativos"):
        validate_required_fields(bad_df)


def test_fractional_tenure_rejected(sample_valid_df: pd.DataFrame) -> None:
    """Verifica se valores fracionários/não-inteiros em tenure são rejeitados (R1)."""
    bad_df = sample_valid_df.copy()
    bad_df["tenure"] = bad_df["tenure"].astype(float)
    bad_df.loc[0, "tenure"] = 1.9
    with pytest.raises(DataValidationError, match="valores fracionários/não-inteiros"):
        validate_required_fields(bad_df)


def test_integer_valued_floats_accepted_in_tenure(sample_valid_df: pd.DataFrame) -> None:
    """Verifica que valores inteiros representados como float (ex: 0.0, 1.0, 72.0) são aceitos (R1)."""
    valid_float_df = sample_valid_df.copy()
    valid_float_df["tenure"] = [0.0, 1.0, 12.0, 24.0, 72.0]
    validate_required_fields(valid_float_df)
    encoded = encode_features_and_target(valid_float_df)
    assert list(encoded["tenure"]) == [0, 1, 12, 24, 72]
    assert np.issubdtype(encoded["tenure"].dtype, np.integer)


def test_non_positive_monthly_charges_rejected(sample_valid_df: pd.DataFrame) -> None:
    """Verifica se valor zero ou negativo em MonthlyCharges é rejeitado."""
    bad_df = sample_valid_df.copy()
    bad_df.loc[0, "MonthlyCharges"] = 0.0
    with pytest.raises(DataValidationError, match="menores ou iguais a zero"):
        validate_required_fields(bad_df)


def test_non_finite_numeric_rejected(sample_valid_df: pd.DataFrame) -> None:
    """Verifica se infinito numérico é rejeitado."""
    bad_df = sample_valid_df.copy()
    bad_df.loc[0, "MonthlyCharges"] = np.inf
    with pytest.raises(DataValidationError, match="valores infinitos"):
        validate_required_fields(bad_df)


# -----------------------------------------------------------------------------
# 4. Testes de Particionamento Estratificado e Isolamento
# -----------------------------------------------------------------------------


def test_partition_split_sizes_and_proportions() -> None:
    """Verifica se a divisão 80/20 atinge as contagens e proporções exatas esperadas."""
    split = load_and_prepare_data(test_size=0.2, random_state=42)
    m = split.metadata

    assert m.total_samples == 7043
    assert m.train_samples == 5634
    assert m.test_samples == 1409

    # Contagens por classe no treino: 4.139 (No) e 1.495 (Yes)
    assert m.train_class_counts["0"] == 4139
    assert m.train_class_counts["1"] == 1495

    # Contagens por classe no teste: 1.035 (No) e 374 (Yes)
    assert m.test_class_counts["0"] == 1035
    assert m.test_class_counts["1"] == 374

    # Proporções preservadas em ambas as partições (~73,46% e ~26,54%)
    assert pytest.approx(m.train_class_proportions["0"], abs=1e-4) == 0.7346
    assert pytest.approx(m.train_class_proportions["1"], abs=1e-4) == 0.2654
    assert pytest.approx(m.test_class_proportions["0"], abs=1e-4) == 0.7346
    assert pytest.approx(m.test_class_proportions["1"], abs=1e-4) == 0.2654


def test_partition_disjoint_and_full_coverage() -> None:
    """Verifica ausência de sobreposição e cobertura total dos registros."""
    split = load_and_prepare_data()

    train_ids = set(split.train_ids)
    test_ids = set(split.test_ids)

    # 1. Ausência de sobreposição
    overlap = train_ids.intersection(test_ids)
    assert len(overlap) == 0, f"Encontrados IDs sobrepostos entre treino e teste: {overlap}"

    # 2. Cobertura completa
    union_ids = train_ids.union(test_ids)
    assert len(union_ids) == EXPECTED_TOTAL_ROWS


def test_partition_reproducibility() -> None:
    """Verifica se a divisão é 100% reproduzível com a mesma semente (random_state=42)."""
    split1 = load_and_prepare_data(random_state=42)
    split2 = load_and_prepare_data(random_state=42)

    assert split1.metadata.train_customer_ids == split2.metadata.train_customer_ids
    assert split1.metadata.test_customer_ids == split2.metadata.test_customer_ids
    pd.testing.assert_frame_equal(split1.X_train, split2.X_train)
    pd.testing.assert_series_equal(split1.y_train, split2.y_train)
    pd.testing.assert_frame_equal(split1.X_test, split2.X_test)
    pd.testing.assert_series_equal(split1.y_test, split2.y_test)


def test_features_exclude_customer_id() -> None:
    """Verifica que customerID é estritamente excluído da matriz de características."""
    split = load_and_prepare_data()

    assert list(split.X_train.columns) == FEATURE_COLS
    assert list(split.X_test.columns) == FEATURE_COLS
    assert ID_COL not in split.X_train.columns
    assert ID_COL not in split.X_test.columns
    assert split.y_train.name == TARGET_COL
    assert split.y_test.name == TARGET_COL


def test_configurable_data_path(tmp_path: Path, sample_valid_df: pd.DataFrame) -> None:
    """Verifica que caminho configurável funciona e caminho inexistente levanta erro."""
    # Duplica para ter 10 amostras com IDs distintos, viabilizando estratificação com test_size=0.2
    df1 = sample_valid_df.copy()
    df2 = sample_valid_df.copy()
    df2["customerID"] = [f"C{i+6:02d}" for i in range(len(df2))]
    df10 = pd.concat([df1, df2], ignore_index=True)

    custom_csv = tmp_path / "custom_telco.csv"
    df10.to_csv(custom_csv, index=False)

    split = load_and_prepare_data(data_path=custom_csv)
    assert split.metadata.total_samples == 10
    assert split.metadata.csv_path == str(custom_csv.resolve())

    with pytest.raises(FileNotFoundError):
        resolve_dataset_path(tmp_path / "non_existent.csv")


def test_lote_1_scope_boundaries() -> None:
    """Confirma que nenhum modelo, estimador ou classificador foi implementado ou ajustado no Lote 1."""
    # Garante que data_loader e DataSplit não expõem estimadores ou modelos ajustados
    split = load_and_prepare_data()
    assert not hasattr(split, "model")
    assert not hasattr(split, "classifier")
    assert not hasattr(split, "priors")


def test_format_report_canonical_and_alternative() -> None:
    """Verifica que format_report reflete dinamicamente a configuração canônica e alternativas (R3)."""
    from run_experiments import format_report

    # Caso 1: Configuração canônica (80/20, seed 42)
    canonical_split = load_and_prepare_data(test_size=0.2, random_state=42)
    report_canonical = format_report(canonical_split)
    assert "configuração canônica reproduzível" in report_canonical
    assert "80/20" in report_canonical
    assert "semente 42" in report_canonical

    # Caso 2: Configuração alternativa (70/30, seed 7)
    alt_split = load_and_prepare_data(test_size=0.3, random_state=7)
    report_alt = format_report(alt_split)
    assert "configuração customizada/alternativa" in report_alt
    assert "70/30" in report_alt
    assert "semente 7" in report_alt
    assert "configuração canônica reproduzível" not in report_alt


