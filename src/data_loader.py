"""Módulo de carga, validação e particionamento dos dados.

Este módulo implementa o carregamento do dataset Telco Customer Churn,
a validação rigorosa dos campos obrigatórios (sem descarte de registros
por colunas não utilizadas como TotalCharges), a codificação das variáveis
e o particionamento estratificado entre treino e teste.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Mapeamentos oficiais do projeto (Seção 4.1 da Análise 2.0 / GUIA.md)
FEATURE_COLS: List[str] = ["tenure", "MonthlyCharges", "gender"]
TARGET_COL: str = "Churn"
ID_COL: str = "customerID"
REQUIRED_COLS: List[str] = [ID_COL] + FEATURE_COLS + [TARGET_COL]

GENDER_MAP: Dict[str, int] = {
    "Female": 0,
    "Male": 1,
}

CHURN_MAP: Dict[str, int] = {
    "No": 0,
    "Yes": 1,
}

# Hash SHA-256 de referência do arquivo de dados auditado
EXPECTED_SHA256: str = "88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a"
EXPECTED_TOTAL_ROWS: int = 7043
EXPECTED_NO_COUNT: int = 5174
EXPECTED_YES_COUNT: int = 1869


class DataValidationError(ValueError):
    """Exceção levantada quando os dados violam requisitos de integridade."""
    pass


@dataclass(frozen=True)
class SplitMetadata:
    """Metadados descritivos da partição de dados."""

    csv_path: str
    csv_sha256: str
    random_state: int
    test_size: float
    stratified: bool
    total_samples: int
    train_samples: int
    test_samples: int
    total_class_counts: Dict[str, int]
    total_class_proportions: Dict[str, float]
    train_class_counts: Dict[str, int]
    train_class_proportions: Dict[str, float]
    test_class_counts: Dict[str, int]
    test_class_proportions: Dict[str, float]
    train_customer_ids: List[str]
    test_customer_ids: List[str]

    def to_dict(self, include_ids: bool = True) -> Dict[str, Any]:
        """Converte metadados para dicionário serializável."""
        data = asdict(self)
        if not include_ids:
            data.pop("train_customer_ids", None)
            data.pop("test_customer_ids", None)
        return data


@dataclass
class DataSplit:
    """Estrutura contendo as partições de treino e teste e seus metadados."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_ids: pd.Series
    test_ids: pd.Series
    metadata: SplitMetadata


def compute_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Calcula o hash SHA-256 de um arquivo em modo binário.

    Args:
        file_path: Caminho para o arquivo.
        chunk_size: Tamanho do bloco para leitura em bytes.

    Returns:
        String hexadecimal com o hash SHA-256.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def resolve_dataset_path(provided_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve o caminho do dataset local de forma configurável e portável.

    Tenta resolver na seguinte ordem:
    1. Caminho fornecido explicitamente (se existir).
    2. Caminhos relativos padrão no repositório.

    Args:
        provided_path: Caminho opcional informado pelo usuário/CLI.

    Returns:
        Path absoluto para o arquivo CSV existente.

    Raises:
        FileNotFoundError: Se nenhuma localização válida for encontrada.
    """
    if provided_path:
        p = Path(provided_path).resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"Caminho especificado para o dataset não existe: {provided_path}")

    candidates: List[Path] = [
        Path("dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv").resolve(),
        Path("../dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv").resolve(),
        (Path(__file__).resolve().parent.parent / "dataset" / "WA_Fn-UseC_-Telco-Customer-Churn.csv").resolve(),
        Path("../../dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv").resolve(),
    ]

    for cand in candidates:
        if cand.is_file():
            return cand

    searched_paths = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Arquivo de dataset não localizado nas buscas automáticas:\n{searched_paths}\n"
        "Informe o caminho explicitamente via parâmetro --data-path."
    )


def load_raw_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Carrega o arquivo CSV sem transformações destrutivas.

    Args:
        file_path: Caminho do arquivo CSV.

    Returns:
        DataFrame carregado diretamente do CSV.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        DataValidationError: Se o CSV estiver vazio.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as err:
        raise DataValidationError(f"O arquivo CSV está vazio: {path}") from err

    if df.empty:
        raise DataValidationError(f"O arquivo CSV carregado não possui registros: {path}")

    return df


def validate_required_fields(df: pd.DataFrame) -> None:
    """Valida a integridade dos campos estritamente necessários ao modelo.

    Regras aplicadas:
    - Presença das colunas: customerID, tenure, MonthlyCharges, gender, Churn.
    - customerID deve ser único (sem duplicatas).
    - Ausência de valores nulos/vazios nos 5 campos necessários.
    - tenure deve ser numérico, finito e não-negativo.
    - MonthlyCharges deve ser numérico, finito e estritamente positivo.
    - gender deve conter apenas categorias mapeadas ('Female', 'Male').
    - Churn deve conter apenas categorias mapeadas ('No', 'Yes').
    - Nenhuma validação ou exclusão é feita sobre TotalCharges.

    Args:
        df: DataFrame bruto a ser validado.

    Raises:
        DataValidationError: Se qualquer regra de validação for violada.
    """
    # 1. Checagem de colunas obrigatórias
    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        raise DataValidationError(
            f"Colunas obrigatórias ausentes no dataset: {missing_cols}. "
            f"Esperadas: {REQUIRED_COLS}."
        )

    # 2. Unicidade de customerID
    duplicated_ids = df[ID_COL].duplicated()
    if duplicated_ids.any():
        n_dups = int(duplicated_ids.sum())
        sample_dups = df[ID_COL][duplicated_ids].head(5).tolist()
        raise DataValidationError(
            f"Foram encontrados {n_dups} IDs de cliente duplicados em '{ID_COL}'. Exemplos: {sample_dups}."
        )

    # 3. Nulos e strings em branco nos campos obrigatórios
    for col in REQUIRED_COLS:
        null_mask = df[col].isna()
        if null_mask.any():
            n_nulls = int(null_mask.sum())
            raise DataValidationError(
                f"Coluna obrigatória '{col}' contém {n_nulls} valores nulos (NaN/None)."
            )

        if df[col].dtype == object or isinstance(df[col].iloc[0], str):
            blank_mask = df[col].astype(str).str.strip() == ""
            if blank_mask.any():
                n_blanks = int(blank_mask.sum())
                raise DataValidationError(
                    f"Coluna obrigatória '{col}' contém {n_blanks} valores em branco/espaços vazios."
                )

    # 4. Validação de categorias permitidas para gender
    valid_genders = set(GENDER_MAP.keys())
    observed_genders = set(df["gender"].unique())
    invalid_genders = observed_genders - valid_genders
    if invalid_genders:
        raise DataValidationError(
            f"Valores inválidos encontrados na coluna 'gender': {sorted(invalid_genders)}. "
            f"Categorias válidas: {sorted(valid_genders)}."
        )

    # 5. Validação de categorias permitidas para Churn
    valid_churns = set(CHURN_MAP.keys())
    observed_churns = set(df[TARGET_COL].unique())
    invalid_churns = observed_churns - valid_churns
    if invalid_churns:
        raise DataValidationError(
            f"Valores inválidos encontrados na coluna '{TARGET_COL}': {sorted(invalid_churns)}. "
            f"Categorias válidas: {sorted(valid_churns)}."
        )

    # 6. Validação numérica de tenure
    try:
        tenure_numeric = pd.to_numeric(df["tenure"], errors="raise")
    except (ValueError, TypeError) as err:
        raise DataValidationError("Coluna 'tenure' não pôde ser convertida para valores numéricos.") from err

    if not np.all(np.isfinite(tenure_numeric)):
        raise DataValidationError("Coluna 'tenure' contém valores infinitos ou não-finitos.")

    if (tenure_numeric < 0).any():
        n_neg = int((tenure_numeric < 0).sum())
        raise DataValidationError(f"Coluna 'tenure' contém {n_neg} valores negativos.")

    # Rejeição estrita de valores fracionários (tenure representa meses inteiros)
    fractional_mask = tenure_numeric != np.floor(tenure_numeric)
    if fractional_mask.any():
        n_frac = int(fractional_mask.sum())
        sample_frac = tenure_numeric[fractional_mask].head(5).tolist()
        raise DataValidationError(
            f"Coluna 'tenure' contém {n_frac} valores fracionários/não-inteiros: {sample_frac}. "
            "A variável tenure representa meses completos e deve conter apenas valores inteiros."
        )

    # 7. Validação numérica de MonthlyCharges
    try:
        monthly_numeric = pd.to_numeric(df["MonthlyCharges"], errors="raise")
    except (ValueError, TypeError) as err:
        raise DataValidationError("Coluna 'MonthlyCharges' não pôde ser convertida para valores numéricos.") from err

    if not np.all(np.isfinite(monthly_numeric)):
        raise DataValidationError("Coluna 'MonthlyCharges' contém valores infinitos ou não-finitos.")

    if (monthly_numeric <= 0).any():
        n_non_pos = int((monthly_numeric <= 0).sum())
        raise DataValidationError(f"Coluna 'MonthlyCharges' contém {n_non_pos} valores menores ou iguais a zero.")


def encode_features_and_target(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica as codificações canônicas às variáveis do modelo.

    Mapeamentos:
    - gender: Female -> 0, Male -> 1
    - Churn: No -> 0, Yes -> 1
    - tenure e MonthlyCharges são convertidos explicitamente para float/int.

    Retorna uma cópia do DataFrame com apenas as colunas estritamente
    necessárias: [customerID, tenure, MonthlyCharges, gender, Churn].

    Args:
        df: DataFrame validado contendo as colunas brutas.

    Returns:
        DataFrame limpo e codificado.
    """
    processed = pd.DataFrame(index=df.index)
    processed[ID_COL] = df[ID_COL].astype(str)
    processed["tenure"] = pd.to_numeric(df["tenure"], errors="raise").astype(int)
    processed["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="raise").astype(float)
    processed["gender"] = df["gender"].map(GENDER_MAP).astype(int)
    processed[TARGET_COL] = df[TARGET_COL].map(CHURN_MAP).astype(int)

    return processed


def create_stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    csv_path: str = "",
    csv_sha256: str = "",
) -> DataSplit:
    """Realiza a divisão estratificada 80/20 do dataset com base no alvo Churn.

    Garante isolamento estrito entre treino e teste, calculando proporções
    e coletando listas de IDs de cada partição para auditabilidade.

    Args:
        df: DataFrame codificado contendo as colunas [customerID, tenure, MonthlyCharges, gender, Churn].
        test_size: Proporção para partição de teste (padrão 0.2 para 80/20).
        random_state: Semente pseudoaleatória para reprodutibilidade.
        csv_path: Caminho do CSV de origem para rastreabilidade.
        csv_sha256: Hash SHA-256 do CSV de origem.

    Returns:
        DataSplit com DataFrames, matrizes de features, séries de alvo,
        listas de IDs e metadados completos.

    Raises:
        DataValidationError: Se a divisão falhar em critérios de particionamento.
    """
    total_samples = len(df)
    y_all = df[TARGET_COL]

    train_indices, test_indices = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=y_all,
        shuffle=True,
    )

    train_df = df.loc[train_indices].copy().reset_index(drop=True)
    test_df = df.loc[test_indices].copy().reset_index(drop=True)

    # Validação de invariantes da partição
    train_ids_set = set(train_df[ID_COL])
    test_ids_set = set(test_df[ID_COL])
    overlap = train_ids_set.intersection(test_ids_set)
    if overlap:
        raise DataValidationError(
            f"Violação de partição: {len(overlap)} IDs compartilhados entre treino e teste."
        )

    if len(train_df) + len(test_df) != total_samples:
        raise DataValidationError(
            f"Inconsistência na cobertura: soma das partições ({len(train_df)} + {len(test_df)}) "
            f"difere do total ({total_samples})."
        )

    # Features: APENAS tenure, MonthlyCharges e gender (customerID NUNCA entra como feature)
    X_train = train_df[FEATURE_COLS].copy()
    y_train = train_df[TARGET_COL].copy()

    X_test = test_df[FEATURE_COLS].copy()
    y_test = test_df[TARGET_COL].copy()

    # Contagens e proporções descritivas
    total_counts = {str(k): int(v) for k, v in y_all.value_counts().sort_index().items()}
    total_props = {str(k): float(v / total_samples) for k, v in total_counts.items()}

    train_counts = {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()}
    train_props = {str(k): float(v / len(train_df)) for k, v in train_counts.items()}

    test_counts = {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()}
    test_props = {str(k): float(v / len(test_df)) for k, v in test_counts.items()}

    metadata = SplitMetadata(
        csv_path=str(csv_path),
        csv_sha256=csv_sha256,
        random_state=random_state,
        test_size=test_size,
        stratified=True,
        total_samples=total_samples,
        train_samples=len(train_df),
        test_samples=len(test_df),
        total_class_counts=total_counts,
        total_class_proportions=total_props,
        train_class_counts=train_counts,
        train_class_proportions=train_props,
        test_class_counts=test_counts,
        test_class_proportions=test_props,
        train_customer_ids=train_df[ID_COL].tolist(),
        test_customer_ids=test_df[ID_COL].tolist(),
    )

    return DataSplit(
        train_df=train_df,
        test_df=test_df,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_ids=train_df[ID_COL],
        test_ids=test_df[ID_COL],
        metadata=metadata,
    )


def load_and_prepare_data(
    data_path: Optional[Union[str, Path]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DataSplit:
    """Função de alto nível que orquestra a carga, validação e divisão dos dados.

    Args:
        data_path: Caminho opcional do arquivo CSV. Se None, resolve automaticamente.
        test_size: Proporção da partição de teste (padrão 0.2).
        random_state: Semente pseudoaleatória (padrão 42).

    Returns:
        DataSplit pronto para os experimentos.
    """
    resolved_path = resolve_dataset_path(data_path)
    sha256_hash = compute_sha256(resolved_path)

    raw_df = load_raw_data(resolved_path)
    validate_required_fields(raw_df)
    encoded_df = encode_features_and_target(raw_df)

    split = create_stratified_split(
        df=encoded_df,
        test_size=test_size,
        random_state=random_state,
        csv_path=str(resolved_path),
        csv_sha256=sha256_hash,
    )

    return split
