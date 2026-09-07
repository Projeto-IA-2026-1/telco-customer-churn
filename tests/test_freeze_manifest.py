import json
import pytest
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from run_experiments import create_model_freeze_manifest, verify_and_save_freeze_manifest
from src.naive_bayes import HybridNaiveBayesClassifier

class DummyModel:
    def to_dict(self):
        return {"dummy": "params"}

def test_freeze_manifest_identities(tmp_path: Path):
    """Testa que a reprodução exata preserva e passa no congelamento."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    meta = MagicMock()
    meta.total_samples = 100
    meta.train_samples = 80
    meta.test_samples = 20
    meta.test_size = 0.2
    meta.random_state = 42
    meta.stratified = True
    meta.csv_sha256 = "abc"
    
    X_train = pd.DataFrame({"A": [1, 2]}, index=[0, 1])
    X_test = pd.DataFrame({"A": [3]}, index=[2])
    
    split = MagicMock()
    split.X_train = X_train
    split.X_test = X_test
    split.train_ids = pd.Series([0, 1])
    split.test_ids = pd.Series([2])
    split.metadata = meta
    
    clf = DummyModel()
    
    # 1. Cria o primeiro manifesto
    manifest1 = create_model_freeze_manifest(clf, split) # type: ignore
    verify_and_save_freeze_manifest(manifest1, output_dir)
    
    assert (output_dir / "model_freeze_manifest.json").exists()
    
    # 2. Reprodução exata deve passar
    manifest2 = create_model_freeze_manifest(clf, split) # type: ignore
    verify_and_save_freeze_manifest(manifest2, output_dir) # Não deve lançar erro
    
    # 3. Altera parâmetros
    class DummyModel2:
        def to_dict(self): return {"dummy": "params_changed"}
        
    clf2 = DummyModel2()
    manifest_param = create_model_freeze_manifest(clf2, split) # type: ignore
    with pytest.raises(RuntimeError, match="Parâmetros do modelo mudaram"):
        verify_and_save_freeze_manifest(manifest_param, output_dir)
        
    # 4. Altera IDs de treino
    split_train_ids = MagicMock()
    split_train_ids.X_train = X_train
    split_train_ids.X_test = X_test
    split_train_ids.train_ids = pd.Series([1, 0]) # Ordem invertida
    split_train_ids.test_ids = pd.Series([2])
    split_train_ids.metadata = meta
    manifest_train_ids = create_model_freeze_manifest(clf, split_train_ids) # type: ignore
    with pytest.raises(RuntimeError, match="IDs do conjunto de treinamento mudaram"):
        verify_and_save_freeze_manifest(manifest_train_ids, output_dir)
        
    # 5. Altera IDs de teste
    split_test_ids = MagicMock()
    split_test_ids.X_train = X_train
    split_test_ids.X_test = X_test
    split_test_ids.train_ids = pd.Series([0, 1])
    split_test_ids.test_ids = pd.Series([99]) # Index alterado
    split_test_ids.metadata = meta
    manifest_test_ids = create_model_freeze_manifest(clf, split_test_ids) # type: ignore
    with pytest.raises(RuntimeError, match="IDs do conjunto de teste mudaram"):
        verify_and_save_freeze_manifest(manifest_test_ids, output_dir)
        
    # 6. Altera hash do CSV
    meta_mod = MagicMock()
    meta_mod.total_samples = 100
    meta_mod.train_samples = 80
    meta_mod.test_samples = 20
    meta_mod.test_size = 0.2
    meta_mod.random_state = 42
    meta_mod.stratified = True
    meta_mod.csv_sha256 = "xyz"
    split_csv = MagicMock()
    split_csv.X_train = X_train
    split_csv.X_test = X_test
    split_csv.train_ids = pd.Series([0, 1])
    split_csv.test_ids = pd.Series([2])
    split_csv.metadata = meta_mod
    manifest_csv = create_model_freeze_manifest(clf, split_csv) # type: ignore
    with pytest.raises(RuntimeError, match="CSV de origem mudou"):
        verify_and_save_freeze_manifest(manifest_csv, output_dir)
        
    # 7. Altera código fonte
    manifest_code = create_model_freeze_manifest(clf, split) # type: ignore
    manifest_code["source_code_sha256"]["src/naive_bayes.py"] = "fake_hash"
    with pytest.raises(RuntimeError, match="Arquivos fonte \\(código\\) foram alterados"):
        verify_and_save_freeze_manifest(manifest_code, output_dir)


def test_legacy_manifest_migration(tmp_path: Path):
    """Testa se manifesto legado (sem IDs) é migrado corretamente."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    legacy = {
        "status": "FROZEN_FOR_FINAL_EVALUATION",
        "dataset_sha256": "abc",
        "split_configuration": {
            "total_samples": 100,
            "train_samples": 80,
            "test_samples": 20,
            "test_size": 0.2,
            "random_state": 42,
            "stratified": True,
        },
        "model_parameters": {"dummy": "params"},
        "source_code_sha256": {"src/naive_bayes.py": "abc"},
        "zero_leakage_attestation": "..."
    }
    manifest_path = output_dir / "model_freeze_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(legacy, f)
        
    meta = MagicMock()
    meta.total_samples = 100
    meta.train_samples = 80
    meta.test_samples = 20
    meta.test_size = 0.2
    meta.random_state = 42
    meta.stratified = True
    meta.csv_sha256 = "abc"
    X_train = pd.DataFrame({"A": [1, 2]}, index=[0, 1])
    X_test = pd.DataFrame({"A": [3]}, index=[2])
    split = MagicMock()
    split.X_train = X_train
    split.X_test = X_test
    split.metadata = meta
    clf = DummyModel()
    
    # Novo manifesto vai ser gerado
    manifest_new = create_model_freeze_manifest(clf, split) # type: ignore
    verify_and_save_freeze_manifest(manifest_new, output_dir)
    
    legacy_path = output_dir / "model_freeze_manifest_legacy.json"
    assert legacy_path.exists()
    
    with open(legacy_path, "r") as f:
        saved_legacy = json.load(f)
    assert saved_legacy == legacy
