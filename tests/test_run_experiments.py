import subprocess
import sys
import shutil
import os
from pathlib import Path

def test_cli_help():
    project_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "run_experiments.py", "--help"],
        cwd=project_root,
        capture_output=True, text=True
    )
    assert result.returncode == 0

def test_independent_execution_without_agents_folder_permissions(tmp_path):
    """L5: Teste para garantir que a execução não falhe se a pasta .agents não existir ou não for gravável."""
    # Copia os arquivos do projeto para o tmp_path
    project_root = Path(__file__).resolve().parent.parent
    
    # Criamos um dir fake_project e clonamos o src e run_experiments.py
    fake_project = tmp_path / "fake_project"
    fake_project.mkdir()
    
    shutil.copytree(project_root / "src", fake_project / "src")
    shutil.copy(project_root / "run_experiments.py", fake_project)
    
    # Criamos a pasta .agents apenas leitura no parente
    agents_dir = tmp_path / ".agents"
    agents_dir.mkdir()
    # Não vamos mexer nas permissões do python, apenas o fato de ele tentar 
    # usar um .agents sem ser o correto já prova a independência.
    # Vamos fazer a pasta agents_dir sem permissão de escrita
    os.chmod(agents_dir, 0o555)
    
    # Executa o run_experiments.py --stage all no fake_project
    env = os.environ.copy()
    env["PYTHONPATH"] = str(fake_project)
    
    # Precisamos apontar para o dataset real usando flag CLI
    from src.data_loader import resolve_dataset_path
    real_csv = resolve_dataset_path()
    
    result = subprocess.run(
        [sys.executable, "run_experiments.py", "--stage", "all", "--data-path", str(real_csv)],
        cwd=fake_project,
        capture_output=True, text=True, env=env
    )
    
    # Restaura permissões para limpeza
    os.chmod(agents_dir, 0o777)
    
    assert result.returncode == 0
    # removido pois a flag é opt-in agora

def test_save_artifacts_lote4_zero_division(tmp_path):
    """Testa se save_artifacts_lote4 suporta classes zeradas sem ZeroDivisionError."""
    from run_experiments import save_artifacts_lote4
    import pandas as pd
    import numpy as np
    import json
    from src.metrics import compute_metrics
    from unittest.mock import MagicMock
    
    # 1. Test missing class 1 (only actual class 0)
    y_true = np.array([0, 0])
    y_pred = np.array([0, 0])
    metrics = compute_metrics(y_true, y_pred)
    
    clf = MagicMock()
    clf.predict.return_value = y_pred
    clf.predict_proba.return_value = np.array([[0.8, 0.2], [0.9, 0.1]])
    clf.predict_joint_log_likelihood.return_value = np.array([[-1.0, -2.0], [-1.0, -2.0]])
    
    split = MagicMock()
    split.X_test = pd.DataFrame({"A": [1, 2]})
    split.y_test = pd.Series(y_true)
    split.test_ids = pd.Series(["id1", "id2"])
    
    output_dir = tmp_path / "out_no_class1"
    save_artifacts_lote4(clf, metrics, split, output_dir)
    
    cm_file = output_dir / "confusion_matrix.json"
    assert cm_file.exists()
    
    with open(cm_file, "r") as f:
        data = json.load(f)
        row_pct = data["row_percentages"]
        assert row_pct["class_1_fn_rate"] is None
        assert row_pct["class_1_vp_rate"] is None
        assert "Nenhuma instância da classe 1" in row_pct["class_1_undefined_reason"]
        
    # 2. Test missing class 0 (only actual class 1)
    y_true2 = np.array([1, 1])
    y_pred2 = np.array([1, 1])
    metrics2 = compute_metrics(y_true2, y_pred2)
    
    split2 = MagicMock()
    split2.X_test = pd.DataFrame({"A": [1, 2]})
    split2.y_test = pd.Series(y_true2)
    split2.test_ids = pd.Series(["id1", "id2"])
    
    output_dir2 = tmp_path / "out_no_class0"
    save_artifacts_lote4(clf, metrics2, split2, output_dir2)
    
    cm_file2 = output_dir2 / "confusion_matrix.json"
    assert cm_file2.exists()
    
    with open(cm_file2, "r") as f:
        data2 = json.load(f)
        row_pct2 = data2["row_percentages"]
        assert row_pct2["class_0_vn_rate"] is None
        assert row_pct2["class_0_fp_rate"] is None
        assert "Nenhuma instância da classe 0" in row_pct2["class_0_undefined_reason"]

