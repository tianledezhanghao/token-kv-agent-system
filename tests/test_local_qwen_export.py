import importlib.util
import json
from pathlib import Path
import pytest


def test_resume_export_preserves_old_history_and_refuses_duplicate(monkeypatch,tmp_path):
 path=Path(__file__).resolve().parents[1]/'experiments/summarize_qwen_78a.py'
 spec=importlib.util.spec_from_file_location('qwen_export_test',path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 monkeypatch.chdir(tmp_path)
 docs=tmp_path/'docs';docs.mkdir()
 old='Historic zero-generation evidence\n'
 target=docs/'local-qwen3b-raw-outputs.md';target.write_text(old)
 root=tmp_path/'experiments';run=root/'run';run.mkdir(parents=True)
 (run/'single-16.json').write_text(json.dumps(dict(text='mock output  \nsecond line  ',stop_reason='max_new_tokens',input_ids=[1],output_ids=[2])))
 module.main(['--root',str(root),'--append'])
 output=target.read_text()
 assert output.startswith(old) and 'mock output' in output
 assert '7.8A-R' in output and '1 / single' in output
 assert 'input_origin=single_model' in output
 assert 'mock output  ' not in output and 'second line  ' not in output
 with pytest.raises(ValueError,match='already exists'):module.main(['--root',str(root),'--append'])
 assert target.read_text()==output
