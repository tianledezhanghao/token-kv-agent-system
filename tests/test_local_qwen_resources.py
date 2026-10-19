from dataclasses import replace
import pytest
from token_kv_agent_system.local_qwen_resources import ResourceConfig,ResourcePolicy,classify_failure,ResourceStop


def sample(t,level=1,swap=2000):return dict(monotonic=t,pressure_level=level,swap_used_mib=swap)


def test_warning_and_stable_swap_never_stop():
 p=ResourcePolicy(sample(0))
 for t in range(0,120,2):
  d=p.observe(sample(t,2,5500))
  assert d['warning'] and d['stop_reason'] is None
 assert d['swap_rate_mib_per_minute']==0


def test_sustained_critical_stops_but_transient_recovers():
 p=ResourcePolicy(sample(0))
 assert p.observe(sample(0,4))['stop_reason'] is None
 assert p.observe(sample(8,4))['stop_reason'] is None
 assert p.observe(sample(9,1))['stop_reason'] is None
 assert p.observe(sample(10,4))['stop_reason'] is None
 assert p.observe(sample(20,4))['stop_reason']=='sustained_critical_pressure'


def test_rapid_swap_requires_pressure_and_full_window():
 for level in (1,2):
  p=ResourcePolicy(sample(0))
  for t in range(0,30,2):assert p.observe(sample(t,level,2000+20*t))['stop_reason'] is None
  d=p.observe(sample(30,level,2600))
  assert d['swap_rate_mib_per_minute']==1200
  assert d['stop_reason']==('sustained_pressure_and_rapid_swap' if level==2 else None)


def test_stalled_forward_only_stops_with_sustained_pressure():
 for level in (1,2):
  p=ResourcePolicy(sample(0))
  p.observe(sample(0,level))
  d=p.observe(sample(60,level),active_forward_since=0)
  assert d['stop_reason']==('stalled_forward_with_sustained_pressure' if level==2 else None)


@pytest.mark.parametrize('name',ResourceConfig.__dataclass_fields__)
def test_thresholds_positive(name):
 with pytest.raises(ValueError):replace(ResourceConfig(),**{name:0})


def test_failure_category():
 assert classify_failure(RuntimeError('MPS backend out of memory'))=='oom'
 assert classify_failure(ResourceStop('sustained_critical_pressure'))=='resource_policy'
 assert classify_failure(ValueError('shape mismatch'))=='execution_exception'


def test_monitor_records_forward_failure_and_closes(tmp_path):
 import torch,json
 from token_kv_agent_system.local_qwen_resources import ResourceMonitor
 m=ResourceMonitor(lambda:sample(0),sample(0),tmp_path)
 m.start()
 def fail(**kwargs):raise RuntimeError('MPS out of memory')
 with pytest.raises(RuntimeError,match='out of memory'):
  m.forward(fail,input_ids=torch.tensor([[1,2]]))
 m.close()
 r=json.loads((tmp_path/'forward-failure.json').read_text())
 assert r['failure_category']=='oom' and r['input_ids']==[1,2]
 assert not m.thread.is_alive()


def test_monitor_boundary_stop_never_calls_forward(tmp_path):
 from token_kv_agent_system.local_qwen_resources import ResourceMonitor
 m=ResourceMonitor(lambda:sample(0),sample(0),tmp_path)
 m.stop_reason='sustained_critical_pressure'
 with pytest.raises(ResourceStop):m.forward(lambda **kw:pytest.fail('should not execute'))


def test_invalid_sample_interval():
 with pytest.raises(ValueError):ResourceConfig(sample_seconds=11)


def test_watchdog_persists_before_terminating_only_own_process(monkeypatch,tmp_path):
 import time,json,threading
 import token_kv_agent_system.local_qwen_resources as resources
 config=ResourceConfig(sample_seconds=.002,critical_seconds=.004,trend_seconds=.006,
                       pressure_seconds=.006,stalled_forward_seconds=.008,stop_grace_seconds=.004)
 m=resources.ResourceMonitor(lambda:sample(time.monotonic(),4),sample(time.monotonic()),tmp_path,config)
 called=threading.Event();codes=[]
 def exit_own(code):
  codes.append(code);m.done.set();called.set()
 monkeypatch.setattr(resources.os,'_exit',exit_own)
 m.start()
 assert called.wait(2)
 m.close()
 r=json.loads((tmp_path/'watchdog-failure.json').read_text())
 assert codes==[2] and r['monitor_terminated_own_process']
 assert r['error']=='sustained_critical_pressure' and r['samples']


def test_runtime_wrapped_resource_error_classification():
 assert classify_failure(RuntimeError('ResourceStop: sustained_critical_pressure'))=='resource_policy'
