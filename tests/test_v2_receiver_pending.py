"""Receiver-owned pending draft boundaries for V2; real tiny CPU forward."""
import pytest
from test_incremental_kv_receive import build
from test_stateful_token_cache import copies, unchanged
from token_kv_agent_system.communication import KVStrategy

V2 = KVStrategy.SUFFIX_RECOMPUTE_DELTA

@pytest.mark.parametrize('raw', [(100,101), (100,2), (100,3)])
@pytest.mark.parametrize('k', [0,1,99])
def test_receiver_pending_close_then_v2_and_reverse_delivery(raw,k):
    b,r=build();s=r.session
    b.outputs=list(raw);s.generate_message('A','B',max_new_tokens=len(raw))
    b.outputs=list(raw);s.generate_message('B','A',max_new_tokens=len(raw))
    old={n:s.state(n) for n in ('A','B')};saved={n:copies(old[n].cache) for n in old}
    assert old['B'].pending_token_id==raw[-1]
    receipt=s.handoff_kv('A','B',strategy=V2,recompute_tokens=k)
    assert receipt.receiver_pending_materialized
    assert s.state('B').pending_token_id is None
    assert s.state('B').draft_message==old['B'].draft_message
    content_count=len(raw)-int(raw[-1] in b.eos_token_ids)
    assert receipt.receiver_message_prefill_count==min(k,content_count)
    expected_start=old['B'].cached_token_count+1+s._closing_extra(old['B'])+len(old['B'].framing.incoming('A','B').prefix_ids)
    assert receipt.receiver_target_span==(expected_start,expected_start+content_count)
    for n in old:unchanged(old[n].cache,saved[n]);s.state(n).validate()
    # Already accepted B draft was closed locally, never silently replaced.
    reverse=s.handoff_kv('B','A',strategy=V2,recompute_tokens=k)
    assert reverse.received_token_ids==raw
    assert s.state('B').draft_message is None
    assert len(s.transport.records)==2
    s.generate_message('B','A',max_new_tokens=1)


def test_bad_v2_count_rejected_without_transport_or_forward():
    b,r=build();s=r.session;s.generate_message('A','B',max_new_tokens=2)
    calls=len(b.calls);states={n:s.state(n) for n in ('A','B')}
    with pytest.raises(ValueError):s.handoff_kv('A','B',strategy=V2,recompute_tokens=-1)
    assert len(b.calls)==calls and not s.transport.records
    assert all(s.state(n) is states[n] for n in states)
