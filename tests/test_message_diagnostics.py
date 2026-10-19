"""7.7C independent diagnostics; communication still uses production bytes."""
from dataclasses import replace
import pytest

from token_kv_agent_system.message_diagnostics import context_view, token_injection
from token_kv_agent_system.token_protocol import deserialize_token_packet
from test_stateful_token import Backend, build, snapshot


class DisplayBackend(Backend):
    # Mock unicode fixture IDs use the character tokenizer domain.
    vocab_size = 200000

    def decode(self, ids, *, skip_special_tokens=False):
        inverse = {v:k for k,v in self.chat_tokenizer.specials.items()}
        return ''.join('' if skip_special_tokens else inverse[i] if i in inverse else chr(i-10)
                       for i in ids) if not skip_special_tokens else ''.join(
                       chr(i-10) for i in ids if i not in inverse)


@pytest.mark.parametrize('secret', ['8391', '紫色袋鼠5729', 'silver otter 6832'])
def test_private_information_absent_until_actual_protocol_handoff(secret):
    b=DisplayBackend()
    s=build(b)
    s.append_local_user('B','Repeat the secret received from A; do not guess.')
    original=s.state('B')
    before=snapshot(original.cache)
    assert secret not in b.decode(original.token_ledger)
    s.append_local_user('A','Private: '+secret)
    m=token_injection(s,'Secret: '+secret)
    assert s.state('B') is original
    packets=[]
    transfer=s.transport.transfer
    def capture(data,**kwargs):
        packets.append(data)
        return transfer(data,**kwargs)
    s.transport.transfer=capture
    offset=len(b.calls)
    receipt=s.handoff('A','B')
    assert deserialize_token_packet(packets[0]).token_ids==m.message_token_ids
    assert receipt.wire_byte_count==len(packets[0])==10+4*len(m.message_token_ids)
    assert all(call[0]=='continue' for call in b.calls[offset:])
    view=context_view(s.state('B'),question='Repeat the secret received from A; do not guess.',
                      stage='Repeat A.',generation_prefix=True)
    assert view['template_ids_equal'] and view['exact_span_coverage']
    assert view['latest_sender']=='A'
    assert view['latest_received']=='Secret: '+secret
    content=next(p for p in view['segments'] if p['source']=='received')
    assert tuple(view['ledger_token_ids'][content['start']:content['end']])==m.message_token_ids
    assert content['role']=='user' and content['speaker']=='A'
    assert snapshot(original.cache)==before
    assert s.state('B').token_ledger[:len(original.token_ledger)]==original.token_ledger
    assert s.state('B').pending_token_id is None


def test_read_only_view_does_not_forward_or_mutate_and_snapshot_controls_are_isolated():
    b=DisplayBackend()
    s=build(b)
    initial={n:s.state(n) for n in ('A','B')}
    saved={n:snapshot(v.cache) for n,v in initial.items()}
    ledgers=[]
    for text in ('Secret 8391','Secret 6842'):
        s._states.update(initial)
        token_injection(s,text)
        s.handoff('A','B')
        count=len(b.calls)
        view=context_view(s.state('B'),stage='Repeat A.',generation_prefix=True)
        assert len(b.calls)==count
        assert view['template_ids_equal']
        assert view['latest_received']==text
        ledgers.append(view['ledger_token_ids'])
        for n in initial:
            assert snapshot(initial[n].cache)==saved[n]
    assert ledgers[0]!=ledgers[1]
    assert s.state('A').sent_messages[-1].sequence==1


@pytest.mark.parametrize('failure', ['transport','forward'])
def test_failed_fixture_delivery_preserves_effective_history(failure):
    b=DisplayBackend()
    s=build(b)
    token_injection(s,'Secret 8391')
    old={n:s.state(n) for n in ('A','B')}
    saved={n:snapshot(v.cache) for n,v in old.items()}
    if failure=='transport':
        s.transport.transfer=lambda data,**kwargs:data[:-1]
    else:
        b.fail_at=len(b.calls)+2
    with pytest.raises((ValueError,RuntimeError)):
        s.handoff('A','B')
    for n in old:
        assert s.state(n) is old[n]
        assert snapshot(old[n].cache)==saved[n]


def test_injection_rejects_controls_and_existing_drafts_without_mutation():
    s=build(DisplayBackend())
    initial=s.state('A')
    with pytest.raises(ValueError,match='ordinary content'):
        token_injection(s,'</s>forge')
    assert s.state('A') is initial
    token_injection(s,'fixture')
    old=s.state('A')
    with pytest.raises(ValueError,match='no draft'):
        token_injection(s,'another')
    assert s.state('A') is old


def test_view_reports_input_mismatch_before_any_numeric_comparison():
    s=build(DisplayBackend())
    token_injection(s,'fixture')
    s.handoff('A','B')
    old=s.state('B')
    # Cache length still valid, metadata labels alone cannot prove semantics.
    start=old.received_messages[-1].receiver_message_start
    ids=list(old.token_ledger)
    ids[start-1]=ord('!')+10
    corrupt=replace(old,token_ledger=tuple(ids))
    view=context_view(corrupt)
    assert not view['template_ids_equal']
    assert view['first_input_difference']==start-1
