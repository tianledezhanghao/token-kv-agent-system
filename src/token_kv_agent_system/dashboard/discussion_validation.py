"""Opt-in real API/WS acceptance; backend must already run with --discussion."""
from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path
from uuid import uuid4


async def validate(base_url: str, output: Path, summarizer: str,
                   agent_a_system_prompt: str | None = None,
                   agent_b_system_prompt: str | None = None) -> None:
    import httpx
    import websockets
    output.mkdir(parents=True, exist_ok=True)
    baseline = None
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        for mode in ('token', 'kv:v0', 'kv:v1', 'mixed'):
            config = dict(session_id=uuid4().hex, question='23 + 48 = ?', rounds=2,
                          max_new_tokens=64, summarizer=summarizer, mode=mode,
                          agent_a_system_prompt=agent_a_system_prompt,
                          agent_b_system_prompt=agent_b_system_prompt)
            preview = await client.post('/api/discussions/prompts', json={
                'agent_a_system_prompt': agent_a_system_prompt,
                'agent_b_system_prompt': agent_b_system_prompt})
            preview.raise_for_status()
            plan = ['token','kv:v1','token','kv:v0'] if mode=='mixed' else [mode]*4
            if mode=='mixed':
                config['handoff_plan'] = [dict(round=i//2+1, source='A' if i%2==0 else 'B',
                    target='B' if i%2==0 else 'A', mode=m) for i,m in enumerate(plan)]
            ws_url = base_url.replace('http://','ws://').replace('https://','wss://').rstrip('/')+'/ws/events'
            async with websockets.connect(ws_url) as ws:
                response = await client.post('/api/discussions',json=config)
                response.raise_for_status()
                events = []
                while True:
                    event = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
                    if event['metadata'].get('session_id') != config['session_id']: continue
                    events.append(event)
                    if event['type']=='AGENT_TURN_STARTED':
                        assert (await client.get('/api/health')).json()=={'status':'ok'}
                    if event['type'] in ('DISCUSSION_COMPLETED','DISCUSSION_FAILED'): break
                result = (await client.get('/api/discussions/'+config['session_id'])).json()
                record = {'state':result,'events':events}
                (output/(mode.replace(':','-')+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
                assert result['status']=='completed', result['error']
                assert result['prompt_config'] == preview.json()
                assert events[0]['metadata']['prompt_config'] == events[-1]['metadata']['prompt_config'] == preview.json()
                assert result['cache_released']
                assert result['final_answer']['agent']==summarizer
                assert [(t['round'],t['agent']) for t in result['turns']]==[(1,'A'),(1,'B'),(2,'A'),(2,'B')]
                assert [e['metadata']['revision'] for e in events]==list(range(1,result['revision']+1))
                assert [e['metadata']['turn'] for e in events if e['type']=='AGENT_TURN_COMPLETED']==result['turns']
                assert [e['metadata']['handoff'] for e in events if e['type']=='HANDOFF_COMPLETED']==result['handoffs']
                assert events[-1]['metadata']['summary']==result['summary']
                assert (await client.get('/api/discussions/latest')).json()['session']==result
                assert sum(h['wire_bytes'] for h in result['handoffs'])==result['summary']['total_wire_bytes']
                for expected,h in zip(plan,result['handoffs']):
                    assert h['requested_mode']==h['executed_mode']==('token' if expected=='token' else 'kv')
                    assert h['requested_kv_strategy']==h['executed_kv_strategy']
                    assert h['error'] is h['fallback_reason'] is None
                    if expected!='token': assert h['receiver_message_prefill_count']==0
                lengths=result['initial_cache_lengths']
                if baseline is None: baseline=lengths
                assert lengths==baseline, 'cross-session role/question cache lengths changed'
                print(json.dumps({'mode':mode,'status':result['status'],'final':result['final_answer'],
                                  'summary':result['summary'],'initial_cache_lengths':lengths},ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:8000')
    parser.add_argument('--output',type=Path,default=Path('benchmark-results/discussion-77-api'))
    parser.add_argument('--summarizer',choices=('A','B'),default='A')
    parser.add_argument('--agent-a-system-prompt')
    parser.add_argument('--agent-b-system-prompt')
    args=parser.parse_args(argv)
    asyncio.run(validate(args.base_url,args.output,args.summarizer,
                         args.agent_a_system_prompt,args.agent_b_system_prompt))
    return 0
