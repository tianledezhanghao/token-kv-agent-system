"""Read-only 7.8C export: exact IDs/source hashes plus independently entered manual B1-B7 ratings."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from .private_exchange_validation import frozen_spec, POSITIVES, NEGATIVES, REPEAT, score_texts

HISTORY = (
    ('P0-17-26','benchmark-results/qwen3b-78ar/3b-secret/secret-17-26.json','291624af184c72aadb691b44bd98c7b1ded1a3b7a2cfc725973a921b722d4218'),
    ('P0-17-31','benchmark-results/qwen3b-78ar/3b-secret/secret-17-31.json','a4166bb9eb4e37035ffdd548247df3223839fe93e238d69c8423c02b885e7ea8'),
    ('P1-17-26','benchmark-results/secret-sum-78b/3b-candidate/candidate-S4-26-token.json','354d329c1f0bcfd922997a4eb7b6ce5429480b700b43765925259aaa22cd09df'),
    ('P1-17-31','benchmark-results/secret-sum-78b/3b-candidate/candidate-S4-31-token.json','3e172d318408e552e7b23627540994eba306a7ee6b15297c7bcac8875d432e2e'),
)


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def history_scene(path, prompt):
    x=json.loads(path.read_text())
    if prompt=='P0':
        outputs=[dict(message=t['message'],text=t['text'],origin='sender_decode') for t in x['result']['turns']]
        outputs.append(dict(message=x['result']['final_message'],text=x['result']['final_text'],origin='sender_decode'))
        contexts=[t['context'] for t in x['generation_contexts']]
        receipts=[t['receipt'] for t in x['result']['turns']]
        assert x['result']['status']=='completed'
    else:
        outputs=x['outputs'];contexts=x['contexts'];receipts=x['receipts']
    assert len(outputs)==len(contexts)==3 and len(receipts)==len(x['wire_records'])==2
    for output,receipt in zip(outputs,receipts):
        assert output['message']['message_token_ids']==receipt['received_token_ids']
        assert receipt['receiver_cached_count_before']>0
        assert receipt['receiver_cached_count_after']-receipt['receiver_cached_count_before']==len(receipt['receiver_appended_ids'])
    b=x['config']['secret_b'] if prompt=='P0' else x['secret_b']
    texts=[o['text'] for o in outputs]
    return dict(prompt=prompt,control='positive',secret_a=17,secret_b=b,positive_real_exchange=True,
                outputs=outputs,contexts=contexts,receipts=receipts,wire_records=x['wire_records'],
                score=score_texts(17,b,texts,received_a=17,received_b=b if prompt=='P1' else None,engineering=True))


def read_reviews(path):
    rows=list(csv.DictReader(path.open()))
    assert len({r['case'] for r in rows})==len(rows), 'duplicate manual assessment'
    allowed={'yes','partial','no','unknown','not_run'}
    for r in rows:
        assert all(r[f'B{i}'] in allowed for i in range(1,8))
        assert r['reason'].strip(), 'manual evidence explanation is required'
    return {r['case']:r for r in rows}


def export(root, docs):
    reviews=read_reviews(docs/'private-exchange-78c-assessments.csv')
    scenes=[]
    for name,relative,digest in HISTORY:
        path=Path(relative)
        assert file_digest(path)==digest, 'historical evidence changed'
        scenes.append((name,path,history_scene(path,name[:2]),'historical_reuse'))
    expected=[('core','positive',a,b) for a,b in POSITIVES]+[('core',n,a,b) for a,b,n in NEGATIVES]+[('repeat','positive',a,b) for a,b in REPEAT]
    for phase,control,a,b in expected:
        name=f'P2-{phase}-{control}-{a}-{b}';path=root/phase/f'{control}-{a}-{b}.json'
        if not path.exists():
            scenes.append((name,path,None,'not_run'));continue
        row=json.loads(path.read_text())
        assert row['spec_sha256']==frozen_spec()['sha256']
        scenes.append((name,path,row,'new_run'))
    assert set(reviews)=={name for name,_,_,_ in scenes}, 'missing or extra reviews'
    index=[];raw=['# 7.8C 私有交换完整文本与核心IDs\n',
                 'P0/P1是历史复用，不算本轮新生成。P2的N2是fixed_fixture，不算真实A Decode。Markdown仅规范化行尾空格；精确输出以IDs和原JSON为准。逐forward/mask/position/cache/wire/资源在对应忽略JSON；下列全量输出及输入IDs供Git内复核。\n']
    for name,path,x,origin in scenes:
        review=reviews[name]
        if x is None:
            assert all(review[f'B{i}']=='not_run' for i in range(1,8)), 'unexecuted scene cannot be graded as failure or success'
        row=dict(case=name,file=str(path),sha256=file_digest(path) if x is not None else '',origin=origin,
                 **{f'B{i}':review[f'B{i}'] for i in range(1,8)},reason=review['reason'])
        row['strict_positive_pass']=bool(x and x['positive_real_exchange'] and all(review[f'B{i}']=='yes' for i in range(1,8)))
        row['control_pass']=bool(x and not x['positive_real_exchange'] and all(review[f'B{i}']=='yes' for i in range(1,8)))
        index.append(row)
        raw.append(f"## {name}\n\n来源：`{path}`；SHA256：`{row['sha256']}`；{origin}。\n\n判定："+json.dumps({f'B{i}':review[f'B{i}'] for i in range(1,8)},ensure_ascii=False)+'。'+review['reason']+'\n')
        if x is None:
            raw.append('not_run\n');continue
        for i,(o,c) in enumerate(zip(x['outputs'],x['contexts'])):
            m=o['message'];stop=m['termination_reason'];label=('A首次','B回复','A最终')[i]
            raw.append(f"### {label}\n\norigin={o.get('origin','sender_decode')}；停止={stop}；输出IDs数={len(m['message_token_ids'])}。\n\n```text\n{o['text']}\n```\n\nInput IDs：\n```json\n"+json.dumps(c['ledger_token_ids'])+'\n```\nOutput IDs：\n```json\n'+json.dumps(m['message_token_ids'])+'\n```\n')
            raw.append('逻辑模型消息/来源边界：\n```json\n'+json.dumps(c['logical_messages'],ensure_ascii=False,indent=2)+'\n```\n')
        receipts=[d['receipt'] for d in x['deliveries']] if 'deliveries' in x else x['receipts']
        raw.append('真实交接received IDs：\n```json\n'+json.dumps([dict(source=r['message']['source_agent'],received_ids=r['received_token_ids'],receiver_new_prefill_ids=r['receiver_appended_ids'],before=r['receiver_cached_count_before'],after=r['receiver_cached_count_after']) for r in receipts])+'\n```\n')
    (docs/'private-exchange-78c-raw-outputs.md').write_text('\n'.join(line.rstrip() for line in '\n'.join(raw).splitlines())+'\n')
    with (docs/'private-exchange-78c-index.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(index[0]),lineterminator='\n');w.writeheader();w.writerows(index)
    return index


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir',type=Path,default=Path('benchmark-results/secret-sum-78c'))
    p.add_argument('--docs-dir',type=Path,default=Path('docs'))
    a=p.parse_args(argv)
    rows=export(a.results_dir,a.docs_dir)
    print(json.dumps(dict(cases=len(rows),strict_positive_pass=sum(r['strict_positive_pass'] for r in rows),control_pass=sum(r['control_pass'] for r in rows))))
    return 0


if __name__=='__main__':raise SystemExit(main())
