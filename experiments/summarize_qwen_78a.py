"""Export ALL completed 7.8A outputs; baseline file hashes preserve provenance."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def turns(row):
    if 'output_ids' in row:
        return [dict(agent=row.get('agent','single'), text=row['text'], stop=row['stop_reason'],
                     input_ids=row['input_ids'], output_ids=row['output_ids'])]
    if 'turns' in row:
        return [dict(agent=t['agent'],text=t['text'],stop=t['stop_reason'],
                     input_ids=t['input_ids'],output_ids=t['output_ids']) for t in row['turns']]
    if 'discussion' in row or 'result' in row:
        r=row.get('discussion',row.get('result'))
        messages=[t['message'] for t in r['turns']]+([r['final_message']] if r.get('final_message') else [])
        texts=[t['text'] for t in r['turns']]+([r['final_text']] if r.get('final_message') else [])
        contexts=row.get('contexts',row.get('generation_contexts',[]))
        return [dict(agent=m['source_agent'],text=text,stop=m['termination_reason'],
                     input_ids=c['context']['ledger_token_ids'],output_ids=m['message_token_ids'])
                for m,text,c in zip(messages,texts,contexts)]
    if 'texts' in row:
        return [dict(agent=m['source_agent'],text=text,stop=m['termination_reason'],
                     input_ids=c['ledger_token_ids'],output_ids=m['message_token_ids'])
                for m,text,c in zip(row['messages'],row['texts'],row['contexts'])]
    return []


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('benchmark-results/qwen3b-78a'))
    p.add_argument('--append',action='store_true',help='Append a new 7.8A-R section; preserve original startup evidence')
    a=p.parse_args(argv)
    lines=['# 7.8A 全部真实输出','', '成功、失败与资源中止均保留。固定来源标记与完整输入/输出IDs保存在忽略目录；这里逐条转录所有已保存生成，SHA256可复核原始JSON。若生成前资源中止，只展示启动诊断，不虚构输出或IDs。Markdown只规范化行尾空白，精确原文以JSON/IDs为准。人工评分见主报告。','']
    if a.append:
        lines[0]='## 7.8A-R — 调整资源保护后的续跑结果'
        old=Path('docs/local-qwen3b-raw-outputs.md').read_text()
        if lines[0] in old:raise ValueError('resume section already exists; refusing duplicate/overwrite')
    section="### " if a.append else "## "
    index=[]
    for path in sorted(a.root.glob('*/*.json')):
        row=json.loads(path.read_text())
        trace=turns(row)
        if not trace:
            if path.stem in ('preload','failure','settings'):
                fields = ('model','repo','revision','checkpoint','kv_config','parameter_bytes',
                          'load_seconds','memory_after_load','type','error','elapsed_seconds','memory',
                          'swap_used_mib','pressure_level','rss_kib','mps_current_bytes','mps_driver_bytes')
                diagnostic={k:row[k] for k in fields if k in row}
                lines += [section+str(path.relative_to(a.root)), '',
                          '执行配置/诊断；实际回答见对应生成记录。JSON SHA256 `'+hashlib.sha256(path.read_bytes()).hexdigest()+'`。',
                          '', '```json',json.dumps(diagnostic,ensure_ascii=False,indent=2),'```','']
            continue
        rel=path.relative_to(a.root)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        origin=row.get('input_origin', 'single_model' if 'output_ids' in row and 'agent' not in row else 'real_local_decode_no_handoff' if row.get('wire_bytes')==0 else 'sender_decode')
        lines += [section+str(rel),'', f'JSON SHA256 `{digest}`；input_origin={origin}。','']
        for i,t in enumerate(trace,1):
            lines += [f'**{i} / {t["agent"]} / {t["stop"]}**；input/output={len(t["input_ids"])}/{len(t["output_ids"])}。','', '```text','\n'.join(line.rstrip() for line in t['text'].splitlines()).rstrip(),'```','']
            index.append(dict(file=str(rel),json_sha256=digest,turn=i,agent=t['agent'],stop=t['stop'],
                input_tokens=len(t['input_ids']),output_tokens=len(t['output_ids']),
                input_sha256=hashlib.sha256(json.dumps(t['input_ids']).encode()).hexdigest(),
                output_sha256=hashlib.sha256(json.dumps(t['output_ids']).encode()).hexdigest()))
    output='\n'.join(lines).rstrip()+'\n'
    Path('docs/local-qwen3b-raw-outputs.md').write_text(old.rstrip()+'\n\n'+output if a.append else output)
    with Path('docs/local-qwen3b-artifact-index.csv').open('w') as f:
        fields=list(index[0]) if index else ['file','json_sha256','turn','agent','stop','input_tokens','output_tokens','input_sha256','output_sha256']
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(index)
    print(len(index),'generations exported')


if __name__=='__main__':main()
