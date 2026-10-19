"""Export every 7.7G output and a hashed trace index; no semantic auto-scoring.

Run after completed raw directories exist. Large JSON/IDs stay gitignored. This
writes reviewable Markdown and CSV, never model inputs or production state.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def generations(record):
    if 'input_ids' in record:
        return [record]
    if 'turns' in record and record['turns'] and 'input_ids' in record['turns'][0]:
        return record['turns']
    if 'contexts' in record and 'discussion' in record:
        return [dict(input_ids=r['context']['ledger_token_ids'], messages=r['context']['logical_messages'],
                     output_ids=r['output']['message_token_ids'], agent=r['context']['agent'],
                     text=record['discussion']['turns'][i]['text'] if i<len(record['discussion']['turns']) else record['discussion']['final_text'],
                     stop_reason=r['output']['termination_reason'],
                     template_ids_equal=r['context']['template_ids_equal'])
                for i,r in enumerate(record['contexts'])]
    if 'generation_contexts' in record and 'result' in record:
        texts=[t['text'] for t in record['result']['turns']]+[record['result']['final_text']]
        messages=[t['message'] for t in record['result']['turns']]+[record['result']['final_message']]
        return [dict(input_ids=r['context']['ledger_token_ids'],messages=r['context']['logical_messages'],
                     output_ids=m['message_token_ids'],agent=r['context']['agent'],text=text,
                     stop_reason=m['termination_reason']) for r,text,m in zip(record['generation_contexts'],texts,messages)]
    if 'modes' in record:
        return [dict(input_ids=r['context']['ledger_token_ids'],messages=r['context']['logical_messages'],
                     output_ids=r['reply']['message_token_ids'],agent='B',text=r['reply_text'],
                     stop_reason=r['reply']['termination_reason'],mode=r['mode']) for r in record['modes']]
    return []


def export(directories, output_dir):
    from token_kv_agent_system.autogen_style_reference import overlap
    output_dir.mkdir(parents=True,exist_ok=True)
    lines=['# 7.7G 全部正式真实输出','',
           '由 experiments/summarize_autogen_77g.py 生成；没有按成功筛选。所有输入/输出 IDs 原始 JSON 留在忽略实验目录；CSV保存每条输入/输出的SHA256。展示行尾空白规范化，精确原文以JSON及Token IDs为准。固定Sender为fixture，B输出是真实Decode；真实对话所有发言均为Decode。停止=eos或预算，仅是执行原因，不是质量评分。','']
    rows=[]; catalog=[]
    for directory in directories:
        if not (directory/'completion.json').exists():
            raise ValueError(f'not completed: {directory}')
        for path in sorted(directory.glob('*.json')):
            record=json.loads(path.read_text())
            catalog.append(dict(directory=directory.name,record=path.name,
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            turns=generations(record)
            if not turns:
                continue
            lines += [f'## {directory.name}/{path.stem}','']
            if record.get('input_origin'):
                lines += [f"Sender origin={record['input_origin']}；fixture={record.get('fixture',record.get('sender_text'))!r}。",'']
            for i,turn in enumerate(turns):
                adjacent=overlap(turns[i-1]['text'],turn['text']) if i else None
                rows.append(dict(directory=directory.name,record=path.stem,turn=i+1,
                    agent=turn.get('agent','B'),mode=turn.get('mode',record.get('mode','')),
                    stop=turn['stop_reason'],input_count=len(turn['input_ids']),output_count=len(turn['output_ids']),
                    input_sha256=digest(turn['input_ids']),output_sha256=digest(turn['output_ids']),
                    adjacent_text_exact=adjacent['exact_match'] if adjacent else '',
                    adjacent_token_exact=turns[i-1]['output_ids']==turn['output_ids'] if i else '',
                    character_overlap=adjacent['character_overlap'] if adjacent else '',
                    long_exact_copy=adjacent['long_exact_copy'] if adjacent else '',
                    template_ids_equal=turn.get('template_ids_equal','')))
                lines += [f"**{i+1} / {turn.get('agent','B')} / {turn.get('mode',record.get('mode',''))}**；stop={turn['stop_reason']}；input/output={len(turn['input_ids'])}/{len(turn['output_ids'])}。",'',
                          '```text','\n'.join(line.rstrip() for line in turn['text'].split('\n')),'```','']
    (output_dir/'autogen-style-raw-outputs.md').write_text('\n'.join(lines).rstrip()+'\n')
    for filename,data in [('autogen-style-trace-index.csv',rows),('autogen-style-artifact-index.csv',catalog)]:
        with (output_dir/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');writer.writeheader();writer.writerows(data)
    print(f'{len(rows)} generations, {len(catalog)} raw artifacts; exported all outputs')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories',type=Path,nargs='+')
    parser.add_argument('--output-dir',type=Path,default=Path('docs'))
    args=parser.parse_args()
    export(args.directories,args.output_dir)
