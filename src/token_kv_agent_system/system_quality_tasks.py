"""8.3C-v1 preregistered tasks and output-only conservative scorers.

No answer returned by a scorer is ever used as model input. These tasks are
independent of all historical benchmarks and production prompt defaults.
"""
from dataclasses import asdict, dataclass
import re

VARIANTS = ('token', 'kv:v0', 'kv:v1', 'kv:v2', 'kv:v3')
PAIR_SYSTEM_A = ('You are Agent A. Send the private information supplied to you accurately. '
                 'Do not invent information. Follow the local instruction briefly.')
PAIR_SYSTEM_B = ('You are Agent B. Read information received from Agent A. '
                 'Use only received information to answer the local question. '
                 'If the information has not been received, answer UNKNOWN. Answer briefly.')
PUBLIC = 'Agent A will send private task information to Agent B.'
SEND_STAGE = 'Repeat the private information from your previous user message exactly. Do not add anything.'

@dataclass(frozen=True)
class PairTask:
    name: str
    category: str
    private: str
    question: str
    extraction: tuple[str, ...]
    answer: str
    numeric: bool = False

# Fixed values chosen before any 8.3C model execution. Three per category.
PAIRS = tuple(
    [PairTask('number-'+str(n), 'number', f'The access number is {n}.',
              'What is the access number sent by Agent A? Output only the number.', (str(n),), str(n), True)
     for n in (27, 64, 138)] +
    [PairTask('code-'+c, 'code', f'The access code is {c}.',
              'Repeat the exact access code sent by Agent A. Output only the code.', (c,), c)
     for c in ('K7M2', 'Q4R9', 'T8N5X3')] +
    [PairTask('fact-'+c, 'fact', f'The {obj} is {c}.',
              f'According to Agent A, what color is the {obj}? Output only its color.', (c,), c)
     for obj,c in (('cup','blue'),('door','green'),('bag','purple'))] +
    [PairTask('facts-'+str(n), 'multiple_facts', f'Locker {n} is {c}. The key is in the {place}.',
              'Extract the locker number, its color, and the key location from Agent A. State all three facts.',
              (str(n),c,place), f'{n} {c} {place}')
     for n,c,place in ((7,'blue','box'),(12,'red','drawer'),(23,'green','bag'))] +
    [PairTask('compute-'+str(n), 'calculation', f'The parcel contains {n} coins.',
              'State the number of coins reported by Agent A, then add 3 coins. Write the addition as an equation.',
              (str(n),), str(n+3), True)
     for n in (8,16,24)]
)
PILOTS = (
    PairTask('pilot-number','number','The access number is 52.',
             'What is the access number sent by Agent A? Output only the number.',('52',),'52',True),
    PairTask('pilot-fact','fact','The hat is yellow.',
             'According to Agent A, what color is the hat? Output only its color.',('yellow',),'yellow'),
)
SECRETS = ((17,26),(17,31),(13,29))
DISCUSSIONS = (
    dict(name='reasoning', category='objective', question='What is 23 + 48? Give the result and a brief check in English.', answer='71'),
    dict(name='integration', category='objective_private',
         question='Combine the two agents\' private inventory facts. State both quantities with ownership and their total in English.',
         a='You are Agent A. Your private inventory fact is: Agent A has 12 red boxes. You do not know Agent B\'s inventory. Exchange facts and calculate the total.',
         b='You are Agent B. Your private inventory fact is: Agent B has 7 blue boxes. You do not know Agent A\'s inventory. Exchange facts and calculate the total.', answer='19'),
    dict(name='choice', category='objective_choice',
         question='Choose one plan satisfying BOTH requirements: cost at most 8 and capacity at least 10. Plan X costs 6 and has capacity 8. Plan Y costs 8 and has capacity 12. Explain briefly in English.', answer='Y'),
    dict(name='open', category='manual_only',
         question='How would you design an efficient multi-agent communication system? Discuss trade-offs briefly in English.', answer=None),
)

def contains(text, item):
    return bool(re.search(r'(?<![A-Za-z0-9])'+re.escape(item)+r'(?![A-Za-z0-9])', text, re.I))

def score_pair(task, sender_text, receiver_text, *, delivered):
    """Extraction is a necessary content check, not ownership/reasoning proof.

    Recall accepts the requested atom, tolerating explanatory words, but rejects
    competing numbers/codes/colors. Calculation requires the reported operand
    and exactly the requested correct equation. Unclear prose stays unknown.
    """
    sender_correct = all(contains(sender_text, x) for x in task.extraction)
    if receiver_text is None:
        return dict(sender_correct=sender_correct, extraction=None, final=None, use=None)
    extraction = all(contains(receiver_text, x) for x in task.extraction)
    if task.category == 'calculation':
        equations = re.findall(r'(?<![\d.])(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)(?![\d.])', receiver_text)
        expected = (int(task.extraction[0]),3,int(task.answer))
        final = (all((int(a),int(b),int(c)) in (expected,(3,expected[0],expected[2])) for a,b,c in equations)
                 if equations else None)
        use = bool(delivered and sender_correct and extraction and final is True)
    elif task.category == 'multiple_facts':
        nums = set(re.findall(r'\b\d+\b',receiver_text))
        final = extraction and nums == {task.extraction[0]}
        use = bool(delivered and sender_correct and final)
    else:
        final = extraction
        if task.category == 'number':
            final = extraction and set(re.findall(r'\b\d+\b',receiver_text)) == {task.answer}
        elif task.category == 'code':
            atoms = set(re.findall(r'\b(?=[A-Za-z0-9]*[0-9])(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]+\b',receiver_text))
            final = extraction and {x.upper() for x in atoms} == {task.answer.upper()}
        elif task.category == 'fact':
            final = extraction and {c for c in ('red','blue','green','purple','yellow') if contains(receiver_text,c)} == {task.answer}
        use = bool(delivered and sender_correct and final)
    return dict(sender_correct=sender_correct, extraction=extraction, final=final, use=use)


def freeze_tasks():
    return dict(version='8.3C-v2', paired=[asdict(t) for t in PAIRS],pilot=[asdict(t) for t in PILOTS],
        systems=[PAIR_SYSTEM_A,PAIR_SYSTEM_B], public=PUBLIC,sender_stage=SEND_STAGE,
        secrets=SECRETS, discussions=DISCUSSIONS, variants=VARIANTS,
        scoring='8.3C-v1 output-only exact atom/conflict checks; calculation exact equation; ownership manual + frozen historical parser; unknown retained',
        sample_plan=dict(pilot=2,paired=15,secret_sessions=15,discussion_sessions=24,absent_controls=2),
        stop='No prompt/answer/scorer changes after formal freeze. Stop on resource policy/OOM; retain partials. All predeclared cases including failures retained. No extra sampling for significance.',
        conditional_analysis='TOKEN-success paired subset alongside all 15; not an exclusion criterion',
        manual_rubric=['substantive new information','unproductive repetition','incorrect facts','error correction when applicable',
                       'important omission','answers original question','budget truncation','explicit ownership without contradiction'])
