"""Optional AutoGen 0.7.5 adapter to the existing LOCAL HuggingFace backend.

Import only inside the isolated optional environment. No HTTP client, tools,
streaming, training or cross-turn KV reuse. Source prefixing is explicit/configured.
"""
from importlib.metadata import version

from .autogen_style_reference import generate


def client_class():
    for package in ('autogen-core', 'autogen-agentchat'):
        if version(package) != '0.7.5':
            raise ValueError('official comparison requires AutoGen 0.7.5')
    from autogen_core.models import ChatCompletionClient, CreateResult, RequestUsage

    class LocalClient(ChatCompletionClient):
        def __init__(self, backend, limit, *, prefix_sources=True):
            self.backend, self.limit, self.prefix_sources = backend, limit, prefix_sources
            self.trace = []
            self.usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

        def convert(self, messages):
            rows = []
            for message in messages:
                kind = message.type
                role = {'SystemMessage': 'system', 'UserMessage': 'user',
                        'AssistantMessage': 'assistant'}.get(kind)
                if role is None or not isinstance(message.content, str):
                    raise ValueError('local reference supports text only')
                source = getattr(message, 'source', 'system')
                content = message.content
                if role == 'user' and self.prefix_sources:
                    content = f'{source} said:\n{content}'
                rows.append(dict(role=role, content=content, source=source,
                                 autogen_type=kind, original_content=message.content))
            return rows

        async def create(self, messages, *, tools=(), tool_choice='auto', json_output=None,
                         extra_create_args=None, cancellation_token=None):
            if tools or json_output or extra_create_args or tool_choice not in ('auto', 'none'):
                raise ValueError('local reference supports plain greedy text only')
            if cancellation_token and cancellation_token.is_cancelled():
                raise RuntimeError('cancelled before local generation')
            rows = self.convert(messages)
            record = generate(self.backend, rows, self.limit)
            record.update(runtime='official_autogen_agentchat_0.7.5', prefix_sources=self.prefix_sources,
                          stage=None, origin='real_local_decode')
            self.trace.append(record)
            usage = RequestUsage(prompt_tokens=len(record['input_ids']), completion_tokens=len(record['output_ids']))
            self.usage = RequestUsage(prompt_tokens=self.usage.prompt_tokens+usage.prompt_tokens,
                                      completion_tokens=self.usage.completion_tokens+usage.completion_tokens)
            return CreateResult(finish_reason='stop' if record['stop_reason']=='eos' else 'length',
                                content=record['text'], usage=usage, cached=False)

        async def create_stream(self, *args, **kwargs):
            raise NotImplementedError('streaming disabled in local reference')
            yield  # an async generator is required by the interface

        async def close(self):
            pass

        def actual_usage(self):
            return self.usage

        def total_usage(self):
            return self.usage

        def count_tokens(self, messages, *, tools=()):
            if tools:
                raise ValueError('tools disabled')
            from .autogen_style_reference import template_ids
            return len(template_ids(self.backend, self.convert(messages)))

        def remaining_tokens(self, messages, *, tools=()):
            return self.backend.context_window_tokens-self.count_tokens(messages, tools=tools)

        @property
        def model_info(self):
            return dict(vision=False, function_calling=False, json_output=False,
                        structured_output=False, family='unknown')

        @property
        def capabilities(self):
            return self.model_info

    return LocalClient


async def official_fixed(backend, question, message, system, stage, limit):
    from autogen_core import CancellationToken
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.messages import TextMessage
    client = client_class()(backend, limit)
    agent = AssistantAgent('B', model_client=client, system_message=system)
    incoming = [TextMessage(source='user', content=question)]
    if message is not None:
        incoming.append(TextMessage(source='A', content=message))
    incoming.append(TextMessage(source='controller', content=stage))
    response = await agent.on_messages(incoming, CancellationToken())
    return dict(runtime='official_autogen_agentchat_0.7.5_direct_on_messages',
                incoming=[m.model_dump(mode='json') for m in incoming],
                output=response.chat_message.model_dump(mode='json'), turns=client.trace,
                context=[m.model_dump(mode='json') for m in await agent.model_context.get_messages()])


async def official_team(backend, question, prompts, limit, *, prefix_sources=True):
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from .autogen_style_reference import repetition_report
    clients = {n: client_class()(backend, limit, prefix_sources=prefix_sources) for n in ('A', 'B')}
    agents = {n: AssistantAgent(n, model_client=clients[n], system_message=prompts[n]) for n in ('A', 'B')}
    team = RoundRobinGroupChat(list(agents.values()), max_turns=5)
    result = await team.run(task=question)
    # The scheduling order is fixed; client traces are per-agent, not global.
    turns = []
    for i, name in enumerate(('A', 'B', 'A', 'B', 'A')):
        row = dict(clients[name].trace[i//2], agent=name)
        turns.append(row)
    return dict(runtime='official_autogen_agentchat_0.7.5_round_robin',
                turns=turns, result=result.model_dump(mode='json') if hasattr(result,'model_dump') else
                dict(messages=[m.model_dump(mode='json') for m in result.messages], stop_reason=result.stop_reason),
                final_contexts={n: [m.model_dump(mode='json') for m in await a.model_context.get_messages()]
                                for n, a in agents.items()}, repetition=repetition_report(turns),
                stage_policy='no_per_turn_stage; role_system_only; five_agent_turns; final_speaker_A',
                prefix_sources=prefix_sources)
