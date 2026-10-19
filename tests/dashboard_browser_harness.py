"""Local browser acceptance harness, never mounted in the production API.

Serves the production Vite build, forwards real REST/WS, and exposes a local WS
fault control. --fixture serves explicitly labelled JSON snapshots for rare UI
states; no inference is claimed. Run only on loopback.
"""
from __future__ import annotations
import argparse
import asyncio
from pathlib import Path
import json
import httpx
import uvicorn
import websockets
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles


def create_harness(fixture: Path | None = None, backend_url: str = "http://127.0.0.1:8000"):
    backend_url = backend_url.rstrip("/")
    app = FastAPI()
    sockets: set[WebSocket] = set()
    stats = {'posts': [], 'ws_connections': 0, 'forwarded_event_types': [], 'drops': 0}
    def snapshot(): return json.loads(fixture.read_text()) if fixture else None

    @app.get('/__test/state')
    def state(): return {**stats, 'active_sockets': len(sockets), 'fixture': bool(fixture)}

    @app.post('/__test/drop_ws')
    async def drop():
        stats['drops'] += 1
        for ws in tuple(sockets):
            try: await ws.close(code=1012)
            except RuntimeError: pass
        return {'dropped': True}

    @app.api_route('/api/{path:path}', methods=['GET','POST','PATCH'])
    async def api(path: str, request: Request):
        if request.method == 'POST': stats['posts'].append(path)
        if fixture:
            data = snapshot()
            if path == 'discussions/latest': return data
            if path == 'discussions/prompts':
                from token_kv_agent_system.discussion_prompts import resolve_prompts
                return data['session']['prompt_config'] if data.get('session') else resolve_prompts().to_dict()
            if path == 'events': return {'events': [], 'count': 0}
            if path.startswith('discussions/'): return data['session']
            return Response('Fixture is read-only', status_code=405)
        async with httpx.AsyncClient(trust_env=False, timeout=20) as client:
            r = await client.request(request.method, backend_url + '/api/' + path,
                params=request.query_params, content=await request.body(), headers={'Content-Type': 'application/json'})
            return Response(r.content, status_code=r.status_code, media_type='application/json')

    @app.websocket('/ws/events')
    async def events(ws: WebSocket):
        await ws.accept()
        sockets.add(ws)
        stats['ws_connections'] += 1
        try:
            if fixture:
                while True: await ws.receive_text()
            else:
                async with websockets.connect(backend_url.replace('http://', 'ws://').replace('https://', 'wss://') + '/ws/events', proxy=None) as upstream:
                    async def downstream():
                        async for raw in upstream:
                            e = json.loads(raw)
                            stats['forwarded_event_types'].append(e.get('type'))
                            await ws.send_text(raw)
                    async def wait_close():
                        while True: await ws.receive_text()
                    tasks = [asyncio.create_task(downstream()), asyncio.create_task(wait_close())]
                    try:
                        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                        for task in pending: task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                    finally:
                        for task in tasks: task.cancel()
        except Exception:
            pass  # Intentional local fault injection closes both sides.
        finally:
            sockets.discard(ws)

    app.mount('/', StaticFiles(directory=Path(__file__).resolve().parents[1] / 'frontend/dist', html=True), name='frontend')
    return app

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=5174)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--backend-url', default='http://127.0.0.1:8000')
    args = parser.parse_args()
    uvicorn.run(create_harness(args.fixture, args.backend_url), host='127.0.0.1', port=args.port)
