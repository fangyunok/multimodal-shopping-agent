from io import BytesIO
import json
import threading
import uuid

from fastapi.testclient import TestClient
from PIL import Image
import pytest
from shopping_agent import app as module
from shopping_agent.session_store import InMemorySessionStore


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv('AUTH_MODE', 'demo')
    monkeypatch.setenv('REASONER_BACKEND', 'rule')
    monkeypatch.setenv('RETRIEVER_BACKEND', 'baseline')
    monkeypatch.setenv('INVENTORY_BACKEND', 'catalog')
    module.get_tools.cache_clear()
    module.get_planner.cache_clear()
    monkeypatch.setattr(module, 'get_session_store', lambda: store)
    store = InMemorySessionStore()
    with TestClient(module.app) as api:
        yield api
    module.get_tools.cache_clear()
    module.get_planner.cache_clear()


def send(client, session='ui', query='运动鞋', **kwargs):
    r = client.post('/chat', json=dict(session_id=session, query=query, **kwargs))
    assert r.status_code == 200, r.text
    return r.json()


def test_cards_targeted_inventory_compare_and_history(client):
    first = send(client, max_price=500)
    assert {p['id'] for p in first['cards']} == {'shoe-001', 'shoe-002'}
    assert all(p['stock'] is not None and p['price'] <= 500 for p in first['cards'])
    inventory = send(client, query='查 shoe-002 库存')
    assert inventory['cards'][0]['id'] == 'shoe-002'
    assert inventory['cards'][0]['stock'] == 8
    comparison = send(client, query='对比', intent='compare', product_ids=['shoe-002','shoe-001'])
    assert [p['id'] for p in comparison['cards']] == ['shoe-002','shoe-001']
    history = client.get('/chat/history', params={'session_id':'ui'}).json()
    assert len(history['messages']) == 6
    assert history['messages'][-1]['reply'] == comparison
    assert history['constraints']['max_price'] == 500


def test_clear_constraints_and_exclude(client):
    send(client, max_price=500, category='运动鞋')
    excluded = send(client, query='排除 shoe-001，推荐其他商品')
    assert 'shoe-001' not in [p['id'] for p in excluded['cards']]
    cleared = send(client, query='背包', max_price=None, category=None)
    assert cleared['constraints']['max_price'] is None
    assert cleared['constraints']['category'] is None
    assert cleared['constraints']['excluded_product_ids'] == ['shoe-001']


def test_stream_observations_and_committed_result(client):
    r = client.post('/chat/stream', json={'session_id':'stream','query':'白色运动鞋'})
    assert r.status_code == 200
    frames = r.text.split('\n\n')
    assert any('event: progress' in frame and 'start' in frame for frame in frames)
    assert any('event: progress' in frame and 'end' in frame for frame in frames)
    results = [json.loads(frame.split('data: ',1)[1]) for frame in frames if frame.startswith('event: result')]
    assert len(results) == 1
    assert client.get('/chat/history?session_id=stream').json()['messages'][-1]['reply'] == results[0]
    assert '/tmp/' not in r.text


def test_cancel_rolls_back(client):
    flag = threading.Event()
    def progress(data):
        flag.set()
    request = module.ChatRequest(session_id='cancel', query='运动鞋')
    with pytest.raises(InterruptedError):
        module._chat(request, 'demo', progress, flag)
    assert client.get('/chat/history?session_id=cancel').json()['messages'] == []


def test_stock_failure_is_unknown(client, monkeypatch):
    tools = module.get_tools()
    def fail(*args, **kwargs):
        raise RuntimeError('secret provider error')
    monkeypatch.setattr(tools, 'check_inventory', fail)
    reply = send(client)
    assert reply['cards']
    assert all(p['stock'] is None for p in reply['cards'])
    assert 'secret' not in json.dumps(reply)


def png():
    content = BytesIO()
    Image.new('RGB', (8,8), 'white').save(content, format='PNG')
    return ('reference.png', content.getvalue(), 'image/png')


def test_image_followup_and_clear_filters(client):
    send(client, max_price=100)
    r = client.post('/chat/image', data={'session_id':'ui','query':'运动鞋','filters':'{"max_price":null}'}, files={'image':png()})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['cards']
    assert body['constraints']['max_price'] is None
    assert '仅按文字检索' in body['answer']
    assert '/tmp/' not in r.text
    followup = send(client, query='有货吗')
    assert followup['cards'][0]['id'] == body['cards'][0]['id']


def test_bad_images_and_filters(client):
    assert client.post('/chat/image', data={'session_id':'x'}, files={'image':png()}).status_code == 422
    assert client.post('/chat/image', data={'session_id':'x','query':'鞋'}, files={'image':('x.png',b'bad','image/png')}).status_code == 422
    assert client.post('/chat/image', data={'session_id':'x','query':'鞋','filters':'{"max_price":-1}'}, files={'image':png()}).status_code == 422
    assert client.get('/chat/history?session_id=x').json()['messages'] == []


def test_history_isolation_and_public_assets(client, monkeypatch):
    monkeypatch.setenv('AUTH_MODE','api_key')
    monkeypatch.setenv('API_KEYS_JSON',json.dumps({'a':'a'*32,'b':'b'*32}))
    assert client.get('/').status_code == 200
    assert client.get('/ui/config').status_code == 401
    assert client.get('/chat/history?session_id=same').status_code == 401
    assert client.post('/chat',headers={'Authorization':'Bearer '+'a'*32},json={'session_id':'same','query':'运动鞋'}).status_code == 200
    assert client.get('/chat/history?session_id=same',headers={'Authorization':'Bearer '+'b'*32}).json()['messages'] == []
    assert len(client.get('/chat/history?session_id=same',headers={'Authorization':'Bearer '+'a'*32}).json()['messages']) == 2


def test_ui_history_bounded(client):
    for _ in range(22):
        send(client, query='shoe-001 库存')
    assert len(client.get('/chat/history?session_id=ui').json()['messages']) == 40


def test_corrupt_png_checksum_is_validation_error(client):
    import base64
    damaged = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aD1sAAAAASUVORK5CYII=')
    r = client.post('/chat/image', data={'session_id':'broken','query':'鞋'},
                    files={'image':('broken.png', damaged, 'image/png')})
    assert r.status_code == 422
    assert client.get('/chat/history?session_id=broken').json()['messages'] == []
