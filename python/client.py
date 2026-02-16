#!/usr/bin/env python3
import time
import requests

BASE = 'http://127.0.0.1:8799'

print('health', requests.get(BASE + '/health').json())
print('reset', requests.post(BASE + '/reset').json())
print('wait', requests.get(BASE + '/wait_ready').json())

for i in range(20):
    la = requests.get(BASE + '/legal_actions').json()
    acts = la.get('actions', [])
    print('legal', i, acts)
    # pick a simple policy
    if 'PLAY_FIRST_LAND' in acts:
        act = 'PLAY_FIRST_LAND'
    elif 'CAST_FIRST_SPELL' in acts:
        act = 'CAST_FIRST_SPELL'
    elif 'ATTACK_ALL' in acts:
        act = 'ATTACK_ALL'
    elif 'BLOCK_NONE' in acts:
        act = 'BLOCK_NONE'
    else:
        act = 'PASS'
    r = requests.post(BASE + '/step', json={'action': act}).json()
    print('step', act, r)
    time.sleep(0.2)
