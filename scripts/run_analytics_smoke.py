#!/usr/bin/env python3
import json
import sys

endpoints = [
    '/api/analytics/accuracy-metrics',
    '/api/analytics/top-patterns',
    '/api/analytics/volume-analysis',
    '/api/analytics/risk-trend',
    '/api/analytics/alert-timeline',
    '/api/analytics/hourly-heatmap'
]

base = 'http://127.0.0.1:5000'

try:
    import requests
except Exception:
    requests = None


def fetch_with_requests(url):
    try:
        r = requests.get(url, timeout=5)
        ct = r.headers.get('Content-Type', '')
        if 'application/json' in ct:
            return r.status_code, r.json()
        else:
            # try to parse as json anyway
            try:
                return r.status_code, r.json()
            except Exception:
                return r.status_code, {'_raw_text': r.text}
    except Exception as e:
        return None, {'error': str(e)}


def fetch_with_urllib(url):
    try:
        from urllib.request import urlopen, Request
        req = Request(url, headers={'User-Agent': 'smoketest-agent'})
        with urlopen(req, timeout=5) as r:
            raw = r.read().decode('utf-8')
            # try json
            try:
                return r.status, json.loads(raw)
            except Exception:
                return r.status, {'_raw_text': raw}
    except Exception as e:
        return None, {'error': str(e)}


if __name__ == '__main__':
    print('Running analytics smoke tests against', base)
    fetch = fetch_with_requests if requests else fetch_with_urllib

    results = {}
    for ep in endpoints:
        url = base + ep
        print('\n---')
        print('Requesting', url)
        status, body = fetch(url)
        results[ep] = {'status': status, 'body': body}
        print('Status:', status)
        try:
            print(json.dumps(body, indent=2, ensure_ascii=False))
        except Exception:
            print(body)

    # Dump summary
    print('\n=== SUMMARY JSON ===')
    print(json.dumps(results, indent=2, ensure_ascii=False))

