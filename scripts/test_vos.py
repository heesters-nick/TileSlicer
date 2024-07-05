import time

import requests

urls = [
    'https://ws-cadc.canfar.net/vault/capabilities',
    'https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/reg/resource-caps',
    'https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca:443',
]


def test_url(url):
    print('testing url:', url)
    try:
        start = time.time()
        response = requests.get(url, timeout=2000000)
        print(f'Got response in {time.time() - start:.2f}s')
        response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code
        print('Successfully accessed the URL')
        print('Content:', response.text)
    except requests.exceptions.RequestException as e:
        print(f'Failed to access the URL: {e}')


for url in urls:
    test_url(url)
