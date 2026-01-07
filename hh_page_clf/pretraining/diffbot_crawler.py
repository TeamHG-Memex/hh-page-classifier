import gzip
import json
from urllib.parse import urlencode

import scrapy


class Spider(scrapy.Spider):
    name = __name__
    custom_settings = {
        'DOWNLOAD_DELAY': 1.2,
        'CONCURRENT_REQUESTS': 1,
    }
    base_api_url = 'http://api.diffbot.com/v3/analyze?'

    def __init__(self, pages, token):
        self.token = token
        with gzip.open(pages, 'rt') as f:
            pages = json.load(f)['pages']
            self.start_urls = list(filter(None, (
                self.api_url(page['url']) for page in pages)))
        super().__init__()

    def api_url(self, page_url):
        if '.onion' in page_url:
            return None
        return self.base_api_url + urlencode({
            'url': page_url,
            'token': self.token,
            'fallback': 'article',
        })

    def parse(self, response):
        return json.loads(response.text)
