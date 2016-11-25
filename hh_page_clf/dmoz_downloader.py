import csv
import gzip
import random

import scrapy


class Spider(scrapy.Spider):
    name = __name__
    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Ubuntu Chromium/43.0.2357.130 '
                      'Chrome/43.0.2357.130 Safari/537.36',
        'CONCURRENT_REQUESTS': 64,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 10,
        'DOWNLOAD_DELAY': 0.0,
    }

    def __init__(self, filename):
        with gzip.open(filename, 'rt') as f:
            # self.start_urls = [url for url, _ in csv.reader(f)
            #                   if url.startswith('http')]
            self.start_urls = []
            try:
                for url, _ in csv.reader(f):
                    if url.startswith('http'):
                        self.start_urls.append(url)
            except Exception:
                pass
            print('{:,} start urls'.format(len(self.start_urls)))
            random.shuffle(self.start_urls)
        super().__init__()

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(
                url, dont_filter=True, meta={'original_url': url})

    def parse(self, response):
        if hasattr(response, 'text'):
            yield {
                'url': response.meta['original_url'],
                'final_url': response.url,
                'html': response.text,
            }
