# -*- coding:utf-8 -*-
"""
@author: zhanglun <zhanglun.me@gmail.com>
@github:  mrzhangboss
@date: 2017/06/09

"""
from io import StringIO
import requests
from lxml import etree

html_parser = etree.HTMLParser()


def create_start_urls(province, year, month):
    url = 'http://lishi.tianqi.com/{}/{}{:02}.html'.format(province, year, month)
    return url


def parse_ul(ul):
    data = ul.xpath('.//a/text()')
    weather = ul.xpath('.//li/text()')
    return data + weather


def parse_month(elements):
    result = []
    xpath = "//div[@class='tqtongji2']/ul"
    for ul in elements.xpath(xpath)[1:]:
        r = parse_ul(ul)
        result.append(r)
    return result


if __name__ == '__main__':
    province = 'yangzhong'
    filename = '{}.csv'.format(province)
    for year in range(2015, 2017):
        for month in range(1, 13):
            url = create_start_urls(province, year, month)
            res = requests.get(url)
            elements = etree.parse(StringIO(res.text), parser=html_parser)
            result = parse_month(elements)
            print(len(result), result)
            with open(filename, 'a') as f:
                f.write('\n'.join((','.join(x) for x in result)))
                f.write('\n')