# -*- coding: utf-8 -*-
import os
from scrapy import cmdline


def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    cmdline.execute("scrapy crawl androidcentral -o androidcentral.json".split())


if __name__ == '__main__':
    main()
