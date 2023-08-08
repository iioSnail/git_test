import os
import time
from pathlib import Path
from urllib.parse import unquote

import requests
import urllib3
from pyquery import PyQuery as pq

site = "https://pinyin.sogou.com"


def get_title():
    doc = pq(url='https://pinyin.sogou.com/dict/', encodings='utf-8')

    titles = []
    for title in doc(".dict_category_list_title a"):
        titles.append({
            'title_name': title.text,
            'title_link': site + title.get("href")
        })

    return titles


def get_categories_by_title(title):
    categories = []

    title_link = title['title_link']
    doc = pq(url=title_link, encodings='utf-8')

    for cate in doc(".cate_words_list div.no_select a"):
        categories.append({
            'cate_name': cate.text,
            'cate_link': cate.get("href"),
        })

    return categories


def get_lexicons_by_category(cate, page=1):
    lexicons = []

    cate_link = site + cate['cate_link'] + "/default/%d" % page
    doc = pq(url=cate_link, encodings='utf-8')

    div_list = doc(".dict_detail_block")
    for div in doc(div_list):
        name = div.cssselect(".detail_title a")[0].text
        link = div.cssselect(".dict_dl_btn a")[0].get('href')

        lexicons.append({
            "name": name,
            "link": link
        })

    if len(div_list) >= 10:
        lexicons += get_lexicons_by_category(cate, page+1)

    return lexicons


def download_file(url, filename, path: Path):
    os.makedirs(path, exist_ok=True)

    file_name = path / filename

    if os.path.exists(file_name):
        print(filename + " 已下载！")
        return

    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful

        with open(file_name, "wb") as file:
            file.write(response.content)

        print(filename, "下载成功！")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: ", filename, "下载失败")
    except Exception as err:
        print(f"An error occurred: ", filename, "下载失败")

    time.sleep(0.5)


def main(output_path):
    output_path = Path(output_path)

    titles = get_title()
    for title in titles:
        categories = get_categories_by_title(title)

        for cate in categories:
            lexicons = get_lexicons_by_category(cate)

            for lexicon in lexicons:
                download_file(lexicon['link'],
                              lexicon['name'] + ".scel",
                              output_path / title['title_name'] / cate['cate_name'])


    print("全部下载完成！！！")


if __name__ == '__main__':
    main("output")
