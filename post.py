import re
import toml
import requests
import argparse

"""
post article to qiita
https://qiita.com/iimuz/items/4837e9c8043ce7c1262b

markdown format

+++
title = "テスト投稿"
draft = true
tags = ["test"]

[qiita]
coediting = false
gist = false
tweet = false
id = ""
+++

hogehoge

"""


def load_file(filepath):
    # ファイル内容の読み込み
    with open(filepath) as f:
        buf = f.read()

    # ヘッダ部分と投稿する内容を分離します。
    # ここでは、 '+++' がヘッダ部分の範囲を指定すると決め打ちしています。
    # また、 hugo の関係上ヘッダが toml で記載されているので、
    # toml パーサで内容を読み込んでいます。
    header = re.match(
        r'^\+\+\+$.+?^\+\+\+$',
        buf,
        flags=(re.MULTILINE | re.DOTALL))
    body = buf[header.end() + 1:]
    header = buf[header.start() + 4:header.end() - 4]
    header = toml.loads(header)

    # ヘッダ情報から Qiita へ投稿するヘッダ情報へ修正します
    item = {
        'title': header['title'],
        'private': header['draft'],
        'tags': [{'name': tag} for tag in header['tags']],
        'coediting': header['qiita']['coediting'],
        'gist': header['qiita']['coediting'],
        'tweet': header['qiita']['tweet'],
        'id': header['qiita']['id'] if 'id' in header['qiita'] else '',
    }

    # 投稿内容を格納
    item['body'] = body

    return item


def post(item):
    url = 'https://qiita.com/api/v2/items'
    with open('token') as f:
        token = f.read()
        token = token.replace("\n", "")
    headers = {'Authorization': 'Bearer {}'.format(token)}

    res = requests.post(url, headers=headers, json=item)

    return res


def add_qiita_id(filepath, item, res):
    item['id'] = res.json()['id']
    body = item.pop('body')
    data = '+++\n{}+++\n{}'.format(toml.dumps(item), body)

    with open(filepath, mode='w') as f:
        f.write(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', type=str)
    args = parser.parse_args()
    filepath = args.filepath
    item = load_file(filepath)
    res = post(item)
    print(res.json())
