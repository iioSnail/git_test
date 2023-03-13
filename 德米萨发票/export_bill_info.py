import time

import pandas as pd
from pyquery import PyQuery as pq
from pandas import DataFrame
import requests
from tqdm import tqdm


def get_session():
    # FIXME
    return "0A75C9746E644B5AC6A30FD05EFCC650"


def get_cookies():
    cookies = {
        "JSESSIONID": get_session(),
        "username": "zxx"
    }

    return cookies


def request_bill_table():
    url = "http://b.dimix.net.cn:226/gonghui_4737/dimix/jxc/cwgl/fpgl/dkfpcx.jsp?cdmc=&cddm=09050313&rFlag=&queryWhere=1&KHID=&khall=&moreQueryConditions_height=67&KHMC=&KDDH=&KPRQ_S=&KPRQ_E=&FBZT=-1&ZXZT=-1&KHBH=&ywytextarea=&YWY=&DDDH=&BDQK_S=&BDQK_E=&DWTT=38&SFTZKP=1&FPLXDM=-1&DDHTH=&rowsPerPage=9999&jumpPage=1"

    resp = requests.get(url, cookies=get_cookies())
    html = resp.text.replace("\n", "").replace("\r", "")
    doc = pq(html)
    rows = doc("#pageTableView tr")

    # 解析html
    columns = []
    datas = []
    for i, tr in enumerate(rows.items()):
        if i == 0:
            columns = tr.text().split("\n")
            columns.insert(0, "id")
            continue

        data = []

        if len(tr("td:first a")) <= 0:
            continue

        data.append(tr("td:first a").attr('onclick').split(",")[1])
        for td in tr("td").items():
            data.append(td.text())

        if len(data) < len(columns):
            continue

        datas.append(data)

    df = DataFrame(datas)
    df.columns = columns

    df['通知内容'] = ''

    return df


def request_bill_detail(id, danhao):
    url = "http://b.dimix.net.cn:226/gonghui_4737/dimix/jxc/xsgl/xskd_ckmx.jsp?urls=/gonghui_4737/dimix/jxc/cwgl/fpgl/dkfpcx.jsp&KDID={id}&{danhao}&dkfp=yes".format(
        id=id, danhao=danhao)

    resp = requests.get(url, cookies=get_cookies())
    html = resp.text.replace("\n", "").replace("\r", "")
    doc = pq(html)
    rows = doc("#tablePageView tr")

    # 解析html
    columns = []
    datas = []
    for i, tr in enumerate(rows.items()):
        if i == 0:
            columns = tr.text().split("\n")
            columns.insert(0, "id")
            columns.insert(1, "单号")
            continue

        data = []
        data.append(str(id))
        data.append(danhao)
        for td in tr("td").items():
            data.append(td.text())

        datas.append(data)
    datas = datas[:-2]
    df = DataFrame(datas, columns=columns)

    df["数量"] = df['数量'].str.replace(r"\【.*?\】", "", regex=True)

    return df


def request_bill_content(id):
    url = "http://b.dimix.net.cn:226/gonghui_4737/dimix/jxc/xsgl/tzkp.jsp?id={id}&flag=ck".format(id=id)
    resp = requests.get(url, cookies=get_cookies())

    html = resp.text
    doc = pq(html)
    text = doc("textarea")

    return text.text()


if __name__ == '__main__':
    df = request_bill_table()

    df_details = []
    for i, row in tqdm(df.iterrows(), desc="导出中", total=len(df)):
        df_detail = request_bill_detail(row['id'], row['单号'])

        df_details.append(df_detail)

        content = request_bill_content(row['id'])

        df.loc[i, '通知内容'] = content

        time.sleep(1)

    df_details = pd.concat(df_details)

    filename = '发票明细.xlsx'
    writer = pd.ExcelWriter(filename)
    df.to_excel(writer, '普票', index=False,
                columns=['id', '单号', '客户', '业务员', '总金额', '通知内容'])
    df_details.to_excel(writer, "明细", index=False, columns=['id', '单号', '产品名称', '产品型号', '单位', '数量', '含税单价', '合计'])

    writer.save()

    print("导出成功，文件为：", filename)
