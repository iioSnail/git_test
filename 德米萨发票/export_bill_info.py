import sys, os
sys.path.insert(1, os.path.abspath(".."))

import session
import time

import pandas as pd
from pyquery import PyQuery as pq
from pandas import DataFrame
from tqdm import tqdm

tt_map = {
    "38": "工惠",
    "39": "雪贝",
    "40": "工运",
    "41": "智多",
    "42": "易华",
    "43": "康悠",
    "44": "易悠",
    "45": "缤乐",
    "46": "诚光",
}

def request_bill_table(DWTT):
    """
    DWTT: 单位抬头
    上海工惠文化传媒有限公司: 38
    上海雪贝医疗科技中心: 39
    上海工运实业发展有限公司: 40
    上海智多邦实业发展有限公司: 41
    上海易华文化传媒中心: 42
    上海康悠商贸中心: 43
    上海易悠生物科技中心: 44
    上海缤乐实业发展有限公司: 45
    上海诚光教育培训有限公司: 46
    中犇国际旅行社（上海）有限公司: 47
    """

    url = "http://b.dimix.net.cn:226/gonghui_4737/dimix/jxc/cwgl/fpgl/dkfpcx.jsp?cdmc=&cddm=09050313&rFlag=&queryWhere=1&KHID=&khall=&moreQueryConditions_height=67&KHMC=&KDDH=&KPRQ_S=&KPRQ_E=&FBZT=-1&ZXZT=-1&KHBH=&ywytextarea=&YWY=&DDDH=&BDQK_S=&BDQK_E=&DWTT={DWTT}&SFTZKP=1&FPLXDM=-1&DDHTH=&rowsPerPage=9999&jumpPage=1".format(
        DWTT=DWTT)

    resp = session.get(url)
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

    if len(datas) <= 0:
        print(tt_map[DWTT], "无数据")
        return None

    df = DataFrame(datas)
    df.columns = columns

    df['通知内容'] = ''

    print(tt_map[DWTT], "%d条数据" % len(df))

    return df


def request_bill_detail(id, danhao):
    url = "http://b.dimix.net.cn:226/gonghui_4737/dimix/jxc/xsgl/xskd_ckmx.jsp?urls=/gonghui_4737/dimix/jxc/cwgl/fpgl/dkfpcx.jsp&KDID={id}&{danhao}&dkfp=yes".format(
        id=id, danhao=danhao)

    resp = session.get(url)
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
    resp = session.get(url)

    html = resp.text
    doc = pq(html)
    text = doc("textarea")

    return text.text()


if __name__ == '__main__':
    """
    DWTT: 单位抬头
    上海工惠文化传媒有限公司: 38
    上海雪贝医疗科技中心: 39
    上海工运实业发展有限公司: 40
    上海智多邦实业发展有限公司: 41
    上海易华文化传媒中心: 42
    上海康悠商贸中心: 43
    上海易悠生物科技中心: 44
    上海缤乐实业发展有限公司: 45
    上海诚光教育培训有限公司: 46
    中犇国际旅行社（上海）有限公司: 47
    """
    dwtt = sys.argv[1]

    if dwtt == "0":
        df_list = []
        for tt in ['38', '40', '42', '46']:
            df = request_bill_table(tt)
            if df is None:
                continue

            df_list.append(df)

        df = pd.concat(df_list)
    else:
        df = request_bill_table(dwtt)

    if df is None:
        exit()

    df_details = []
    for i, row in tqdm(df.iterrows(), desc="导出中", total=len(df)):
        df_detail = request_bill_detail(row['id'], row['单号'])

        df_details.append(df_detail)

        content = request_bill_content(row['id'])

        df.loc[i, '通知内容'] = content

        time.sleep(1)

    df_details = pd.concat(df_details)

    if dwtt == '38':
        filename = '发票明细(工惠).xlsx'
    elif dwtt == '40':
        filename = '发票明细(工运).xlsx'
    elif dwtt == '42':
        filename = '发票明细(易华).xlsx'
    elif dwtt == '46':
        filename = '发票明细(诚光).xlsx'
    else:
        filename = '发票明细(全部).xlsx'

    writer = pd.ExcelWriter(filename)
    df.to_excel(writer, '普票', index=False,
                columns=['id', '单号', '客户', '业务员', '总金额', '通知内容'])
    df_details.to_excel(writer, "明细", index=False, columns=['id', '单号', '产品名称', '产品型号', '单位', '数量', '含税单价', '合计'])

    writer.save()

    print("导出成功，文件为：", filename)
