import time
import traceback

import pandas as pd
from pyquery import PyQuery as pq

import re

from tqdm import tqdm

import session


def read_html():
    with open('../test.html', encoding='utf-8') as f:
        return f.read()


def resolve_xxfpgl():
    """
    解析销项发票管理
    """
    url = "http://b.dimix.net.cn:226/gonghui_4737/dimix/jxc/cwgl/fpgl/xxfpgl.jsp?cdmc=%CF%FA%CF%EE%B7%A2%C6%B1%B9%DC%C0%ED&cddm=09050302&rFlag=&KHID=&khall=&queryWhere=0&moreQueryConditions_height=25&DYYWDJH=&FPRQ_S=&FPRQ_E=&FPDM=&FPHM=&djztmc=%C8%AB%B2%BF&djzt=-1&SHZT=-1&FPJE=&KPDW=&JLSJ_S=&JLSJ_E=&czytextarea=&CZY=&QSZT=-1&kprtextarea=&KPR=&BZ=&KHBH=&DDHTH=&DWTT=-1&FPLXDM=-1&YLZDINPUT9=&CPPHINPUT=&YLZDINPUT6=&YLZDINPUT6_E=&YLZDINPUT7=&YLZDINPUT7_E=&YLZDINPUT8=&YLZDINPUT8_E=&rowsPerPage=9999&jumpPage=1"
    resp = session.get(url)

    html = resp.text.replace("\n", "").replace("\r", "")
    doc = pq(html)
    rows = doc("#pageTableView tr")

    datas = []
    for i, tr in enumerate(rows.items()):
        if i == 0:  # 表头不要
            continue

        href = tr("div a:first-child").attr("href")
        if href is None:  # 最后还有个合计，跳过
            continue
        # 查看用的id
        id = re.search(r"ID=(\d+)", href).group(1)
        # 单位抬头
        dwtt = tr("td")[1].text.strip()
        # 发票号码
        fphm = tr("td")[2].text.strip()

        datas.append({
            "id": id,
            "单位抬头": dwtt,
            "fphm": fphm
        })

    return datas


def resolve_xxfpmx(id):
    """
    查看销项发票(明细)
    """
    url = "http://b.dimix.net.cn:226/gonghui_4737/dimix/jxc/cwgl/fpgl/xxfpgl_mx.jsp?urls=/gonghui_4737/dimix/jxc/cwgl/fpgl/xxfpgl.jsp&ID={id}&cdmc=%B2%E9%BF%B4%CF%FA%CF%EE%B7%A2%C6%B1&lcsp_bzlx=xxfp&lcsp_djid={id}".format(
        id=id)
    form_data = {
        "cdmc": "%CF%FA%CF%EE%B7%A2%C6%B1%B9%DC%C0%ED",
        "cddm": "09050302",
        "rFlag": "",
        "KHID": "",
        "khall": "",
        "queryWhere": "0",
        "moreQueryConditions_height": "25",
        "DYYWDJH": "",
        "FPRQ_S": "",
        "FPRQ_E": "",
        "FPDM": "",
        "FPHM": "",
        "djztmc": "%C8%AB%B2%BF",
        "djzt": "-1",
        "SHZT": "-1",
        "FPJE": "",
        "KPDW": "",
        "JLSJ_S": "",
        "JLSJ_E": "",
        "czytextarea": "",
        "CZY": "",
        "QSZT": "-1",
        "kprtextarea": "",
        "KPR": "",
        "BZ": "",
        "KHBH": "",
        "DDHTH": "",
        "DWTT": "-1",
        "FPLXDM": "-1",
        "YLZDINPUT9": "",
        "CPPHINPUT": "",
        "YLZDINPUT6": "",
        "YLZDINPUT6_E": "",
        "YLZDINPUT7": "",
        "YLZDINPUT7_E": "",
        "YLZDINPUT8": "",
        "YLZDINPUT8_E": "",
        "rowsPerPage": "12",
        "jumpPage": "1",
    }
    resp = session.form_post(url, form_data)

    html = resp.text.replace("\n", "").replace("\r", "")
    doc = pq(html)
    rows = doc("#pageTableView tr")
    for i, tr in enumerate(rows.items()):
        if i == 1:  # 表头不要
            # 客户单位
            khdw = tr("td")[1].text.strip()

        if i == 2:
            # 发票号码
            fphm = tr("td")[3].text.strip()

        pass

    # 单据号
    djh_pq = doc("#pageTableView tr a")
    djh_id = None
    djh_list = []
    for i, djh_a in enumerate(djh_pq.items()):
        djh_list.append(djh_a.text())
        if djh_a.attr("onclick") is not None and djh_id is None:
            djh_id = djh_a.attr("onclick").split(",")[-1].replace("'", "").replace(")", "")

    datas = {
        "客户单位": khdw,
        "发票号码": fphm,
        "业务单据号": ','.join(djh_list)
    }

    time.sleep(0.3)

    djh_datas = resovle_djhmx(djh_id)

    datas.update(djh_datas)

    return datas


def resovle_djhmx(id):
    """
    点击单据号后，进入的那个明细
    """
    url = "http://b.dimix.net.cn:226/gonghui_4737/dimix/jxc/xsgl/xskd_ckmx.jsp?KDID={id}".format(id=id)
    form_data = {
        "cddm": "09050302",
        "cdmc": "%B2%E9%BF%B4%CF%FA%CF%EE%B7%A2%C6%B1",
        "BHYY": ""
    }
    resp = session.form_post(url, form_data)

    html = resp.text.replace("\n", "").replace("\r", "")
    doc = pq(html)
    rows = doc("#pageTableView tr")

    for i, tr in enumerate(rows.items()):
        if i == 1:
            # 联系人
            lxr = tr("td")[3].text.strip()
            lxrdh = tr("td")[5].text.strip()

        if i == 2:
            # 联系人手机
            lxrsj = tr("td")[1].text.strip()
            ywy = tr("td")[7].text.strip()

    if lxrsj == "" and lxrdh != "":
        lxrsj = lxrdh

    return {
        "联系人": lxr,
        "联系人手机": lxrsj,
        "业务员": ywy
    }


def main():
    # 获取“销项发票管理”中的所有数据
    xxfp_datas = resolve_xxfpgl()
    print("共", len(xxfp_datas), "条数据，开始导出:")
    errors = []
    for item in tqdm(xxfp_datas, desc="正在导出"):
        time.sleep(0.3)
        try:
            mx_datas = resolve_xxfpmx(item['id'])
            item.update(mx_datas)
        except:
            traceback.print_exc()
            errors.append(item['fphm'])

    df = pd.DataFrame(xxfp_datas)
    filename = "联系人导出.xlsx"
    writer = pd.ExcelWriter(filename)
    df.to_excel(writer, index=False,
                columns=['发票号码', '单位抬头', '客户单位', '联系人', '联系人手机', '业务员', '业务单据号'])
    writer.close()
    print("导出成功，文件为：", filename)

    if len(errors) > 0:
        print("部分导出异常，请手动处理！异常发票号码：", errors)

if __name__ == '__main__':
    main()
