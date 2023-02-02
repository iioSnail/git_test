import os
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import SubElement
from xml.etree.ElementTree import ElementTree

import numpy
import pandas as pd

import sys
sys.path.insert(1, os.path.abspath(".."))

from utils import df_strip, str_to_num


def process(val, round_num=2):
    if pd.isnull(val):
        return ""

    if type(val) == numpy.float64 and val % 1 == 0:
        return str(int(val))

    if type(val) == numpy.float64 and val % 1 != 0:
        return str(round(val, round_num))

    return str(val)


def electronic_invoice(info, details):
    business = Element('business', attrib={
        "id": "FPKJ",
        "comment": "发票开具"
    })

    fpkj = SubElement(business, 'REQUEST_COMMON_FPKJ', attrib={
        "class": "REQUEST_COMMON_FPKJ"
    })

    _build_info(info, fpkj, details)

    _build_detail(details, fpkj)

    tree = ElementTree(business)
    filename = "发票%s.xml" % process(info['发票序号'])
    tree.write(filename, encoding='GBK', xml_declaration=True)
    print("发票%s导出完毕~~，文件%s" % (process(info['发票序号']), filename))


def _build_info(info, fpkj, details):
    fpt = SubElement(fpkj, 'COMMON_FPKJ_FPT', attrib={
        "class": "COMMON_FPKJ_FPT"
    })

    FPQQLSH = SubElement(fpt, 'FPQQLSH')
    FPQQLSH.text = process(info['发票序号'])

    KPLX = SubElement(fpt, 'KPLX')
    KPLX.text = "0"

    BMB_BBH = SubElement(fpt, 'BMB_BBH')
    BMB_BBH.text = "42.0"

    XSF_NSRSBH = SubElement(fpt, 'XSF_NSRSBH')
    XSF_NSRSBH.text = "91310109586826634J"

    XSF_MC = SubElement(fpt, 'XSF_MC')
    XSF_MC.text = "上海工惠文化传媒有限公司"

    XSF_DZDH = SubElement(fpt, 'XSF_DZDH')
    XSF_DZDH.text = "虹口区中山北二路1800号海鸥大厦5楼  021-65533215"

    XSF_YHZH = SubElement(fpt, 'XSF_YHZH')
    XSF_YHZH.text = "上海农商银行虹口支行 32403108010107786"

    GMF_NSRSBH = SubElement(fpt, 'GMF_NSRSBH')
    GMF_NSRSBH.text = process(info['识别号'])

    GMF_MC = SubElement(fpt, 'GMF_MC')
    GMF_MC.text = process(info['购买方名称'])

    GMF_DZDH = SubElement(fpt, 'GMF_DZDH')
    GMF_DZDH.text = process(info['地址、电话'])

    GMF_YHZH = SubElement(fpt, 'GMF_YHZH')
    GMF_YHZH.text = process(info['开户行及账号'])

    KPR = SubElement(fpt, 'KPR')
    KPR.text = "王敬龙"

    SKR = SubElement(fpt, 'SKR')
    SKR.text = "刘少华"

    FHR = SubElement(fpt, 'FHR')
    FHR.text = "周幸"

    YFP_DM = SubElement(fpt, 'YFP_DM')
    YFP_DM.text = ""

    YFP_HM = SubElement(fpt, 'YFP_HM')
    YFP_HM.text = ""

    JSHJ = SubElement(fpt, 'JSHJ')
    JSHJ.text = process(details['金额（含税）'].sum())

    HJJE = SubElement(fpt, 'HJJE')
    HJJE.text = process(round(details['金额（含税）'] / (1 + details['税率']), 2).sum())

    HJSE = SubElement(fpt, 'HJSE')
    HJSE.text = process(round((details['金额（含税）'] / (1 + details['税率'])) * details['税率'], 2).sum())

    HSBZ = SubElement(fpt, 'HSBZ')
    HSBZ.text = "0"

    BZ = SubElement(fpt, 'BZ')
    BZ.text = process(info['备注'])


def _build_detail(details, fpkj):
    XMXXS = SubElement(fpkj, 'COMMON_FPKJ_XMXXS', attrib={
        "class": "COMMON_FPKJ_XMXX",
        "size": process(len(details))
    })

    for i in range(len(details)):
        detail = details.iloc[i]

        XMXX = SubElement(XMXXS, 'COMMON_FPKJ_XMXX')

        FPHXZ = SubElement(XMXX, 'FPHXZ')
        FPHXZ.text = "0"

        SPBM = SubElement(XMXX, 'SPBM')
        SPBM.text = process(detail['税收分类编码']).ljust(19, '0')

        ZXBM = SubElement(XMXX, 'ZXBM')
        ZXBM.text = ""

        YHZCBS = SubElement(XMXX, 'YHZCBS')
        YHZCBS.text = process(detail['使用优惠政策标识'])

        LSLBS = SubElement(XMXX, 'LSLBS')
        LSLBS.text = process(detail['零税率标识'])

        ZZSTSGL = SubElement(XMXX, 'ZZSTSGL')
        ZZSTSGL.text = ""

        XMMC = SubElement(XMXX, 'XMMC')
        XMMC.text = process(detail['商品名称'])

        GGXH = SubElement(XMXX, 'GGXH')
        GGXH.text = process(detail['规格型号'])

        DW = SubElement(XMXX, 'DW')
        DW.text = process(detail['计量单位'])

        XMSL = SubElement(XMXX, 'XMSL')
        XMSL.text = process(detail['数量'])

        XMDJ = SubElement(XMXX, 'XMDJ')
        XMDJ.text = process(detail['单价'] / (1 + detail['税率']) + 0.000001, round_num=6)

        XMJE = SubElement(XMXX, 'XMJE')
        XMJE.text = process(detail['金额（含税）'] / (1 + detail['税率']))

        SL = SubElement(XMXX, 'SL')
        SL.text = process(detail['税率'])

        SE = SubElement(XMXX, 'SE')
        SE.text = process((detail['金额（含税）'] / (1 + detail['税率'])) * detail['税率'])

        KCE = SubElement(XMXX, 'KCE')
        KCE.text = ""


def paper_invoice(info, details):
    Kp = Element('Kp')
    Version = SubElement(Kp, 'Version')
    Fpxx = SubElement(Kp, 'Fpxx')
    Zsl = SubElement(Fpxx, 'Zsl')
    Zsl.text = '1'
    Fpsj = SubElement(Fpxx, 'Fpsj')
    Fp = SubElement(Fpsj, 'Fp')

    # 发票信息
    Djh = SubElement(Fp, 'Djh')
    Djh.text = process(info['发票序号'])

    Gfmc = SubElement(Fp, 'Gfmc')
    Gfmc.text = process(info['购买方名称'])

    Gfsh = SubElement(Fp, 'Gfsh')
    Gfsh.text = process(info['识别号'])

    Gfyhzh = SubElement(Fp, 'Gfyhzh')
    Gfyhzh.text = process(info['开户行及账号'])

    Gfdzdh = SubElement(Fp, 'Gfdzdh')
    Gfdzdh.text = process(info['地址、电话'])

    Bz = SubElement(Fp, 'Bz')
    Bz.text = ''    # 备注

    Fhr = SubElement(Fp, 'Fhr')
    Fhr.text = "周幸"

    Skr = SubElement(Fp, 'Skr')
    Skr.text = "刘少华"

    Spbmbbh = SubElement(Fp, 'Spbmbbh')
    Spbmbbh.text = "1"

    Hsbz = SubElement(Fp, 'Hsbz')
    Hsbz.text = "1"

    Spxx = SubElement(Fp, 'Spxx')


    # 商品详情
    for j in range(len(details)):
        Sph = SubElement(Spxx, 'Sph')

        detail = details.iloc[j]

        Xh = SubElement(Sph, 'Xh')
        Xh.text = str(j + 1)

        Spmc = SubElement(Sph, 'Spmc')
        Spmc.text = process(detail['商品名称'])

        Ggxh = SubElement(Sph, 'Ggxh')
        Ggxh.text = process(detail['规格型号'])

        Jldw = SubElement(Sph, 'Jldw')
        Jldw.text = process(detail['计量单位'])

        Spbm = SubElement(Sph, 'Spbm')
        Spbm.text = process(detail['税收分类编码']).ljust(19, '0')

        Qyspbm = SubElement(Sph, 'Qyspbm')
        Qyspbm.text = ''

        Syyhzcbz = SubElement(Sph, 'Syyhzcbz')
        Syyhzcbz.text = '0'

        Lslbz = SubElement(Sph, 'Lslbz')
        Lslbz.text = ''

        Yhzcsm = SubElement(Sph, 'Yhzcsm')
        Yhzcsm.text = ''

        Dj = SubElement(Sph, 'Dj')
        Dj.text = process(detail['单价'] / (1 + detail['税率']) + 0.000001, round_num=6)

        Sl = SubElement(Sph, 'Sl')
        Sl.text = process(detail['数量'])

        Je = SubElement(Sph, 'Je')
        Je.text = process(detail['金额（含税）'] / (1 + detail['税率']))

        Slv = SubElement(Sph, 'Slv')
        Slv.text = process(detail['税率'])

        Kce = SubElement(Sph, 'Kce')
        Kce.text = '0'

        Se = SubElement(Sph, 'Se')
        Se.text = process((detail['金额（含税）'] / (1 + detail['税率'])) * detail['税率'])

    tree = ElementTree(Kp)
    filename = "发票%s(纸质).xml" % process(info['发票序号'])
    tree.write(filename, encoding='GBK', xml_declaration=True)
    print("发票%s(纸质)导出完毕~~，文件%s" % (process(info['发票序号']), filename))




if __name__ == '__main__':
    excel = pd.read_excel("dianzi.xlsx", sheet_name=None)

    infos = excel['电子普票']
    details_list = excel['电子明细']

    infos = df_strip(infos)
    details_list = df_strip(details_list)

    details_list['税率'] = details_list['税率'].map(lambda x: str_to_num(x))

    for i in infos['发票序号']:
        details = details_list[details_list['发票序号'] == i]
        info = infos[infos['发票序号'] == i].iloc[0]

        if pd.isnull(info['是否纸质']) or '' == info['是否纸质']:
            electronic_invoice(info, details)
        else:
            paper_invoice(info, details)
    print("全部导出完毕！")
