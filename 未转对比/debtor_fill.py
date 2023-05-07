"""
根据发票号填充联系人、联系电话和接单人
"""
import pandas as pd

from utils import fill_merge_cells, is_null, merge_cell_and_export


def read_debtor_datas():
    df = pd.read_excel("联系人导出.xlsx")

    datas = {}
    for i, row in df.iterrows():
        datas[row['发票号码']] = {
            'lxr': row['联系人'],
            'lxrsj': row['联系人手机'],
            'ywy': row['业务员'],
        }

    return datas


def fill_excel():
    datas = read_debtor_datas()
    df = pd.read_excel("weishoukuan.xlsx", header=1, dtype=str)
    df = fill_merge_cells(df, '发票号码',
                          exclude_columns=['商品名称', '规格', '单位', '数量', '单价', '金额', '税率', '税收编码'])

    for i, row in df.iterrows():
        fphm = row['发票号码']
        if fphm not in datas:
            continue

        data = datas[fphm]
        if is_null(row['联系人']):
            row['联系人'] = data['lxr']

        if is_null(row['电话']):
            row['电话'] = data['lxrsj']

        if is_null(row['接单人']):
            row['接单人'] = data['ywy']

    filename = "weishoukuan_final.xlsx"
    merge_cell_and_export(df, filename, index_column='发票号码')
    print("生成完毕，文件名：", filename)

if __name__ == '__main__':
    #
    fill_excel()
    pass
