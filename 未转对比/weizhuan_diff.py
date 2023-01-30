import sys, os
sys.path.insert(1, os.path.abspath(".."))

import pandas as pd
from utils import fill_merge_cells, columns_strip, merge_cell_and_export


def not_transfer_diff():
    df1 = pd.read_excel('weizhuan1.xlsx', header=0, dtype=str)
    df2 = pd.read_excel('weizhuan2.xlsx', header=0, dtype=str)

    columns_strip(df1)
    columns_strip(df2)

    columns = ['发票号码', '购方企业名称', '开票日期', '商品名称', '单位', '数量',
               '单价', '金额', '税率', '税额', '价税合计', '联系人', '电话', '接单人', '收款情况']

    df1 = df1[columns]
    df2 = df2[columns]

    exclude_columns = ['商品名称', '单位', '数量', '单价', '金额', '税率', '税额']
    df1 = fill_merge_cells(df1, '发票号码', exclude_columns)
    df2 = fill_merge_cells(df2, '发票号码', exclude_columns)

    id_list1 = set(df1['发票号码'])
    id_list2 = set(df2['发票号码'])

    df1 = df1[df1['发票号码'].isin(list(id_list1 - id_list2))]
    df2 = df2[df2['发票号码'].isin(list(id_list2 - id_list1))]

    df1.insert(loc=len(df1.columns), column='备注', value='weizhuan1有，weizhuan2没')
    df2.insert(loc=len(df2.columns), column='备注', value='weizhuan1没，weizhuan2有')

    df = pd.concat([df1, df2])

    merge_cell_and_export(df, 'weizhuan_chayi.xlsx', '发票号码', exclude_columns)


if __name__ == '__main__':
    not_transfer_diff()
    print("对比完成，文件：weizhuan_chayi.xlsx")