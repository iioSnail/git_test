import pandas as pd
import sys, os
sys.path.insert(1, os.path.abspath("../../invoice_helper"))

from utils import fill_merge_cells, columns_strip, merge_cell_and_export, fuzzy_compare_str

daifang = '贷方'
zhanghu = '账户'
jizhang_date = '记账日期'

def find_row(ys_df, price, name):
    rows = []
    for index, row in ys_df.iterrows():
        if price != float(row['价税合计']):
            continue

        if not fuzzy_compare_str(name, row['购方企业名称']):
            continue

        rows.append(row)

    return rows


def liushui_duibi_export():
    ls_df = pd.read_excel("liushui.xlsx")
    excel = pd.read_excel("yingshou.xlsx", sheet_name=None)
    columns = ['发票代码', '发票号码', '开票日期', '购方企业名称', '价税合计', '收款情况']

    result_df_list = []
    index_column = '发票号码'

    for key, df in excel.items():
        df.columns = list(df.iloc[4])
        columns_strip(df)
        if df.columns[0] != '发票代码':
            print("错误：sheet页“%s”第6行第一个单元格不是发票代码，请确保表头在第6行!" % key)

        df = df[5:]

        df = df[columns]

        # 找出需要删除的列
        drop_idx = []
        for i, value in enumerate(list(df['发票代码'])):
            if "发票类别" in str(value):
                drop_idx.append(i + 5)

            if '发票代码' in str(value):
                drop_idx.append(i + 5)

        df = df.drop(drop_idx)

        # 填补缺失项（合并单元格）
        df = fill_merge_cells(df, index_column)

        df = df[df['收款情况'] == '未转']

        result_df_list.append(df)

        print('“%s”处理成功.' % key)

    df = pd.concat(result_df_list)

    ys_df = df[['发票号码', '开票日期', '购方企业名称', '价税合计']]

    result_df = pd.DataFrame(columns=['序号', jizhang_date, daifang, zhanghu, '发票号码', '开票日期', '购方企业名称', '价税合计'])

    i = 1
    for index, row in ls_df.iterrows():
        if pd.isnull(row[daifang]) or pd.isnull(row[zhanghu]):
            continue

        rows = find_row(ys_df, row[daifang], row[zhanghu])

        if len(rows) <= 0:
            continue

        row['序号'] = i

        for item in rows:
            result_df = result_df.append(pd.concat([row, item]), ignore_index=True)

        i += 1

    # 调整顺序
    result_df = result_df[['序号', jizhang_date, daifang, '价税合计', zhanghu, '购方企业名称', '发票号码', '开票日期']]
    result_df = result_df.drop_duplicates()

    merge_cell_and_export(result_df, "liushui_duibi.xlsx", '序号',
                          exclude_columns=['价税合计', '购方企业名称', '发票号码', '开票日期'])
    print("对比完成，文件：liushui_duibi.xlsx")


if __name__ == '__main__':
    liushui_duibi_export()
