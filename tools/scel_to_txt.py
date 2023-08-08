import struct
import os

# 拼音表偏移，
from pathlib import Path

startPy = 0x1540

# 汉语词组表偏移
startChinese = 0x2628

# 全局拼音表
GPy_Table = {}


# 原始字节码转为字符串
def byte2str(data):
    pos = 0
    str = ''
    while pos < len(data):
        c = chr(struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0])
        if c != chr(0):
            str += c
        pos += 2
    return str


# 获取拼音表
def getPyTable(data):
    data = data[4:]
    pos = 0
    while pos < len(data):
        index = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
        pos += 2
        lenPy = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
        pos += 2
        py = byte2str(data[pos:pos + lenPy])
        GPy_Table[index] = py
        pos += lenPy


# 获取一个词组的拼音
def getWordPy(data):
    pos = 0
    pinyins = []
    while pos < len(data):
        index = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
        pinyins.append(GPy_Table[index])
        pos += 2
    return ','.join(pinyins)


# 读取中文表
def getChinese(data):
    # 解析结果
    # 元组(词频,拼音,中文词组)的列表
    GTable = []
    pos = 0
    while pos < len(data):
        # 同音词数量
        same = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]

        # 拼音索引表长度
        pos += 2
        py_table_len = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]

        # 拼音索引表
        pos += 2
        py = getWordPy(data[pos: pos + py_table_len])

        # 中文词组
        pos += py_table_len
        for i in range(same):
            # 中文词组长度
            c_len = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
            # 中文词组
            pos += 2
            word = byte2str(data[pos: pos + c_len])
            # 扩展数据长度
            pos += c_len
            ext_len = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
            # 词频
            pos += 2
            count = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]

            # 保存
            GTable.append((count, py, word))

            # 到下个词的偏移位置
            pos += ext_len

    return GTable


def scel2txt(file_name, output_path):
    # 分隔符
    print('-' * 60)
    # 读取文件
    with open(file_name, 'rb') as f:
        data = f.read()

    getPyTable(data[startPy:startChinese])
    result = getChinese(data[startChinese:])

    GPy_Table.clear()

    file_name = file_name.replace(".scel", ".txt")
    file_name = Path(output_path) / file_name
    os.makedirs(file_name.parent, exist_ok=True)

    f = open(file_name, 'w')
    for count, py, word in result:
        word = word.strip().replace(" ", "").replace("\t", "")
        f.write("%s\t%s\n" % (word, py))
    f.close()

    print(file_name, "，转换完成！")


def get_scel_files(path: Path):
    files = []
    for filename in os.listdir(path):
        if filename.endswith(".scel"):
            files.append(path / filename)
            continue

        if os.path.isdir(path / filename):
            files.extend(get_scel_files(path / filename))
            continue

    return files


def main():
    in_path = Path("output")
    out_path = "txt_output"
    scel_files = get_scel_files(in_path)
    for file in scel_files:
        try:
            scel2txt(str(file), out_path)
        except:
            print(str(file), "转换失败！！！！！！！！！！！！！！")


if __name__ == '__main__':
    main()
