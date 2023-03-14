import os
import time

import requests

session = None

session_file = "../session.txt"


def get_cookies():
    cookies = {
        "JSESSIONID": session,
        "username": "zxx"
    }

    return cookies


def login():
    global session
    resp1 = requests.get("http://b.dimix.net.cn:226/gonghui_4737/dimix/welcome/login.jsp")
    session = resp1.cookies.get("JSESSIONID")

    time.sleep(0.05)

    resp1_1 = requests.post("http://b.dimix.net.cn:226/gonghui_4737/ScglControl", data={
        "otype": "getUserInfo",
    }, cookies=get_cookies())

    resp1_2 = requests.post("http://b.dimix.net.cn:226/gonghui_4737/LoginControl", data={
        "oType": "doGetDlfs",
        "zymc": "zxx",
        "ZTID": "0"
    }, cookies=get_cookies())

    resp1_3 = requests.post("http://b.dimix.net.cn:226/gonghui_4737/LoginControl", data={
        "oType": "setZt",
        "ZTID": "0",
        "ZTMC": "账套1"
    }, cookies=get_cookies())

    resp1_4 = requests.post("http://b.dimix.net.cn:226/gonghui_4737/ScglControl", data={
        "otype": "getYhsyrq",
        "yhbm": "zxx",
    }, cookies=get_cookies())

    resp1_5 = requests.post("http://b.dimix.net.cn:226/gonghui_4737/LoginControl", data={
        "oType": "doGetDlfs",
        "zymc": "zxx",
        "ZTID": "0",
    }, cookies=get_cookies())

    resp2 = requests.post("http://b.dimix.net.cn:226/gonghui_4737/LoginControl", data={
        "oType": "GetZybmFromRsryjbxx",
        "yhdm": "zxx",
        "yhkl": "123456",
        "ZTID": "0"
    }, cookies=get_cookies())

    time.sleep(0.05)
    print("")

    resp3 = requests.post("http://b.dimix.net.cn:226/gonghui_4737/LoginControl", data={
        "oType": "dofiveCheck",
        "yhdm": "100083",
        "yhmm": "123456",
        "EnableEnterpriseCode": "0",
        "EnterpriseCode": "0",
        "EnableAuthenticationCode": "0",
        "QYBM": "",
        "YZM": "",
        "dlfs": "1",
        "phoneAuthenticationCode": "",
        "UUID": "d83c3464",
    }, cookies=get_cookies(), headers={
        "Referer": "http://b.dimix.net.cn:226/gonghui_4737/Login"
    })

    resp4 = requests.post("http://b.dimix.net.cn:226/gonghui_4737/SystemsafesetServlet", data={
        "oType": "doUpdatePasswordTheFirstLogin",
        "ZYBM": "100083",
    }, cookies=get_cookies())

    resp5 = requests.post("http://b.dimix.net.cn:226/gonghui_4737/SystemsafesetServlet", data={
        "oType": "doLimitLoginTimes",
        "ZYBM": "100083",
    }, cookies=get_cookies())

    resp6 = requests.post("http://b.dimix.net.cn:226/gonghui_4737/LoginControl", data={
        "oType": "docomparemmyxq",
        "yhdm": "100083",
    }, cookies=get_cookies())

    resp7 = requests.post("http://b.dimix.net.cn:226/gonghui_4737/LoginControl", data={
        "oType": "docomparemmyxq",
        "yhdm": "100083",
    }, cookies=get_cookies())

    resp8 = requests.post("http://b.dimix.net.cn:226/gonghui_4737/LoginControl", data={
        "yhdmTemp": "zxx",
        "yhdm": "100083",
        "LOGTYPE": "1",
        "UUID": "d83c3464",
        "yhkl": "123456",
        "SJYZM": "",
        "QYBM": "",
        "YZM": "",
        "SJYZM": "",
        "ZTID": "",
        "ZTMC": "",  # TODO
        "dlfsmc": "1"
    }, cookies=get_cookies())

    print("自动登录德米萨系统成功...")

    with open(session_file, mode='w') as f:
        f.write(session)


def get(url, retry=False):
    resp = requests.get(url, cookies=get_cookies())

    if "重新登录" in resp.text:
        if retry:
            print("自动登录失败，请重试或联系你老公！")
            exit()

        login()
        return get(url, True)

    return resp


if os.path.exists(session_file):
    with open(session_file, mode='r') as f:
        session = f.read().strip()

if session is None or session == "":
    login()
