# 百度通用翻译API,不包含词典、tts语音合成等资源，如有相关需求请联系translate_api@baidu.com
# coding=utf-8

import http.client
import hashlib
import urllib
import random
import json
import time


class Translator:
    def __init__(self):
        self.app_dict = {'your id': 'your secret key'}
        self.httpClient = None
        self.myurl = '/api/trans/vip/translate'

    def translate(self, from_lang, to_lang, inputs):
        salt = random.randint(32768, 65536)
        q = '\n'.join(input for input in inputs)
        myurls = []
        for id, key in self.app_dict.items():
            sign = id + q + str(salt) + key
            sign = hashlib.md5(sign.encode()).hexdigest()
            myurl = self.myurl + '?appid=' + id + '&q=' + urllib.parse.quote(
                q) + '&from=' + from_lang + '&to=' + to_lang + '&salt=' + str(
                salt) + '&sign=' + sign
            myurls.append(myurl)

        time_out = False
        while True:
            for myurl in myurls:
                try:
                    httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
                    httpClient.request('GET', myurl)
                    # response是HTTPResponse对象
                    response = httpClient.getresponse()
                    result_all = response.read().decode("utf-8")

                    try:
                        result = json.loads(result_all)
                    except Exception:
                        return None

                    translations = [sentence['dst'] for sentence in result['trans_result']]
                    if httpClient:
                        httpClient.close()
                    return translations

                except Exception as e:
                    try:
                        if result['error_code'] == '52001':
                            if not time_out:
                                time_out = True
                            else:
                                return None
                    except Exception as t:
                        return None
