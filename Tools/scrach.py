import os
import requests

# 写一个脚本，从 "https://dcapswoz.ict.usc.edu/wwwdaicwoz/" 爬取编号从320到492的数据集
# 保存到当前路径的文件夹中
# 采用并发的方式爬取数据集
# 可以不采用爬虫，用一些命令行下载工具也可以，比如wget，请自行判断两种方式的优劣

import concurrent.futures

def download_file(url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    print(dest_folder)
    filename = os.path.join(dest_folder, url.split('/')[-1])
    print("尝试建立 request 连接...")
    try:
        response = requests.get(url, stream=True, timeout=10)
        print(f"建立 request 连接成功！开始尝试下载文件 {filename} ...")
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"下载成功：{filename}")
    except Exception as e:
        print(f"下载失败：{url}，错误信息：{e}")

def main():
    base_url = "https://dcapswoz.ict.usc.edu/wwwdaicwoz/"
    dest_folder = os.path.abspath(os.path.dirname(__file__))

    urls = [base_url + file for file in ['322_P.zip', '331_P.zip', '337_P.zip', '341_P.zip', '344_P.zip', '363_P.zip', '374_P.zip', '376_P.zip', '380_P.zip', '383_P.zip', '384_P.zip', '386_P.zip']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(download_file, urls, [dest_folder]*len(urls))

if __name__ == "__main__":
    main()

