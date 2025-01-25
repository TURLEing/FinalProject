# 编写函数，给定一个文件夹路径，要求仅保留有对应后缀的文件，删除其他文件

import os
import shutil
def keep_files_with_suffix(path, keyword):
    for root, _, files in os.walk(path):
        for file in files:
            if keyword not in file:
                # print("Deleting %s" % file)
                os.remove(os.path.join(root, file))

if __name__ == "__main__":
    path = "/home/xtz/GPT-SoVITS/SoVITS_weights_v2"
    keyword = "_e12_"
    keep_files_with_suffix(path, keyword)
