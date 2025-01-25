import os

def check_and_cleanup_folders(base_path):
    """
    检查文件夹中是否存在以 AUDIO 和 TRANSCRIPT 结尾的文件，并删除含 _CLNF_ 的文件。
    
    Args:
        base_path (str): 根目录路径，包含多个子文件夹。
    """
    for root, dirs, files in os.walk(base_path):
        file_names = [os.path.splitext(file)[0] for file in files]

        # 检查是否存在以 AUDIO 和 TRANSCRIPT 结尾的文件名
        has_audio = any(name.endswith("AUDIO") for name in file_names)
        has_transcript = any(name.endswith("TRANSCRIPT") for name in file_names)
        
        # 如果文件夹中缺少所需文件，打印具体缺少的文件
        if not has_audio:
              print(f"WARNING: Missing AUDIO file in folder: {root}")
        if not has_transcript:
            print(f"WARNING: Missing TRANSCRIPT file in folder: {root}")    
            file_lost.append(root[-5:]+".zip") 
        
        # 删除包含 _CLNF_ 的文件
        for file in files:
            if "_CLNF_" in file:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

# 调用函数
file_lost = []
base_path = "D:\\下载\\datasets"  # 修改为你的根目录路径
check_and_cleanup_folders(base_path)
print(file_lost)