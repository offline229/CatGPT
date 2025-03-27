import os

def delete_files_with_keyword(directory, keyword='aug', dry_run=True, case_insensitive=True):
    """
    递归删除指定目录下所有包含指定关键字的文件
    
    参数:
    directory (str): 要搜索的根目录路径
    keyword (str): 文件名中需要包含的关键词（默认"aug"）
    dry_run (bool): 是否预览删除（True=仅显示，False=实际删除）
    case_insensitive (bool): 是否不区分大小写匹配（True=不区分）
    """
    
    # 安全检查：确认目录存在
    if not os.path.exists(directory):
        print(f"❌ 目标目录 {directory} 不存在！")
        return
    
    # 确认操作
    confirmation = input(f"⚠ 警告：即将{'预览' if dry_run else '实际'}删除目录 {directory} 下所有包含 '{keyword}' 的文件，确认继续？(输入 yes 继续): ")
    if confirmation != 'yes':
        print("操作已取消。")
        return
    
    # 开始搜索
    print(f"\n{'='*50}\n开始{'预览' if dry_run else '执行'}删除操作...\n{'='*50}")
    
    # 遍历目录
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # 处理大小写
            if case_insensitive:
                if keyword.lower() in filename.lower():
                    process_file(root, filename, dry_run)
            else:
                if keyword in filename:
                    process_file(root, filename, dry_run)
    
    print(f"\n{'='*50}\n操作完成！\n{'='*50}")

def process_file(root, filename, dry_run):
    """处理单个文件的删除"""
    file_path = os.path.join(root, filename)
    print(f"文件路径: {file_path}")
    
    if not dry_run:
        try:
            os.remove(file_path)
            print(f"✅ 已删除：{file_path}")
        except Exception as e:
            print(f"❌ 删除失败：{file_path} - {str(e)}")
    else:
        print(f"⏳ 预览删除：{file_path}")

# 使用示例（请先修改路径！）
# 注意：运行前请确保已备份重要数据！

# 修改为你的目标目录
target_directory = "./Defence"

# # 调用函数（推荐先预览）
# delete_files_with_keyword(target_directory, keyword='aug', dry_run=True, case_insensitive=True)

# 如果确认无误，再执行实际删除
delete_files_with_keyword(target_directory, keyword='aug', dry_run=False, case_insensitive=True)