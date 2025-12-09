#!/usr/bin/env python3
"""
修复parquet文件的metadata格式问题
直接使用pyarrow读取数据并重新保存，生成兼容的格式
"""
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path

def fix_parquet_file(input_path, output_path):
    """修复parquet文件的metadata"""
    print(f"Reading {input_path}...")
    
    # 使用pyarrow直接读取数据（忽略metadata）
    table = pq.read_table(input_path)
    
    print(f"  Shape: {table.shape}")
    print(f"  Columns: {table.column_names}")
    
    # 重新保存，让datasets库生成新的metadata
    print(f"Writing to {output_path}...")
    pq.write_table(table, output_path, use_deprecated_int96_timestamps=False)
    
    print(f"  ✓ Fixed {output_path}")

def main():
    base_dir = Path("/root/UserRL/data")
    
    # 修复训练和测试数据
    files_to_fix = [
        (base_dir / "alltrain_multiturn" / "train.parquet", 
         base_dir / "alltrain_multiturn" / "train.parquet.bak"),
        (base_dir / "alltest_multiturn" / "test.parquet",
         base_dir / "alltest_multiturn" / "test.parquet.bak"),
    ]
    
    for input_path, backup_path in files_to_fix:
        if input_path.exists():
            # 先备份
            import shutil
            shutil.copy2(input_path, backup_path)
            print(f"Backed up to {backup_path}")
            
            # 修复
            fix_parquet_file(input_path, input_path)
        else:
            print(f"Warning: {input_path} does not exist")

if __name__ == "__main__":
    main()

