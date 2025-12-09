#!/usr/bin/env python3
"""
使用datasets库重新加载并保存parquet文件，确保metadata格式正确
"""
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
import json

def fix_parquet_with_pyarrow(input_path, output_path):
    """使用pyarrow读取并重新保存，移除有问题的metadata"""
    print(f"Reading {input_path}...")
    
    # 读取数据
    table = pq.read_table(input_path)
    
    print(f"  Shape: {table.shape}")
    print(f"  Columns: {table.column_names}")
    
    # 创建新的schema，移除有问题的metadata
    new_schema = table.schema.remove_metadata()
    
    # 创建新表，使用无metadata的schema
    new_table = pa.Table.from_arrays(
        [table[col] for col in table.column_names],
        schema=new_schema
    )
    
    # 保存，不包含任何metadata
    print(f"Writing to {output_path}...")
    pq.write_table(new_table, output_path, use_deprecated_int96_timestamps=False)
    
    print(f"  ✓ Fixed {output_path} (metadata removed)")

def main():
    base_dir = Path("/root/UserRL/data")
    
    files_to_fix = [
        base_dir / "alltrain_multiturn" / "train.parquet",
        base_dir / "alltest_multiturn" / "test.parquet",
    ]
    
    for file_path in files_to_fix:
        if file_path.exists():
            # 备份
            backup_path = file_path.with_suffix(file_path.suffix + '.bak2')
            import shutil
            shutil.copy2(file_path, backup_path)
            print(f"Backed up to {backup_path}")
            
            # 修复（移除metadata）
            fix_parquet_with_pyarrow(file_path, file_path)
        else:
            print(f"Warning: {file_path} does not exist")

if __name__ == "__main__":
    main()

