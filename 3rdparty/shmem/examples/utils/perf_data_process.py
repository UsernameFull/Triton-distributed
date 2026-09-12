#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
import math
import argparse
import os
import glob
import csv
from math import ceil
from collections import defaultdict

def bytes_to_human_readable(x, pos):
    """Convert bytes to human-readable units (B, KB, MB, etc.)"""
    if x < 1024:
        return f"{x} B"
    elif x < 1024 ** 2:
        return f"{x // 1024} KB"
    elif x < 1024 ** 3:
        return f"{x // (1024 ** 2)} MB"
    else:
        return f"{x // (1024 ** 3)} GB"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process CSV data and generate performance analysis charts')
    # Directory argument: process all CSV files in the directory
    parser.add_argument(
        '--dir', '-d',
        type=str,
        default=None,
        help='Directory path containing CSV files (process all .csv files in this directory)'
    )
    # CSV path argument: process a single CSV file
    parser.add_argument(
        '--csv-path', '-c',
        type=str,
        default=None,
        help='Full/relative path of a single CSV file (process this specific file)'
    )
    # Optional UBsize filter
    parser.add_argument(
        '--ubsize', '-u',
        type=int,
        nargs='+',
        default=None,
        help='Filter UBsize values to plot (multiple values allowed, e.g., -u 16 32)'
    )
    # Optional Coresize filter
    parser.add_argument(
        '--coresize', '-s',
        type=int,
        nargs='+',
        default=None,
        help='Filter Coresize values to plot (multiple values allowed, e.g., -s 8 16)'
    )
    # Plot control
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable generating plots'
    )
    # Markdown control
    parser.add_argument(
        '--no-markdown',
        action='store_true',
        help='Disable generating Markdown report'
    )
    args = parser.parse_args()

    # Validate mutual exclusivity of -d and -c
    if args.dir is None and args.csv_path is None:
        parser.error('Either --dir (-d) or --csv-path (-c) must be specified')
    if args.dir is not None and args.csv_path is not None:
        parser.error('Only one of --dir (-d) or --csv-path (-c) can be specified')

    return args

def get_subplot_layout(n, default_cols=5):
    """
    Calculate subplot rows and columns based on the number of items
    :param n: Number of items to plot
    :param default_cols: Default number of columns for layout
    :return: (rows, cols)
    """
    if n == 0:
        return 1, 1  # Fallback layout for no data
    cols = min(default_cols, n)
    rows = ceil(n / cols)
    return rows, cols

def process_csv(csv_file, output_dir, ubsize_filter, coresize_filter, no_plot=False):
    """Process a single CSV file and generate charts to the specified output directory"""
    generated_images = []
    df = None

    # 定义表头
    headers = ['DataSize/B', 'Npus', 'Blocks', 'UBsize/KB', 'Bandwidth/GB/s', 'CoreMaxTime/us']
    
    try:
        # 手动读取CSV文件，只保留前6列
        data_rows = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # 跳过第一行表头
            next(reader, None)
            # 读取数据行，只取前6列
            for row in reader:
                if len(row) >= 6:
                    data_rows.append(row[:6])
                else:
                    # 如果行数据不足6列，填充空值
                    padded = row + [''] * (6 - len(row))
                    data_rows.append(padded)
    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {e}")
        return generated_images, df
        
        # 创建DataFrame
    df = pd.DataFrame(data_rows, columns=headers)
    
    # 转换列类型
    for col in headers:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            print(f"Failed to convert column '{col}' to numeric type: {e}")
    
    required_cols = ['DataSize/B', 'Blocks', 'UBsize/KB', 'Bandwidth/GB/s']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: CSV file {csv_file} missing required column: {col}")
            return generated_images, df

    # 如果不绘图，直接返回数据
    if no_plot:
        return generated_images, df

    os.makedirs(output_dir, exist_ok=True)

    df['y'] = df['Bandwidth/GB/s']
    ymin, ymax = 0, df['y'].max()
    ymax = math.ceil(ymax / 10) * 10

    # Get CSV basename for chart naming
    csv_basename = os.path.splitext(os.path.basename(csv_file))[0]

    # -------------------------- Figure 1: UBsize_compare (CoreSize subplots) --------------------------
    blocks_values = sorted(df['Blocks'].unique())
    if coresize_filter is not None:
        blocks_values = [v for v in blocks_values if v in coresize_filter]

    fig1_path = None
    if blocks_values:
        core_rows, core_cols = get_subplot_layout(len(blocks_values), default_cols=5)
        fig, axs = plt.subplots(core_rows, core_cols, figsize=(core_cols*4, core_rows*4))
        fig.subplots_adjust(right=0.8, hspace=0.5, wspace=0.3)

        if core_rows == 1 and core_cols == 1:
            axs = np.array([axs])
        elif core_rows == 1 or core_cols == 1:
            axs = axs.flatten()
        else:
            axs = axs.flatten()

        for i, blocks in enumerate(blocks_values):
            if i >= len(axs):
                break
            ax = axs[i]
            subset = df[df['Blocks'] == blocks]
            for ubsize, group in subset.groupby('UBsize/KB'):
                if ubsize_filter is not None and ubsize not in ubsize_filter:
                    continue
                ax.plot(group['DataSize/B'], group['y'], label=f'UBsize={ubsize}')
            ax.set_title(f'Blocks={blocks}')
            ax.set_xlabel('DataSize')
            ax.set_ylabel('Bandwidth(GB/s)')
            ax.set_ylim(ymin, ymax)
            ax.set_xscale('log')
            ax.grid(True)
            max_val = subset['DataSize/B'].max()
            min_val = subset['DataSize/B'].min()
            ticks = [2 ** i for i in range(int(np.log2(min_val)), int(np.log2(max_val)) + 1)]
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(bytes_to_human_readable))
            ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
            ax.tick_params(axis='x', rotation=45)

        for i in range(len(blocks_values), len(axs)):
            axs[i].set_visible(False)

        if len(axs) > 0:
            handles, labels = axs[0].get_legend_handles_labels()
            if labels:
                fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.5), fontsize=8)

        fig1_path = os.path.join(output_dir, f'{csv_basename}_UBsize_compare.png')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(fig1_path, bbox_inches='tight')
        plt.close()
        print(f"Generated: {fig1_path}")
        generated_images.append({
            'name': 'UBsize_compare',
            'path': fig1_path,
            'description': '按 Blocks 分组的 UBsize 对比图',
            'subplots': [f'Blocks={v}' for v in blocks_values],
            'lines': [f'UBsize={v}' for v in sorted(df['UBsize/KB'].unique())]
        })
    else:
        print(f"WARN: {csv_file} - No eligible Blocks values found, skipping UBsize_compare chart")

    # -------------------------- Figure 2: Core_compare (UBsize subplots) --------------------------
    ubsize_values = sorted(df['UBsize/KB'].unique())
    if ubsize_filter is not None:
        ubsize_values = [v for v in ubsize_values if v in ubsize_filter]

    fig2_path = None
    if ubsize_values:
        ub_rows, ub_cols = get_subplot_layout(len(ubsize_values), default_cols=2)
        fig, axs = plt.subplots(ub_rows, ub_cols, figsize=(ub_cols*8, ub_rows*8))
        fig.subplots_adjust(right=0.8, hspace=0.5, wspace=0.3)

        if ub_rows == 1 and ub_cols == 1:
            axs = np.array([axs])
        elif ub_rows == 1 or ub_cols == 1:
            axs = axs.flatten()
        else:
            axs = axs.flatten()

        yc = np.arange(21)
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=np.min(yc), vmax=np.max(yc))

        for i, ubsize in enumerate(ubsize_values):
            if i >= len(axs):
                break
            ax = axs[i]
            subset = df[df['UBsize/KB'] == ubsize]
            for blocks, group in subset.groupby('Blocks'):
                if coresize_filter is not None and blocks not in coresize_filter:
                    continue
                ax.plot(group['DataSize/B'], group['y'], color=cmap(norm(blocks)), label=f'Blocks={blocks}')
            ax.set_title(f'UBsize={ubsize}')
            ax.set_xlabel('DataSize')
            ax.set_ylabel('Bandwidth(GB/s)')
            ax.set_ylim(ymin, ymax)
            ax.set_xscale('log')
            ax.grid(True)
            max_val = subset['DataSize/B'].max()
            min_val = subset['DataSize/B'].min()
            ticks = [2 ** i for i in range(int(np.log2(min_val)), int(np.log2(max_val)) + 1)]
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(bytes_to_human_readable))
            ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
            ax.tick_params(axis='x', rotation=45)

        for i in range(len(ubsize_values), len(axs)):
            axs[i].set_visible(False)

        if len(axs) > 0:
            handles, labels = axs[0].get_legend_handles_labels()
            if labels:
                fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.5), fontsize=8)

        fig2_path = os.path.join(output_dir, f'{csv_basename}_Core_compare.png')
        plt.tight_layout(rect=[0, 0, 0.8, 1])
        plt.savefig(fig2_path, bbox_inches='tight')
        plt.close()
        print(f"Generated: {fig2_path}")
        generated_images.append({
            'name': 'Core_compare',
            'path': fig2_path,
            'description': '按 UBsize 分组的 Blocks 对比图',
            'subplots': [f'UBsize={v}' for v in ubsize_values],
            'lines': [f'Blocks={v}' for v in sorted(df['Blocks'].unique())]
        })
    else:
        print(f"WARN: {csv_file} - No eligible UBsize values found, skipping Core_compare chart")

    # -------------------------- Heatmaps (Filtered Data) --------------------------
    heatmap_df = df.copy()
    if ubsize_filter is not None:
        heatmap_df = heatmap_df[heatmap_df['UBsize/KB'].isin(ubsize_filter)]
    if coresize_filter is not None:
        heatmap_df = heatmap_df[heatmap_df['Blocks'].isin(coresize_filter)]

    if heatmap_df.empty:
        print(f"WARN: {csv_file} - No data left after filtering, skipping heatmaps")
        return generated_images, df

    # -------------------------- Heatmap 1: Max Bandwidth --------------------------
    max_values = heatmap_df.groupby(['Blocks', 'UBsize/KB'])['y'].max().reset_index()
    heatmap1_path = None
    if not max_values.empty:
        max_values_pivot = max_values.pivot(index='UBsize/KB', columns='Blocks', values='y')
        
        # 动态计算图的大小和字体大小
        num_rows = len(max_values_pivot)
        num_cols = len(max_values_pivot.columns)
        fig_width = max(12, num_cols * 1.5)
        fig_height = max(8, num_rows * 1.2)
        annot_kws = {"size": 8} if num_cols * num_rows > 30 else {"size": 10}
        
        plt.figure(figsize=(fig_width, fig_height))
        ax = sns.heatmap(max_values_pivot, annot=True, cmap='coolwarm', fmt='.2f',
                        annot_kws=annot_kws, cbar_kws={"shrink": 0.8})
        ax.invert_yaxis()
        plt.title('Bandwidth Max (GB/s)', fontsize=14, pad=20)
        plt.xlabel('Blocks', fontsize=12)
        plt.ylabel('UBsize/KB', fontsize=12)
        plt.xticks(rotation=45 if num_cols > 8 else 0, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        heatmap1_path = os.path.join(output_dir, f'{csv_basename}_bandwidth_max_heatmap.png')
        plt.savefig(heatmap1_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated: {heatmap1_path}")
        generated_images.append({
            'name': 'bandwidth_max_heatmap',
            'path': heatmap1_path,
            'description': '最大带宽热力图',
            'subplots': [],
            'lines': []
        })
    else:
        print(f"WARN: {csv_file} - No max bandwidth data available, skipping max heatmap")

    # -------------------------- Heatmap 2: Mean Bandwidth (>1MB) --------------------------
    filtered_data = heatmap_df[heatmap_df['DataSize/B'] >= 1048576].groupby(['Blocks', 'UBsize/KB'])['y'].mean().reset_index()
    heatmap2_path = None
    if not filtered_data.empty:
        filtered_data_pivot = filtered_data.pivot(index='UBsize/KB', columns='Blocks', values='y')
        
        # 动态计算图的大小和字体大小
        num_rows = len(filtered_data_pivot)
        num_cols = len(filtered_data_pivot.columns)
        fig_width = max(12, num_cols * 1.5)
        fig_height = max(8, num_rows * 1.2)
        annot_kws = {"size": 8} if num_cols * num_rows > 30 else {"size": 10}
        
        plt.figure(figsize=(fig_width, fig_height))
        ax = sns.heatmap(filtered_data_pivot, annot=True, cmap='coolwarm', fmt='.2f',
                        annot_kws=annot_kws, cbar_kws={"shrink": 0.8})
        ax.invert_yaxis()
        plt.title('Bandwidth Mean (DataSize > 1MB) (GB/s)', fontsize=14, pad=20)
        plt.xlabel('Blocks', fontsize=12)
        plt.ylabel('UBsize/KB', fontsize=12)
        plt.xticks(rotation=45 if num_cols > 8 else 0, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        heatmap2_path = os.path.join(output_dir, f'{csv_basename}_bandwidth_mean_heatmap.png')
        plt.savefig(heatmap2_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated: {heatmap2_path}")
        generated_images.append({
            'name': 'bandwidth_mean_heatmap',
            'path': heatmap2_path,
            'description': '平均带宽热力图（数据量>1MB）',
            'subplots': [],
            'lines': []
        })
    else:
        print(f"WARN: {csv_file} - No mean bandwidth data available, skipping mean heatmap")

    return generated_images, df

def generate_markdown(root_dir, csv_data_dict):
    """Generate Markdown document from processed CSV data"""
    md_content = []
    md_content.append("# Performance Test Report\n")
    md_content.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_content.append("\n---\n")

    # 按子目录和CSV文件组织数据
    for subdir in sorted(csv_data_dict.keys()):
        if subdir:
            md_content.append(f"\n## {subdir}\n")
        
        csv_files = csv_data_dict[subdir]
        for csv_name in sorted(csv_files.keys()):
            csv_info = csv_files[csv_name]
            csv_basename = os.path.basename(csv_name)
            md_content.append(f"\n### {csv_basename}\n")
            
            # 添加表格数据
            df = csv_info['dataframe']
            display_cols = []
            for col in ['DataSize/B', 'Npus', 'Blocks', 'UBsize/KB', 'Bandwidth/GB/s', 'CoreMaxTime/us']:
                if col in df.columns:
                    display_cols.append(col)
            
            if display_cols:
                md_content.append("\n#### 性能数据\n")
                # 创建一个副本用于显示，格式化DataSize/B列避免科学计数法
                display_df = df[display_cols].copy()
                if 'DataSize/B' in display_df.columns:
                    display_df['DataSize/B'] = display_df['DataSize/B'].apply(lambda x: f"{int(x)}" if pd.notna(x) and float(x).is_integer() else f"{x}")
                md_content.append(display_df.to_markdown(index=False))
                md_content.append("\n")
            
            # 添加图片链接和注解
            images = csv_info['images']
            if images:
                md_content.append("\n#### 图表\n")
                for img in images:
                    img_path = img['path']
                    # 计算相对路径
                    rel_img_path = os.path.relpath(img_path, root_dir)
                    md_content.append(f"\n##### {img['name']}\n")
                    md_content.append(f"- **描述**: {img['description']}\n")
                    if img['subplots']:
                        md_content.append(f"- **子图**: {', '.join(img['subplots'])}\n")
                    if img['lines']:
                        md_content.append(f"- **折线**: {', '.join(img['lines'])}\n")
                    md_content.append(f"\n![{img['name']}]({rel_img_path})\n")
    
    # 写入Markdown文件
    md_path = os.path.join(root_dir, 'performance_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_content))
    print(f"\nGenerated Markdown report: {md_path}")

def main():
    args = parse_arguments()
    ubsize_filter = args.ubsize
    coresize_filter = args.coresize
    no_plot = args.no_plot
    no_markdown = args.no_markdown

    # 用于存储所有CSV数据的字典，结构：{子目录: {csv文件路径: {'dataframe': df, 'images': []}}}
    csv_data_dict = defaultdict(lambda: {})
    root_dir = None

    if args.dir:
        dir_path = os.path.abspath(args.dir)
        root_dir = dir_path
        if not os.path.isdir(dir_path):
            print(f"Error: Directory {dir_path} does not exist")
            return
        
        # 查找所有CSV文件（包括子目录）
        all_csv_files = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.csv'):
                    all_csv_files.append(os.path.join(root, file))
        
        if not all_csv_files:
            print(f"Warning: No CSV files found in directory {dir_path}")
            return
        
        base_output_dir = os.path.join(dir_path, 'picture')
        for csv_file in all_csv_files:
            print(f"\nProcessing CSV file: {csv_file}")
            csv_basename = os.path.splitext(os.path.basename(csv_file))[0]
            csv_dir = os.path.dirname(csv_file)
            output_dir = os.path.join(csv_dir, 'picture', csv_basename)
            generated_images, df = process_csv(csv_file, output_dir, ubsize_filter, coresize_filter, no_plot)
            if df is not None:
                subdir = os.path.relpath(csv_dir, dir_path)
                if subdir == '.':
                    subdir = ''
                csv_data_dict[subdir][csv_file] = {
                    'dataframe': df,
                    'images': generated_images
                }
    elif args.csv_path:
        csv_file = os.path.abspath(args.csv_path)
        root_dir = os.path.dirname(csv_file)
        if not os.path.isfile(csv_file):
            print(f"Error: CSV file {csv_file} does not exist")
            return
        csv_basename = os.path.splitext(os.path.basename(csv_file))[0]
        output_dir = os.path.join(os.path.dirname(csv_file), 'picture', csv_basename)
        print(f"\nProcessing CSV file: {csv_file}")
        generated_images, df = process_csv(csv_file, output_dir, ubsize_filter, coresize_filter, no_plot)
        if df is not None:
            csv_data_dict[''][csv_file] = {
                'dataframe': df,
                'images': generated_images
            }

    # 生成Markdown报告
    if not no_markdown and csv_data_dict and root_dir:
        generate_markdown(root_dir, csv_data_dict)

    print("\nAll CSV files processed successfully!")

if __name__ == "__main__":
    main()
