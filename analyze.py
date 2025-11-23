import re
import pandas as pd

# 复制你的训练日志到这里
log_text = """
testing acc 8.45%
保存最佳模型
epoch 0 finished
testing acc 10.85%
保存最佳模型
epoch 1 finished
testing acc 13.22%
保存最佳模型
epoch 2 finished
testing acc 15.46%
保存最佳模型
epoch 3 finished
testing acc 17.34%
保存最佳模型
epoch 4 finished
testing acc 18.82%
保存最佳模型
epoch 5 finished
testing acc 21.39%
保存最佳模型
epoch 6 finished
testing acc 23.71%
保存最佳模型
epoch 7 finished
testing acc 24.2%
保存最佳模型
epoch 8 finished
testing acc 24.99%
保存最佳模型
epoch 9 finished
testing acc 25.07%
保存最佳模型
epoch 10 finished
testing acc 26.56%
保存最佳模型
epoch 11 finished
testing acc 28.69%
保存最佳模型
epoch 12 finished
testing acc 29.79%
保存最佳模型
epoch 13 finished
testing acc 30.18%
保存最佳模型
epoch 14 finished
testing acc 30.73%
保存最佳模型
epoch 15 finished
testing acc 31.82%
保存最佳模型
epoch 16 finished
testing acc 32.38%
保存最佳模型
epoch 17 finished
testing acc 34.55%
保存最佳模型
epoch 18 finished
testing acc 34.29%
epoch 19 finished
testing acc 34.34%
epoch 20 finished
testing acc 34.83%
保存最佳模型
epoch 21 finished
testing acc 36.31%
保存最佳模型
epoch 22 finished
testing acc 35.67%
epoch 23 finished
testing acc 36.46%
保存最佳模型
epoch 24 finished
testing acc 38.38%
保存最佳模型
epoch 25 finished
testing acc 37.45%
epoch 26 finished
testing acc 36.98%
epoch 27 finished
testing acc 39.91%
保存最佳模型
epoch 28 finished
testing acc 40.08%
保存最佳模型
epoch 29 finished
testing acc 39.28%
epoch 30 finished
testing acc 40.98%
保存最佳模型
epoch 31 finished
testing acc 40.5%
epoch 32 finished
testing acc 40.67%
epoch 33 finished
testing acc 41.71%
保存最佳模型
epoch 34 finished
testing acc 42.01%
保存最佳模型
epoch 35 finished
testing acc 41.14%
epoch 36 finished
testing acc 41.05%
epoch 37 finished
testing acc 42.03%
保存最佳模型
epoch 38 finished
testing acc 42.05%
保存最佳模型
epoch 39 finished
testing acc 42.71%
保存最佳模型
epoch 40 finished
testing acc 41.97%
epoch 41 finished
testing acc 41.98%
epoch 42 finished
testing acc 42.65%
epoch 43 finished
testing acc 43.87%
保存最佳模型
epoch 44 finished
testing acc 44.0%
保存最佳模型
epoch 45 finished
testing acc 42.81%
epoch 46 finished
testing acc 43.08%
epoch 47 finished
testing acc 44.81%
保存最佳模型
epoch 48 finished
testing acc 45.05%
保存最佳模型
epoch 49 finished
testing acc 44.17%
epoch 50 finished
testing acc 44.54%
epoch 51 finished
testing acc 45.51%
保存最佳模型
epoch 52 finished
testing acc 45.73%
保存最佳模型
epoch 53 finished
testing acc 45.8%
保存最佳模型
epoch 54 finished
testing acc 45.28%
epoch 55 finished
testing acc 45.31%
epoch 56 finished
testing acc 45.91%
保存最佳模型
epoch 57 finished
testing acc 44.96%
epoch 58 finished
testing acc 44.55%
epoch 59 finished
testing acc 45.43%
epoch 60 finished
testing acc 45.73%
epoch 61 finished
testing acc 46.98%
保存最佳模型
epoch 62 finished
testing acc 46.83%
epoch 63 finished
testing acc 44.73%
epoch 64 finished
testing acc 47.22%
保存最佳模型
epoch 65 finished
testing acc 47.03%
epoch 66 finished
testing acc 45.94%
epoch 67 finished
testing acc 47.69%
保存最佳模型
epoch 68 finished
testing acc 47.51%
epoch 69 finished
testing acc 46.51%
epoch 70 finished
testing acc 47.51%
epoch 71 finished
testing acc 46.67%
epoch 72 finished
testing acc 46.7%
epoch 73 finished
testing acc 46.53%
epoch 74 finished
testing acc 50.02%
保存最佳模型
epoch 75 finished
testing acc 50.48%
保存最佳模型
epoch 76 finished
testing acc 50.6%
保存最佳模型
epoch 77 finished
testing acc 51.03%
保存最佳模型
epoch 78 finished
testing acc 50.65%
epoch 79 finished
testing acc 50.18%
epoch 80 finished
testing acc 51.35%
保存最佳模型
epoch 81 finished
testing acc 51.62%
保存最佳模型
epoch 82 finished
testing acc 51.22%
epoch 83 finished
testing acc 50.39%
epoch 84 finished
testing acc 51.12%
epoch 85 finished
testing acc 50.75%
epoch 86 finished
testing acc 50.99%
epoch 87 finished
testing acc 51.1%
epoch 88 finished
testing acc 52.9%
保存最佳模型
epoch 89 finished
testing acc 52.62%
epoch 90 finished
testing acc 52.31%
epoch 91 finished
testing acc 53.32%
保存最佳模型
epoch 92 finished
testing acc 53.26%
epoch 93 finished
testing acc 54.0%
保存最佳模型
epoch 94 finished
testing acc 53.56%
epoch 95 finished
testing acc 54.26%
保存最佳模型
epoch 96 finished
testing acc 54.31%
保存最佳模型
epoch 97 finished
testing acc 54.18%
epoch 98 finished
testing acc 53.65%
epoch 99 finished
best acc is 54.31%
"""

# 解析日志文件
data = []
lines = log_text.strip().split('\n')

i = 0
while i < len(lines):
    line = lines[i].strip()

    # 查找测试准确率
    if line.startswith('testing acc'):
        acc_match = re.search(r'testing acc ([\d\.]+)%', line)
        if acc_match:
            acc = float(acc_match.group(1))

            # 检查下一行是否是"保存最佳模型"
            is_best = False
            if i + 1 < len(lines) and '保存最佳模型' in lines[i + 1]:
                is_best = True
                i += 1  # 跳过"保存最佳模型"行

            # 移动到epoch行
            i += 1
            if i < len(lines):
                epoch_match = re.search(r'epoch (\d+) finished', lines[i])
                if epoch_match:
                    epoch = int(epoch_match.group(1))
                    data.append([epoch, acc, is_best])

    i += 1

# 创建DataFrame
df = pd.DataFrame(data, columns=['Epoch', 'Testing Accuracy (%)', 'Is Best Model'])

# 计算累计最佳准确率
df['Best Accuracy (%)'] = df['Testing Accuracy (%)'].cummax()

# 调整列顺序
df = df[['Epoch', 'Testing Accuracy (%)', 'Best Accuracy (%)', 'Is Best Model']]

# 保存到Excel文件
excel_filename = 'drop=0.3.xlsx'
df.to_excel(excel_filename, index=False)

# 打印总结信息
print(f"✅ Excel文件已成功保存: {excel_filename}")
print(f"📊 共记录 {len(df)} 个epoch的数据")
print(f"🏆 最终最佳准确率: {df['Best Accuracy (%)'].max():.2f}%")
print(f"🎯 达到最佳准确率的epoch: {df.loc[df['Best Accuracy (%)'].idxmax(), 'Epoch']}")
