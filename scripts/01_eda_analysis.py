"""
============================================================
消费者购物行为探索性数据分析 (EDA)
数据集: Kaggle Customer Shopping Behavior Dataset
工具: pandas + matplotlib + seaborn
============================================================
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

OUTPUT = '../output'
os.makedirs(OUTPUT, exist_ok=True)

# ===================== 1. 数据加载与清洗 =====================
print("=" * 60)
print("1. 数据加载与清洗")
print("=" * 60)

df = pd.read_csv('../data/shopping_behavior.csv')
print(f"原始数据: {df.shape[0]} 行 × {df.shape[1]} 列")

# 检查缺失值
missing = df.isnull().sum()
print(f"缺失值: {missing[missing > 0].to_dict() if missing.sum() > 0 else '无'}")

# 检查重复值
dups = df.duplicated().sum()
print(f"重复行: {dups}")

# 统一列名（去掉空格，替换为下划线）
df.columns = df.columns.str.strip().str.replace(' ', '_')
print(f"清洗后列名: {list(df.columns)}")

# ===================== 2. 描述统计概览 =====================
print("\n" + "=" * 60)
print("2. 描述统计概览")
print("=" * 60)

# 数值型变量统计
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"数值型变量 ({len(num_cols)}): {num_cols}")
print(df[num_cols].describe().round(2).to_string())

# 分类变量统计
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\n分类型变量 ({len(cat_cols)}): {cat_cols}")
for col in cat_cols:
    print(f"\n  [{col}] Top5:")
    print(df[col].value_counts().head(5).to_string())

# ===================== 3. 消费者人口画像 =====================
print("\n" + "=" * 60)
print("3. 消费者人口画像分析")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 3.1 年龄分布
axes[0, 0].hist(df['Age'], bins=25, color='steelblue', edgecolor='white', alpha=0.8)
axes[0, 0].axvline(df['Age'].mean(), color='red', linestyle='--', label=f'Mean={df["Age"].mean():.0f}')
axes[0, 0].set_title('Age Distribution', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Age')
axes[0, 0].legend()

# 3.2 性别分布
gender_counts = df['Gender'].value_counts()
colors = ['#ff9999', '#66b3ff']
axes[0, 1].pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90, explode=(0.02, 0.02))
axes[0, 1].set_title('Gender Distribution', fontsize=13, fontweight='bold')

# 3.3 品类分布
cat_counts = df['Category'].value_counts()
axes[0, 2].bar(cat_counts.index, cat_counts.values, color=sns.color_palette("Set2", len(cat_counts)))
axes[0, 2].set_title('Product Category Distribution', fontsize=13, fontweight='bold')
axes[0, 2].set_ylabel('Count')
axes[0, 2].tick_params(axis='x', rotation=30)

# 3.4 季节分布
season_counts = df['Season'].value_counts()
season_order = ['Spring', 'Summer', 'Fall', 'Winter']
season_counts = season_counts.reindex(season_order)
axes[1, 0].bar(season_counts.index, season_counts.values, color=sns.color_palette("coolwarm", 4))
axes[1, 0].set_title('Purchase by Season', fontsize=13, fontweight='bold')
axes[1, 0].set_ylabel('Count')

# 3.5 购买金额分布
axes[1, 1].hist(df['Purchase_Amount_(USD)'], bins=30, color='coral', edgecolor='white', alpha=0.8)
axes[1, 1].axvline(df['Purchase_Amount_(USD)'].mean(), color='blue', linestyle='--',
                   label=f'Mean=${df["Purchase_Amount_(USD)"].mean():.0f}')
axes[1, 1].set_title('Purchase Amount Distribution', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Purchase Amount (USD)')
axes[1, 1].legend()

# 3.6 评分分布
rating_counts = df['Review_Rating'].value_counts().sort_index()
axes[1, 2].bar(rating_counts.index, rating_counts.values, color='gold', edgecolor='gray')
axes[1, 2].set_title('Review Rating Distribution', fontsize=13, fontweight='bold')
axes[1, 2].set_xlabel('Rating')
axes[1, 2].set_ylabel('Count')
axes[1, 2].axvline(df['Review_Rating'].mean(), color='red', linestyle='--',
                   label=f'Mean={df["Review_Rating"].mean():.2f}')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, '01_demographic_overview.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[OK] 已保存: 01_demographic_overview.png")

# ===================== 4. 品类与季节交叉分析 =====================
print("\n" + "=" * 60)
print("4. 品类 × 季节 交叉分析")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 4.1 各品类购买金额对比
cat_amount = df.groupby('Category')['Purchase_Amount_(USD)'].agg(['mean', 'std', 'count']).round(2)
print(cat_amount.to_string())
axes[0].bar(cat_amount.index, cat_amount['mean'], yerr=cat_amount['std'],
            color=sns.color_palette("Set2", len(cat_amount)), capsize=5)
axes[0].set_title('Average Purchase Amount by Category', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Average Amount (USD)')
axes[0].tick_params(axis='x', rotation=30)

# 4.2 各季节品类热力图
season_cat = pd.crosstab(df['Season'], df['Category'])
season_cat = season_cat.reindex(['Spring', 'Summer', 'Fall', 'Winter'])
sns.heatmap(season_cat, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1],
            linewidths=0.5, cbar_kws={'label': 'Count'})
axes[1].set_title('Season × Category Heatmap', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, '02_category_season_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[OK] 已保存: 02_category_season_analysis.png")

# ===================== 5. 促销与会员分析 =====================
print("\n" + "=" * 60)
print("5. 促销与会员效果分析")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5.1 折扣对购买金额的影响
discount_amounts = [df[df['Discount_Applied'] == 'Yes']['Purchase_Amount_(USD)'],
                    df[df['Discount_Applied'] == 'No']['Purchase_Amount_(USD)']]
bp1 = axes[0, 0].boxplot(discount_amounts, tick_labels=['Discount', 'No Discount'],
                          patch_artist=True, widths=0.5)
bp1['boxes'][0].set_facecolor('#ff9999')
bp1['boxes'][1].set_facecolor('#66b3ff')
axes[0, 0].set_title('Purchase Amount: Discount vs No Discount', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Purchase Amount (USD)')

# 5.2 会员 vs 非会员消费
member_amounts = [df[df['Subscription_Status'] == 'Yes']['Purchase_Amount_(USD)'],
                  df[df['Subscription_Status'] == 'No']['Purchase_Amount_(USD)']]
bp2 = axes[0, 1].boxplot(member_amounts, tick_labels=['Member', 'Non-Member'],
                          patch_artist=True, widths=0.5)
bp2['boxes'][0].set_facecolor('#90EE90')
bp2['boxes'][1].set_facecolor('#D3D3D3')
axes[0, 1].set_title('Purchase Amount: Member vs Non-Member', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Purchase Amount (USD)')

# 5.3 购买频次分布
freq_order = ['Weekly', 'Fortnightly', 'Bi-Weekly', 'Monthly', 'Quarterly', 'Every 3 Months', 'Annually']
freq_counts = df['Frequency_of_Purchases'].value_counts()
# 对齐顺序
freq_available = [f for f in freq_order if f in freq_counts.index]
freq_counts = freq_counts.reindex(freq_available)
axes[1, 0].barh(freq_counts.index, freq_counts.values, color=sns.color_palette("viridis", len(freq_counts)))
axes[1, 0].set_title('Purchase Frequency Distribution', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Count')

# 5.4 支付方式分布
pay_counts = df['Payment_Method'].value_counts()
axes[1, 1].pie(pay_counts, labels=pay_counts.index, autopct='%1.1f%%',
               colors=sns.color_palette("pastel", len(pay_counts)),
               startangle=90, explode=[0.02]*len(pay_counts))
axes[1, 1].set_title('Payment Method Distribution', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, '03_promotion_membership_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[OK] 已保存: 03_promotion_membership_analysis.png")

# ===================== 6. 相关性分析 =====================
print("\n" + "=" * 60)
print("6. 数值变量相关性分析")
print("=" * 60)

num_df = df.select_dtypes(include=[np.number])
corr = num_df.corr()

plt.figure(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5,
            vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numerical Variables', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, '04_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[OK] 已保存: 04_correlation_heatmap.png")

# ===================== 7. 输出EDA关键发现摘要 =====================
print("\n" + "=" * 60)
print("7. EDA 关键发现摘要")
print("=" * 60)

findings = {
    '样本规模': f'{len(df)} 条消费记录',
    '年龄范围': f'{df["Age"].min()} – {df["Age"].max()} 岁 (均值 {df["Age"].mean():.1f})',
    '性别比例': f'{df["Gender"].value_counts(normalize=True)["Male"]*100:.1f}% 男 / {df["Gender"].value_counts(normalize=True).get("Female", 0)*100:.1f}% 女',
    '平均消费金额': f'${df["Purchase_Amount_(USD)"].mean():.2f} (中位数 ${df["Purchase_Amount_(USD)"].median():.0f})',
    '平均评分': f'{df["Review_Rating"].mean():.2f} / 5.0',
    '会员占比': f'{df["Subscription_Status"].value_counts(normalize=True)["Yes"]*100:.1f}%',
    '折扣使用率': f'{df["Discount_Applied"].value_counts(normalize=True)["Yes"]*100:.1f}%',
    '促销码使用率': f'{df["Promo_Code_Used"].value_counts(normalize=True)["Yes"]*100:.1f}%',
    '最常见的品类': f'{df["Category"].mode()[0]}',
    '最常见的购买频次': f'{df["Frequency_of_Purchases"].mode()[0]}',
    '平均历史购买次数': f'{df["Previous_Purchases"].mean():.1f} 次',
}

for k, v in findings.items():
    print(f"  - {k}: {v}")

# 保存摘要
with open(os.path.join(OUTPUT, 'eda_summary.txt'), 'w', encoding='utf-8') as f:
    for k, v in findings.items():
        f.write(f'{k}: {v}\n')

print(f"\n[OK] EDA分析完成！所有图表已保存至 {OUTPUT}/")
