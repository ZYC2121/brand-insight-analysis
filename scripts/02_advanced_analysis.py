"""
============================================================
高级消费者分析: 客户细分 + 促销效果检验 + 评分驱动因素
============================================================
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

OUTPUT = '../output'
os.makedirs(OUTPUT, exist_ok=True)

df = pd.read_csv('../data/shopping_behavior.csv')
df.columns = df.columns.str.strip().str.replace(' ', '_')

# ============================================================
# PART A: 客户细分 (K-Means Clustering)
# ============================================================
print("=" * 60)
print("PART A: 基于 K-Means 的消费者细分")
print("=" * 60)

# 构造聚类特征: 消费金额、历史购买次数、评分、年龄
cluster_features = df[['Purchase_Amount_(USD)', 'Previous_Purchases', 'Review_Rating', 'Age']].copy()

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_features)

# 肘部法则确定最优K
inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# 选择K=4
k_optimal = 4
km = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
df['Cluster'] = km.fit_predict(X_scaled)

print(f"最优聚类数 K = {k_optimal}")
print(f"各聚类样本量:")
for i in range(k_optimal):
    count = (df['Cluster'] == i).sum()
    pct = count / len(df) * 100
    mean_amt = df[df['Cluster'] == i]['Purchase_Amount_(USD)'].mean()
    mean_rating = df[df['Cluster'] == i]['Review_Rating'].mean()
    mean_age = df[df['Cluster'] == i]['Age'].mean()
    print(f"  Cluster {i}: {count} 人 ({pct:.1f}%) | "
          f"平均消费=${mean_amt:.0f} | 评分={mean_rating:.2f} | 年龄={mean_age:.0f}")

# 聚类画像
cluster_profile = df.groupby('Cluster').agg({
    'Purchase_Amount_(USD)': 'mean',
    'Previous_Purchases': 'mean',
    'Review_Rating': 'mean',
    'Age': 'mean',
    'Customer_ID': 'count'
}).round(2)
cluster_profile.columns = ['Avg_Purchase', 'Avg_PrevPurchases', 'Avg_Rating', 'Avg_Age', 'Count']
print("\n聚类画像详情:")
print(cluster_profile.to_string())

# 聚类可视化
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# PCA降维可视化
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'],
                          cmap='viridis', alpha=0.6, s=20)
axes[0].set_title(f'Customer Segmentation (PCA, K={k_optimal})', fontsize=13, fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.colorbar(scatter, ax=axes[0], label='Cluster')

# 肘部法则图
axes[1].plot(K_range, inertias, 'bo-', markersize=8, linewidth=2)
axes[1].axvline(x=k_optimal, color='red', linestyle='--', label=f'Optimal K={k_optimal}')
axes[1].set_title('Elbow Method for Optimal K', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Inertia')
axes[1].legend()

# 聚类消费金额对比
cluster_means = df.groupby('Cluster')['Purchase_Amount_(USD)'].mean()
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
axes[2].bar(range(k_optimal), cluster_means, color=colors[:k_optimal])
axes[2].set_title('Average Purchase Amount by Cluster', fontsize=13, fontweight='bold')
axes[2].set_xlabel('Cluster')
axes[2].set_ylabel('Average Purchase Amount (USD)')
axes[2].set_xticks(range(k_optimal))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, '05_customer_segmentation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[OK] 已保存: 05_customer_segmentation.png")

# ============================================================
# PART B: 促销效果 A/B 检验
# ============================================================
print("\n" + "=" * 60)
print("PART B: 促销策略效果统计检验")
print("=" * 60)

# B1: 折扣组 vs 非折扣组 消费金额 t检验
discount_yes = df[df['Discount_Applied'] == 'Yes']['Purchase_Amount_(USD)']
discount_no = df[df['Discount_Applied'] == 'No']['Purchase_Amount_(USD)']
t_stat_d, p_val_d = stats.ttest_ind(discount_yes, discount_no)

print(f"\n【折扣效果检验】")
print(f"  折扣组均值: ${discount_yes.mean():.2f} (n={len(discount_yes)})")
print(f"  非折扣组均值: ${discount_no.mean():.2f} (n={len(discount_no)})")
print(f"  t统计量 = {t_stat_d:.3f}, p值 = {p_val_d:.4f}")
print(f"  结论: {'[OK] 差异显著' if p_val_d < 0.05 else '[X] 无显著差异'} (α=0.05)")

# B2: 促销码使用效果
promo_yes = df[df['Promo_Code_Used'] == 'Yes']['Purchase_Amount_(USD)']
promo_no = df[df['Promo_Code_Used'] == 'No']['Purchase_Amount_(USD)']
t_stat_p, p_val_p = stats.ttest_ind(promo_yes, promo_no)

print(f"\n【促销码效果检验】")
print(f"  促销码组均值: ${promo_yes.mean():.2f} (n={len(promo_yes)})")
print(f"  无促销码组均值: ${promo_no.mean():.2f} (n={len(promo_no)})")
print(f"  t统计量 = {t_stat_p:.3f}, p值 = {p_val_p:.4f}")
print(f"  结论: {'[OK] 差异显著' if p_val_p < 0.05 else '[X] 无显著差异'} (α=0.05)")

# B3: 会员 vs 非会员
member_yes = df[df['Subscription_Status'] == 'Yes']['Purchase_Amount_(USD)']
member_no = df[df['Subscription_Status'] == 'No']['Purchase_Amount_(USD)']
t_stat_m, p_val_m = stats.ttest_ind(member_yes, member_no)

print(f"\n【会员效果检验】")
print(f"  会员组均值: ${member_yes.mean():.2f} (n={len(member_yes)})")
print(f"  非会员组均值: ${member_no.mean():.2f} (n={len(member_no)})")
print(f"  t统计量 = {t_stat_m:.3f}, p值 = {p_val_m:.4f}")
print(f"  结论: {'[OK] 差异显著' if p_val_m < 0.05 else '[X] 无显著差异'} (α=0.05)")

# B4: 品类消费差异 ANOVA
categories = df['Category'].unique()
cat_groups = [df[df['Category'] == c]['Purchase_Amount_(USD)'] for c in categories]
f_stat_cat, p_val_cat = stats.f_oneway(*cat_groups)

print(f"\n【品类消费差异 ANOVA 检验】")
print(f"  F统计量 = {f_stat_cat:.3f}, p值 = {p_val_cat:.4f}")
print(f"  结论: {'[OK] 不同品类消费金额存在显著差异' if p_val_cat < 0.05 else '[X] 无显著差异'} (α=0.05)")

# B5: 季节消费差异 ANOVA
seasons = ['Spring', 'Summer', 'Fall', 'Winter']
season_groups = [df[df['Season'] == s]['Purchase_Amount_(USD)'] for s in seasons if s in df['Season'].values]
f_stat_s, p_val_s = stats.f_oneway(*season_groups)

print(f"\n【季节消费差异 ANOVA 检验】")
print(f"  F统计量 = {f_stat_s:.3f}, p值 = {p_val_s:.4f}")
print(f"  结论: {'[OK] 不同季节消费金额存在显著差异' if p_val_s < 0.05 else '[X] 无显著差异'} (α=0.05)")

# 促销效果汇总可视化
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 折扣效果
labels_d = ['Discount', 'No Discount']
means_d = [discount_yes.mean(), discount_no.mean()]
axes[0].bar(labels_d, means_d, color=['#ff9999', '#66b3ff'])
axes[0].set_title(f'Discount Effect\n(t={t_stat_d:.2f}, p={p_val_d:.4f})', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Avg Purchase Amount (USD)')

# 促销码效果
labels_p = ['Promo Code', 'No Promo']
means_p = [promo_yes.mean(), promo_no.mean()]
axes[1].bar(labels_p, means_p, color=['#99ff99', '#D3D3D3'])
axes[1].set_title(f'Promo Code Effect\n(t={t_stat_p:.2f}, p={p_val_p:.4f})', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Avg Purchase Amount (USD)')

# 会员效果
labels_m = ['Member', 'Non-Member']
means_m = [member_yes.mean(), member_no.mean()]
axes[2].bar(labels_m, means_m, color=['#ffcc99', '#D3D3D3'])
axes[2].set_title(f'Membership Effect\n(t={t_stat_m:.2f}, p={p_val_m:.4f})', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Avg Purchase Amount (USD)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, '06_promotion_statistical_tests.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[OK] 已保存: 06_promotion_statistical_tests.png")

# ============================================================
# PART C: 评分驱动因素分析
# ============================================================
print("\n" + "=" * 60)
print("PART C: 消费者评分驱动因素分析")
print("=" * 60)

# C1: 各品类评分对比
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cat_rating = df.groupby('Category')['Review_Rating'].agg(['mean', 'std']).sort_values('mean', ascending=False)
print("\n各品类平均评分:")
print(cat_rating.round(3).to_string())
axes[0].barh(cat_rating.index, cat_rating['mean'], xerr=cat_rating['std'],
             color=sns.color_palette("Set2", len(cat_rating)), capsize=3)
axes[0].set_title('Average Review Rating by Category', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Average Rating')

# C2: 折扣与评分的关系
disc_rating_yes = df[df['Discount_Applied'] == 'Yes']['Review_Rating']
disc_rating_no = df[df['Discount_Applied'] == 'No']['Review_Rating']
t_r, p_r = stats.ttest_ind(disc_rating_yes, disc_rating_no)
print(f"\n折扣组平均评分: {disc_rating_yes.mean():.3f} vs 非折扣组: {disc_rating_no.mean():.3f}")
print(f"  评分差异 t检验: t={t_r:.3f}, p={p_r:.4f}")

# C3: 会员与评分的关系
mem_rating_yes = df[df['Subscription_Status'] == 'Yes']['Review_Rating']
mem_rating_no = df[df['Subscription_Status'] == 'No']['Review_Rating']
t_rm, p_rm = stats.ttest_ind(mem_rating_yes, mem_rating_no)
print(f"会员组平均评分: {mem_rating_yes.mean():.3f} vs 非会员组: {mem_rating_no.mean():.3f}")
print(f"  评分差异 t检验: t={t_rm:.3f}, p={p_rm:.4f}")

# 评分驱动因素汇总图
drivers = ['Category', 'Discount\nApplied', 'Membership', 'Season']
effects = [p_val_cat, p_val_d, p_val_m, p_val_s]
colors_bar = ['green' if p < 0.05 else 'gray' for p in effects]
axes[1].bar(drivers, [-np.log10(p) if p > 0 else 10 for p in effects], color=colors_bar)
axes[1].axhline(y=-np.log10(0.05), color='red', linestyle='--', label='α=0.05 threshold')
axes[1].set_title('Review Rating Drivers (Statistical Significance)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('-log10(p-value)')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, '07_review_rating_drivers.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[OK] 已保存: 07_review_rating_drivers.png")

# ============================================================
# 汇总报告
# ============================================================
print("\n" + "=" * 60)
print("分析完成！汇总输出:")
print("=" * 60)
for f in os.listdir(OUTPUT):
    fpath = os.path.join(OUTPUT, f)
    size = os.path.getsize(fpath)
    print(f"  >> {f} ({size/1024:.1f} KB)")
