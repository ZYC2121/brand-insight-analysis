"""
============================================================
通用消费者数据分析引擎 — AutoAnalyzer
============================================================
自动识别 CSV 列类型 → 匹配分析方法 → 输出结构化分析结果
可被 CLI / Streamlit / Jupyter 复用调用
============================================================
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from datetime import datetime
import os, warnings, json
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class AutoAnalyzer:
    """通用数据分析引擎：输入CSV路径，输出完整分析结果"""

    def __init__(self, csv_path, output_dir='output', max_categories=15, top_n=10,
                 id_unique_ratio=0.9, id_min_unique=100, text_avg_len_threshold=50):
        """
        参数:
            csv_path: CSV 文件路径
            output_dir: 输出目录
            max_categories: 超过此数的分类变量不参与分组检验
            top_n: 分类柱状图展示前 N 个类别
            id_unique_ratio: ID列判定 — 唯一值占比阈值（默认0.9，即唯一值超过90%触发）
            id_min_unique:  ID列判定 — 唯一值绝对数量阈值（默认100，配合unique_ratio使用）
            text_avg_len_threshold: 文本列判定 — 平均字符串长度阈值（默认50字符）
        """
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.max_categories = max_categories  # 超过此数的分类列不参与分组分析
        self.top_n = top_n                    # 分类变量展示top N
        # 列类型检测阈值（可根据数据特征调整）
        # ID列: high基数 + 非数值 → 很可能是唯一标识符而非有意义的分类维度
        self.id_unique_ratio = id_unique_ratio
        self.id_min_unique = id_min_unique
        # 文本列: 长字符串 → 自由文本而非类别标签（如产品描述、用户评论）
        self.text_avg_len_threshold = text_avg_len_threshold
        self.df = None
        self.meta = {}                        # 列类型元数据
        self.findings = []                    # 关键发现列表
        os.makedirs(output_dir, exist_ok=True)

    # ======================== 1. 数据加载与列类型识别 ========================
    def load_and_detect(self):
        """加载CSV并自动识别每列类型"""
        self.df = pd.read_csv(self.csv_path)
        self.df.columns = self.df.columns.str.strip().str.replace(' ', '_')

        numeric_cols, categorical_cols, datetime_cols, id_cols, text_cols = [], [], [], [], []

        for col in self.df.columns:
            series = self.df[col].dropna()
            if len(series) == 0:
                continue
            dtype = series.dtype

            # 数值型
            if pd.api.types.is_numeric_dtype(dtype):
                if series.nunique() <= 2:
                    categorical_cols.append(col)  # 二值变量视为分类
                else:
                    numeric_cols.append(col)
                continue

            # 日期型
            if pd.api.types.is_datetime64_any_dtype(dtype):
                datetime_cols.append(col)
                continue
            try:
                pd.to_datetime(series, errors='raise')
                datetime_cols.append(col)
                continue
            except:
                pass

            # ID型 (高基数但非数值)
            # 判定逻辑: 唯一值占比高 + 绝对数量大 → 很可能是标识符而非有意义的分类
            unique_ratio = series.nunique() / len(series)
            if unique_ratio > self.id_unique_ratio and series.nunique() > self.id_min_unique:
                id_cols.append(col)
                continue

            # 文本型 (长字符串)
            # 判定逻辑: 平均字符长度超过阈值 → 很可能是自由文本而非类别标签
            if series.apply(lambda x: len(str(x)) if pd.notna(x) else 0).mean() > self.text_avg_len_threshold:
                text_cols.append(col)
                continue

            # 其余 → 分类型
            categorical_cols.append(col)

        self.meta = {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'datetime': datetime_cols,
            'id': id_cols,
            'text': text_cols,
            'total_rows': len(self.df),
            'total_cols': len(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicated_rows': int(self.df.duplicated().sum()),
        }

        # 对分类列做值频次统计
        cat_info = {}
        for col in categorical_cols:
            vc = self.df[col].value_counts()
            cat_info[col] = {'unique': len(vc), 'top': vc.head(self.top_n).to_dict()}
        self.meta['cat_info'] = cat_info

        return self.meta

    # ======================== 1.5 数据质量检测 ========================
    def data_quality(self):
        """检测缺失值和异常值，返回数据质量报告"""
        dq = {'missing': {}, 'outliers': {}}

        # --- 缺失值检测 ---
        for col in self.df.columns:
            n_miss = int(self.df[col].isnull().sum())
            if n_miss > 0:
                pct = n_miss / len(self.df) * 100
                dq['missing'][col] = {
                    'count': n_miss,
                    'percentage': round(pct, 1),
                    'severity': 'high' if pct > 20 else 'medium' if pct > 5 else 'low'
                }

        # --- 异常值检测 (IQR方法，仅对数值列) ---
        for col in self.meta['numeric']:
            series = self.df[col].dropna()
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            n_outliers = int(((series < lower) | (series > upper)).sum())
            if n_outliers > 0:
                pct = n_outliers / len(series) * 100
                dq['outliers'][col] = {
                    'count': n_outliers,
                    'percentage': round(pct, 1),
                    'lower_bound': round(float(lower), 2),
                    'upper_bound': round(float(upper), 2),
                    'severity': 'high' if pct > 10 else 'medium' if pct > 3 else 'low'
                }

        self.meta['data_quality'] = dq
        return dq

    # ======================== 2. 描述统计分析 ========================
    def descriptive_stats(self):
        """数值型变量描述统计"""
        numeric_cols = self.meta['numeric']
        if not numeric_cols:
            return None

        stats_df = self.df[numeric_cols].describe().round(2)
        # 增加偏度峰度
        skew_kurt = pd.DataFrame({
            'skewness': self.df[numeric_cols].skew().round(3),
            'kurtosis': self.df[numeric_cols].kurtosis().round(3),
        })
        result = pd.concat([stats_df.T, skew_kurt], axis=1).T

        # 保存
        stats_path = os.path.join(self.output_dir, 'descriptive_stats.csv')
        result.to_csv(stats_path)
        return {'table': result, 'path': stats_path, 'n_cols': len(numeric_cols)}

    # ======================== 3. 单变量可视化 ========================
    def univariate_plots(self):
        """为每个数值列生成直方图+箱线图，每个分类列生成柱状图"""
        numeric_cols = self.meta['numeric']
        categorical_cols = self.meta['categorical']
        charts = []

        # --- 数值列: 直方图+箱线图 ---
        for col in numeric_cols[:12]:  # 最多12个数值列
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            # 直方图+KDE
            axes[0].hist(self.df[col].dropna(), bins=30, color='steelblue', edgecolor='white', alpha=0.8)
            axes[0].axvline(self.df[col].mean(), color='red', linestyle='--', linewidth=1.5,
                            label=f'Mean={self.df[col].mean():.1f}')
            axes[0].axvline(self.df[col].median(), color='orange', linestyle='-', linewidth=1.5,
                            label=f'Median={self.df[col].median():.1f}')
            axes[0].set_title(f'{col} — Distribution', fontsize=12, fontweight='bold')
            axes[0].legend(fontsize=8)

            # 箱线图
            axes[1].boxplot(self.df[col].dropna(), vert=True, patch_artist=True,
                            boxprops=dict(facecolor='lightblue'))
            axes[1].set_title(f'{col} — Boxplot', fontsize=12, fontweight='bold')
            axes[1].set_ylabel(col)

            plt.tight_layout()
            fname = f'hist_box_{self._safe_name(col)}.png'
            plt.savefig(os.path.join(self.output_dir, fname), dpi=120, bbox_inches='tight')
            plt.close()
            charts.append({'file': fname, 'column': col, 'type': 'numeric_distribution'})

        # --- 分类列: 柱状图 ---
        for col in categorical_cols[:12]:
            vc = self.df[col].value_counts().head(self.top_n)
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(range(len(vc)), vc.values, color=sns.color_palette("Set2", len(vc)))
            ax.set_xticks(range(len(vc)))
            ax.set_xticklabels(vc.index, rotation=30, ha='right', fontsize=9)
            ax.set_title(f'{col} — Category Distribution (Top {self.top_n})', fontsize=12, fontweight='bold')
            ax.set_ylabel('Count')
            # 添加数值标签
            for bar, val in zip(bars, vc.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vc.values)*0.01,
                        str(val), ha='center', fontsize=8)

            plt.tight_layout()
            fname = f'bar_{self._safe_name(col)}.png'
            plt.savefig(os.path.join(self.output_dir, fname), dpi=120, bbox_inches='tight')
            plt.close()
            charts.append({'file': fname, 'column': col, 'type': 'categorical_distribution'})

        return charts

    # ======================== 4. 相关性分析 ========================
    def correlation_analysis(self):
        """数值变量相关性矩阵 — Pearson + Spearman 双方法"""
        numeric_cols = self.meta['numeric']
        if len(numeric_cols) < 2:
            return None

        pearson = self.df[numeric_cols].corr(method='pearson')
        spearman = self.df[numeric_cols].corr(method='spearman')

        # 高相关对（综合两种方法）
        high_corr = []
        pearson_high = []
        for i in range(len(pearson.columns)):
            for j in range(i+1, len(pearson.columns)):
                p_val = pearson.iloc[i, j]
                s_val = spearman.iloc[i, j]
                pair = f'{pearson.columns[i]} × {pearson.columns[j]}'
                if abs(p_val) > 0.3 or abs(s_val) > 0.3:
                    high_corr.append({
                        'pair': pair,
                        'pearson': round(p_val, 3),
                        'spearman': round(s_val, 3),
                        'correlation': round(p_val, 3)  # backward compat
                    })
                if abs(p_val) > 0.3:
                    pearson_high.append({
                        'pair': pair,
                        'pearson': round(p_val, 3),
                        'spearman': round(s_val, 3)
                    })

        high_corr.sort(key=lambda x: abs(x['pearson']), reverse=True)
        pearson_high.sort(key=lambda x: abs(x['pearson']), reverse=True)

        # 检测 Pearson 和 Spearman 差异大的变量对（提示非线性关系）
        nonlinear_hints = []
        for i in range(len(pearson.columns)):
            for j in range(i+1, len(pearson.columns)):
                diff = abs(abs(pearson.iloc[i, j]) - abs(spearman.iloc[i, j]))
                if diff > 0.2:  # 差异超过 0.2 提示可能存在非线性关系
                    nonlinear_hints.append({
                        'pair': f'{pearson.columns[i]} × {pearson.columns[j]}',
                        'pearson': round(pearson.iloc[i, j], 3),
                        'spearman': round(spearman.iloc[i, j], 3),
                        'difference': round(diff, 3)
                    })
        nonlinear_hints.sort(key=lambda x: x['difference'], reverse=True)

        # Pearson 热力图
        fig, axes = plt.subplots(1, 2, figsize=(max(12, len(numeric_cols)*1.6),
                                                 max(5.5, len(numeric_cols)*0.7)))
        mask = np.triu(np.ones_like(pearson, dtype=bool))
        sns.heatmap(pearson, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, square=True, linewidths=0.5, vmin=-1, vmax=1, ax=axes[0])
        axes[0].set_title('Pearson Correlation', fontsize=13, fontweight='bold')
        sns.heatmap(spearman, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, square=True, linewidths=0.5, vmin=-1, vmax=1, ax=axes[1])
        axes[1].set_title('Spearman Correlation', fontsize=13, fontweight='bold')
        plt.tight_layout()
        fname = 'correlation_heatmap.png'
        plt.savefig(os.path.join(self.output_dir, fname), dpi=150, bbox_inches='tight')
        plt.close()

        return {
            'pearson_matrix': pearson, 'spearman_matrix': spearman,
            'corr_matrix': pearson,  # backward compat
            'high_correlations': pearson_high,
            'all_high_correlations': high_corr,
            'nonlinear_hints': nonlinear_hints,
            'chart_file': fname
        }

    # ======================== 5. 分组对比分析 (t检验/ANOVA) ========================
    def group_comparison(self):
        """分类变量 vs 数值变量的统计检验"""
        numeric_cols = self.meta['numeric']
        categorical_cols = [c for c in self.meta['categorical']
                           if self.meta['cat_info'][c]['unique'] <= self.max_categories
                           and self.meta['cat_info'][c]['unique'] >= 2]
        if not numeric_cols or not categorical_cols:
            return None

        results = []
        charts = []
        for num_col in numeric_cols[:8]:  # 控制组合数
            for cat_col in categorical_cols[:8]:
                n_cats = self.meta['cat_info'][cat_col]['unique']
                groups = [self.df[self.df[cat_col] == v][num_col].dropna()
                         for v in self.df[cat_col].unique() if len(self.df[self.df[cat_col] == v]) >= 5]
                if len(groups) < 2:
                    continue

                test_result = {'numeric': num_col, 'categorical': cat_col, 'n_groups': n_cats}

                # ===== 正态性检验 (Shapiro-Wilk) =====
                # 策略: n<3 无法检验; 3<=n<=2000 用 Shapiro-Wilk;
                # n>2000 跳过检验，依赖中心极限定理(CLT)，默认使用参数方法
                SHAPIRO_MAX_N = 2000
                all_normal = True
                normality_details = []
                for g in groups:
                    n_g = len(g)
                    if n_g < 3:
                        all_normal = False
                        normality_details.append({'n': n_g, 'normal': False, 'note': 'n<3'})
                    elif n_g > SHAPIRO_MAX_N:
                        normality_details.append({'n': n_g, 'normal': True,
                                                  'note': f'n>{SHAPIRO_MAX_N}, CLT applies'})
                    else:
                        s_stat, s_p = stats.shapiro(g)
                        is_normal = s_p > 0.05
                        normality_details.append({
                            'n': n_g, 'shapiro_stat': round(s_stat, 3),
                            'shapiro_p': round(s_p, 4), 'normal': is_normal
                        })
                        if not is_normal:
                            all_normal = False
                test_result['normality_met'] = all_normal
                test_result['normality_details'] = normality_details

                # ===== 根据正态性选择参数/非参数检验 =====
                if n_cats == 2:
                    if all_normal:
                        t_stat, p_val = stats.ttest_ind(groups[0], groups[1])
                        test_result['method'] = 't-test'
                        test_result['test_statistic'] = round(t_stat, 3)
                        test_result['statistic_name'] = 't'
                        test_result['t_statistic'] = round(t_stat, 3)
                    else:
                        u_stat, p_val = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                        test_result['method'] = 'Mann-Whitney U'
                        test_result['test_statistic'] = round(u_stat, 3)
                        test_result['statistic_name'] = 'U'
                else:
                    if all_normal:
                        f_stat, p_val = stats.f_oneway(*groups)
                        test_result['method'] = 'ANOVA'
                        test_result['test_statistic'] = round(f_stat, 3)
                        test_result['statistic_name'] = 'F'
                        test_result['f_statistic'] = round(f_stat, 3)
                    else:
                        h_stat, p_val = stats.kruskal(*groups)
                        test_result['method'] = 'Kruskal-Wallis'
                        test_result['test_statistic'] = round(h_stat, 3)
                        test_result['statistic_name'] = 'H'

                test_result['p_value'] = round(p_val, 4)
                test_result['significant'] = bool(p_val < 0.05)

                # ===== 事后检验 =====
                if test_result['significant']:
                    unique_vals = sorted(self.df[cat_col].dropna().unique())
                    posthoc_groups, posthoc_labels = [], []
                    for v in unique_vals:
                        g = self.df[self.df[cat_col] == v][num_col].dropna()
                        if len(g) >= 5:
                            posthoc_groups.append(g.values)
                            posthoc_labels.append(str(v))

                    if test_result['method'] == 'ANOVA' and len(posthoc_groups) >= 3:
                        # Tukey HSD（参数方法）
                        try:
                            from scipy.stats import tukey_hsd
                            tukey_res = tukey_hsd(*posthoc_groups)
                            posthoc = []
                            for i in range(len(posthoc_labels)):
                                for j in range(i + 1, len(posthoc_labels)):
                                    if tukey_res.pvalue[i][j] < 0.05:
                                        posthoc.append({
                                            'group1': posthoc_labels[i],
                                            'group2': posthoc_labels[j],
                                            'mean_diff': round(float(
                                                posthoc_groups[i].mean() - posthoc_groups[j].mean()), 3),
                                            'p_value': round(float(tukey_res.pvalue[i][j]), 4),
                                            'method': 'Tukey HSD'
                                        })
                            if posthoc:
                                test_result['posthoc_pairs'] = posthoc
                        except ImportError:
                            pass

                    elif test_result['method'] == 'Kruskal-Wallis' and len(posthoc_groups) >= 3:
                        # Pairwise Mann-Whitney U + Bonferroni 校正（非参数方法）
                        n_pairs = len(posthoc_groups) * (len(posthoc_groups) - 1) // 2
                        bonf_alpha = 0.05 / n_pairs if n_pairs > 0 else 0.05
                        posthoc = []
                        for i in range(len(posthoc_labels)):
                            for j in range(i + 1, len(posthoc_labels)):
                                u, p = stats.mannwhitneyu(posthoc_groups[i], posthoc_groups[j],
                                                         alternative='two-sided')
                                if p < bonf_alpha:
                                    posthoc.append({
                                        'group1': posthoc_labels[i],
                                        'group2': posthoc_labels[j],
                                        'mean_diff': round(float(
                                            posthoc_groups[i].mean() - posthoc_groups[j].mean()), 3),
                                        'p_value': round(float(p), 4),
                                        'bonferroni_alpha': round(bonf_alpha, 6),
                                        'method': 'Mann-Whitney U (Bonferroni)'
                                    })
                        if posthoc:
                            test_result['posthoc_pairs'] = posthoc

                # 组均值
                group_means = self.df.groupby(cat_col)[num_col].mean().round(2).to_dict()
                test_result['group_means'] = {k: float(v) for k, v in group_means.items()}

                # ===== 标准化效应量 =====
                n_total = sum(len(g) for g in groups)
                if n_cats == 2:
                    # Cohen's d
                    n1, n2 = len(groups[0]), len(groups[1])
                    var1, var2 = groups[0].var(), groups[1].var()
                    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
                    if pooled_std > 0:
                        d_val = abs(groups[0].mean() - groups[1].mean()) / pooled_std
                    else:
                        d_val = 0.0
                    test_result['effect_size'] = round(d_val, 3)
                    test_result['effect_size_name'] = "Cohen's d"
                    test_result['effect_size_interpretation'] = (
                        'large' if d_val > 0.8 else 'medium' if d_val > 0.5 else 'small'
                    )
                else:
                    if test_result['method'] in ('ANOVA', 'Kruskal-Wallis'):
                        grand_mean = np.concatenate([g for g in groups]).mean()
                        if test_result['method'] == 'ANOVA':
                            ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
                            ss_total = sum(((g - grand_mean)**2).sum() for g in groups)
                            eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
                            test_result['effect_size_name'] = 'η² (eta-squared)'
                        else:
                            # Kruskal-Wallis: η²_H approximation
                            H = test_result['test_statistic']
                            eta_sq = H / (n_total - 1) if n_total > 1 else 0.0
                            test_result['effect_size_name'] = 'η²_H (Kruskal-Wallis)'
                        test_result['effect_size'] = round(eta_sq, 3)
                        test_result['effect_size_interpretation'] = (
                            'large' if eta_sq > 0.14 else 'medium' if eta_sq > 0.06 else 'small'
                        )
                    else:
                        # Mann-Whitney U: r = Z / sqrt(N)
                        z_val = abs(stats.norm.ppf(max(p_val / 2, 1e-10)))
                        r_val = z_val / np.sqrt(n_total)
                        test_result['effect_size'] = round(r_val, 3)
                        test_result['effect_size_name'] = 'r (rank-biserial)'
                        test_result['effect_size_interpretation'] = (
                            'large' if r_val > 0.5 else 'medium' if r_val > 0.3 else 'small'
                        )

                results.append(test_result)

                # 生成箱线图
                if test_result['significant']:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    plot_data = [self.df[self.df[cat_col] == v][num_col].dropna()
                                for v in sorted(self.df[cat_col].unique())]
                    bp = ax.boxplot(plot_data, labels=sorted(self.df[cat_col].unique()),
                                    patch_artist=True, widths=0.5)
                    for patch in bp['boxes']:
                        patch.set_facecolor('lightcoral' if p_val < 0.05 else 'lightgray')
                    ax.set_title(f'{num_col} by {cat_col}\n({test_result["method"]}, p={p_val:.4f})',
                                fontsize=12, fontweight='bold')
                    ax.set_xlabel(cat_col)
                    ax.set_ylabel(num_col)
                    plt.tight_layout()
                    fname = f'comparison_{self._safe_name(num_col)}_{self._safe_name(cat_col)}.png'
                    plt.savefig(os.path.join(self.output_dir, fname), dpi=120, bbox_inches='tight')
                    plt.close()
                    charts.append({'file': fname, 'test_result': test_result})

        # 排序：显著的排前面
        results.sort(key=lambda x: x['p_value'])
        # Bonferroni 多重比较校正
        n_tests = len(results)
        bonferroni_alpha = 0.05 / n_tests if n_tests > 0 else 0.05
        n_sig_bonferroni = 0
        for r in results:
            r['bonferroni_alpha'] = round(bonferroni_alpha, 6)
            r['significant_bonferroni'] = r['p_value'] < bonferroni_alpha
            if r['significant_bonferroni']:
                n_sig_bonferroni += 1
        # 保存
        def _make_json_safe(obj):
            """递归转换 numpy 类型为 Python 原生类型"""
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, dict):
                return {str(k): _make_json_safe(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_make_json_safe(v) for v in obj]
            return obj

        with open(os.path.join(self.output_dir, 'group_comparisons.json'), 'w', encoding='utf-8') as f:
            json.dump(_make_json_safe(results), f, ensure_ascii=False, indent=2)
        return {'results': results, 'charts': charts,
                'n_significant': sum(1 for r in results if r['significant']),
                'n_significant_bonferroni': n_sig_bonferroni}

    # ======================== 6. 客户聚类 ========================
    def clustering(self):
        """K-Means 聚类 (≥2个数值变量时)"""
        numeric_cols = self.meta['numeric']
        if len(numeric_cols) < 2:
            return None

        # 选择聚类特征
        cluster_cols = [c for c in numeric_cols if self.df[c].nunique() > 2][:6]
        if len(cluster_cols) < 2:
            return None

        X = self.df[cluster_cols].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 肘部法则 + 轮廓系数
        inertias = []
        silhouettes = []
        K_range = range(1, min(11, len(X)))
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)
            if k >= 2:
                silhouettes.append(silhouette_score(X_scaled, labels))

        # 自动选K：肘部法则 + 轮廓系数交叉验证
        if len(inertias) >= 4:
            diffs = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
            diffs2 = [diffs[i] - diffs[i+1] for i in range(len(diffs)-1)]
            k_elbow = diffs2.index(max(diffs2)) + 2
        else:
            k_elbow = 3

        # 找出轮廓系数最高的 K
        if silhouettes:
            k_silhouette = silhouettes.index(max(silhouettes)) + 2  # +2 because K_range starts at 2
        else:
            k_silhouette = k_elbow

        # 综合选择：优先轮廓系数，但如果肘部 K 更大且轮廓系数差距小(<0.05)，选更大的
        best_sil = max(silhouettes) if silhouettes else 0
        sil_at_elbow = silhouettes[k_elbow - 2] if silhouettes and k_elbow >= 2 and k_elbow - 2 < len(silhouettes) else 0
        if k_elbow > k_silhouette and best_sil - sil_at_elbow < 0.05:
            k_optimal = k_elbow
        else:
            k_optimal = k_silhouette

        k_optimal = max(2, min(6, k_optimal))

        km = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
        clusters = km.fit_predict(X_scaled)
        final_silhouette = silhouette_score(X_scaled, clusters) if k_optimal >= 2 else 0

        # 聚类画像
        cluster_df = X.copy()
        cluster_df['Cluster'] = clusters
        profile = cluster_df.groupby('Cluster').agg(['mean', 'std', 'count']).round(2)

        # PCA可视化
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # PCA图
        scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters,
                                  cmap='viridis', alpha=0.6, s=15)
        axes[0].set_title(f'Customer Segmentation — PCA (K={k_optimal})', fontsize=13, fontweight='bold')
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.colorbar(scatter, ax=axes[0], label='Cluster')

        # 肘部法则 + 轮廓系数
        ax2 = axes[1]
        ax2.plot(list(K_range), inertias, 'bo-', markersize=8, linewidth=2, label='Inertia')
        ax2.axvline(x=k_optimal, color='red', linestyle='--', label=f'Optimal K={k_optimal}')
        ax2.set_title('Elbow Method + Silhouette', fontsize=13, fontweight='bold')
        ax2.set_xlabel('K')
        ax2.set_ylabel('Inertia', color='blue')
        ax2.legend(loc='upper right')

        if silhouettes:
            ax3 = ax2.twinx()
            ax3.plot(list(K_range)[1:], silhouettes, 'rs-', markersize=6, linewidth=1.5, label='Silhouette')
            ax3.set_ylabel('Silhouette Score', color='red')
            ax3.legend(loc='lower right')

        plt.tight_layout()
        fname = 'clustering.png'
        plt.savefig(os.path.join(self.output_dir, fname), dpi=150, bbox_inches='tight')
        plt.close()

        # 聚类质量评估
        silhouette_quality = 'good' if final_silhouette > 0.5 else 'fair' if final_silhouette > 0.25 else 'poor'

        # 各聚类均值（用于报告）
        cluster_summary = cluster_df.groupby('Cluster').mean().round(2).to_dict()

        return {
            'k_optimal': k_optimal,
            'k_elbow': k_elbow,
            'k_silhouette': k_silhouette,
            'silhouette_score': round(final_silhouette, 4),
            'silhouette_quality': silhouette_quality,
            'all_silhouettes': [round(s, 4) for s in silhouettes],
            'cluster_sizes': pd.Series(clusters).value_counts().to_dict(),
            'cluster_means': cluster_summary,
            'cluster_features': cluster_cols,
            'pca_variance': pca.explained_variance_ratio_.tolist(),
            'chart_file': fname,
            'profile': profile
        }

    # ======================== 7. 驱动力分析 ========================
    def driver_analysis(self, target_cols=None, max_onehot_features=20):
        """
        驱动力分析: 使用多元线性回归 + 标准化系数排名，识别影响目标变量的关键驱动因素。

        与相关性分析的区别：相关性是一对一的，回归是"控制其他变量后"的净效应。
        标准化系数（Beta权重）可以直接比较不同量纲特征的相对重要性。

        参数:
            target_cols: 目标变量列表。None 时自动选择 KPI 相关列
                         (含 amount/value/revenue/score/rating/rate/ROI 等关键词的列)
            max_onehot_features: one-hot 编码分类变量数量上限
        返回:
            dict with per-target: r², adj_r², 模型显著性, 特征重要性排名
        """
        numeric_cols = self.meta['numeric']
        categorical_cols = [c for c in self.meta['categorical']
                           if self.meta['cat_info'][c]['unique'] <= self.max_categories
                           and self.meta['cat_info'][c]['unique'] >= 2]

        if len(numeric_cols) < 2:
            return None

        # 自动选择目标变量: KPI 关键词匹配
        if target_cols is None:
            kpi_keywords = ['amount', 'value', 'revenue', 'score', 'rating', 'spend',
                           'purchase', 'sales', 'price', 'cost', 'roi', 'rate',
                           'conversion', 'frequency', 'order', 'click']
            target_cols = []
            for c in numeric_cols:
                c_lower = c.lower().replace(' ', '_').replace('(', '').replace(')', '')
                if any(kw in c_lower for kw in kpi_keywords):
                    target_cols.append(c)
            if not target_cols:
                target_cols = numeric_cols[:3]  # fallback: 取前三个数值列

        target_cols = [t for t in target_cols if t in numeric_cols][:3]

        all_results = []
        for target in target_cols:
            feature_numeric = [c for c in numeric_cols if c != target]
            feature_cat = categorical_cols[:max_onehot_features]

            # 构建建模数据集
            cols_needed = [target] + feature_numeric + feature_cat
            df_model = self.df[cols_needed].dropna()

            n_samples = len(df_model)
            if n_samples < 30:
                continue

            # 数值特征标准化
            X_num = df_model[feature_numeric].values
            scaler_X = StandardScaler()
            X_num_scaled = scaler_X.fit_transform(X_num)

            # 分类特征 One-Hot
            X_cat_parts = []
            cat_names = []
            for cc in feature_cat:
                dummies = pd.get_dummies(df_model[cc], prefix=cc, drop_first=True).astype(float)
                X_cat_parts.append(dummies.values)
                cat_names.extend(list(dummies.columns))

            all_feature_names = feature_numeric + cat_names

            X = np.hstack([X_num_scaled] + X_cat_parts) if X_cat_parts else X_num_scaled
            y = df_model[target].values
            y_scaled = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()

            n_features = X.shape[1]
            if n_features < 1 or n_features >= n_samples - 1:
                continue  # 特征太多或太少，跳过

            # 拟合回归
            model = LinearRegression()
            model.fit(X, y_scaled)
            y_pred = model.predict(X)

            # 模型拟合度
            r2 = float(model.score(X, y_scaled))
            adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
            f_stat = (r2 / n_features) / ((1 - r2) / (n_samples - n_features - 1)) if r2 < 1 else 0
            f_pvalue = float(stats.f.sf(f_stat, n_features, n_samples - n_features - 1))

            # 系数标准误 & p值（用于判断单个特征是否显著）
            residuals = y_scaled - y_pred
            mse = np.sum(residuals ** 2) / (n_samples - n_features - 1)
            X_aug = np.hstack([np.ones((n_samples, 1)), X])
            try:
                cov_mat = mse * np.linalg.inv(X_aug.T @ X_aug)
            except np.linalg.LinAlgError:
                cov_mat = mse * np.linalg.pinv(X_aug.T @ X_aug)
            se = np.sqrt(np.diag(cov_mat))[1:]  # 去掉截距项

            coefs = model.coef_
            t_stats = coefs / np.maximum(se, 1e-10)
            p_values = 2 * stats.t.sf(np.abs(t_stats), n_samples - n_features - 1)

            # 构建驱动因素排名
            drivers = []
            for i, name in enumerate(all_feature_names):
                drivers.append({
                    'feature': name,
                    'coefficient': round(float(coefs[i]), 4),
                    'std_error': round(float(se[i]), 4),
                    't_statistic': round(float(t_stats[i]), 3),
                    'p_value': round(float(p_values[i]), 4),
                    'significant': bool(p_values[i] < 0.05),
                    'abs_importance': abs(float(coefs[i]))
                })
            drivers.sort(key=lambda x: x['abs_importance'], reverse=True)

            # 可视化: Top 15 驱动因素
            top_n = min(15, len(drivers))
            top_drivers = drivers[:top_n]
            fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
            colors = ['#2ecc71' if d['coefficient'] > 0 else '#e74c3c' for d in top_drivers]
            ax.barh(range(top_n), [d['coefficient'] for d in top_drivers], color=colors)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels([d['feature'] for d in top_drivers], fontsize=9)
            ax.axvline(x=0, color='black', linewidth=0.5)
            ax.set_xlabel('Standardized Coefficient (Beta)', fontsize=11)
            ax.set_title(f'Driver Analysis — {target}\n'
                        f'(R²={r2:.3f}, Adj R²={adj_r2:.3f}, n={n_samples})',
                        fontsize=13, fontweight='bold')
            ax.invert_yaxis()
            # 添加系数标签
            for idx, d in enumerate(top_drivers):
                sign = '+' if d['coefficient'] > 0 else ''
                ax.text(d['coefficient'] + (0.01 if d['coefficient'] >= 0 else -0.01),
                       idx, f'{sign}{d["coefficient"]:.3f}', va='center', fontsize=8)
            plt.tight_layout()
            fname = f'driver_{self._safe_name(target)}.png'
            plt.savefig(os.path.join(self.output_dir, fname), dpi=150, bbox_inches='tight')
            plt.close()

            all_results.append({
                'target': target,
                'n_samples': n_samples,
                'n_features': n_features,
                'feature_names_used': all_feature_names,
                'r_squared': round(r2, 4),
                'adj_r_squared': round(adj_r2, 4),
                'f_statistic': round(f_stat, 3),
                'f_pvalue': round(f_pvalue, 4),
                'model_significant': bool(f_pvalue < 0.05),
                'drivers': drivers,
                'chart_file': fname
            })

        return {'results': all_results} if all_results else None

    # ======================== 8. 综合运行 ========================
    def run_all(self):
        """一键运行全部分析"""
        print(f"[AutoAnalyzer] Loading: {self.csv_path}")
        meta = self.load_and_detect()
        print(f"  Detected: {len(meta['numeric'])} numeric, {len(meta['categorical'])} categorical, "
              f"{len(meta['datetime'])} datetime, {len(meta['id'])} id columns")
        print(f"  Rows: {meta['total_rows']}, Duplicates: {meta['duplicated_rows']}")

        print("[AutoAnalyzer] Running data quality checks...")
        dq = self.data_quality()
        n_miss = len(dq['missing'])
        n_out = len(dq['outliers'])
        print(f"  Missing value issues: {n_miss} columns, Outlier issues: {n_out} columns")

        results = {'meta': meta}

        print("[AutoAnalyzer] Running descriptive statistics...")
        results['descriptive'] = self.descriptive_stats()

        print("[AutoAnalyzer] Generating univariate charts...")
        results['univariate_charts'] = self.univariate_plots()

        print("[AutoAnalyzer] Running correlation analysis...")
        results['correlation'] = self.correlation_analysis()

        print("[AutoAnalyzer] Running group comparisons...")
        results['group_comparison'] = self.group_comparison()

        print("[AutoAnalyzer] Running clustering...")
        results['clustering'] = self.clustering()

        print("[AutoAnalyzer] Running driver analysis...")
        results['driver_analysis'] = self.driver_analysis()

        print(f"[AutoAnalyzer] Done! All outputs in: {self.output_dir}/")
        return results

    # ======================== Helper ========================
    @staticmethod
    def _safe_name(s):
        return s.replace('/', '_').replace('\\', '_').replace(' ', '_')[:50]


# ======================== CLI入口 ========================
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python auto_analyzer.py <path_to_csv> [output_dir]")
        print("Example: python auto_analyzer.py data/my_data.csv output")
        sys.exit(1)

    csv_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else 'output'
    analyzer = AutoAnalyzer(csv_path, output_dir=out_dir)
    analyzer.run_all()
