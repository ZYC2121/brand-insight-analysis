"""
============================================================
自动分析报告生成器
基于 AutoAnalyzer 的输出结果，填充模板生成 Markdown 报告
============================================================
"""
import os, json
from datetime import datetime


def generate_report(analysis_results, output_path='output/analysis_report.md'):
    """根据分析结果自动生成 Markdown 报告"""
    meta = analysis_results['meta']
    desc = analysis_results.get('descriptive')
    corr = analysis_results.get('correlation')
    grp = analysis_results.get('group_comparison')
    clust = analysis_results.get('clustering')

    lines = []
    def w(s=''): lines.append(s)

    # ===== 标题页 =====
    w(f'# 数据分析报告')
    w(f'> 自动生成于 {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    w(f'> 数据规模: {meta["total_rows"]:,} 行 × {meta["total_cols"]} 列')
    w('')

    # ===== 1. 数据概览 =====
    w('---')
    w('## 一、数据概览')
    w('')
    w(f'- **总记录数**: {meta["total_rows"]:,}')
    w(f'- **总变量数**: {meta["total_cols"]}')
    w(f'- **数值型变量**: {len(meta["numeric"])} 个 — {", ".join(meta["numeric"][:8])}')
    w(f'- **分类型变量**: {len(meta["categorical"])} 个 — {", ".join(meta["categorical"][:8])}')
    if meta['datetime']:
        w(f'- **日期型变量**: {len(meta["datetime"])} 个 — {", ".join(meta["datetime"])}')
    if meta['id']:
        w(f'- **ID型变量**: {len(meta["id"])} 个 — {", ".join(meta["id"])}')
    w(f'- **重复行**: {meta["duplicated_rows"]}')
    w('')

    # 数据质量报告
    dq = meta.get('data_quality', {})
    if dq:
        missing_info = dq.get('missing', {})
        outlier_info = dq.get('outliers', {})

        has_missing = any(v['severity'] in ('high', 'medium') for v in missing_info.values())
        has_outliers = any(v['severity'] in ('high', 'medium') for v in outlier_info.values())

        if has_missing or has_outliers or missing_info or outlier_info:
            w('### ⚠️ 数据质量报告')
            w('')

        if missing_info:
            high_miss = {k: v for k, v in missing_info.items() if v['severity'] == 'high'}
            med_miss = {k: v for k, v in missing_info.items() if v['severity'] == 'medium'}
            if high_miss or med_miss:
                w('**缺失值警告:**')
                w('| 列名 | 缺失数 | 缺失率 | 严重程度 |')
                w('|------|--------|--------|----------|')
                for k, v in {**high_miss, **med_miss}.items():
                    flag = '🔴 高' if v['severity'] == 'high' else '🟡 中'
                    w(f'| {k} | {v["count"]} | {v["percentage"]}% | {flag} |')
                w('')

        if outlier_info:
            high_out = {k: v for k, v in outlier_info.items() if v['severity'] == 'high'}
            med_out = {k: v for k, v in outlier_info.items() if v['severity'] == 'medium'}
            if high_out or med_out:
                w('**异常值警告 (IQR 方法):**')
                w('| 列名 | 异常值数 | 占比 | 正常范围 | 严重程度 |')
                w('|------|----------|------|----------|----------|')
                for k, v in {**high_out, **med_out}.items():
                    flag = '🔴 高' if v['severity'] == 'high' else '🟡 中'
                    w(f'| {k} | {v["count"]} | {v["percentage"]}% | '
                      f'{v["lower_bound"]} ~ {v["upper_bound"]} | {flag} |')
                w('')

        if not (has_missing or has_outliers):
            w('> ✅ 未发现严重的数据质量问题。')
            w('')

    # ===== 2. 描述统计 =====
    if desc:
        w('---')
        w('## 二、描述统计分析')
        w('')
        w('### 数值变量统计摘要')
        w('')
        w('| 变量 | 均值 | 标准差 | 最小值 | 25% | 50% | 75% | 最大值 | 偏度 | 峰度 |')
        w('|------|------|--------|--------|-----|-----|-----|--------|------|------|')
        stats_table = desc['table']
        for col in meta['numeric']:
            try:
                row = stats_table[col]
                w(f'| {col} | {row.get("mean","-"):.2f} | {row.get("std","-"):.2f} | {row.get("min","-"):.1f} | '
                  f'{row.get("25%","-"):.1f} | {row.get("50%","-"):.1f} | {row.get("75%","-"):.1f} | '
                  f'{row.get("max","-"):.1f} | {row.get("skewness","-"):.2f} | {row.get("kurtosis","-"):.2f} |')
            except:
                w(f'| {col} | - | - | - | - | - | - | - | - | - |')
        w('')

    # ===== 3. 相关性分析 =====
    if corr and corr.get('high_correlations'):
        w('---')
        w('## 三、相关性分析')
        w('')
        w('> 同时计算 **Pearson**（线性相关）和 **Spearman**（单调相关），两者差异大提示可能存在非线性关系。')
        w('')
        w(f'![相关性热力图]({corr["chart_file"]})')
        w('')
        if corr.get('high_correlations'):
            w('### 显著相关变量对 (|Pearson| > 0.3)')
            w('| 变量对 | Pearson | Spearman |')
            w('|--------|---------|----------|')
            for hc in corr['high_correlations'][:10]:
                p_dir = ':small_red_triangle:' if hc['pearson'] > 0 else ':small_red_triangle_down:'
                s_dir = ':small_red_triangle:' if hc['spearman'] > 0 else ':small_red_triangle_down:'
                w(f'| {hc["pair"]} | {p_dir} {hc["pearson"]:.3f} | {s_dir} {hc["spearman"]:.3f} |')
            w('')

        if corr.get('nonlinear_hints'):
            w('### ⚠️ 非线性关系提示')
            w('以下变量对的 Pearson 和 Spearman 差异较大 (>0.2)，提示可能存在**非线性单调关系**：')
            w('| 变量对 | Pearson | Spearman | 差异 |')
            w('|--------|---------|----------|------|')
            for nh in corr['nonlinear_hints'][:8]:
                w(f'| {nh["pair"]} | {nh["pearson"]:.3f} | {nh["spearman"]:.3f} | {nh["difference"]:.3f} |')
            w('')

    # ===== 4. 分组对比分析 =====
    if grp and grp.get('results'):
        w('---')
        w('## 四、分组对比分析（统计检验）')
        w('')
        w('> **方法选择逻辑**: 先对每组数据做 Shapiro-Wilk 正态性检验 →')
        w('> 全部通过正态性 → 使用参数检验（t-test / ANOVA + Tukey HSD）；')
        w('> 任一组未通过 → 自动回退为非参数检验（Mann-Whitney U / Kruskal-Wallis + pairwise MWU with Bonferroni）。')
        w('')

        # Bonferroni 校正说明
        n_total = len(grp['results'])
        if n_total > 1:
            w(f'> ⚠️ 共进行 **{n_total}** 次统计检验，采用 **Bonferroni 校正**控制多重比较错误率'
              f'（校正后 α = 0.05/{n_total} ≈ {grp["results"][0].get("bonferroni_alpha", 0.05/n_total):.6f}）。'
              f'通过 Bonferroni 校正的发现可信度更高。')
            w('')

        sig_results = [r for r in grp['results'] if r['significant']]
        bonf_results = [r for r in grp['results'] if r.get('significant_bonferroni', False)]

        if bonf_results:
            w(f'### ✅ 通过 Bonferroni 校正的显著发现 ({len(bonf_results)} 项)')
            w('')
            es_name = bonf_results[0].get('effect_size_name', 'Effect Size') if bonf_results else 'Effect Size'
            w(f'| 数值变量 | 分组变量 | 检验方法 | p值 | {es_name} | 解读 | Bonferroni |')
            w('|----------|----------|----------|-----|--------|------|------------|')
            for r in bonf_results:
                es_val = r.get('effect_size', 0)
                es_interp = r.get('effect_size_interpretation', '')
                w(f'| {r["numeric"]} | {r["categorical"]} | {r["method"]} | {r["p_value"]:.4f} | '
                  f'{es_val:.3f} | {es_interp} | ✅ 通过 |')
            w('')
            for r in bonf_results[:5]:
                w(f'**{r["numeric"]} by {r["categorical"]}** (p={r["p_value"]:.4f})')
                for grp_name, grp_mean in r['group_means'].items():
                    w(f'- {grp_name}: {grp_mean}')
                # Tukey HSD 事后检验结果
                if r.get('posthoc_pairs'):
                    w('')
                    w('*事后检验（Tukey HSD）— 显著差异组对：*')
                    for pair in r['posthoc_pairs'][:10]:
                        direction = '>' if pair['mean_diff'] > 0 else '<'
                        w(f'- {pair["group1"]} {direction} {pair["group2"]} '
                          f'(均值差={pair["mean_diff"]:.2f}, p={pair["p_value"]:.4f})')
                w('')

        # 未校正显著但未通过Bonferroni的（弱信号）
        weak_sig = [r for r in sig_results if not r.get('significant_bonferroni', False)]
        if weak_sig:
            w(f'### ⚡ 弱信号 — 未校正显著但未通过 Bonferroni 校正 ({len(weak_sig)} 项)')
            w('以下发现 p<0.05 但未通过多重比较校正，建议谨慎解读，可在更大样本中验证：')
            w('')
            w('| 数值变量 | 分组变量 | 检验方法 | p值 | 效应量 |')
            w('|----------|----------|----------|-----|--------|')
            for r in weak_sig:
                w(f'| {r["numeric"]} | {r["categorical"]} | {r["method"]} | '
                  f'{r["p_value"]:.4f} | {r["effect_size"]:.3f} |')
            w('')

        non_sig = [r for r in grp['results'] if not r['significant']]
        if non_sig:
            w(f'### 无显著差异 ({len(non_sig)} 项，p>=0.05)')
            w(f'以下变量组合未通过显著性检验，提示这些因素对目标变量的单独影响有限：')
            for r in non_sig[:5]:
                w(f'- {r["numeric"]} × {r["categorical"]}:  {r["method"]} p={r["p_value"]:.4f}')
            w('')

    # ===== 5. 驱动力分析 =====
    driver = analysis_results.get('driver_analysis')
    if driver and driver.get('results'):
        w('---')
        w('## 五、驱动力分析 — 什么因素在驱动关键指标？')
        w('')
        w('> 使用多元线性回归 + 标准化系数（Beta权重），控制其他变量后衡量每个因素的**独立贡献**。')
        w('> 与相关性分析的区别：相关性是"一对一"的，回归是"控制其他变量后"的净效应。')
        w('')

        for d_result in driver['results']:
            target = d_result['target']
            w(f'### 目标变量: {target}')
            w(f'- 建模样本: {d_result["n_samples"]} 条')
            w(f'- 特征数量: {d_result["n_features"]} 个')
            w(f'- R² = {d_result["r_squared"]:.3f}（模型解释了 {d_result["r_squared"]*100:.1f}% 的变异）')
            w(f'- 调整 R² = {d_result["adj_r_squared"]:.3f}')
            w(f'- 模型显著性: {"p < 0.05 ✅" if d_result["model_significant"] else "未通过显著性检验"} '
              f'(F={d_result["f_statistic"]:.3f}, p={d_result["f_pvalue"]:.4f})')
            w('')
            w(f'![驱动力分析]({d_result["chart_file"]})')
            w('')
            w('### 驱动因素排名')
            w('')
            w('| 排名 | 特征 | 标准化系数 | 方向 | p值 | 显著性 |')
            w('|------|------|-----------|------|-----|--------|')
            for rank, d in enumerate(d_result['drivers'][:15], 1):
                direction = ':arrow_up: 正向' if d['coefficient'] > 0 else ':arrow_down: 负向'
                sig_mark = '✅' if d['significant'] else ''
                w(f'| {rank} | {d["feature"]} | {d["coefficient"]:.4f} | {direction} | '
                  f'{d["p_value"]:.4f} | {sig_mark} |')
            w('')

            # 业务解读
            top3 = d_result['drivers'][:3]
            if top3:
                w('**关键洞察：**')
                for i, d in enumerate(top3):
                    direction_text = '正向推动' if d['coefficient'] > 0 else '负向抑制'
                    sig_text = '显著' if d['significant'] else '未达显著水平'
                    w(f'{i+1}. **{d["feature"]}** 对 {target} 的{direction_text}力最强 '
                      f'(Beta={d["coefficient"]:.3f}, {sig_text})')
            w('')

    # ===== 6. 客户聚类 =====
    if clust:
        w('---')
        w('## 六、客户聚类分析')
        w('')
        w(f'使用 K-Means 算法（K={clust["k_optimal"]}）对 {len(clust["cluster_features"])} 个特征进行聚类。')
        w(f'')
        if clust.get('silhouette_score'):
            sil = clust['silhouette_score']
            sil_q = clust.get('silhouette_quality', '')
            w(f'- **轮廓系数 (Silhouette Score)**: {sil:.4f}（{sil_q}）')
            w(f'  > 轮廓系数范围 [-1, 1]，越接近 1 聚类质量越好。>0.5 良好，>0.25 可接受，<0.25 聚类结构较弱。')
        w(f'')
        w(f'![聚类分析]({clust["chart_file"]})')
        w('')
        w('### 聚类画像')
        w('')
        w(f'| 聚类 | 人数 | 占比 |')
        w(f'|------|------|------|')
        total = sum(clust['cluster_sizes'].values())
        for c_id in sorted(clust['cluster_sizes'].keys()):
            size = clust['cluster_sizes'][c_id]
            w(f'| Cluster {c_id} | {size} | {size/total*100:.1f}% |')
        w('')

    # ===== 7. 关键发现总结 =====
    w('---')
    w('## 七、关键发现总结')
    w('')
    findings = []

    # 从驱动力分析提取
    driver = analysis_results.get('driver_analysis')
    if driver and driver.get('results'):
        for d_result in driver['results']:
            top3 = d_result['drivers'][:3]
            r2_pct = d_result['r_squared'] * 100
            findings.append(f'### 发现一：{d_result["target"]} 的关键驱动因素')
            findings.append(f'- 模型解释了 **{r2_pct:.1f}%** 的变异（R²={d_result["r_squared"]:.3f}）')
            for i, d in enumerate(top3, 1):
                direction = '正向推动' if d['coefficient'] > 0 else '负向抑制'
                findings.append(f'- **#{i}: {d["feature"]}**（{direction}，Beta={d["coefficient"]:.3f}）')
            findings.append('')
        finding_offset = 2  # 后续发现从"发现二"开始
    else:
        finding_offset = 1  # 无驱动力分析，从"发现一"开始

    # 从显著性检验提取（优先使用 Bonferroni 校正后结果）
    if grp and grp.get('results'):
        bonf_count = sum(1 for r in grp['results'] if r.get('significant_bonferroni', False))
        raw_sig_count = sum(1 for r in grp['results'] if r['significant'])
        total_count = len(grp['results'])
        if bonf_count > 0:
            top_sig = [r for r in grp['results'] if r.get('significant_bonferroni', False)][:3]
            findings.append(f'### 发现二：{bonf_count}/{total_count} 项分组检验存在显著差异（Bonferroni校正后）')
            for r in top_sig:
                ratio = max(r["group_means"].values())/min(r["group_means"].values()) if min(r["group_means"].values()) > 0 else float('inf')
                findings.append(f'- **{r["categorical"]}** 对 **{r["numeric"]}** 具有显著影响 (p={r["p_value"]:.4f})，'
                               f'最高组均值是最低组的 {ratio:.1f} 倍')
            # 附上最关键的事后检验发现
            for r in top_sig:
                if r.get('posthoc_pairs') and len(r['posthoc_pairs']) > 0:
                    top_pair = r['posthoc_pairs'][0]
                    findings.append(f'  事后检验：差异最大的组对为 {top_pair["group1"]} vs {top_pair["group2"]} '
                                   f'(均值差={top_pair["mean_diff"]:.2f}, p={top_pair["p_value"]:.4f})')
                    break
        elif raw_sig_count > 0:
            findings.append(f'### 发现二：{raw_sig_count}/{total_count} 项未校正显著，但均未通过Bonferroni校正')
            findings.append(f'- 提示这些差异信号较弱，建议扩大样本量或通过更严格的实验设计进一步验证')
        else:
            findings.append(f'### 发现二：未发现显著的分组差异')
            findings.append(f'- 全部 {total_count} 项分组检验均未通过显著性阈值 (p>=0.05)，')
            findings.append(f'  提示各组在目标指标上的差异可能由随机因素造成。')

    # 从相关性提取
    if corr and corr.get('high_correlations'):
        top_corr = corr['high_correlations'][:3]
        findings.append(f'### 发现三：{len(corr["high_correlations"])} 对变量存在中等以上相关性')
        for hc in top_corr:
            findings.append(f'- {hc["pair"]}: r={hc["correlation"]:.3f}')

    # 从聚类提取
    if clust:
        sizes = list(clust['cluster_sizes'].values())
        findings.append(f'### 发现四：识别出 {clust["k_optimal"]} 个差异化客群')
        findings.append(f'- 最小客群: {min(sizes)} 人 ({min(sizes)/sum(sizes)*100:.1f}%)')
        findings.append(f'- 最大客群: {max(sizes)} 人 ({max(sizes)/sum(sizes)*100:.1f}%)')

    for f_text in findings:
        w(f_text)
        w('')

    # ===== 8. 业务建议 =====
    w('---')
    w('## 八、策略建议')
    w('')
    w('> 基于上述分析结果，自动生成以下可操作的业务建议。')
    w('')
    recommendations = []

    # 基于驱动力分析
    if driver and driver.get('results'):
        for d_result in driver['results']:
            if d_result.get('model_significant'):
                top3 = d_result['drivers'][:3]
                top3_names = [d['feature'] for d in top3 if d['significant']]
                if top3_names:
                    target = d_result['target']
                    recommendations.append(
                        f'- **聚焦 {target} 优化**: 优先关注 **{top3_names[0]}** '
                        f'(Beta={top3[0]["coefficient"]:.3f})，其驱动力远超其他因素。'
                        f'建议将资源配置向该方向倾斜。'
                    )

    # 基于分组对比
    if grp and grp.get('results'):
        bonf_sig = [r for r in grp['results'] if r.get('significant_bonferroni', False)]
        for r in bonf_sig[:2]:
            top_group = max(r['group_means'], key=r['group_means'].get)
            bottom_group = min(r['group_means'], key=r['group_means'].get)
            recommendations.append(
                f'- **差异化策略**: **{r["categorical"]}** 对 **{r["numeric"]}** 有显著影响 '
                f'({r["method"]}, p={r["p_value"]:.4f})。'
                f'"{top_group}" 组表现最优，"{bottom_group}" 组有最大提升空间，'
                f'建议对后者进行专项干预。'
            )
            # 事后检验
            if r.get('posthoc_pairs'):
                top_pair = r['posthoc_pairs'][0]
                recommendations.append(
                    f'  事后检验确认: {top_pair["group1"]} vs {top_pair["group2"]} 差异显著 '
                    f'(p={top_pair["p_value"]:.4f}, {top_pair.get("method", "")})。'
                )

    # 基于聚类
    if clust:
        sizes = clust['cluster_sizes']
        sil = clust.get('silhouette_score', 0)
        if len(sizes) >= 2 and sil > 0.25:
            largest_k = max(sizes, key=sizes.get)
            smallest_k = min(sizes, key=sizes.get)
            recommendations.append(
                f'- **分群运营**: 识别出 {clust["k_optimal"]} 个差异化客群（轮廓系数={sil:.3f}）。'
                f'最大客群 Cluster {largest_k} 共 {sizes[largest_k]} 人，建议做规模化触达；'
                f'最小客群 Cluster {smallest_k} 共 {sizes[smallest_k]} 人，可能是高价值小众群体，建议深度运营。'
            )

    # 基于相关性非线性提示
    if corr and corr.get('nonlinear_hints'):
        top_nl = corr['nonlinear_hints'][0]
        recommendations.append(
            f'- **非线性关系值得关注**: **{top_nl["pair"]}** 的 Pearson ({top_nl["pearson"]:.3f}) '
            f'与 Spearman ({top_nl["spearman"]:.3f}) 差异达 {top_nl["difference"]:.3f}，'
            f'提示两者之间可能存在非线性关系，建议进一步探索（如分段分析或转换变量）。'
        )

    if recommendations:
        for rec in recommendations[:8]:
            w(rec)
            w('')
    else:
        w('> 当前数据中未检测到足够显著的信号来生成具体策略建议。建议扩大样本量或补充更多特征维度的数据。')
        w('')

    w('---')
    w(f'*报告由 AutoAnalyzer 自动生成 | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*')

    report_text = '\n'.join(lines)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"[ReportGenerator] Report saved to: {output_path}")
    return report_text
