"""
============================================================
Brand Insight AI — 通用消费者数据分析工作台
============================================================
Streamlit Web App | 上传任意CSV → 自动分析 → 生成报告
启动命令: streamlit run app.py
============================================================
"""
import streamlit as st
import pandas as pd
import os, sys, tempfile, json
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auto_analyzer import AutoAnalyzer
from report_generator import generate_report

st.set_page_config(
    page_title="Brand Insight AI — 消费者数据分析工作台",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================== SIDEBAR ========================
with st.sidebar:
    st.title("Brand Insight AI")
    st.caption("通用消费者数据分析工作台")
    st.divider()

    st.markdown("### 1. 上传数据")
    uploaded_file = st.file_uploader(
        "上传 CSV 文件",
        type=['csv'],
        help="支持任意 CSV 格式数据。引擎会自动识别列类型并匹配分析方法。"
    )

    st.markdown("### 2. 分析配置")
    max_categories = st.slider("分类变量最大类别数", 5, 30, 15,
                               help="超过此数的类别会在分组检验时被跳过")
    top_n = st.slider("分类展示 Top N", 5, 20, 10)

    st.markdown("### 3. 启动分析")
    run_btn = st.button("Run Analysis", type="primary", use_container_width=True,
                        disabled=(uploaded_file is None))

    st.divider()
    st.caption("Powered by Claude Agent + Streamlit")
    st.caption("人定框架 :middle_dot: AI 提效")

# ======================== MAIN ========================
st.title("Brand Insight AI — 消费者数据分析工作台")
st.caption(
    "上传任意 CSV 数据文件，自动完成描述统计、相关性分析、分组检验和客户聚类，"
    "一键生成分析报告。你定义分析框架，AI 执行计算与可视化。"
)

if uploaded_file is None:
    # 空状态：展示指引
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### 1. 上传数据
        支持任意 CSV 格式数据文件。
        引擎会自动识别数值型、分类型、日期型列。
        """)
    with col2:
        st.markdown("""
        #### 2. 自动分析
        根据数据类型自动选择分析方法：
        - 描述统计 & 分布可视化
        - 相关性矩阵 & 热力图
        - t检验 / ANOVA 分组对比
        - K-Means 客户聚类
        """)
    with col3:
        st.markdown("""
        #### 3. 生成报告
        一键生成结构化 Markdown 分析报告，
        包含图表、统计结果和关键发现总结。
        """)

    st.info(":point_left: 请在左侧上传 CSV 文件开始分析")

    # Demo：展示示例输出
    with st.expander("查看示例分析输出（使用内置数据集）"):
        st.markdown("""
        内置数据集包含 3,900 条消费者购物记录（18列），涵盖：
        - **人口统计**: 年龄、性别、地理位置
        - **购买行为**: 品类、金额、频次、支付方式
        - **营销触点**: 折扣、促销码、会员状态
        - **满意度**: 评分

        点击 👈 侧边栏上传文件即可开始分析，或运行 `python auto_analyzer.py data/shopping_behavior.csv` 在命令行中体验。
        """)

else:
    if run_btn:
        # 保存上传的文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='wb') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        # 创建输出目录
        output_dir = os.path.join('output', uploaded_file.name.replace('.csv', ''))
        os.makedirs(output_dir, exist_ok=True)

        # ===== 运行分析 =====
        with st.spinner("Analyzing data... This may take 1-2 minutes."):
            status_container = st.empty()

            progress = st.progress(0, text="Loading data...")
            analyzer = AutoAnalyzer(tmp_path, output_dir=output_dir,
                                   max_categories=max_categories, top_n=top_n)

            # 阶段1: 加载 + 数据质量
            progress.progress(10, text="Detecting column types...")
            meta = analyzer.load_and_detect()
            dq = analyzer.data_quality()
            progress.progress(20, text="Checking data quality...")

            # 阶段2: 描述统计
            progress.progress(30, text="Computing descriptive statistics...")
            desc = analyzer.descriptive_stats()

            # 阶段3: 可视化
            progress.progress(50, text="Generating univariate charts...")
            charts = analyzer.univariate_plots()

            # 阶段4: 相关性
            progress.progress(65, text="Running correlation analysis...")
            corr = analyzer.correlation_analysis()

            # 阶段5: 分组对比
            progress.progress(80, text="Running statistical tests (t-test/ANOVA)...")
            grp = analyzer.group_comparison()

            # 阶段6: 聚类
            progress.progress(85, text="Running K-Means clustering...")
            clust = analyzer.clustering()

            # 阶段7: 驱动力分析
            progress.progress(92, text="Running driver analysis (regression)...")
            driver = analyzer.driver_analysis()

            # 阶段8: 报告
            progress.progress(95, text="Generating analysis report...")
            results = {
                'meta': meta, 'descriptive': desc,
                'univariate_charts': charts, 'correlation': corr,
                'group_comparison': grp, 'clustering': clust,
                'driver_analysis': driver
            }
            report_path = os.path.join(output_dir, 'analysis_report.md')
            report_text = generate_report(results, report_path)

            progress.progress(100, text="Analysis complete!")
            status_container.success("Analysis complete!")

        # ======================== 展示结果 ========================
        st.divider()
        st.header("Analysis Results")

        # ---- Tab布局 ----
        tabs = st.tabs([
            "Overview", "Descriptive Stats", "Correlation",
            "Group Comparisons", "Driver Analysis", "Clustering", "Report"
        ])

        # Tab 1: Overview
        with tabs[0]:
            st.subheader("Data Overview")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Rows", f"{meta['total_rows']:,}")
            col2.metric("Total Columns", meta['total_cols'])
            col3.metric("Numeric Columns", len(meta['numeric']))
            col4.metric("Categorical Columns", len(meta['categorical']))

            st.markdown("**Numerical Columns:** " + ", ".join(meta['numeric'][:10]) if meta['numeric'] else "None")
            st.markdown("**Categorical Columns:** " + ", ".join(meta['categorical'][:10]) if meta['categorical'] else "None")
            if meta['datetime']:
                st.markdown("**Date/Time Columns:** " + ", ".join(meta['datetime']))
            if meta['id']:
                st.markdown("**ID Columns (excluded from analysis):** " + ", ".join(meta['id']))

            st.caption(f"Duplicated rows: {meta['duplicated_rows']}")

            # 数据质量警告
            dq_meta = meta.get('data_quality', {})
            if dq_meta:
                missing_info = dq_meta.get('missing', {})
                outlier_info = dq_meta.get('outliers', {})
                high_miss = {k: v for k, v in missing_info.items() if v['severity'] == 'high'}
                high_out = {k: v for k, v in outlier_info.items() if v['severity'] == 'high'}
                if high_miss or high_out:
                    st.warning(
                        f"⚠️ Data Quality: {len(high_miss)} column(s) with high missing rate, "
                        f"{len(high_out)} column(s) with high outlier rate"
                    )
                with st.expander("Data Quality Details"):
                    if missing_info:
                        st.markdown("**Missing Values**")
                        st.dataframe(pd.DataFrame([
                            {'Column': k, 'Missing': v['count'], 'Rate': f'{v[\"percentage\"]}%',
                             'Severity': v['severity']}
                            for k, v in sorted(missing_info.items(), key=lambda x: -x[1]['percentage'])
                        ]), use_container_width=True, hide_index=True)
                    if outlier_info:
                        st.markdown("**Outliers (IQR method)**")
                        st.dataframe(pd.DataFrame([
                            {'Column': k, 'Count': v['count'], 'Rate': f'{v[\"percentage\"]}%',
                             'Range': f'{v[\"lower_bound\"]} ~ {v[\"upper_bound\"]}', 'Severity': v['severity']}
                            for k, v in sorted(outlier_info.items(), key=lambda x: -x[1]['percentage'])
                        ]), use_container_width=True, hide_index=True)

            # 数据预览
            with st.expander("Data Preview (first 20 rows)"):
                st.dataframe(analyzer.df.head(20), use_container_width=True)

        # Tab 2: Descriptive Stats
        with tabs[1]:
            st.subheader("Descriptive Statistics")
            if desc:
                st.dataframe(desc['table'], use_container_width=True)
            else:
                st.info("No numerical columns found.")

            st.subheader("Univariate Charts")
            if charts:
                # 按类型分组展示
                num_charts = [c for c in charts if c['type'] == 'numeric_distribution']
                cat_charts = [c for c in charts if c['type'] == 'categorical_distribution']

                if num_charts:
                    st.markdown("**Numeric Distributions**")
                    cols_per_row = 2
                    for i in range(0, len(num_charts), cols_per_row):
                        row_charts = num_charts[i:i+cols_per_row]
                        cols = st.columns(cols_per_row)
                        for j, chart in enumerate(row_charts):
                            chart_path = os.path.join(output_dir, chart['file'])
                            if os.path.exists(chart_path):
                                cols[j].image(chart_path, caption=chart['column'], use_container_width=True)

                if cat_charts:
                    st.markdown("**Category Distributions**")
                    for chart in cat_charts[:6]:
                        chart_path = os.path.join(output_dir, chart['file'])
                        if os.path.exists(chart_path):
                            st.image(chart_path, caption=chart['column'], use_container_width=True)
            else:
                st.info("No charts generated.")

        # Tab 3: Correlation
        with tabs[2]:
            st.subheader("Correlation Analysis — Pearson + Spearman")
            if corr:
                chart_path = os.path.join(output_dir, corr['chart_file'])
                if os.path.exists(chart_path):
                    st.image(chart_path, use_container_width=True)

                if corr.get('high_correlations'):
                    st.markdown("**Significant Pearson Correlations (|r| > 0.3)**")
                    corr_rows = []
                    for hc in corr['high_correlations'][:10]:
                        corr_rows.append({
                            'Variable Pair': hc['pair'],
                            'Pearson': hc['pearson'],
                            'Spearman': hc.get('spearman', 'N/A')
                        })
                    st.dataframe(pd.DataFrame(corr_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No strong correlations found between numerical variables.")

                if corr.get('nonlinear_hints'):
                    st.warning("**⚠️ Nonlinear Relationship Detected** — Pearson vs Spearman differ by >0.2:")
                    nl_rows = [{'Variable Pair': nh['pair'], 'Pearson': nh['pearson'],
                               'Spearman': nh['spearman'], 'Difference': nh['difference']}
                              for nh in corr['nonlinear_hints'][:5]]
                    st.dataframe(pd.DataFrame(nl_rows), use_container_width=True, hide_index=True)
            else:
                st.info("Not enough numerical columns for correlation analysis (need >= 2).")

        # Tab 4: Group Comparisons
        with tabs[3]:
            st.subheader("Group Comparisons — Statistical Tests")
            st.caption(
                "Method: Shapiro-Wilk normality test → parametric (t-test/ANOVA) if all groups normal, "
                "otherwise non-parametric (Mann-Whitney/Kruskal-Wallis). "
                "Bonferroni correction applied for multiple comparisons."
            )
            if grp and grp.get('results'):
                sig_results = [r for r in grp['results'] if r['significant']]
                bonf_results = [r for r in grp['results'] if r.get('significant_bonferroni', False)]
                total_tests = len(grp['results'])

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Tests", total_tests)
                col2.metric("Raw p<0.05", len(sig_results))
                col3.metric("Bonferroni Corrected", grp.get('n_significant_bonferroni', len(bonf_results)),
                           delta=f"{grp.get('n_significant_bonferroni', 0)}/{total_tests}" if total_tests > 0 else "")

                if total_tests > 1:
                    st.caption(
                        f"Bonferroni correction applied: α = 0.05/{total_tests} = "
                        f"{sig_results[0].get('bonferroni_alpha', 0.05/total_tests):.6f}" if sig_results
                        else f"Bonferroni correction: α = 0.05/{total_tests} ≈ {0.05/total_tests:.6f}"
                    )

                if bonf_results:
                    st.markdown("#### ✅ Bonferroni-Significant Findings")
                    sig_df = pd.DataFrame([{
                        'Numeric': r['numeric'], 'Group By': r['categorical'],
                        'Method': r['method'], 'p-value': r['p_value'],
                        'Effect Size': r['effect_size'], 'Bonferroni': '✅'
                    } for r in bonf_results])
                    st.dataframe(sig_df, use_container_width=True, hide_index=True)

                    for r in bonf_results[:3]:
                        chart_name = f'comparison_{analyzer._safe_name(r["numeric"])}_{analyzer._safe_name(r["categorical"])}.png'
                        chart_path = os.path.join(output_dir, chart_name)
                        if os.path.exists(chart_path):
                            st.image(chart_path, use_container_width=True)
                        # 展示事后检验结果
                        if r.get('posthoc_pairs'):
                            st.caption(
                                "Tukey HSD post-hoc: " +
                                ", ".join([f"{p['group1']} vs {p['group2']}"
                                          f" (p={p['p_value']:.4f})"
                                          for p in r['posthoc_pairs'][:6]])
                            )

                weak_sig = [r for r in sig_results if not r.get('significant_bonferroni', False)]
                if weak_sig:
                    with st.expander(f"⚡ Weak Signals — p<0.05 but failed Bonferroni ({len(weak_sig)} items)"):
                        st.caption("These may be false positives; interpret with caution.")
                        for r in weak_sig:
                            st.caption(f"{r['numeric']} × {r['categorical']}: "
                                      f"{r['method']} p={r['p_value']:.4f}")

                if [r for r in grp['results'] if not r['significant']]:
                    with st.expander(f"Non-significant Tests ({total_tests - len(sig_results)} items)"):
                        for r in grp['results']:
                            if not r['significant']:
                                st.caption(f"{r['numeric']} × {r['categorical']}: {r['method']} p={r['p_value']:.4f}")
            else:
                st.info("Not enough categorical+ numerical column combinations for statistical testing.")

        # Tab 5: Driver Analysis
        with tabs[4]:
            st.subheader("Driver Analysis — What Drives Your KPIs?")
            driver = results.get('driver_analysis')
            if driver and driver.get('results'):
                st.caption(
                    "Multiple linear regression with standardized coefficients (Beta weights). "
                    "Unlike correlation (one-to-one), this shows each factor's **independent contribution** "
                    "after controlling for all other variables."
                )
                for d_result in driver['results']:
                    target = d_result['target']
                    st.markdown(f"### Target: {target}")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("R²", f"{d_result['r_squared']:.3f}",
                               help="Proportion of variance explained by the model")
                    col2.metric("Adj R²", f"{d_result['adj_r_squared']:.3f}")
                    col3.metric("Features", d_result['n_features'])
                    col4.metric("Model Sig", "✅ p<0.05" if d_result['model_significant']
                               else f"p={d_result['f_pvalue']:.4f}")

                    chart_path = os.path.join(output_dir, d_result['chart_file'])
                    if os.path.exists(chart_path):
                        st.image(chart_path, use_container_width=True)

                    # Top drivers table
                    st.markdown("**Top 10 Drivers (ranked by absolute Beta):**")
                    top10 = d_result['drivers'][:10]
                    driver_df = pd.DataFrame([{
                        'Rank': i+1,
                        'Feature': d['feature'],
                        'Beta (Std Coef)': d['coefficient'],
                        'Direction': '↑ Positive' if d['coefficient'] > 0 else '↓ Negative',
                        'p-value': d['p_value'],
                        'Significant': '✅' if d['significant'] else ''
                    } for i, d in enumerate(top10)])
                    st.dataframe(driver_df, use_container_width=True, hide_index=True)

                    # Business insight
                    top3 = d_result['drivers'][:3]
                    if top3:
                        st.markdown("**💡 Key Insight:**")
                        for i, d in enumerate(top3):
                            direction = 'drives UP' if d['coefficient'] > 0 else 'drives DOWN'
                            st.markdown(
                                f"{i+1}. **{d['feature']}** {direction} {target} "
                                f"(Beta={d['coefficient']:.3f}, p={d['p_value']:.4f})"
                            )
                    st.divider()
            else:
                st.info(
                    "Not enough data for driver analysis. "
                    "Need >= 2 numeric columns and >= 30 complete records."
                )

        # Tab 6: Clustering
        with tabs[5]:
            st.subheader("Customer Segmentation — K-Means Clustering")
            if clust:
                chart_path = os.path.join(output_dir, clust['chart_file'])
                if os.path.exists(chart_path):
                    st.image(chart_path, use_container_width=True)

                st.markdown(f"**K = {clust['k_optimal']}** clusters identified from "
                           f"{len(clust['cluster_features'])} features: "
                           f"{', '.join(clust['cluster_features'])}")

                if clust.get('silhouette_score'):
                    sil = clust['silhouette_score']
                    sil_q = clust.get('silhouette_quality', '')
                    st.metric("Silhouette Score", f"{sil:.4f}",
                             help=f"Quality: {sil_q}. >0.5 good, >0.25 fair, <0.25 poor")

                sizes = pd.DataFrame([{
                    'Cluster': k, 'Size': v,
                    'Percentage': f"{v/sum(clust['cluster_sizes'].values())*100:.1f}%"
                } for k, v in sorted(clust['cluster_sizes'].items())])
                st.dataframe(sizes, use_container_width=True, hide_index=True)

                st.caption(f"PCA explained variance: "
                          f"PC1={clust['pca_variance'][0]*100:.1f}%, "
                          f"PC2={clust['pca_variance'][1]*100:.1f}%")
            else:
                st.info("Not enough numerical columns for clustering (need >= 2).")

        # Tab 7: Report
        with tabs[6]:
            st.subheader("Generated Report")
            if os.path.exists(report_path):
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_content = f.read()
                st.markdown(report_content)

                st.divider()
                # 下载按钮
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Download Report (.md)",
                        data=report_content,
                        file_name="analysis_report.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                with col2:
                    # 打包所有输出
                    st.caption("All output files saved to:")
                    st.code(output_dir)
            else:
                st.info("Report generation failed. Check console for errors.")

        # 清理临时文件
        try:
            os.unlink(tmp_path)
        except:
            pass

        # 侧边栏：输出摘要
        with st.sidebar:
            st.divider()
            st.markdown("### Output Summary")
            n_charts = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
            st.metric("Charts Generated", n_charts)
            st.metric("Output Directory", output_dir)
            if clust:
                st.metric("Clusters Found", clust['k_optimal'])
            if grp:
                n_sig = sum(1 for r in grp['results'] if r['significant'])
                n_bonf = grp.get('n_significant_bonferroni',
                                 sum(1 for r in grp['results'] if r.get('significant_bonferroni', False)))
                st.metric("Significant Tests",
                         f"{n_bonf}/{len(grp['results'])} (Bonferroni)",
                         help=f"Raw p<0.05: {n_sig}/{len(grp['results'])}")
