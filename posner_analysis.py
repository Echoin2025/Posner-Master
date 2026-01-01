import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持 - 修复乱码问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)

class PosnerAnalyzer:
    """Posner注意线索实验数据分析器"""
    
    def __init__(self, data_folder='data'):
        self.data_folder = data_folder
        self.all_data = None
        self.cleaned_data = None
        self.results = {}
        
    def load_data(self):
        """加载所有CSV数据文件"""
        print("=" * 60)
        print("开始加载数据...")
        
        all_files = [f for f in os.listdir(self.data_folder) if f.endswith('.csv')]
        print(f"找到 {len(all_files)} 个数据文件")
        
        dfs = []
        for file in all_files:
            try:
                df = pd.read_csv(os.path.join(self.data_folder, file))
                # 只保留试次数据行（包含反应时的行）
                df = df[df['mouse.time'].notna()]
                if len(df) > 0:
                    dfs.append(df)
            except Exception as e:
                print(f"警告: 文件 {file} 读取失败: {e}")
        
        self.all_data = pd.concat(dfs, ignore_index=True)
        print(f"成功加载 {len(dfs)} 个文件，共 {len(self.all_data)} 条试次数据")
        print(f"被试人数: {self.all_data['participant'].nunique()}")
        print("=" * 60 + "\n")
        
        return self.all_data
    
    def clean_data(self):
        """数据清洗：移除异常值"""
        print("=" * 60)
        print("开始数据清洗...")
        
        df = self.all_data.copy()
        original_count = len(df)
        
        # 1. 移除反应时为空的数据
        df = df[df['mouse.time'].notna()]
        
        # 2. 移除异常快的反应（<200ms，可能是预判）
        df = df[df['mouse.time'] >= 0.2]
        fast_removed = original_count - len(df)
        
        # 3. 移除异常慢的反应（>3s，可能走神）
        df = df[df['mouse.time'] <= 3.0]
        slow_removed = original_count - fast_removed - len(df)
        
        # 4. 使用Z-score方法移除极端异常值（±3SD）
        df['rt_zscore'] = np.abs(stats.zscore(df['mouse.time']))
        df = df[df['rt_zscore'] <= 3]
        outlier_removed = original_count - fast_removed - slow_removed - len(df)
        
        self.cleaned_data = df
        
        print(f"原始数据: {original_count} 条")
        print(f"移除过快反应(<200ms): {fast_removed} 条")
        print(f"移除过慢反应(>3s): {slow_removed} 条")
        print(f"移除统计异常值(±3SD): {outlier_removed} 条")
        print(f"清洗后数据: {len(self.cleaned_data)} 条")
        print(f"数据保留率: {len(self.cleaned_data)/original_count*100:.2f}%")
        print("=" * 60 + "\n")
        
        return self.cleaned_data
    
    def descriptive_statistics(self):
        """描述性统计分析"""
        print("=" * 60)
        print("描述性统计分析")
        print("=" * 60)
        
        df = self.cleaned_data
        
        # 整体统计
        print("\n【整体反应时统计】")
        print(f"平均反应时: {df['mouse.time'].mean()*1000:.2f} ms")
        print(f"标准差: {df['mouse.time'].std()*1000:.2f} ms")
        print(f"中位数: {df['mouse.time'].median()*1000:.2f} ms")
        print(f"最小值: {df['mouse.time'].min()*1000:.2f} ms")
        print(f"最大值: {df['mouse.time'].max()*1000:.2f} ms")
        
        # 按条件分组统计
        print("\n【按线索条件分组统计】")
        print(f"{'条件':<15} {'样本量':<10} {'平均RT(ms)':<15} {'标准差(ms)':<15}")
        print("-" * 60)
        
        for congr in [1, 0]:
            subset = df[df['congr'] == congr]
            label = "一致试次(Valid)" if congr == 1 else "不一致试次(Invalid)"
            print(f"{label:<15} {len(subset):<10} {subset['mouse.time'].mean()*1000:<15.2f} {subset['mouse.time'].std()*1000:<15.2f}")
        
        # 计算Posner效应
        valid_rt = df[df['congr'] == 1]['mouse.time'].mean()
        invalid_rt = df[df['congr'] == 0]['mouse.time'].mean()
        posner_effect = (invalid_rt - valid_rt) * 1000
        
        print("\n【Posner注意线索效应】")
        print(f"注意益处(Benefit): {posner_effect:.2f} ms")
        print(f"解释: 有效线索使反应时{'加快' if posner_effect > 0 else '减慢'} {abs(posner_effect):.2f} ms")
        
        # 保存结果
        self.results['overall_mean'] = df['mouse.time'].mean() * 1000
        self.results['overall_std'] = df['mouse.time'].std() * 1000
        self.results['valid_mean'] = valid_rt * 1000
        self.results['invalid_mean'] = invalid_rt * 1000
        self.results['posner_effect'] = posner_effect
        
        print("=" * 60 + "\n")
        plt.close('all')  # 关闭所有图表释放内存
        
    def inferential_statistics(self):
        """推断性统计分析"""
        print("=" * 60)
        print("推断性统计分析")
        print("=" * 60)
        
        df = self.cleaned_data
        
        # 配对样本t检验
        valid_data = df[df['congr'] == 1]['mouse.time']
        invalid_data = df[df['congr'] == 0]['mouse.time']
        
        # 如果样本量不同，使用独立样本t检验
        if len(valid_data) != len(invalid_data):
            t_stat, p_value = stats.ttest_ind(valid_data, invalid_data)
            test_type = "独立样本t检验"
        else:
            t_stat, p_value = stats.ttest_rel(valid_data, invalid_data)
            test_type = "配对样本t检验"
        
        # 计算效应量Cohen's d
        pooled_std = np.sqrt((valid_data.std()**2 + invalid_data.std()**2) / 2)
        cohens_d = (invalid_data.mean() - valid_data.mean()) / pooled_std
        
        print(f"\n【{test_type}结果】")
        print(f"t值: {t_stat:.4f}")
        print(f"p值: {p_value:.6f}")
        print(f"效应量(Cohen's d): {cohens_d:.4f}")
        
        # 判断显著性
        if p_value < 0.001:
            sig_level = "p < 0.001 (极其显著 ***)"
        elif p_value < 0.01:
            sig_level = "p < 0.01 (非常显著 **)"
        elif p_value < 0.05:
            sig_level = "p < 0.05 (显著 *)"
        else:
            sig_level = "p >= 0.05 (不显著)"
        
        print(f"显著性水平: {sig_level}")
        
        # 解释效应量
        if abs(cohens_d) < 0.2:
            effect_interp = "微小"
        elif abs(cohens_d) < 0.5:
            effect_interp = "小"
        elif abs(cohens_d) < 0.8:
            effect_interp = "中等"
        else:
            effect_interp = "大"
        
        print(f"效应量大小: {effect_interp}")
        
        print("\n【结论】")
        if p_value < 0.05:
            print(f"线索有效性对反应时有显著影响({sig_level})，")
            print(f"无效线索条件下的反应时显著{'长于' if cohens_d > 0 else '短于'}有效线索条件，")
            print(f"差异具有{effect_interp}效应量(d={cohens_d:.3f})。")
            print("这支持了Posner的注意定向理论：空间线索能够引导注意资源的分配。")
        else:
            print(f"线索有效性对反应时无显著影响({sig_level})。")
        
        # 保存结果
        self.results['t_statistic'] = t_stat
        self.results['p_value'] = p_value
        self.results['cohens_d'] = cohens_d
        self.results['significant'] = p_value < 0.05
        
        print("=" * 60 + "\n")
    
    def create_visualizations(self):
        """创建数据可视化图表 - 每个图单独保存"""
        print("=" * 60)
        print("生成可视化图表...")
        
        df = self.cleaned_data
        
        # 创建输出文件夹
        import os
        if not os.path.exists('figures'):
            os.makedirs('figures')
            print("Created 'figures' folder for saving individual charts")
        
        # 图1: 反应时分布对比（箱线图）
        fig1 = plt.figure(figsize=(8, 6))
        fig1.patch.set_facecolor('white')
        ax1 = fig1.add_subplot(111)
        
        data_for_box = [
            df[df['congr'] == 1]['mouse.time'] * 1000,
            df[df['congr'] == 0]['mouse.time'] * 1000
        ]
        bp = ax1.boxplot(data_for_box, labels=['Valid\nCue', 'Invalid\nCue'],
                         patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor('#66c2a5')
        bp['boxes'][1].set_facecolor('#fc8d62')
        ax1.set_ylabel('Reaction Time (ms)', fontsize=12, fontweight='bold')
        ax1.set_title('Fig 1: RT Distribution Boxplot', fontsize=14, fontweight='bold', pad=15)
        ax1.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('figures/fig1_boxplot.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("  √ Fig 1 saved: figures/fig1_boxplot.png")
        plt.close()
        
        # 图2: 反应时条形图（含误差线）
        fig2 = plt.figure(figsize=(8, 6))
        fig2.patch.set_facecolor('white')
        ax2 = fig2.add_subplot(111)
        
        means = [
            df[df['congr'] == 1]['mouse.time'].mean() * 1000,
            df[df['congr'] == 0]['mouse.time'].mean() * 1000
        ]
        sems = [
            df[df['congr'] == 1]['mouse.time'].sem() * 1000,
            df[df['congr'] == 0]['mouse.time'].sem() * 1000
        ]
        bars = ax2.bar(['Valid\nCue', 'Invalid\nCue'], means,
                      yerr=sems, capsize=5, color=['#66c2a5', '#fc8d62'],
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Mean RT (ms)', fontsize=12, fontweight='bold')
        ax2.set_title('Fig 2: Mean RT Comparison (Error bar=SEM)', fontsize=14, fontweight='bold', pad=15)
        
        for i, (bar, mean) in enumerate(zip(bars, means)):
            ax2.text(bar.get_x() + bar.get_width()/2, mean + sems[i] + 20,
                    f'{mean:.1f}ms', ha='center', fontsize=11, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('figures/fig2_barplot.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("  √ Fig 2 saved: figures/fig2_barplot.png")
        plt.close()
        
        # 图3: 反应时分布直方图
        fig3 = plt.figure(figsize=(8, 6))
        fig3.patch.set_facecolor('white')
        ax3 = fig3.add_subplot(111)
        
        ax3.hist(df[df['congr'] == 1]['mouse.time'] * 1000, bins=30, alpha=0.6,
                label='Valid', color='#66c2a5', edgecolor='black')
        ax3.hist(df[df['congr'] == 0]['mouse.time'] * 1000, bins=30, alpha=0.6,
                label='Invalid', color='#fc8d62', edgecolor='black')
        ax3.set_xlabel('Reaction Time (ms)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax3.set_title('Fig 3: RT Distribution Histogram', fontsize=14, fontweight='bold', pad=15)
        ax3.legend(fontsize=11)
        ax3.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('figures/fig3_histogram.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("  √ Fig 3 saved: figures/fig3_histogram.png")
        plt.close()
        
        # 图4: 小提琴图
        fig4 = plt.figure(figsize=(8, 6))
        fig4.patch.set_facecolor('white')
        ax4 = fig4.add_subplot(111)
        
        violin_data = pd.DataFrame({
            'RT(ms)': df['mouse.time'] * 1000,
            'Condition': df['congr'].map({1: 'Valid', 0: 'Invalid'})
        })
        sns.violinplot(data=violin_data, x='Condition', y='RT(ms)', 
                      palette=['#66c2a5', '#fc8d62'], ax=ax4)
        ax4.set_title('Fig 4: RT Distribution Violin Plot', fontsize=14, fontweight='bold', pad=15)
        ax4.set_xlabel('', fontsize=12)
        ax4.set_ylabel('Reaction Time (ms)', fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('figures/fig4_violin.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("  √ Fig 4 saved: figures/fig4_violin.png")
        plt.close()
        
        # 图5: 个体差异图
        fig5 = plt.figure(figsize=(8, 6))
        fig5.patch.set_facecolor('white')
        ax5 = fig5.add_subplot(111)
        
        participant_means = df.groupby(['participant', 'congr'])['mouse.time'].mean().unstack() * 1000
        if len(participant_means) > 0:
            for idx in participant_means.index:
                if 1 in participant_means.columns and 0 in participant_means.columns:
                    ax5.plot([1, 2], [participant_means.loc[idx, 1], participant_means.loc[idx, 0]],
                            'o-', alpha=0.3, color='gray', linewidth=1)
            
            if 1 in participant_means.columns and 0 in participant_means.columns:
                ax5.plot([1, 2], [participant_means[1].mean(), participant_means[0].mean()],
                        'o-', linewidth=3, markersize=10, color='red', label='Group Mean')
        
        ax5.set_xlim(0.5, 2.5)
        ax5.set_xticks([1, 2])
        ax5.set_xticklabels(['Valid', 'Invalid'])
        ax5.set_ylabel('Mean RT (ms)', fontsize=12, fontweight='bold')
        ax5.set_title('Fig 5: Individual Differences', fontsize=14, fontweight='bold', pad=15)
        ax5.legend(fontsize=10)
        ax5.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('figures/fig5_individual.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("  √ Fig 5 saved: figures/fig5_individual.png")
        plt.close()
        
        # 图6: 统计检验结果可视化
        fig6 = plt.figure(figsize=(8, 6))
        fig6.patch.set_facecolor('white')
        ax6 = fig6.add_subplot(111)
        ax6.axis('off')
        
        stats_text = f"""
    STATISTICAL RESULTS SUMMARY
    {'='*45}
    
    Sample Size:
    • Valid trials: {len(df[df['congr'] == 1])}
    • Invalid trials: {len(df[df['congr'] == 0])}
    
    Mean Reaction Time:
    • Valid: {self.results['valid_mean']:.2f} ms
    • Invalid: {self.results['invalid_mean']:.2f} ms
    
    Posner Effect: {self.results['posner_effect']:.2f} ms
    
    t-test:
    • t = {self.results['t_statistic']:.4f}
    • p = {self.results['p_value']:.6f}
    • Cohen's d = {self.results['cohens_d']:.4f}
    
    Conclusion: {'Significant ***' if self.results['significant'] else 'Not Significant'}
        """
        
        ax6.text(0.05, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax6.set_title('Fig 6: Statistical Summary', fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig('figures/fig6_statistics.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("  √ Fig 6 saved: figures/fig6_statistics.png")
        plt.close()
        
        # 图7: 个体Posner效应
        fig7 = plt.figure(figsize=(10, 8))
        fig7.patch.set_facecolor('white')
        ax7 = fig7.add_subplot(111)
        
        participant_effect = df.groupby('participant').apply(
            lambda x: (x[x['congr']==0]['mouse.time'].mean() - x[x['congr']==1]['mouse.time'].mean()) * 1000
        ).sort_values()
        
        if len(participant_effect) > 30:
            colors = ['#fc8d62' if x > 0 else '#66c2a5' for x in participant_effect]
            bars = ax7.barh(range(len(participant_effect)), participant_effect, 
                           color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax7.axvline(x=0, color='black', linestyle='--', linewidth=2)
            
            n_ticks = min(10, len(participant_effect))
            tick_indices = np.linspace(0, len(participant_effect)-1, n_ticks, dtype=int)
            ax7.set_yticks(tick_indices)
            ax7.set_yticklabels([f'P{i+1}' for i in tick_indices], fontsize=8)
            ax7.set_ylabel('Participants (ranked by effect size)', fontsize=10, fontweight='bold')
            
            positive_pct = (participant_effect > 0).sum() / len(participant_effect) * 100
            ax7.text(0.98, 0.98, 
                    f'n = {len(participant_effect)}\n{positive_pct:.1f}% positive effect',
                    transform=ax7.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            colors = ['#fc8d62' if x > 0 else '#66c2a5' for x in participant_effect]
            ax7.barh(range(len(participant_effect)), participant_effect, 
                    color=colors, alpha=0.7, edgecolor='black')
            ax7.axvline(x=0, color='black', linestyle='--', linewidth=2)
            ax7.set_yticks(range(len(participant_effect)))
            ax7.set_yticklabels([f'S{p}' for p in participant_effect.index], fontsize=8)
        
        ax7.set_xlabel('Posner Effect (ms)', fontsize=12, fontweight='bold')
        ax7.set_title('Fig 7: Individual Posner Effects Distribution', fontsize=14, fontweight='bold', pad=15)
        ax7.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('figures/fig7_individual_effects.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("  √ Fig 7 saved: figures/fig7_individual_effects.png")
        plt.close()
        
        # 图8: 反应时随试次变化
        fig8 = plt.figure(figsize=(10, 6))
        fig8.patch.set_facecolor('white')
        ax8 = fig8.add_subplot(111)
        
        df_sorted = df.sort_values(['participant', 'trials.thisN'])
        for congr in [1, 0]:
            subset = df_sorted[df_sorted['congr'] == congr]
            if len(subset) > 0:
                window = min(5, len(subset))
                smoothed = subset.groupby('trials.thisN')['mouse.time'].mean().rolling(window=window, center=True).mean() * 1000
                label = 'Valid' if congr == 1 else 'Invalid'
                color = '#66c2a5' if congr == 1 else '#fc8d62'
                ax8.plot(smoothed.index, smoothed.values, label=label, color=color, linewidth=2.5, alpha=0.8)
        
        ax8.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
        ax8.set_ylabel('Mean RT (ms)', fontsize=12, fontweight='bold')
        ax8.set_title('Fig 8: RT Trend Across Trials', fontsize=14, fontweight='bold', pad=15)
        ax8.legend(fontsize=11, framealpha=0.9)
        ax8.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('figures/fig8_trend.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("  √ Fig 8 saved: figures/fig8_trend.png")
        plt.close()
        
        # 图9: 个体效应分布直方图
        fig9 = plt.figure(figsize=(8, 6))
        fig9.patch.set_facecolor('white')
        ax9 = fig9.add_subplot(111)
        
        ax9.hist(participant_effect, bins=20, color='#8da0cb', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax9.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Effect')
        ax9.axvline(x=participant_effect.mean(), color='green', linestyle='-', linewidth=2, 
                   label=f'Mean = {participant_effect.mean():.1f}ms')
        ax9.set_xlabel('Posner Effect (ms)', fontsize=12, fontweight='bold')
        ax9.set_ylabel('Number of Participants', fontsize=12, fontweight='bold')
        ax9.set_title('Fig 9: Distribution of Individual Effects', fontsize=14, fontweight='bold', pad=15)
        ax9.legend(fontsize=10)
        ax9.grid(axis='y', alpha=0.3)
        
        stats_text = f'n = {len(participant_effect)}\nMean = {participant_effect.mean():.2f} ms\nSD = {participant_effect.std():.2f} ms\nPositive: {(participant_effect > 0).sum()}/{len(participant_effect)}'
        ax9.text(0.98, 0.98, stats_text, transform=ax9.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout()
        plt.savefig('figures/fig9_effect_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("  √ Fig 9 saved: figures/fig9_effect_distribution.png")
        plt.close()
        
        # 图10: 效应方向分布
        fig10 = plt.figure(figsize=(8, 6))
        fig10.patch.set_facecolor('white')
        ax10 = fig10.add_subplot(111)
        
        positive_count = (participant_effect > 0).sum()
        negative_count = (participant_effect <= 0).sum()
        
        bars = ax10.bar(['Positive Effect\n(Invalid > Valid)', 'Negative Effect\n(Valid > Invalid)'],
                       [positive_count, negative_count],
                       color=['#fc8d62', '#66c2a5'], alpha=0.8, edgecolor='black', linewidth=2)
        
        for bar, count in zip(bars, [positive_count, negative_count]):
            height = bar.get_height()
            ax10.text(bar.get_x() + bar.get_width()/2., height,
                     f'{count}\n({count/len(participant_effect)*100:.1f}%)',
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax10.set_ylabel('Number of Participants', fontsize=12, fontweight='bold')
        ax10.set_title('Fig 10: Effect Direction Distribution', fontsize=14, fontweight='bold', pad=15)
        ax10.grid(axis='y', alpha=0.3)
        ax10.set_ylim(0, max(positive_count, negative_count) * 1.15)
        plt.tight_layout()
        plt.savefig('figures/fig10_direction.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("  √ Fig 10 saved: figures/fig10_direction.png")
        plt.close()
        
        # 图11: 效应量分组
        fig11 = plt.figure(figsize=(10, 6))
        fig11.patch.set_facecolor('white')
        ax11 = fig11.add_subplot(111)
        
        bins = [-np.inf, -50, 0, 50, 100, 200, np.inf]
        labels = ['< -50', '-50~0', '0~50', '50~100', '100~200', '> 200']
        colors_cat = ['#66c2a5', '#a6d854', '#ffd92f', '#fc8d62', '#e78ac3', '#8da0cb']
        
        counts, _ = np.histogram(participant_effect, bins=bins)
        bars = ax11.bar(labels, counts, color=colors_cat, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for bar, count in zip(bars, counts):
            if count > 0:
                height = bar.get_height()
                ax11.text(bar.get_x() + bar.get_width()/2., height,
                         f'{count}',
                         ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax11.set_xlabel('Posner Effect Range (ms)', fontsize=12, fontweight='bold')
        ax11.set_ylabel('Number of Participants', fontsize=12, fontweight='bold')
        ax11.set_title('Fig 11: Effect Size Categories', fontsize=14, fontweight='bold', pad=15)
        ax11.grid(axis='y', alpha=0.3)
        plt.setp(ax11.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('figures/fig11_categories.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("  √ Fig 11 saved: figures/fig11_categories.png")
        plt.close()
        
        # 图1: 反应时分布对比（箱线图）
        ax1 = plt.subplot(2, 3, 1)
        data_for_box = [
            df[df['congr'] == 1]['mouse.time'] * 1000,
            df[df['congr'] == 0]['mouse.time'] * 1000
        ]
        bp = ax1.boxplot(data_for_box, labels=['Valid\nCue', 'Invalid\nCue'],
                         patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor('#66c2a5')
        bp['boxes'][1].set_facecolor('#fc8d62')
        ax1.set_ylabel('Reaction Time (ms)', fontsize=12, fontweight='bold')
        ax1.set_title('Fig 1: RT Distribution Boxplot', fontsize=13, fontweight='bold', pad=10)
        ax1.grid(axis='y', alpha=0.3)
        
        # 图2: 反应时条形图（含误差线）
        ax2 = plt.subplot(2, 3, 2)
        means = [
            df[df['congr'] == 1]['mouse.time'].mean() * 1000,
            df[df['congr'] == 0]['mouse.time'].mean() * 1000
        ]
        sems = [
            df[df['congr'] == 1]['mouse.time'].sem() * 1000,
            df[df['congr'] == 0]['mouse.time'].sem() * 1000
        ]
        bars = ax2.bar(['Valid\nCue', 'Invalid\nCue'], means,
                      yerr=sems, capsize=5, color=['#66c2a5', '#fc8d62'],
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Mean RT (ms)', fontsize=12, fontweight='bold')
        ax2.set_title('Fig 2: Mean RT Comparison (Error bar=SEM)', fontsize=13, fontweight='bold', pad=10)
        
        # 在柱子上标注数值
        for i, (bar, mean) in enumerate(zip(bars, means)):
            ax2.text(bar.get_x() + bar.get_width()/2, mean + sems[i] + 20,
                    f'{mean:.1f}ms', ha='center', fontsize=11, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # 图3: 反应时分布直方图
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist(df[df['congr'] == 1]['mouse.time'] * 1000, bins=30, alpha=0.6,
                label='Valid', color='#66c2a5', edgecolor='black')
        ax3.hist(df[df['congr'] == 0]['mouse.time'] * 1000, bins=30, alpha=0.6,
                label='Invalid', color='#fc8d62', edgecolor='black')
        ax3.set_xlabel('Reaction Time (ms)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax3.set_title('Fig 3: RT Distribution Histogram', fontsize=13, fontweight='bold', pad=10)
        ax3.legend(fontsize=11)
        ax3.grid(axis='y', alpha=0.3)
        
        # 图4: 小提琴图
        ax4 = plt.subplot(2, 3, 4)
        violin_data = pd.DataFrame({
            'RT(ms)': df['mouse.time'] * 1000,
            'Condition': df['congr'].map({1: 'Valid', 0: 'Invalid'})
        })
        sns.violinplot(data=violin_data, x='Condition', y='RT(ms)', 
                      palette=['#66c2a5', '#fc8d62'], ax=ax4)
        ax4.set_title('Fig 4: RT Distribution Violin Plot', fontsize=13, fontweight='bold', pad=10)
        ax4.set_xlabel('', fontsize=12)
        ax4.set_ylabel('Reaction Time (ms)', fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        # 图5: 个体差异图
        ax5 = plt.subplot(2, 3, 5)
        participant_means = df.groupby(['participant', 'congr'])['mouse.time'].mean().unstack() * 1000
        if len(participant_means) > 0:
            for idx in participant_means.index:
                if 1 in participant_means.columns and 0 in participant_means.columns:
                    ax5.plot([1, 2], [participant_means.loc[idx, 1], participant_means.loc[idx, 0]],
                            'o-', alpha=0.3, color='gray', linewidth=1)
            
            # 绘制平均线
            if 1 in participant_means.columns and 0 in participant_means.columns:
                ax5.plot([1, 2], [participant_means[1].mean(), participant_means[0].mean()],
                        'o-', linewidth=3, markersize=10, color='red', label='Group Mean')
        
        ax5.set_xlim(0.5, 2.5)
        ax5.set_xticks([1, 2])
        ax5.set_xticklabels(['Valid', 'Invalid'])
        ax5.set_ylabel('Mean RT (ms)', fontsize=12, fontweight='bold')
        ax5.set_title('Fig 5: Individual Differences', fontsize=13, fontweight='bold', pad=10)
        ax5.legend(fontsize=10)
        ax5.grid(axis='y', alpha=0.3)
        
        # 图6: 统计检验结果可视化
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # 显示统计结果
        stats_text = f"""
    STATISTICAL RESULTS SUMMARY
    {'='*45}
    
    Sample Size:
    • Valid trials: {len(df[df['congr'] == 1])}
    • Invalid trials: {len(df[df['congr'] == 0])}
    
    Mean Reaction Time:
    • Valid: {self.results['valid_mean']:.2f} ms
    • Invalid: {self.results['invalid_mean']:.2f} ms
    
    Posner Effect: {self.results['posner_effect']:.2f} ms
    
    t-test:
    • t = {self.results['t_statistic']:.4f}
    • p = {self.results['p_value']:.6f}
    • Cohen's d = {self.results['cohens_d']:.4f}
    
    Conclusion: {'Significant ***' if self.results['significant'] else 'Not Significant'}
        """
        
        ax6.text(0.05, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax6.set_title('Fig 6: Statistical Summary', fontsize=13, fontweight='bold', pad=10)
        
        plt.tight_layout()
        plt.savefig('posner_analysis_results.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("Main analysis chart saved: posner_analysis_results.png")
        plt.close()
        
        # 创建第二张图：更多分析
        fig2 = plt.figure(figsize=(14, 6))
        fig2.patch.set_facecolor('white')
        
        # 按被试分析 - 优化版：分组显示或汇总统计
        ax7 = plt.subplot(1, 2, 1)
        participant_effect = df.groupby('participant').apply(
            lambda x: (x[x['congr']==0]['mouse.time'].mean() - x[x['congr']==1]['mouse.time'].mean()) * 1000
        ).sort_values()
        
        # 方案1：如果被试数量>30，只显示分布而不显示个体标签
        if len(participant_effect) > 30:
            colors = ['#fc8d62' if x > 0 else '#66c2a5' for x in participant_effect]
            bars = ax7.barh(range(len(participant_effect)), participant_effect, 
                           color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax7.axvline(x=0, color='black', linestyle='--', linewidth=2)
            
            # 只显示部分刻度，避免拥挤
            n_ticks = min(10, len(participant_effect))
            tick_indices = np.linspace(0, len(participant_effect)-1, n_ticks, dtype=int)
            ax7.set_yticks(tick_indices)
            ax7.set_yticklabels([f'P{i+1}' for i in tick_indices], fontsize=8)
            ax7.set_ylabel('Participants (ranked by effect size)', fontsize=10, fontweight='bold')
            
            # 添加统计信息
            positive_pct = (participant_effect > 0).sum() / len(participant_effect) * 100
            ax7.text(0.98, 0.98, 
                    f'n = {len(participant_effect)}\n{positive_pct:.1f}% positive effect',
                    transform=ax7.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            # 被试较少时，显示所有标签
            colors = ['#fc8d62' if x > 0 else '#66c2a5' for x in participant_effect]
            ax7.barh(range(len(participant_effect)), participant_effect, 
                    color=colors, alpha=0.7, edgecolor='black')
            ax7.axvline(x=0, color='black', linestyle='--', linewidth=2)
            ax7.set_yticks(range(len(participant_effect)))
            ax7.set_yticklabels([f'S{p}' for p in participant_effect.index], fontsize=8)
        
        ax7.set_xlabel('Posner Effect (ms)', fontsize=12, fontweight='bold')
        ax7.set_title('Fig 7: Individual Posner Effects Distribution', fontsize=13, fontweight='bold', pad=10)
        ax7.grid(axis='x', alpha=0.3)
        
        # 反应时随试次变化
        ax8 = plt.subplot(1, 2, 2)
        df_sorted = df.sort_values(['participant', 'trials.thisN'])
        for congr in [1, 0]:
            subset = df_sorted[df_sorted['congr'] == congr]
            if len(subset) > 0:
                # 使用滑动窗口平滑
                window = min(5, len(subset))
                smoothed = subset.groupby('trials.thisN')['mouse.time'].mean().rolling(window=window, center=True).mean() * 1000
                label = 'Valid' if congr == 1 else 'Invalid'
                color = '#66c2a5' if congr == 1 else '#fc8d62'
                ax8.plot(smoothed.index, smoothed.values, label=label, color=color, linewidth=2.5, alpha=0.8)
        
        ax8.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
        ax8.set_ylabel('Mean RT (ms)', fontsize=12, fontweight='bold')
        ax8.set_title('Fig 8: RT Trend Across Trials', fontsize=13, fontweight='bold', pad=10)
        ax8.legend(fontsize=11, framealpha=0.9)
        ax8.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('posner_analysis_extended.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("Extended analysis chart saved: posner_analysis_extended.png")
        plt.close()
        
        # 创建第三张图：个体差异的详细分析
        fig3 = plt.figure(figsize=(16, 6))
        fig3.patch.set_facecolor('white')
        
        # 图9: 个体Posner效应的分布直方图
        ax9 = plt.subplot(1, 3, 1)
        ax9.hist(participant_effect, bins=20, color='#8da0cb', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax9.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Effect')
        ax9.axvline(x=participant_effect.mean(), color='green', linestyle='-', linewidth=2, 
                   label=f'Mean = {participant_effect.mean():.1f}ms')
        ax9.set_xlabel('Posner Effect (ms)', fontsize=12, fontweight='bold')
        ax9.set_ylabel('Number of Participants', fontsize=12, fontweight='bold')
        ax9.set_title('Fig 9: Distribution of Individual Effects', fontsize=13, fontweight='bold', pad=10)
        ax9.legend(fontsize=10)
        ax9.grid(axis='y', alpha=0.3)
        
        # 添加统计信息
        stats_text = f'n = {len(participant_effect)}\nMean = {participant_effect.mean():.2f} ms\nSD = {participant_effect.std():.2f} ms\nPositive: {(participant_effect > 0).sum()}/{len(participant_effect)}'
        ax9.text(0.98, 0.98, stats_text, transform=ax9.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 图10: 分类统计
        ax10 = plt.subplot(1, 3, 2)
        positive_count = (participant_effect > 0).sum()
        negative_count = (participant_effect <= 0).sum()
        
        bars = ax10.bar(['Positive Effect\n(Invalid > Valid)', 'Negative Effect\n(Valid > Invalid)'],
                       [positive_count, negative_count],
                       color=['#fc8d62', '#66c2a5'], alpha=0.8, edgecolor='black', linewidth=2)
        
        # 添加数值标签
        for bar, count in zip(bars, [positive_count, negative_count]):
            height = bar.get_height()
            ax10.text(bar.get_x() + bar.get_width()/2., height,
                     f'{count}\n({count/len(participant_effect)*100:.1f}%)',
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax10.set_ylabel('Number of Participants', fontsize=12, fontweight='bold')
        ax10.set_title('Fig 10: Effect Direction Distribution', fontsize=13, fontweight='bold', pad=10)
        ax10.grid(axis='y', alpha=0.3)
        ax10.set_ylim(0, max(positive_count, negative_count) * 1.15)
        
        # 图11: 效应量分组
        ax11 = plt.subplot(1, 3, 3)
        
        # 分组标准
        bins = [-np.inf, -50, 0, 50, 100, 200, np.inf]
        labels = ['< -50', '-50~0', '0~50', '50~100', '100~200', '> 200']
        colors_cat = ['#66c2a5', '#a6d854', '#ffd92f', '#fc8d62', '#e78ac3', '#8da0cb']
        
        counts, _ = np.histogram(participant_effect, bins=bins)
        bars = ax11.bar(labels, counts, color=colors_cat, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            if count > 0:
                height = bar.get_height()
                ax11.text(bar.get_x() + bar.get_width()/2., height,
                         f'{count}',
                         ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax11.set_xlabel('Posner Effect Range (ms)', fontsize=12, fontweight='bold')
        ax11.set_ylabel('Number of Participants', fontsize=12, fontweight='bold')
        ax11.set_title('Fig 11: Effect Size Categories', fontsize=13, fontweight='bold', pad=10)
        ax11.grid(axis='y', alpha=0.3)
        plt.setp(ax11.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('posner_individual_differences.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("Individual differences analysis saved: posner_individual_differences.png")
        
        print("=" * 60 + "\n")
    
    def generate_report(self):
        """生成分析报告"""
        print("=" * 60)
        print("生成分析报告...")
        
        report = f"""
# Posner注意线索实验数据分析报告

## 1. 实验概述

**实验名称**: Posner注意线索范式 (Posner Cueing Paradigm)

**理论基础**: Posner (1980) 提出的空间注意定向理论认为，空间线索能够引导注意资源的分配。有效线索（箭头指向目标出现位置）能够加快反应时（注意益处），而无效线索（箭头指向目标相反位置）会延长反应时（注意代价）。

**实验设计**:
- 自变量：线索有效性（一致试次 vs 不一致试次）
- 因变量：反应时（从目标出现到被试点击的时间）
- 被试数量：{self.cleaned_data['participant'].nunique()} 人
- 试次数量：{len(self.cleaned_data)} 个有效试次

---

## 2. 数据清洗

原始数据经过以下清洗步骤：

1. 移除反应时 < 200ms 的试次（可能为预判反应）
2. 移除反应时 > 3000ms 的试次（可能为注意力分散）
3. 使用Z-score方法移除±3SD以外的统计异常值

数据保留率: {len(self.cleaned_data)/len(self.all_data)*100:.2f}%

---

## 3. 描述性统计结果

### 3.1 整体反应时
- **平均值**: {self.results['overall_mean']:.2f} ms
- **标准差**: {self.results['overall_std']:.2f} ms

### 3.2 按条件分组

| 条件 | 平均反应时 (ms) | 样本量 |
|------|----------------|--------|
| 一致试次 (Valid) | {self.results['valid_mean']:.2f} | {len(self.cleaned_data[self.cleaned_data['congr']==1])} |
| 不一致试次 (Invalid) | {self.results['invalid_mean']:.2f} | {len(self.cleaned_data[self.cleaned_data['congr']==0])} |

### 3.3 Posner注意线索效应

**效应量**: {self.results['posner_effect']:.2f} ms

**解释**: {'无效线索条件下的反应时比有效线索条件长' if self.results['posner_effect'] > 0 else '有效线索条件下的反应时比无效线索条件长'} {abs(self.results['posner_effect']):.2f} ms，这表明空间线索确实影响了注意资源的分配。

---

## 4. 推断性统计结果

### 4.1 假设检验

- **H₀** (零假设): 线索有效性对反应时无影响
- **H₁** (备择假设): 线索有效性对反应时有显著影响

### 4.2 t检验结果

- **t统计量**: {self.results['t_statistic']:.4f}
- **p值**: {self.results['p_value']:.6f}
- **效应量 (Cohen's d)**: {self.results['cohens_d']:.4f}

### 4.3 结论

{
'拒绝零假设。线索有效性对反应时有显著影响 (p < 0.05)。无效线索条件下的反应时显著长于有效线索条件，这支持了Posner的注意定向理论。' 
if self.results['significant'] 
else '不能拒绝零假设。线索有效性对反应时无显著影响 (p ≥ 0.05)。'
}

---

## 5. 数据可视化分析

本研究生成了以下可视化图表（详见附图）：

1. **图1**: 反应时分布箱线图 - 展示两种条件下反应时的分布特征
2. **图2**: 平均反应时对比柱状图 - 清晰显示组间差异
3. **图3**: 反应时分布直方图 - 展示数据的正态性
4. **图4**: 小提琴图 - 结合密度分布和箱线图的优势
5. **图5**: 个体差异图 - 展示每个被试在两种条件下的表现
6. **图6**: 统计检验结果摘要
7. **图7**: 各被试的Posner效应 - 展示个体差异
8. **图8**: 反应时随试次变化趋势 - 探索练习效应

---

## 6. 讨论与结论

### 6.1 主要发现

{
f'本研究发现，有效空间线索显著缩短了被试的反应时（平均缩短{abs(self.results["posner_effect"]):.2f}ms），这一发现与Posner (1980)的经典研究结果一致。统计检验显示这一差异具有统计学意义（t={self.results["t_statistic"]:.3f}, p={self.results["p_value"]:.4f}），且效应量为d={abs(self.results["cohens_d"]):.3f}。'
if self.results['significant']
else f'本研究未发现线索有效性对反应时的显著影响（p={self.results["p_value"]:.3f}）。这可能是由于样本量不足、实验操作问题或个体差异较大等原因。'
}

### 6.2 理论意义

Posner注意线索效应表明，人类的视觉注意系统能够根据空间线索预先分配注意资源。有效线索引导注意至目标位置，从而加快信息加工速度；而无效线索则需要注意的重新定向，导致反应时延长。

### 6.3 研究局限

1. 样本量相对较小，可能影响统计检验的效力
2. 实验环境未严格控制，可能存在混淆变量
3. 未考虑个体差异因素（如注意力水平、认知能力等）

### 6.4 未来方向

1. 增加样本量以提高统计效力
2. 加入更多线索-目标间隔时间条件
3. 考察不同年龄组的注意定向能力差异
4. 结合眼动追踪技术研究注意转移的时间进程

---

## 7. 参考文献

Posner, M. I. (1980). Orienting of attention. *Quarterly Journal of Experimental Psychology, 32*(1), 3-25.

---

**报告生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**分析工具**: Python (pandas, scipy, matplotlib, seaborn)
        """
        
        with open('posner_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("分析报告已保存为: posner_analysis_report.txt")
        print("=" * 60 + "\n")
    
    def run_full_analysis(self):
        """运行完整的分析流程"""
        print("\n" + "="*60)
        print(" "*15 + "Posner实验数据分析系统")
        print("="*60 + "\n")
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 数据清洗
        self.clean_data()
        
        # 3. 描述性统计
        self.descriptive_statistics()
        
        # 4. 推断性统计
        self.inferential_statistics()
        
        # 5. 生成可视化
        self.create_visualizations()
        
        # 6. 生成报告
        self.generate_report()
        
        print("\n" + "="*60)
        print("Analysis Complete! Generated files:")
        print("\nFigures (in 'figures' folder):")
        print("  √ fig1_boxplot.png - RT distribution boxplot")
        print("  √ fig2_barplot.png - Mean RT comparison with error bars")
        print("  √ fig3_histogram.png - RT distribution histogram")
        print("  √ fig4_violin.png - RT distribution violin plot")
        print("  √ fig5_individual.png - Individual differences line plot")
        print("  √ fig6_statistics.png - Statistical summary")
        print("  √ fig7_individual_effects.png - Individual Posner effects")
        print("  √ fig8_trend.png - RT trend across trials")
        print("  √ fig9_effect_distribution.png - Effect size distribution")
        print("  √ fig10_direction.png - Effect direction classification")
        print("  √ fig11_categories.png - Effect size categories")
        print("\nOther files:")
        print("  √ posner_analysis_report.txt - Detailed analysis report")
        print("  √ cleaned_data.csv - Cleaned data file")
        print("="*60 + "\n")

# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    # 创建分析器实例
    analyzer = PosnerAnalyzer(data_folder='data')
    
    # 运行完整分析
    analyzer.run_full_analysis()
    
    # 可选：保存清洗后的数据
    analyzer.cleaned_data.to_csv('cleaned_data.csv', index=False, encoding='utf-8-sig')
    print("\nCleaned data saved: cleaned_data.csv")