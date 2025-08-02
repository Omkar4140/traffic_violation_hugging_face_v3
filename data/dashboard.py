import os
import matplotlib.pyplot as plt
from collections import defaultdict
from config.settings import Config

class ViolationDashboard:
    @staticmethod
    def create_dashboard(violations_log):
        """Create violation dashboard charts"""
        if not violations_log:
            return None
            
        violation_counts = defaultdict(int)
        for violation in violations_log:
            violation_counts[violation['violation_type']] += 1
        
        if not violation_counts:
            return None
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        violations = list(violation_counts.keys())
        counts = list(violation_counts.values())
        colors = ['red', 'orange', 'blue', 'purple', 'green']
        
        # Bar chart
        bars = ax1.bar(violations, counts, color=colors[:len(violations)])
        ax1.set_xlabel('Violation Type')
        ax1.set_ylabel('Count')
        ax1.set_title('Violation Count by Type')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(count), ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        if len(violations) > 1:
            ax2.pie(counts, labels=violations, autopct='%1.1f%%', 
                   colors=colors[:len(violations)], startangle=90)
            ax2.set_title('Violation Distribution')
        else:
            ax2.text(0.5, 0.5, f'Only {violations[0]}\ndetected', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Violation Distribution')
        
        plt.tight_layout()
        
        chart_path = os.path.join(Config.TEMP_DIR, "violation_dashboard.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return chart_path
