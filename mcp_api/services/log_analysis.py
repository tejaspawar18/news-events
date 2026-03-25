"""
Log analysis and monitoring utilities
"""
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re

class LogAnalyzer:
    """Analyze application logs for insights and monitoring"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
    
    def parse_log_file(self, filename: str):
        """Parse a JSON log file and return log entries"""
        filepath = os.path.join(self.log_dir, filename)
        if not os.path.exists(filepath):
            return []
        
        entries = []
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
        return entries
    
    def get_error_summary(self, hours: int = 24):
        """Get error summary for the last N hours"""
        entries = self.parse_log_file('error.log')
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_errors = []
        for entry in entries:
            try:
                timestamp = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                if timestamp > cutoff_time:
                    recent_errors.append(entry)
            except (KeyError, ValueError):
                continue
        
        error_counts = Counter(entry.get('message', 'Unknown error') for entry in recent_errors)
        
        return {
            'total_errors': len(recent_errors),
            'unique_errors': len(error_counts),
            'top_errors': error_counts.most_common(10),
            'recent_errors': recent_errors[-10:]  # Last 10 errors
        }
    
    def get_performance_metrics(self, hours: int = 24):
        """Get performance metrics for the last N hours"""
        entries = self.parse_log_file('access.log')
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_requests = []
        for entry in entries:
            try:
                timestamp = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                if timestamp > cutoff_time:
                    recent_requests.append(entry)
            except (KeyError, ValueError):
                continue
        
        if not recent_requests:
            return {}
        
        # Calculate metrics
        processing_times = [entry.get('processing_time', 0) for entry in recent_requests]
        status_codes = [entry.get('status_code', 0) for entry in recent_requests]
        
        return {
            'total_requests': len(recent_requests),
            'avg_response_time': sum(processing_times) / len(processing_times),
            'max_response_time': max(processing_times),
            'min_response_time': min(processing_times),
            'status_code_distribution': Counter(status_codes),
            'requests_per_hour': len(recent_requests) / hours,
            'slow_requests': len([t for t in processing_times if t > 1.0])
        }
    
    def get_usage_stats(self, hours: int = 24):
        """Get usage statistics"""
        entries = self.parse_log_file('app.log')
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        translation_requests = []
        for entry in entries:
            try:
                timestamp = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                if (timestamp > cutoff_time and 
                    entry.get('event') == 'translation_success'):
                    translation_requests.append(entry)
            except (KeyError, ValueError):
                continue
        
        if not translation_requests:
            return {}
        
        # Calculate stats
        text_lengths = [entry.get('source_length', 0) for entry in translation_requests]
        languages = [entry.get('target_language', 'Unknown') for entry in translation_requests]
        
        return {
            'total_translations': len(translation_requests),
            'avg_text_length': sum(text_lengths) / len(text_lengths),
            'language_distribution': Counter(languages),
            'translations_per_hour': len(translation_requests) / hours
        }
    
    def generate_report(self, hours: int = 24):
        """Generate a comprehensive report"""
        report = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'period_hours': hours,
            'errors': self.get_error_summary(hours),
            'performance': self.get_performance_metrics(hours),
            'usage': self.get_usage_stats(hours)
        }
        
        return report

# Usage example
if __name__ == "__main__":
    analyzer = LogAnalyzer()
    report = analyzer.generate_report(24)
    print(json.dumps(report, indent=2))

