import json

with open('data/processed/evaluation_report.json') as f:
    report = json.load(f)

print('=== EVALUATION REPORT ===')

print('\nClassifier Metrics:')
for k, v in report['binary_classifier_metrics'].items():
    print(f'  {k}: {v}')

print('\nRegressor Metrics:')
for k, v in report['regressor_metrics'].items():
    print(f'  {k}: {v}')

print('\nSeverity Classifier Metrics:')
for k, v in report['severity_classifier_metrics'].items():
    print(f'  {k}: {v}')

print('\nROI Results:')
print(f'  Monthly Savings: ${report["roi_results"]["monthly_savings_usd"]:,.2f}')
print(f'  Annual Savings:  ${report["roi_results"]["annual_savings_usd"]:,.2f}')

print(f'\nTotal Test Samples: {report["total_test_samples"]}')
print(f'Total Features:     {report["total_features"]}')