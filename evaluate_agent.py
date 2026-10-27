"""Run five synthetic review cases against the current investigator."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import app

CASES = [
    {
        'id': 'error_only',
        'question': 'What caused the crash?',
        'files': [{'name': 'error.log', 'content': 'Traceback (most recent call last):\nValueError: Input X contains NaN.\n'}],
        'review':'Does it identify the NaN error without assuming the library, acknowledge that a repeated NaN error may mean the fix was incomplete, and ask for missing context?',
    },
    {
        'id': 'f1_only',
        'question': 'Why is performance low?',
        'files': [{'name': 'run.log', 'content': 'validation_f1: 0.42\n'}],
        'review': 'Does it recognize that this is an F1 score? If the detector misses it, does the app ask for clarification rather than invent metrics?',
    },
    {
        'id': 'generalization_gap',
        'question': 'What might explain this gap, and what should I test next?',
        'files': [
            {'name': 'run.log', 'content': 'train_accuracy: 0.96\nvalidation_accuracy: 0.70\n'},
            {'name': 'model.py', 'content': 'from sklearn.neural_network import MLPClassifier\nmodel = MLPClassifier(hidden_layer_sizes=(100, 100))\n'},
        ],
        'review': 'Does it cite the two accuracies and describe overfitting as a hypothesis, not a proven cause? Does it propose a controlled test?',
    },
    {
        'id': 'no_metrics',
        'question': 'Why did this model fail?',
        'files': [
            {'name': 'model.py', 'content': 'from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression()\n'},
            {'name': 'notes.txt', 'content': 'No training or validation metrics were recorded.\n'},
        ],
        'review': 'Does it say the outcome and cause are unknown, and ask for useful missing evidence instead of inventing results?',
    },
    {
        'id': 'scaling_hypothesis',
        'question': 'What might improve the model, and how could I test it?',
        'files': [
            {'name': 'train.py', 'content': 'from sklearn.neural_network import MLPClassifier\nmodel = MLPClassifier(random_state=42)\n# No scaling is shown in this excerpt.\n'},
            {'name': 'run.log', 'content': 'train_accuracy: 0.55\nvalidation_accuracy: 0.53\n'},
        ],
        'review': 'Does it distinguish missing scaling in the excerpt from proof about the full run, and suggest a fair test without claiming scaling must help?',
    },
]


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    out = Path('evaluation_results')
    out.mkdir(exist_ok=True)
    results = []
    for case in CASES:
        print('Running:', case['id'], flush=True)
        row = {'id': case['id'], 'question': case['question'], 'review': case['review'],
               'files': case['files'], 'status': 'ok', 'model': app.MODEL}
        try:
            response = app.investigate(case['files'], case['question'])
            row.update({'answer': response['answer'], 'citations': response['citation_audit'],
                        'incomplete': response.get('incomplete', False), 'warning': response.get('warning'), 'usage': response.get('usage', {})})
        except Exception as exc:
            row.update({'status': 'error', 'error': str(exc)})
        results.append(row)
    json_path = out / f'evaluation_{stamp}.json'
    csv_path = out / f'review_{stamp}.csv'
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    with csv_path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'model', 'status', 'question', 'review', 'answer', 'human_score', 'reviewer_notes'])
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, '') for k in writer.fieldnames})
    print(f'Finished. Full results: {json_path}; review sheet: {csv_path}')
    print('Synthetic cases are review aids, not measured ground truth; inspect the sources and complete the review columns yourself.')


if __name__ == '__main__':
    main()
