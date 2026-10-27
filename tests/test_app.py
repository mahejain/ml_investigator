import base64, io, json, os, tempfile, unittest, zipfile
from types import SimpleNamespace
from unittest.mock import patch
import app
import export_mlflow_run

def archive(entries):
    buf=io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        for name, data in entries.items(): z.writestr(name, data)
    return buf.getvalue()

class InvestigatorTests(unittest.TestCase):
    def test_likely_secret_blocks_network_calls(self):
        files = [{'name':'run.log','content':'validation_accuracy: 0.8\napi_key = "' + ('a' * 24) + '"'}]
        with patch.dict(os.environ, {'NEBIUS_API_KEY':'dummy','TAVILY_API_KEY':'dummy'}, clear=True), patch.object(app,'post') as post:
            with self.assertRaisesRegex(ValueError, 'Possible credential detected'):
                app.investigate(files, 'Why?')
        post.assert_not_called()

    def test_placeholder_is_not_treated_as_credential(self):
        self.assertFalse(app.contains_possible_secret('api_key = "replace_with_your_api_key_here"'))
        self.assertFalse(app.contains_possible_secret('key = os.environ.get("NEBIUS_API_KEY")'))

    def test_possible_secret_in_question_is_blocked(self):
        with self.assertRaisesRegex(ValueError, 'Possible credential detected'):
            app.check_for_possible_secrets([], 'token: sk-' + ('a' * 25))
    def test_mlflow_export_contains_metadata_not_artifacts(self):
        run = SimpleNamespace(info=SimpleNamespace(run_id='id', experiment_id='1', status='FINISHED'), data=SimpleNamespace(params={'model':'mlp'}, metrics={'train_accuracy':0.9,'validation_accuracy':0.7}, tags={'mlflow.source.git.commit':'abc','private':'not-exported'}))
        with tempfile.TemporaryDirectory() as directory:
            path=os.path.join(directory,'run.zip')
            export_mlflow_run.write_run_archive(run, path)
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(archive.namelist(), ['mlflow_run.json'])
                record=json.loads(archive.read('mlflow_run.json'))
                self.assertEqual(record['code_version'], 'abc')
                self.assertNotIn('private', record)
                self.assertEqual(app.computed_findings([{'name':'run.json','content':json.dumps(record)}])[0]['gap'], 0.2)

    def test_zip_ignores_traversal_and_secrets(self):
        files=app.read_zip(archive({'../outside.py':'bad', '.env':'KEY=secret', 'train.py':'print(1)'}))
        self.assertEqual([f['name'] for f in files], ['train.py'])

    def test_rejects_large_expanded_file(self):
        with self.assertRaises(ValueError): app.read_zip(archive({'huge.txt':'a'*70001}))

    def test_metrics_and_source_line(self):
        refs=app.metrics([{'name':'train.py','content':"results = {'train_accuracy': model.score(X, y)}"}, {'name':'run.log','content':'header\ntrain_accuracy: 0.8\n'}])
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]['source'],'run.log:2')

    def test_citation_line_checks(self):
        audit=app.audit_citations('Evidence: train.py:2; also train.py:9.', [{'name':'train.py','content':'one\ntwo'}])
        self.assertEqual([x['exists'] for x in audit['references']], [True, False])

    def test_citations_outside_presented_evidence(self):
        files = [{'name':'run.log','content':'first\nsecond\nthird'}]
        evidence, presented = app.build_evidence(files, limit=30)
        audit = app.audit_citations('run.log:1 and run.log:3', files, presented)
        self.assertTrue(audit['references'][0]['presented'])
        self.assertTrue(audit['references'][1]['exists'])
        self.assertFalse(audit['references'][1]['presented'])

    def test_query_mentions_detected_estimator(self):
        self.assertIn('randomforestclassifier', app.choose_search([{'name':'model.py','content':'RandomForestClassifier()'}]))

    def test_daily_request_limits_are_shared(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'DAILY_REQUEST_LIMIT':'2','DAILY_IP_LIMIT':'1'}):
            db=os.path.join(directory,'limits.db')
            self.assertTrue(app.reserve_request('client-a', db))
            self.assertFalse(app.reserve_request('client-a', db))
            self.assertTrue(app.reserve_request('client-b', db))
            self.assertFalse(app.reserve_request('client-c', db))

    def test_findings_pair_values_only_within_same_run(self):
        files = [{'name':'run.json','content':json.dumps({'a':{'train_accuracy':.96,'validation_accuracy':.70,'convergence_warning':True}, 'b':{'train_accuracy':.5}, 'c':{'validation_accuracy':.2}})}]
        findings = app.computed_findings(files)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]['gap'], .26)
        self.assertTrue(findings[0]['convergence_warning'])
        self.assertEqual(findings[0]['path'], '$.a')

    def test_citation_context_and_missing_filename(self):
        audit=app.audit_citations('See `train.py` (line 2), and train.py:9. A separate line 1.', [{'name':'train.py','content':'one\ntwo'}])
        self.assertEqual([(x['reference'], x['exists']) for x in audit['references']], [('train.py:2', True), ('train.py:9', False)])

    def test_citation_filename_then_parenthesized_lines(self):
        files = [{'name':'run.json','content':'first\nsecond\nthird'}]
        audit = app.audit_citations('See `run.json` (lines 2–3).', files, {('run.json',2), ('run.json',3)})
        self.assertEqual(audit['references'][0]['reference'], 'run.json:2-3')
        self.assertTrue(audit['references'][0]['presented'])

    def test_citation_separate_comma_separated_lines(self):
        files = [{'name':'run.json','content':'\n'.join(str(i) for i in range(1, 29))}]
        audit = app.audit_citations('(run.json:23, 28)', files, {('run.json', 23)})
        self.assertEqual([x['reference'] for x in audit['references']], ['run.json:23', 'run.json:28'])
        self.assertEqual([x['presented'] for x in audit['references']], [True, False])

    def test_model_truncation_is_reported(self):
        files=[{'name':'run.log','content':'train_accuracy: 0.8'}]
        with patch.dict(os.environ, {'NEBIUS_API_KEY':'dummy'}, clear=True), patch.object(app,'call_model',return_value={'choices':[{'finish_reason':'length','message':{'content':'An incomplete answer'}}]}):
            self.assertTrue(app.investigate(files,'Why?')['incomplete'])

    def test_missing_metrics_does_not_guess_or_call_model(self):
        files=[{'name':'model.py','content':'model = LogisticRegression()'}]
        with patch.dict(os.environ, {'NEBIUS_API_KEY':'dummy'}, clear=True), patch.object(app,'call_model') as model:
            result=app.investigate(files,'Why did this fail?')
        model.assert_not_called()
        self.assertIn('does not establish',result['answer'])
        self.assertEqual(result['metrics'],[])

    def test_no_key_shows_clear_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'NEBIUS_API_KEY'): app.investigate([], 'why')

    def test_nebius_and_tavily_workflow(self):
        files=[{'name':'train.py','content':'MLPClassifier()\n'}, {'name':'run.log','content':'validation_accuracy: 0.6\n'}]
        url='https://scikit-learn.org/stable/guide'
        mocked=[{'results':[{'title':'Docs','url':url,'content':'snippet'}]}, {'results':[{'url':url,'raw_content':'Official documentation text'}]}, {'choices':[{'message':{'content':'Evidence: run.log:1'}}]}]
        with patch.dict(os.environ, {'NEBIUS_API_KEY':'dummy','TAVILY_API_KEY':'dummy'}, clear=True), patch.object(app,'post',side_effect=mocked) as post:
            result=app.investigate(files,'Why?')
        self.assertTrue(result['citation_audit']['references'][0]['exists'])
        self.assertEqual(len(result['sources']),1)
        self.assertIn('Official documentation text',post.call_args_list[2].args[1]['messages'][1]['content'])
        self.assertIn('Repeat the filename for every observation', post.call_args_list[2].args[1]['messages'][1]['content'])
        self.assertIn('not proof of a unique cause', post.call_args_list[2].args[1]['messages'][1]['content'])
        self.assertIn('MLPClassifier',post.call_args_list[0].args[1]['query'])

    def test_tavily_failure_does_not_hide_investigation(self):
        files=[{'name':'run.log','content':'train_accuracy: 0.7'}]
        with patch.dict(os.environ, {'NEBIUS_API_KEY':'dummy','TAVILY_API_KEY':'dummy'}, clear=True), patch.object(app,'post',side_effect=[OSError('offline'), {'choices':[{'message':{'content':'Analysis'}}]}]):
            self.assertIn('Tavily search failed',app.investigate(files,'Why?')['warning'])

    def test_error_only_is_investigated(self):
        files=[{'name':'run.log','content':'Traceback (most recent call last):\nValueError: NaN'}]
        with patch.dict(os.environ, {'NEBIUS_API_KEY':'dummy'}, clear=True), patch.object(app,'call_model',return_value={'choices':[{'message':{'content':'Error reported'}}]}) as model:
            result=app.investigate(files,'Why?')
        model.assert_called_once()
        self.assertEqual(result['answer'],'Error reported')

    def test_untrusted_search_domains_excluded(self):
        result={'results':[{'url':'https://scikit-learn.org.evil.example/doc', 'content':'bad'}]}
        with patch.dict(os.environ, {'TAVILY_API_KEY':'dummy'}, clear=True), patch.object(app,'post',return_value=result):
            docs, warning=app.retrieve_docs([{'name':'a.py','content':'MLPClassifier'}])
        self.assertEqual(docs,[])

if __name__ == '__main__': unittest.main()
