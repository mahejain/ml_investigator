import base64, io, json, os, tempfile, unittest, zipfile
from unittest.mock import patch
import app

def archive(entries):
    buf=io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        for name, data in entries.items(): z.writestr(name, data)
    return buf.getvalue()

class InvestigatorTests(unittest.TestCase):
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

    def test_citation_context_and_missing_filename(self):
        audit=app.audit_citations('See `train.py` (line 2), and train.py:9. A separate line 1.', [{'name':'train.py','content':'one\ntwo'}])
        self.assertEqual([(x['reference'], x['exists']) for x in audit['references']], [('train.py:2', True), ('train.py:9', False)])

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
