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
        refs=app.metrics([{'name':'run.log','content':'header\ntrain_accuracy: 0.8\n'}])
        self.assertEqual(refs[0]['source'],'run.log:2')

    def test_citation_line_checks(self):
        audit=app.audit_citations('Evidence: train.py:2; also train.py:9.', [{'name':'train.py','content':'one\ntwo'}])
        self.assertEqual([x['exists'] for x in audit['references']], [True, False])

    def test_no_key_shows_clear_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'NEBIUS_API_KEY'): app.investigate([], 'why')

    def test_nebius_and_tavily_workflow(self):
        files=[{'name':'train.py','content':'MLPClassifier()\n'}, {'name':'run.log','content':'validation_accuracy: 0.6\n'}]
        with patch.dict(os.environ, {'NEBIUS_API_KEY':'dummy','TAVILY_API_KEY':'dummy'}, clear=True), patch.object(app,'post',side_effect=[{'choices':[{'message':{'content':'Evidence: run.log:1'}}]}, {'results':[{'title':'Docs','url':'https://example.org','content':'Details'}]}]) as post:
            result=app.investigate(files,'Why?')
        self.assertTrue(result['citation_audit']['references'][0]['exists'])
        self.assertEqual(len(result['sources']),1)
        self.assertIn('MLPClassifier',post.call_args_list[1].args[1]['query'])

    def test_tavily_failure_does_not_hide_investigation(self):
        files=[{'name':'train.py','content':'pass'}]
        with patch.dict(os.environ, {'NEBIUS_API_KEY':'dummy','TAVILY_API_KEY':'dummy'}, clear=True), patch.object(app,'post',side_effect=[{'choices':[{'message':{'content':'Analysis'}}]}, OSError('offline')]):
            self.assertIn('Tavily search failed',app.investigate(files,'Why?')['warning'])

if __name__ == '__main__': unittest.main()
