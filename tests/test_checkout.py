import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

from embeddingvc import commit_manager as manager, config, objects
from embeddingvc.cli import app
from embeddingvc.commands.add import add
from embeddingvc.commands.branch import branch
from embeddingvc.commands.checkout import checkout, CheckoutError
from embeddingvc.commands.commit import commit
from embeddingvc.commands.config import set_
from embeddingvc.commands.embed import embed
from embeddingvc.commands.status import status
from embeddingvc.object_store import read_object
from embeddingvc.repository import repository_lock
from checkout_fixtures import CheckoutFixture, Encoder, Store


class CheckoutTests(CheckoutFixture, unittest.TestCase):
    def test_restore_detached_exact_snapshot_and_preserve_sources(self):
        source = self.source.read_bytes()
        history = {p.name: p.read_bytes() for p in (self.meta / 'commits').iterdir()}
        with patch.object(Encoder, 'encode', side_effect=AssertionError('encoder called')):
            result = self.checkout(self.first[:12])
        self.assertIn('detached HEAD', result.render())
        self.assertEqual((self.meta / 'HEAD').read_text().strip(), self.first)
        restored = objects.read_index(self.root)
        self.assertEqual(manager.snapshot(restored), manager.snapshot(self.manifest(self.first)))
        self.assertEqual(restored['base_commit'], self.first)
        self.assertEqual(config.load(self.root).data, read_object(self.root, 'configs', self.manifest(self.first)['config_object']))
        self.assertEqual(Store.calls[-1][0], self.first)
        self.assertEqual(len(Store.calls[-1][1]), 2)
        self.assertEqual(self.source.read_bytes(), source)
        self.assertFalse(self.deleted.exists())
        self.assertEqual(history, {p.name: p.read_bytes() for p in (self.meta / 'commits').iterdir()})
        self.assertIn('Working state differs', status(self.root))

    def test_branch_switch_removes_deleted_records_and_leaves_refs(self):
        branch(self.root, 'old', self.first)
        refs = {p.name: p.read_bytes() for p in (self.meta / 'refs/heads').iterdir()}
        self.checkout('old')
        self.assertEqual((self.meta / 'HEAD').read_text().strip(), 'ref: refs/heads/old')
        with self.assertRaisesRegex(CheckoutError, 'Dirty'):
            self.checkout('main')
        self.checkout('main', force=True)
        self.assertEqual(len(Store.calls[-1][1]), 1)
        self.assertEqual((self.meta / 'HEAD').read_text().strip(), 'ref: refs/heads/main')
        self.assertEqual(refs, {p.name: p.read_bytes() for p in (self.meta / 'refs/heads').iterdir()})

    def test_dirty_sources_config_and_staging_are_guarded(self):
        self.source.write_text('local edits')
        with self.assertRaisesRegex(CheckoutError, 'Dirty'):
            self.checkout(self.first)
        self.assertEqual(self.metadata(), self.before)
        self.source.write_text('changed text')
        set_(self.root, 'chunk_size', '900')
        dirty = self.metadata()
        with self.assertRaisesRegex(CheckoutError, 'Dirty'):
            self.checkout(self.first)
        self.assertEqual(self.metadata(), dirty)
        (self.root / 'embeddingvc.yaml').write_bytes(self.before['embeddingvc.yaml'])
        candidate = objects.read_index(self.root)
        candidate['documents']['data/a.txt']['metadata'] = {'title': 'staged edit'}
        objects.publish_index(self.root, candidate)
        with self.assertRaisesRegex(CheckoutError, 'Dirty'):
            self.checkout(self.first)
        self.assertEqual(Store.calls, [])
        self.checkout(self.first, force=True)
        self.assertEqual(self.source.read_text(), 'changed text')

    def test_new_tracked_file_blocks_switch(self):
        (self.root / 'data/new.txt').write_text('new')
        with self.assertRaisesRegex(CheckoutError, 'Tracked source changes'):
            self.checkout(self.first)
        self.assertEqual(self.metadata(), self.before)

    def test_head_repair_preserves_dirty_and_invalid_working_state(self):
        self.source.write_text('local edits')
        (self.root / 'embeddingvc.yaml').write_text('invalid YAML: [')
        (self.meta / 'index.json').write_text('invalid JSON')
        dirty = self.metadata()
        for force in (False, True):
            result = self.checkout('HEAD', force=force)
            self.assertTrue(result.repaired)
            self.assertEqual(result.commit, self.second)
            for name in ('embeddingvc.yaml', '.embeddingvc/index.json', '.embeddingvc/HEAD'):
                self.assertEqual(self.metadata()[name], dirty[name])
        self.assertEqual(self.source.read_text(), 'local edits')
        self.assertEqual(Store.calls[-1][0], self.second)

    def test_force_handles_broken_working_config_and_index(self):
        (self.root / 'embeddingvc.yaml').write_text('broken')
        (self.meta / 'index.json').write_text('broken')
        self.checkout(self.first, force=True)
        self.assertEqual(objects.read_index(self.root)['base_commit'], self.first)

    def test_corruption_and_missing_objects_fail_before_sync(self):
        manifest = self.manifest(self.first)
        vector = manifest['documents']['data/b.txt']['occurrences'][0]['embedding']['vector']
        paths = [self.meta / f'objects/embeddings/{vector}.json',
                 self.meta / f"objects/configs/{manifest['config']}.json",
                 self.meta / f"objects/configs/{manifest['config_object']}.json"]
        for path in paths:
            contents = path.read_bytes()
            try:
                path.unlink()
                with self.assertRaisesRegex(CheckoutError, 'Missing/corrupt'):
                    self.checkout(self.first, force=True)
                self.assertEqual(self.metadata(), self.before)
                self.assertEqual(Store.calls, [])
            finally:
                path.write_bytes(contents)

    def test_sync_failure_preserves_previous_metadata(self):
        with patch.object(Store, 'replace', side_effect=RuntimeError('database unavailable')):
            with self.assertRaisesRegex(CheckoutError, 'database unavailable'):
                self.checkout(self.first)
        self.assertEqual(self.metadata(), self.before)
        self.assertFalse((self.meta / 'transactions/checkout.json').exists())
        self.checkout(self.first)

    def test_head_sync_failure_is_retryable(self):
        with patch.object(Store, 'replace', side_effect=RuntimeError('failed sync')):
            with self.assertRaisesRegex(CheckoutError, 'failed sync'):
                self.checkout('HEAD')
        self.assertEqual(json.loads((self.meta / 'sync.json').read_text())['status'], 'failed')
        self.checkout('HEAD')
        self.assertEqual(json.loads((self.meta / 'sync.json').read_text())['status'], 'synced')
        for name in ('embeddingvc.yaml', '.embeddingvc/index.json', '.embeddingvc/HEAD'):
            self.assertEqual(self.metadata()[name], self.before[name])

    def test_dimensions_change_without_model_and_commit_after_branch(self):
        class TwoDimensions(Encoder):
            dimension = 2
            def encode(self, texts, **kwargs):
                return [[1., 0.] for _ in texts]
        embed(self.root, encoder_factory=TwoDimensions)
        third = commit(self.root, message='two dimensions', adapter_factory=Store).commit
        with patch.object(Encoder, 'encode', side_effect=AssertionError('model called')):
            self.checkout(self.second)
            self.assertEqual(json.loads((self.meta / 'sync.json').read_text())['dimension'], 3)
            self.checkout(third)
            self.assertEqual(json.loads((self.meta / 'sync.json').read_text())['dimension'], 2)
        branch(self.root, 'experiment')
        self.checkout('experiment')
        self.source.write_text('branch source edit')
        embed(self.root, encoder_factory=TwoDimensions)
        fourth = commit(self.root, message='on branch', adapter_factory=Store).commit
        self.assertEqual(self.manifest(fourth)['parent'], third)

    def test_cli_help_failures_and_dispatch(self):
        with contextlib.chdir(self.root):
            for args, code in ((['checkout', '--help'], 0), (['checkout'], 2), (['checkout', 'unknown'], 1)):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                    app(args)
                self.assertEqual(exc.exception.code, code)
            output = io.StringIO()
            with patch('embeddingvc.commit_manager.prepare_sync', wraps=lambda root, digest, **kw: original(root, digest, adapter_factory=Store)), contextlib.redirect_stdout(output):
                app(['checkout', self.first])
            self.assertIn('Checked out', output.getvalue())

    def test_unborn_and_invalid_revisions(self):
        (self.meta / 'refs/heads/empty').write_text('')
        for revision in ('empty', 'HEAD~99', '../main', 'unknown'):
            with self.assertRaises(CheckoutError):
                self.checkout(revision, force=True)
        self.assertEqual(self.metadata(), self.before)
        self.assertEqual(Store.calls, [])

    def test_lock_blocks_checkout(self):
        with repository_lock(self.root):
            with self.assertRaisesRegex(CheckoutError, 'locked'):
                self.checkout(self.first)
        self.assertEqual(self.metadata(), self.before)

    @unittest.skipUnless(importlib.util.find_spec('chromadb'), 'Chroma not installed')
    def test_real_chroma_restores_membership(self):
        script = '''
import json, sys
from pathlib import Path
from embeddingvc.commands.checkout import checkout
from embeddingvc.vector_store.chromadb_adapter import ChromaAdapter
from embeddingvc.config import load
root = Path(sys.argv[1])
for revision, force, count in ((sys.argv[2], False, 2), ('main', True, 1)):
    result = checkout(revision, root, force=force)
    sync = json.loads((root / '.embeddingvc/sync.json').read_text())
    adapter = ChromaAdapter(root, load(root).data['vector_store'])
    collection = adapter.client.get_collection(sync['collection'], embedding_function=None)
    assert collection.count() == count
    assert sync['commit'] == result.commit
    assert sync['status'] == 'synced'
'''
        result = subprocess.run([sys.executable, '-c', script, str(self.root), self.first], capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


original = manager.prepare_sync
