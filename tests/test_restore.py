import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from embeddingvc import commit_manager as manager, objects
from embeddingvc.commands.branch import branch
from embeddingvc.commands.checkout import CheckoutError
from embeddingvc.commands.config import set_
from embeddingvc.commands.status import status
from embeddingvc.repository import repository_lock
from embeddingvc.restore_engine import TRANSACTION
from checkout_fixtures import CheckoutFixture, Store


class RestoreTests(CheckoutFixture, unittest.TestCase):
    def test_every_publication_write_recovers_after_interruption(self):
        original = objects._atomic_write
        # Marker, pending sync, YAML, index, HEAD, ready sync.
        for fail_after in range(1, 7):
            with self.subTest(fail_after=fail_after):
                self.checkout('main', force=True)
                calls = 0
                def interrupted(path, payload):
                    nonlocal calls
                    original(path, payload)
                    calls += 1
                    if calls == fail_after:
                        raise SystemExit('simulated process interruption')
                with patch.object(objects, '_atomic_write', side_effect=interrupted):
                    with self.assertRaises(SystemExit):
                        self.checkout(self.first)
                self.assertTrue((self.root / TRANSACTION).exists())
                self.assertIn('Checkout publication is incomplete', status(self.root))
                with repository_lock(self.root):
                    self.assertFalse((self.root / TRANSACTION).exists())
                self.assertEqual((self.meta / 'HEAD').read_text().strip(), self.first)
                self.assertEqual(objects.read_index(self.root)['base_commit'], self.first)
                self.assertEqual(json.loads((self.meta / 'sync.json').read_text())['status'], 'synced')
                self.assertEqual(self.source.read_text(), 'changed text')

    def test_marker_write_failure_keeps_original_metadata(self):
        original = objects._atomic_write
        def failed(path, payload):
            if path == self.root / TRANSACTION:
                raise OSError('marker write failed')
            original(path, payload)
        with patch.object(objects, '_atomic_write', side_effect=failed):
            with self.assertRaisesRegex(CheckoutError, 'marker write failed'):
                self.checkout(self.first)
        self.assertEqual(self.metadata(), self.before)
        self.assertFalse((self.root / TRANSACTION).exists())

    def interrupt_at_index(self):
        original = objects._atomic_write
        def failed(path, payload):
            if path == self.meta / 'index.json':
                raise OSError('index publication failed')
            original(path, payload)
        with patch.object(objects, '_atomic_write', side_effect=failed):
            with self.assertRaisesRegex(CheckoutError, 'transaction retained'):
                self.checkout(self.first)

    def test_other_mutators_finish_pending_checkout(self):
        self.interrupt_at_index()
        branch(self.root, 'recovered')
        self.assertEqual((self.meta / 'refs/heads/recovered').read_text().strip(), self.first)
        self.assertFalse((self.root / TRANSACTION).exists())
        self.checkout('main', force=True)
        self.interrupt_at_index()
        set_(self.root, 'chunk_size', '900')
        self.assertEqual((self.meta / 'HEAD').read_text().strip(), self.first)
        self.assertFalse((self.root / TRANSACTION).exists())

    def test_recovery_conflict_preserves_later_edits(self):
        self.interrupt_at_index()
        (self.root / 'embeddingvc.yaml').write_text('later editor changes')
        changed = self.metadata()
        with self.assertRaisesRegex(CheckoutError, 'conflicts with edits'):
            self.checkout('HEAD')
        self.assertEqual(self.metadata(), changed)
        self.assertTrue((self.root / TRANSACTION).exists())
        (self.root / 'embeddingvc.yaml').write_bytes(self.before['embeddingvc.yaml'])
        self.checkout('HEAD')
        self.assertFalse((self.root / TRANSACTION).exists())
        self.assertEqual(objects.read_index(self.root)['base_commit'], self.first)

    def test_corrupt_target_blocks_recovery(self):
        self.interrupt_at_index()
        target = self.meta / f'commits/{self.first}.json'
        original = target.read_bytes()
        target.write_text('{}')
        current = self.metadata()
        with self.assertRaisesRegex(CheckoutError, 'corrupt'):
            self.checkout('HEAD')
        self.assertEqual(self.metadata(), current)
        target.write_bytes(original)
        self.checkout('HEAD')

    def test_edits_during_chroma_build_abort_before_publication(self):
        def edited(*args):
            self.source.write_text('edited while building')
            return 'prepared-collection'
        with patch.object(Store, 'replace', side_effect=edited):
            with self.assertRaisesRegex(CheckoutError, 'Dirty'):
                self.checkout(self.first)
        self.assertEqual(self.metadata(), self.before)
        self.assertFalse((self.root / TRANSACTION).exists())
        self.source.write_text('changed text')
        def config_edited(*args):
            (self.root / 'embeddingvc.yaml').write_text('concurrent edit')
            return 'prepared-collection'
        with patch.object(Store, 'replace', side_effect=config_edited):
            with self.assertRaisesRegex(CheckoutError, 'Metadata changed'):
                self.checkout(self.first, force=True)
        self.assertEqual((self.root / 'embeddingvc.yaml').read_text(), 'concurrent edit')
        self.assertEqual((self.meta / 'HEAD').read_bytes(), self.before['.embeddingvc/HEAD'])

    def test_target_branch_change_blocks_recovery(self):
        branch(self.root, 'old', self.first)
        original = objects._atomic_write
        def interrupted(path, payload):
            original(path, payload)
            if path == self.root / TRANSACTION:
                raise SystemExit('interrupted')
        with patch.object(objects, '_atomic_write', side_effect=interrupted):
            with self.assertRaises(SystemExit):
                self.checkout('old')
        (self.meta / 'refs/heads/old').write_text(self.second)
        with self.assertRaisesRegex(CheckoutError, 'Target branch changed'):
            self.checkout('HEAD')
        self.assertEqual(self.metadata(), self.before)

    def test_process_death_releases_lock_and_recovery_finishes(self):
        script = '''
import os, sys
from pathlib import Path
from embeddingvc import objects
from embeddingvc.commands.checkout import checkout
root = Path(sys.argv[1])
class Store:
    def __init__(self, *args): pass
    def replace(self, *args): return 'prepared-test-collection'
original = objects._atomic_write
def interrupted(path, payload):
    original(path, payload)
    if path == root / '.embeddingvc/index.json': os._exit(19)
objects._atomic_write = interrupted
checkout(sys.argv[2], root, adapter_factory=Store)
'''
        result = subprocess.run([sys.executable, '-c', script, str(self.root), self.first], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 19, result.stdout + result.stderr)
        self.assertTrue((self.root / TRANSACTION).exists())
        self.checkout('HEAD')
        self.assertFalse((self.root / TRANSACTION).exists())
        self.assertEqual(objects.read_index(self.root)['base_commit'], self.first)
        self.assertFalse((self.meta / 'lock').exists())
