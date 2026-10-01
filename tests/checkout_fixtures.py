"""Checkout fixtures use real staging/commit code and synthetic vectors."""
import tempfile
from pathlib import Path

from embeddingvc import commit_manager as manager, objects
from embeddingvc.commands.add import add
from embeddingvc.commands.config import set_
from embeddingvc.commands.embed import embed
from embeddingvc.commands.commit import commit
from embeddingvc.commands.checkout import checkout
from embeddingvc.repository import initialize
from test_commit import Encoder, Store


class CheckoutFixture:
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = initialize(Path(temp.name) / 'repo').resolve()
        self.meta = self.root / '.embeddingvc'
        set_(self.root, 'model_revision', 'a' * 40)
        self.source = self.root / 'data/a.txt'
        self.source.write_text('first text', encoding='utf-8')
        self.deleted = self.root / 'data/b.txt'
        self.deleted.write_text('second document', encoding='utf-8')
        add([str(self.root / 'data')], directory=self.root)
        embed(self.root, encoder_factory=Encoder)
        self.first = commit(self.root, message='first', adapter_factory=Store).commit
        self.source.write_text('changed text', encoding='utf-8')
        self.deleted.unlink()
        embed(self.root, encoder_factory=Encoder)
        self.second = commit(self.root, message='second', adapter_factory=Store).commit
        self.before = self.metadata()
        Store.calls = []

    def metadata(self):
        paths = ['embeddingvc.yaml', '.embeddingvc/index.json', '.embeddingvc/HEAD', '.embeddingvc/sync.json']
        return {name: (self.root / name).read_bytes() for name in paths}

    def checkout(self, revision, **kwargs):
        return checkout(revision, self.root, adapter_factory=Store, **kwargs)

    def manifest(self, digest):
        return manager.read_commit(self.root, digest)
