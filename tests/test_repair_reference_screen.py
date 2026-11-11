import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location('repair_reference_screen', Path(__file__).resolve().parents[1]/'experiments/018-repair-verification/reference_screen.py')
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


class ReferenceScreenContracts(unittest.TestCase):
    def test_icons_do_not_hide_real_modules_metadata_or_archive_packages(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            canonical = root/'hyperlink'
            canonical.mkdir()
            (canonical/'_url.py').write_text('target implementation')
            digest = hashlib.sha256((canonical/'_url.py').read_bytes()).hexdigest()
            (root/'hyperlink.png').write_bytes(b'image')
            with zipfile.ZipFile(root/'icons.zip','w') as archive:
                archive.writestr('cmd/hyperlink.svg','image')
            clean = screen.scan([root], 'hyperlink', canonical, {digest})
            self.assertTrue(clean['screen_passed'])
            self.assertEqual(len(clean['nonpackage_substring_name_matches']),1)
            self.assertEqual(len(clean['nonpackage_substring_archive_matches']),1)
            (root/'hyperlink.py').write_text('other version')
            (root/'hyperlink-1.0.dist-info').mkdir()
            (root/'renamed.py').write_text('target implementation')
            with zipfile.ZipFile(root/'dependency.whl','w') as archive:
                archive.writestr('vendor/hyperlink/_url.py','another version')
            exposed = screen.scan([root], 'hyperlink', canonical, {digest})
            self.assertFalse(exposed['screen_passed'])
            self.assertEqual(len(exposed['other_package_name_matches']),2)
            self.assertEqual(len(exposed['duplicate_selected_source_files']),1)
            self.assertEqual(len(exposed['archives_containing_package_name']),1)


if __name__ == '__main__':
    unittest.main()
