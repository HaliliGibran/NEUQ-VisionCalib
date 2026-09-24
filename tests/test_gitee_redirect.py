"""Regression test for credential stripping on cross-origin redirects."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from urllib.request import Request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

from mirror_release_to_gitee import SafeRedirectHandler  # noqa: E402


class SafeRedirectHandlerTests(unittest.TestCase):
    def test_cross_origin_302_drops_authorization(self):
        original = Request(
            'https://gitee.com/api/v5/releases/1/attach_files/2/download',
            headers={'Authorization': 'sentinel'},
        )

        redirected = SafeRedirectHandler().redirect_request(
            original,
            None,
            302,
            'Found',
            {},
            'https://download.example.net/attachment.zip',
        )

        self.assertIsNotNone(redirected)
        self.assertIsNone(redirected.get_header('Authorization'))


if __name__ == '__main__':
    unittest.main()
