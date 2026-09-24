"""Tests for Gitee API retries and idempotent attachment uploads."""
from __future__ import annotations

import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock
from urllib.request import Request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

import mirror_release_to_gitee as mirror  # noqa: E402


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return b'{}'


class _AttachmentApi:
    def __init__(
        self,
        content: bytes,
        *,
        attach_then_fail: bool = False,
        fail_always: bool = False,
    ) -> None:
        self.content = content
        self.attach_then_fail = attach_then_fail
        self.fail_always = fail_always
        self.attachments: list[dict[str, object]] = []
        self.post_count = 0
        self.list_count = 0

    def request_json(self, method, path, *, label, file_path=None):
        if method == 'GET':
            self.list_count += 1
            return list(self.attachments)

        self.post_count += 1
        if self.fail_always:
            raise mirror.SyncError('Gitee API upload failed (timed out).')
        if self.attach_then_fail and self.post_count == 1:
            self.attachments.append(
                {'id': 1, 'name': file_path.name, 'size': len(self.content)}
            )
            raise mirror.SyncError('Gitee API upload failed (timed out).')
        if not self.attach_then_fail and self.post_count == 1:
            raise mirror.SyncError('Gitee API upload failed (timed out).')

        attachment = {'id': 1, 'name': file_path.name, 'size': len(self.content)}
        self.attachments.append(attachment)
        return attachment

    def download_attachment(self, release_id, attachment_id, destination):
        destination.write_bytes(self.content)


class GiteeUploadTests(unittest.TestCase):
    def test_ordinary_gitee_api_uses_short_timeout_and_three_retries(self):
        api = mirror.GiteeApi('owner', 'repo', 'token')
        with mock.patch.object(mirror, '_request', return_value=b'[]') as request:
            result = api.request_json('GET', '/tags', label='list tags')

        self.assertEqual(result, [])
        self.assertEqual(request.call_args.kwargs['timeout'], mirror.API_TIMEOUT_SECONDS)
        self.assertEqual(request.call_args.kwargs['retries'], mirror.API_RETRIES)

    def test_gitee_api_retries_transient_failure_three_times(self):
        opener = mock.Mock()
        opener.open.side_effect = [urllib.error.URLError('temporary')] * 3 + [_Response()]
        request = Request('https://gitee.com/api/v5/repos/owner/repo/tags')

        with mock.patch.object(mirror, 'OPENER', opener), mock.patch.object(
            mirror.time, 'sleep'
        ) as sleep:
            result = mirror._request(
                request,
                service='Gitee API',
                label='list tags',
                timeout=mirror.API_TIMEOUT_SECONDS,
                retries=mirror.API_RETRIES,
            )

        self.assertEqual(result, b'{}')
        self.assertEqual(opener.open.call_count, 4)
        self.assertTrue(all(call.kwargs['timeout'] == 45 for call in opener.open.call_args_list))
        self.assertEqual(sleep.call_count, 3)

    def test_zip_and_small_asset_use_separate_upload_timeouts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            api = mirror.GiteeApi('owner', 'repo', 'token')
            for name, expected_timeout in (
                ('NEUQ-VisionCalib-v1.1.0-Windows-x64.zip', 900),
                ('release-manifest.json', 45),
            ):
                path = directory / name
                path.write_bytes(b'asset')
                with self.subTest(name=name), mock.patch.object(
                    mirror, '_request', return_value=b'{"name":"' + name.encode() + b'"}'
                ) as request:
                    api.request_json(
                        'POST',
                        '/releases/1/attach_files',
                        label=f'upload {name}',
                        file_path=path,
                    )

                self.assertEqual(request.call_args.kwargs['timeout'], expected_timeout)
                self.assertEqual(request.call_args.kwargs['retries'], 0)

    def test_lost_upload_response_is_reconciled_without_duplicate_upload(self):
        content = b'zip bytes'
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            path = directory / 'NEUQ-VisionCalib-v1.1.0-Windows-x64.zip'
            path.write_bytes(content)
            api = _AttachmentApi(content, attach_then_fail=True)

            mirror._sync_attachments(api, 7, [path], mirror._sha256(path), directory)

        self.assertEqual(api.post_count, 1)
        self.assertGreaterEqual(api.list_count, 3)

    def test_missing_attachment_is_queried_before_retry(self):
        content = b'zip bytes'
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            path = directory / 'NEUQ-VisionCalib-v1.1.0-Windows-x64.zip'
            path.write_bytes(content)
            api = _AttachmentApi(content)

            mirror._sync_attachments(api, 7, [path], mirror._sha256(path), directory)

        self.assertEqual(api.post_count, 2)
        self.assertGreaterEqual(api.list_count, 3)

    def test_existing_mismatched_attachment_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            path = directory / 'NEUQ-VisionCalib-v1.1.0-Windows-x64.zip'
            path.write_bytes(b'original')
            api = _AttachmentApi(b'oriGinal')
            api.attachments.append({'id': 1, 'name': path.name, 'size': path.stat().st_size})

            with self.assertRaisesRegex(mirror.SyncError, 'different content'):
                mirror._sync_attachments(api, 7, [path], mirror._sha256(path), directory)

        self.assertEqual(api.post_count, 0)

    def test_zip_upload_has_at_most_three_retries(self):
        content = b'zip bytes'
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            path = directory / 'NEUQ-VisionCalib-v1.1.0-Windows-x64.zip'
            path.write_bytes(content)
            api = _AttachmentApi(content, fail_always=True)

            with self.assertRaisesRegex(mirror.SyncError, 'after 4 attempts'):
                mirror._sync_one_attachment(
                    api,
                    7,
                    path,
                    directory,
                    expected_attempts=mirror.ZIP_UPLOAD_ATTEMPTS,
                )

        self.assertEqual(api.post_count, 4)
        self.assertEqual(api.list_count, 5)


if __name__ == '__main__':
    unittest.main()
