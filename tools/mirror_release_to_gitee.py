"""Mirror the three original GitHub Release assets to the matching Gitee Release."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

GITHUB_API = 'https://api.github.com'
GITEE_API = 'https://gitee.com/api/v5'
API_TIMEOUT_SECONDS = 300
ASSET_PATTERN = re.compile(r'^NEUQ-VisionCalib-.+-Windows-x64\.zip$')
SHA256_PATTERN = re.compile(r'^([0-9a-fA-F]{64})(?:\s+\*?(.+))?$')


class SyncError(RuntimeError):
    """A safe, user-facing error that never includes request credentials or URLs."""


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Never forward an Authorization header across HTTPS origins."""

    def redirect_request(
        self,
        request: urllib.request.Request,
        response: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> urllib.request.Request | None:
        redirected = super().redirect_request(
            request, response, code, message, headers, new_url
        )
        if redirected is None:
            return None

        old = urllib.parse.urlsplit(request.full_url)
        new = urllib.parse.urlsplit(redirected.full_url)
        if new.scheme != 'https' or (old.scheme, old.netloc) != (new.scheme, new.netloc):
            redirected.remove_header('Authorization')
        return redirected


OPENER = urllib.request.build_opener(SafeRedirectHandler())


def _request(
    request: urllib.request.Request,
    *,
    service: str,
    label: str,
    destination: Path | None = None,
    allow_not_found: bool = False,
) -> bytes | None:
    try:
        with OPENER.open(request, timeout=API_TIMEOUT_SECONDS) as response:
            if destination is None:
                return response.read()
            with destination.open('wb') as output:
                shutil.copyfileobj(response, output)
            return b''
    except urllib.error.HTTPError as exc:
        if allow_not_found and exc.code == 404:
            return None
        raise SyncError(f'{service} {label} failed (HTTP {exc.code}).') from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # Do not print the underlying exception: urllib errors can include the full URL.
        reason = 'timed out' if isinstance(exc, TimeoutError) else 'network or I/O error'
        raise SyncError(f'{service} {label} failed ({reason}).') from None


def _json_bytes(response: bytes | None, *, service: str, label: str) -> Any:
    if response is None:
        return None
    try:
        return json.loads(response)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise SyncError(f'{service} {label} returned invalid JSON.') from None


class GitHubApi:
    def __init__(self, repository: str, token: str) -> None:
        parts = repository.split('/')
        if len(parts) != 2 or not all(parts):
            raise SyncError('GITHUB_REPOSITORY must have the form owner/repository.')
        self.owner, self.repo = parts
        self.token = token

    def get_json(self, path: str, *, label: str) -> Any:
        request = urllib.request.Request(
            f'{GITHUB_API}{path}',
            headers={
                'Accept': 'application/vnd.github+json',
                'Authorization': f'Bearer {self.token}',
                'X-GitHub-Api-Version': '2022-11-28',
                'User-Agent': 'NEUQ-VisionCalib-release-mirror',
            },
        )
        return _json_bytes(
            _request(request, service='GitHub API', label=label),
            service='GitHub API',
            label=label,
        )

    def release(self, tag: str) -> dict[str, Any]:
        path = (
            f'/repos/{urllib.parse.quote(self.owner, safe="")}/'
            f'{urllib.parse.quote(self.repo, safe="")}/releases/tags/'
            f'{urllib.parse.quote(tag, safe="/")}'
        )
        release = self.get_json(path, label=f'get GitHub Release {tag}')
        if not isinstance(release, dict):
            raise SyncError(f'GitHub Release {tag} returned an unexpected response.')
        if release.get('draft') or not release.get('published_at'):
            raise SyncError(f'GitHub Release {tag} is not published.')
        if release.get('tag_name') != tag:
            raise SyncError(f'GitHub returned a different tag for Release {tag}.')
        return release

    def tag_commit(self, tag: str) -> str:
        path = (
            f'/repos/{urllib.parse.quote(self.owner, safe="")}/'
            f'{urllib.parse.quote(self.repo, safe="")}/git/ref/tags/'
            f'{urllib.parse.quote(tag, safe="/")}'
        )
        ref = self.get_json(path, label=f'resolve GitHub tag {tag}')
        if not isinstance(ref, dict) or not isinstance(ref.get('object'), dict):
            raise SyncError(f'GitHub tag {tag} has no resolvable target.')

        target = ref['object']
        for _ in range(10):
            sha = target.get('sha')
            object_type = target.get('type')
            if not isinstance(sha, str) or not re.fullmatch(r'[0-9a-fA-F]{40,64}', sha):
                raise SyncError(f'GitHub tag {tag} has an invalid target SHA.')
            if object_type == 'commit':
                return sha.lower()
            if object_type != 'tag':
                raise SyncError(f'GitHub tag {tag} does not ultimately point to a commit.')
            annotated_tag = self.get_json(
                f'/repos/{urllib.parse.quote(self.owner, safe="")}/'
                f'{urllib.parse.quote(self.repo, safe="")}/git/tags/{sha}',
                label=f'resolve annotated GitHub tag {tag}',
            )
            if not isinstance(annotated_tag, dict) or not isinstance(
                annotated_tag.get('object'), dict
            ):
                raise SyncError(f'Annotated GitHub tag {tag} has no resolvable target.')
            target = annotated_tag['object']
        raise SyncError(f'GitHub tag {tag} has too many nested annotated tags.')

    def download_asset(self, asset: dict[str, Any], destination: Path) -> None:
        url = asset.get('browser_download_url')
        parsed = urllib.parse.urlsplit(url or '')
        if parsed.scheme != 'https' or parsed.hostname != 'github.com':
            raise SyncError('GitHub Release contains an unexpected asset download URL.')
        request = urllib.request.Request(
            url,
            headers={'User-Agent': 'NEUQ-VisionCalib-release-mirror'},
        )
        _request(request, service='GitHub asset download', label=str(asset.get('name')), destination=destination)
        if destination.stat().st_size != asset.get('size'):
            raise SyncError(f'GitHub asset {asset.get("name")} has an unexpected size.')


class GiteeApi:
    def __init__(self, owner: str, repo: str, token: str) -> None:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', owner) or not re.fullmatch(
            r'[A-Za-z0-9_.-]+', repo
        ):
            raise SyncError('GITEE_OWNER and GITEE_REPO must be valid single path segments.')
        self.owner = owner
        self.repo = repo
        self.token = token
        self.base = (
            f'{GITEE_API}/repos/{urllib.parse.quote(owner, safe="")}/'
            f'{urllib.parse.quote(repo, safe="")}'
        )

    def request_json(
        self,
        method: str,
        path: str,
        *,
        label: str,
        fields: dict[str, Any] | None = None,
        file_path: Path | None = None,
        allow_not_found: bool = False,
    ) -> Any:
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.token}',
            'User-Agent': 'NEUQ-VisionCalib-release-mirror',
        }
        body: bytes | None = None
        if file_path is not None:
            boundary = f'----NEUQVisionCalib{uuid.uuid4().hex}'
            headers['Content-Type'] = f'multipart/form-data; boundary={boundary}'
            body = _multipart_file(file_path, boundary)
        elif fields is not None:
            headers['Content-Type'] = 'application/x-www-form-urlencoded; charset=utf-8'
            body = urllib.parse.urlencode(fields).encode('utf-8')

        request = urllib.request.Request(
            f'{self.base}{path}', data=body, headers=headers, method=method
        )
        response = _request(
            request,
            service='Gitee API',
            label=label,
            allow_not_found=allow_not_found,
        )
        return _json_bytes(response, service='Gitee API', label=label)

    def download_attachment(self, release_id: int, attachment_id: int, destination: Path) -> None:
        _request(
            urllib.request.Request(
                f'{self.base}/releases/{release_id}/attach_files/{attachment_id}/download',
                headers={
                    'Authorization': f'Bearer {self.token}',
                    'User-Agent': 'NEUQ-VisionCalib-release-mirror',
                },
            ),
            service='Gitee attachment download',
            label=f'attachment {attachment_id}',
            destination=destination,
        )


def _multipart_file(path: Path, boundary: str) -> bytes:
    filename = path.name
    content_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
    data = path.read_bytes()
    prefix = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f'Content-Type: {content_type}\r\n\r\n'
    ).encode('utf-8')
    return prefix + data + f'\r\n--{boundary}--\r\n'.encode('ascii')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_release_assets(
    release: dict[str, Any], github: GitHubApi, directory: Path
) -> tuple[list[Path], str, dict[str, Any]]:
    assets = release.get('assets')
    if not isinstance(assets, list):
        raise SyncError('GitHub Release has no valid asset list.')
    archives = [
        asset
        for asset in assets
        if isinstance(asset, dict) and ASSET_PATTERN.fullmatch(str(asset.get('name', '')))
    ]
    manifests = [
        asset
        for asset in assets
        if isinstance(asset, dict) and asset.get('name') == 'release-manifest.json'
    ]
    if len(archives) != 1 or len(manifests) != 1:
        raise SyncError(
            'GitHub Release must contain exactly one Windows-x64 ZIP and one release-manifest.json.'
        )

    archive_asset = archives[0]
    archive_name = str(archive_asset['name'])
    checksum_name = f'{archive_name}.sha256'
    checksums = [
        asset
        for asset in assets
        if isinstance(asset, dict) and asset.get('name') == checksum_name
    ]
    if len(checksums) != 1:
        raise SyncError(f'GitHub Release must contain exactly one {checksum_name}.')

    downloaded: list[Path] = []
    for asset in (archive_asset, checksums[0], manifests[0]):
        if asset.get('state', 'uploaded') != 'uploaded':
            raise SyncError(f'GitHub asset {asset.get("name")} is not fully uploaded.')
        path = directory / str(asset['name'])
        github.download_asset(asset, path)
        downloaded.append(path)

    archive_path, checksum_path, manifest_path = downloaded
    checksum_lines = checksum_path.read_text(encoding='ascii').splitlines()
    if len(checksum_lines) != 1:
        raise SyncError(f'{checksum_name} must contain exactly one SHA256 line.')
    checksum_match = SHA256_PATTERN.fullmatch(checksum_lines[0].strip())
    if not checksum_match:
        raise SyncError(f'{checksum_name} has an invalid SHA256 line.')
    expected_sha = checksum_match.group(1).lower()
    if checksum_match.group(2) and Path(checksum_match.group(2)).name != archive_name:
        raise SyncError(f'{checksum_name} refers to a different archive.')

    actual_sha = _sha256(archive_path)
    if actual_sha != expected_sha:
        raise SyncError('Downloaded GitHub ZIP does not match its .sha256 file.')
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise SyncError('GitHub release-manifest.json is invalid JSON.') from None
    if not isinstance(manifest, dict):
        raise SyncError('GitHub release-manifest.json must contain a JSON object.')
    if manifest.get('archive') != archive_name or manifest.get('sha256', '').lower() != expected_sha:
        raise SyncError('GitHub ZIP SHA256 does not match release-manifest.json.')
    if manifest.get('archive_bytes') != archive_path.stat().st_size:
        raise SyncError('GitHub ZIP size does not match release-manifest.json.')

    api_digest = archive_asset.get('digest')
    if api_digest and api_digest != f'sha256:{expected_sha}':
        raise SyncError('GitHub API ZIP digest disagrees with the Release checksum.')

    return downloaded, expected_sha, manifest


def _list_paginated(
    api: GiteeApi, path: str, *, label: str, page_size: int = 100
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for page in range(1, 1001):
        rows = api.request_json(
            'GET',
            f'{path}?page={page}&per_page={page_size}',
            label=f'{label}, page {page}',
        )
        if not isinstance(rows, list):
            raise SyncError(f'Gitee API {label} returned an unexpected response.')
        if any(not isinstance(row, dict) for row in rows):
            raise SyncError(f'Gitee API {label} returned a malformed entry.')
        result.extend(rows)
        if len(rows) < page_size:
            return result
    raise SyncError(f'Gitee API {label} exceeded the pagination safety limit.')


def _tag_sha(tag: dict[str, Any]) -> str | None:
    commit = tag.get('commit')
    if isinstance(commit, dict):
        commit = commit.get('sha')
    if isinstance(commit, str):
        return commit.lower()
    return None


def _ensure_tag(api: GiteeApi, tag: str, commit_sha: str) -> None:
    tags = _list_paginated(api, '/tags', label='list Gitee tags')
    matches = [item for item in tags if item.get('name') == tag]
    if len(matches) > 1:
        raise SyncError(f'Gitee contains duplicate tag entries for {tag}.')
    if matches:
        if _tag_sha(matches[0]) != commit_sha:
            raise SyncError(
                f'Gitee tag {tag} points to a different commit; refusing to move an existing tag.'
            )
        print(f'Gitee tag {tag} already matches the GitHub commit.')
        return

    print(f'Creating Gitee tag {tag}.')
    api.request_json(
        'POST',
        '/tags',
        label=f'create Gitee tag {tag}',
        fields={'tag_name': tag, 'refs': commit_sha, 'tag_message': f'Mirror GitHub tag {tag}'},
    )
    tags = _list_paginated(api, '/tags', label='verify Gitee tag')
    matches = [item for item in tags if item.get('name') == tag]
    if len(matches) != 1 or _tag_sha(matches[0]) != commit_sha:
        raise SyncError(f'Gitee tag {tag} was not created at the expected GitHub commit.')


def _ensure_release(
    api: GiteeApi,
    release: dict[str, Any],
    tag: str,
    commit_sha: str,
) -> dict[str, Any]:
    title = release.get('name') or tag
    body = release.get('body') or ''
    prerelease = bool(release.get('prerelease'))
    path = f'/releases/tags/{urllib.parse.quote(tag, safe="/")}'
    current = api.request_json(
        'GET', path, label=f'get Gitee Release {tag}', allow_not_found=True
    )
    if current is not None and not isinstance(current, dict):
        raise SyncError(f'Gitee Release {tag} returned an unexpected response.')
    fields = {
        'tag_name': tag,
        'name': title,
        'body': body,
        'prerelease': str(prerelease).lower(),
    }
    if current is None:
        print(f'Creating Gitee Release {tag}.')
        current = api.request_json(
            'POST',
            '/releases',
            label=f'create Gitee Release {tag}',
            fields={**fields, 'target_commitish': commit_sha},
        )
    elif (
        current.get('tag_name') != tag
        or current.get('name') != title
        or (current.get('body') or '') != body
        or bool(current.get('prerelease')) != prerelease
    ):
        release_id = current.get('id')
        if not isinstance(release_id, int):
            raise SyncError(f'Existing Gitee Release {tag} has no valid ID.')
        print(f'Updating Gitee Release metadata for {tag}.')
        current = api.request_json(
            'PATCH',
            f'/releases/{release_id}',
            label=f'update Gitee Release {tag}',
            fields=fields,
        )

    if not isinstance(current, dict) or not isinstance(current.get('id'), int):
        raise SyncError(f'Gitee Release {tag} returned no valid release ID.')
    if current.get('tag_name') != tag:
        raise SyncError(f'Gitee Release {tag} was created for a different tag.')
    return current


def _list_attachments(api: GiteeApi, release_id: int) -> list[dict[str, Any]]:
    return _list_paginated(
        api,
        f'/releases/{release_id}/attach_files',
        label=f'list attachments for Gitee Release {release_id}',
    )


def _download_gitee_attachment(
    api: GiteeApi, release_id: int, attachment: dict[str, Any], destination: Path
) -> None:
    attachment_id = attachment.get('id')
    if not isinstance(attachment_id, int):
        raise SyncError(f'Gitee attachment {attachment.get("name")} has no valid ID.')
    api.download_attachment(release_id, attachment_id, destination)
    if destination.stat().st_size != attachment.get('size'):
        raise SyncError(f'Gitee attachment {attachment.get("name")} has an unexpected size.')


def _sync_attachments(
    api: GiteeApi,
    release_id: int,
    files: list[Path],
    expected_zip_sha: str,
    directory: Path,
) -> None:
    attachments = _list_attachments(api, release_id)
    for path in files:
        matches = [item for item in attachments if item.get('name') == path.name]
        if len(matches) > 1:
            raise SyncError(f'Gitee Release has duplicate attachments named {path.name}.')
        if matches:
            existing_path = directory / f'gitee-existing-{path.name}'
            _download_gitee_attachment(api, release_id, matches[0], existing_path)
            if _sha256(existing_path) != _sha256(path):
                raise SyncError(
                    f'Gitee already has {path.name} with different content; refusing to overwrite it.'
                )
            print(f'Skipping identical Gitee attachment {path.name}.')
            continue

        print(f'Uploading original GitHub asset {path.name}.')
        uploaded = api.request_json(
            'POST',
            f'/releases/{release_id}/attach_files',
            label=f'upload Gitee attachment {path.name}',
            file_path=path,
        )
        if not isinstance(uploaded, dict) or uploaded.get('name') != path.name:
            raise SyncError(f'Gitee did not confirm upload of {path.name}.')
        attachments.append(uploaded)

    # Read-after-write verification catches incomplete or corrupt uploads. Retry briefly for
    # Gitee's attachment listing to become visible, but never convert persistent failure to success.
    verified: list[dict[str, Any]] | None = None
    for attempt in range(4):
        verified = _list_attachments(api, release_id)
        if all(sum(item.get('name') == path.name for item in verified) == 1 for path in files):
            break
        if attempt < 3:
            time.sleep(2)
    if verified is None or any(
        sum(item.get('name') == path.name for item in verified) != 1 for path in files
    ):
        raise SyncError('Gitee Release is missing one or more uploaded assets.')

    for path in files:
        item = next(item for item in verified if item.get('name') == path.name)
        downloaded = directory / f'gitee-verified-{path.name}'
        _download_gitee_attachment(api, release_id, item, downloaded)
        remote_sha = _sha256(downloaded)
        if remote_sha != _sha256(path):
            raise SyncError(f'Re-downloaded Gitee attachment {path.name} differs from GitHub.')
        if path.name.endswith('-Windows-x64.zip') and remote_sha != expected_zip_sha:
            raise SyncError('Gitee ZIP SHA256 does not match the GitHub checksum and manifest.')


def _validate_tag(tag: str) -> None:
    if not tag or tag.startswith('-') or any(ord(char) < 32 for char in tag):
        raise SyncError('Release tag is empty or contains control characters.')


def main() -> int:
    tag = os.environ.get('RELEASE_TAG', '').strip()
    github_token = os.environ.get('GITHUB_TOKEN', '')
    gitee_token = os.environ.get('GITEE_TOKEN', '')
    owner = os.environ.get('GITEE_OWNER', '').strip()
    repo = os.environ.get('GITEE_REPO', '').strip()
    repository = os.environ.get('GITHUB_REPOSITORY', '')

    _validate_tag(tag)
    if not github_token:
        raise SyncError('GitHub Actions did not provide GITHUB_TOKEN.')
    if not gitee_token:
        raise SyncError('Repository secret GITEE_TOKEN is missing.')
    if not owner or not repo:
        raise SyncError(
            'Set the Actions Repository Variables GITEE_OWNER and GITEE_REPO to the Gitee mirror.'
        )

    github = GitHubApi(repository, github_token)
    gitee = GiteeApi(owner, repo, gitee_token)
    release = github.release(tag)
    commit_sha = github.tag_commit(tag)

    with tempfile.TemporaryDirectory(prefix='neuq-gitee-release-') as temporary_directory:
        directory = Path(temporary_directory)
        files, expected_sha, _manifest = _resolve_release_assets(release, github, directory)
        print(f'Mirroring GitHub Release {tag} to Gitee {owner}/{repo}.')
        _ensure_tag(gitee, tag, commit_sha)
        gitee_release = _ensure_release(gitee, release, tag, commit_sha)
        _sync_attachments(gitee, gitee_release['id'], files, expected_sha, directory)

    print(f'Gitee Release {tag} is synchronized and all three assets passed SHA256 verification.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except SyncError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1) from None
