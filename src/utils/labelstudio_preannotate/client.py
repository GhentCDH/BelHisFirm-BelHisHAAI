"""Minimal Label Studio API client: upload images to a project's file
storage and import them as tasks with attached predictions. Everything goes
over HTTPS to the Label Studio API — no shared/mounted storage needed.

Adapted from glashelder's labelstudio_metadata_import (specifically its
`sample` command's upload path — see that project's importer.py). That
project only uses this upload-then-reference path for projects it knows
have a single data key (just $image); it embeds images as base64 for
anything else, because a live instance was observed to reject direct file
upload (commit_to_project=false) for a labeling config with more than one
data key. This project's config (single <Image name="image" value="$image"/>
plus a RectangleLabels referencing it by `toName`, not by a second $-key) has
exactly one data key, so that restriction doesn't apply here.

Separately, Label Studio's direct upload also validates the file extension
and rejects some outright (confirmed live: ".tif extension is not
supported") — see _upload_payload for the PNG transcoding that works around it.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image

_SERVER_PATHS_RETRY_ATTEMPTS = 5
_SERVER_PATHS_RETRY_DELAY_SECONDS = 1.5

# Label Studio's direct file upload validates the extension server-side and
# rejects anything outside its own allow-list (confirmed against a live
# instance: uploading a .tif returned {"non_field_errors": [".tif extension
# is not supported"]}). Browsers can't render TIFF/BMP natively either, so
# even a format LS did accept wouldn't display in the labeling UI. Anything
# outside this safe set is transcoded to PNG in memory before upload.
_WEB_SAFE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif"}


def _upload_payload(path: Path) -> tuple[bytes, str]:
    """Return (bytes, filename) to send for PATH's upload — the raw file
    as-is if its extension is web-safe, otherwise transcoded to PNG.
    """
    if path.suffix.lower() in _WEB_SAFE_EXTENSIONS:
        return path.read_bytes(), path.name
    buf = io.BytesIO()
    with Image.open(path) as im:
        im.convert("RGB").save(buf, format="PNG")
    return buf.getvalue(), f"{path.stem}.png"


@dataclass
class HealthCheckResult:
    label: str
    ok: bool
    detail: str = ""


def _raise_for_status(resp: requests.Response) -> None:
    if not resp.ok:
        raise requests.HTTPError(f"{resp.status_code} {resp.reason} for {resp.url}: {resp.text[:500]}", response=resp)


class LabelStudioClient:
    def __init__(self, base_url: str, api_key: str, project_id: int):
        self.base_url = base_url.rstrip("/")
        self.project_id = project_id
        self.headers = {"Authorization": f"Token {api_key}"}

    def upload_images(self, paths: list[Path]) -> dict[str, str]:
        """Upload PATHS to Label Studio's file storage without creating tasks
        (commit_to_project=false). Files with an extension LS's upload
        rejects (or a browser can't render) are transcoded to PNG first —
        see _upload_payload.

        One file per request: Label Studio's import endpoint only registers
        the first file when several are sent under the repeated "file" field
        of a single multipart request.

        Returns {original_filename: "/data/upload/{project}/{stored_name}"},
        keyed by each path's *original* filename even where the upload itself
        was renamed to a converted .png.
        """
        if not paths:
            raise ValueError("No images to upload")

        upload_names: dict[str, str] = {}  # original filename -> filename actually sent
        for i, p in enumerate(paths, start=1):
            payload, upload_name = _upload_payload(p)
            upload_names[p.name] = upload_name
            resp = requests.post(
                f"{self.base_url}/api/projects/{self.project_id}/import",
                headers=self.headers,
                params={"commit_to_project": "false"},
                files={"file": (upload_name, payload, "application/octet-stream")},
                timeout=300,
            )
            _raise_for_status(resp)
            if i % 20 == 0 or i == len(paths):
                print(f"  uploaded {i}/{len(paths)}")

        by_upload_name = self._server_paths_covering(set(upload_names.values()))
        return {original: by_upload_name[uploaded] for original, uploaded in upload_names.items() if uploaded in by_upload_name}

    def _server_paths_covering(self, expected: set[str]) -> dict[str, str]:
        """server_paths(), retried briefly if EXPECTED filenames we just
        uploaded haven't shown up yet — Label Studio can register a batch
        asynchronously, so an immediate GET can lag behind the POST that just
        succeeded.
        """
        paths = self.server_paths()
        for attempt in range(1, _SERVER_PATHS_RETRY_ATTEMPTS):
            if expected <= paths.keys():
                break
            missing = len(expected - paths.keys())
            print(f"  {missing} of {len(expected)} just-uploaded file(s) not visible yet, retrying ({attempt}/{_SERVER_PATHS_RETRY_ATTEMPTS - 1})...")
            time.sleep(_SERVER_PATHS_RETRY_DELAY_SECONDS)
            paths = self.server_paths()
        return paths

    def server_paths(self) -> dict[str, str]:
        """Fetch the stored filenames and build /data/upload/{project}/... paths.

        Label Studio prefixes uploads with a uuid, e.g.
        'upload/1/a1b2c3-page_0001.jpg'; that prefix is stripped so results can
        be matched back to the original filename. Follows pagination if the
        endpoint returns a paginated {"results": [...], "next": ...} envelope.
        """
        paths: dict[str, str] = {}
        url = f"{self.base_url}/api/projects/{self.project_id}/file-uploads"
        params = {"all": "true", "page_size": 10000}

        while url:
            resp = requests.get(url, headers=self.headers, params=params, timeout=60)
            _raise_for_status(resp)
            data = resp.json()
            items = data if isinstance(data, list) else data.get("results", [])

            for item in items:
                stored = Path(item["file"]).name
                original = stored.split("-", 1)[-1]
                paths[original] = f"/data/upload/{self.project_id}/{stored}"

            url = data.get("next") if isinstance(data, dict) else None
            params = None  # 'next' already carries the full query string

        return paths

    def download_file(self, server_path: str) -> bytes:
        """Download a file previously returned by upload_images/server_paths,
        e.g. "/data/upload/{project}/{stored_name}".
        """
        resp = requests.get(f"{self.base_url}{server_path}", headers=self.headers, timeout=120)
        _raise_for_status(resp)
        return resp.content

    def import_tasks(self, tasks: list[dict]) -> dict:
        resp = requests.post(
            f"{self.base_url}/api/projects/{self.project_id}/import",
            headers=self.headers,
            json=tasks,
            timeout=300,
        )
        _raise_for_status(resp)
        return resp.json()

    def health_check(self) -> list[HealthCheckResult]:
        """Verify the base URL is reachable, the token is valid, and the
        project is accessible — the same three calls the real import depends on.
        """
        results = []

        def get(label: str, path: str, **kwargs) -> requests.Response | None:
            try:
                resp = requests.get(f"{self.base_url}{path}", headers=self.headers, timeout=15, **kwargs)
            except requests.exceptions.RequestException as exc:
                results.append(HealthCheckResult(label, False, f"{type(exc).__name__}: {exc}"))
                return None
            if not resp.ok:
                results.append(HealthCheckResult(label, False, f"HTTP {resp.status_code}: {resp.text[:200]}"))
                return None
            return resp

        who_resp = get("auth (whoami)", "/api/current-user/whoami")
        if who_resp is None:
            return results
        who = who_resp.json()
        results.append(HealthCheckResult("auth (whoami)", True, who.get("email") or who.get("username") or ""))

        project_resp = get(f"project {self.project_id}", f"/api/projects/{self.project_id}/")
        if project_resp is not None:
            data = project_resp.json()
            results.append(HealthCheckResult(f"project {self.project_id}", True, f"'{data.get('title')}', {data.get('task_number')} task(s)"))

        return results
