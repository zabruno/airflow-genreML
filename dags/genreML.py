"""
### GenreML - capa Bronce

se define el DAG, resuelve la fuente de Hugging Face y descarga o
reutiliza los archivos Parquet crudos en Bronce, particionados por revision.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pendulum
from airflow.sdk import Param, dag, task

log = logging.getLogger(__name__)

OUTPUT_DIR = Path("/usr/local/airflow/include/output/genreML")
BRONZE_DIR = OUTPUT_DIR / "bronze"
SILVER_DIR = OUTPUT_DIR / "silver"

HF_REPO_ID = "maharshipandya/spotify-tracks-dataset"
HF_API_URL = f"https://huggingface.co/api/datasets/{HF_REPO_ID}"
HF_PARQUET_API_URL = f"{HF_API_URL}/parquet/default/train"
HF_CONFIG = "default"
HF_SPLIT = "train"


def _http_json(url: str) -> Any:
    request = Request(url, headers={"User-Agent": "GenreML-Airflow/1.0"})
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _download_to_file(url: str, destination: Path) -> int:
    request = Request(url, headers={"User-Agent": "GenreML-Airflow/1.0"})
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")

    total_bytes = 0
    with urlopen(request, timeout=180) as response, open(tmp_path, "wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total_bytes += len(chunk)
            out.write(chunk)

    tmp_path.replace(destination)
    return total_bytes


def _bronze_revision_dir(resolved_revision: str) -> Path:
    return BRONZE_DIR / f"revision={resolved_revision}"


def _bronze_part_path(resolved_revision: str, index: int, url: str) -> Path:
    filename = Path(urlparse(url).path).name or f"part-{index:05d}.parquet"
    if not filename.endswith(".parquet"):
        filename = f"part-{index:05d}.parquet"
    return _bronze_revision_dir(resolved_revision) / filename


def _bronze_manifest_path(resolved_revision: str) -> Path:
    return _bronze_revision_dir(resolved_revision) / "manifest.json"


@dag(
    dag_id="GenreML",
    schedule=None,
    start_date=pendulum.datetime(2026, 9, 1, tz="America/Argentina/Buenos_Aires"),
    catchup=False,
    tags=[
        "ciencia-de-datos",
        "utn-frm",
        "hugging-face",
        "Proyecto Integrador: GenreML",
        "5k10 - 02",
        "Entrega 1",
    ],
    doc_md=__doc__,
    params={
        "force": Param(
            False,
            type="boolean",
            title="Forzar descarga",
            description=(
                "Si es True, vuelve a descargar los Parquet aunque ya existan en Bronce "
                "para la revision resuelta."
            ),
        ),
    },
)
def genreML_hf_ingest():

    @task
    def resolve_source_revision(**context) -> dict:
        """Resuelve la version de la fuente en Hugging Face."""
        requested_revision = "main"
        parquet_api_url = HF_PARQUET_API_URL

        try:
            metadata = _http_json(HF_API_URL)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(
                "No se pudo consultar la API de Hugging Face para resolver "
                f"la revision actual de {HF_REPO_ID}: {exc}"
            ) from exc
        resolved_revision = metadata.get("sha") or requested_revision

        try:
            parquet_files = _http_json(parquet_api_url)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(
                "No se pudo consultar la API Parquet de Hugging Face. "
                f"URL={parquet_api_url} error={exc}"
            ) from exc

        if not isinstance(parquet_files, list) or not parquet_files:
            raise ValueError(
                "La API Parquet de Hugging Face no devolvio archivos para "
                f"{HF_REPO_ID}/{HF_CONFIG}/{HF_SPLIT}. Respuesta: {parquet_files}"
            )

        log.info(
            "Fuente resuelta: repo=%s config=%s split=%s revision_pedida=%s "
            "revision_usada=%s partes_parquet=%s",
            HF_REPO_ID,
            HF_CONFIG,
            HF_SPLIT,
            requested_revision,
            resolved_revision,
            len(parquet_files),
        )
        return {
            "repo_id": HF_REPO_ID,
            "config": HF_CONFIG,
            "split": HF_SPLIT,
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision,
            "parquet_api_url": parquet_api_url,
            "parquet_files": parquet_files,
        }

    @task(retries=2, retry_delay=pendulum.duration(seconds=30))
    def land_bronze(source: dict, **context) -> dict:
        """Descarga o reutiliza los Parquet crudos en la capa Bronce."""
        revision = source["resolved_revision"]
        manifest_path = _bronze_manifest_path(revision)
        force = bool(context["params"]["force"])

        parts = []
        downloaded_parts = 0
        reused_parts = 0
        downloaded_bytes = 0
        for index, url in enumerate(source["parquet_files"]):
            bronze_part = _bronze_part_path(revision, index, url)
            if bronze_part.exists() and not force:
                reused_parts += 1
                status = "reused"
                part_bytes = None
                log.info("Parte Bronce reutilizada: %s", bronze_part)
            else:
                try:
                    part_bytes = _download_to_file(url, bronze_part)
                except (HTTPError, URLError, TimeoutError) as exc:
                    raise RuntimeError(
                        "No se pudo descargar una parte Parquet desde Hugging Face. "
                        f"URL={url} error={exc}"
                    ) from exc
                downloaded_parts += 1
                downloaded_bytes += part_bytes
                status = "downloaded"
                log.info(
                    "Parte Bronce descargada: %s bytes crudos -> %s",
                    part_bytes,
                    bronze_part,
                )
            parts.append({
                "source_url": url,
                "bronze_path": str(bronze_part),
                "status": status,
                "downloaded_bytes": part_bytes,
            })

        manifest = {
            **source,
            "bronze_dir": str(_bronze_revision_dir(revision)),
            "bronze_paths": [part["bronze_path"] for part in parts],
            "storage_format": "parquet",
            "downloaded_parts": downloaded_parts,
            "reused_parts": reused_parts,
            "downloaded_bytes": downloaded_bytes,
            "parts": parts,
            "created_at": pendulum.now("America/Argentina/Buenos_Aires").to_iso8601_string(),
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        log.info("Manifiesto Bronce: %s", manifest_path)
        return {
            "bronze_dir": str(_bronze_revision_dir(revision)),
            "bronze_paths": [part["bronze_path"] for part in parts],
            "manifest_path": str(manifest_path),
            **source,
        }

    source = resolve_source_revision()
    land_bronze(source)


genreML_hf_ingest()
