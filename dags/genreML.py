"""
### GenreML - resolucion de fuente 

Esta version define el DAG y agrega la resolucion automatica de la fuente.
Todavia no descarga datos: solamente identifica el repositorio, la revision
actual y las URLs Parquet informadas por Hugging Face.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
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

    resolve_source_revision()


genreML_hf_ingest()
