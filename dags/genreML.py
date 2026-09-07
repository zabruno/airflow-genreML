"""
### GenreML - estructura del DAG

Estructura minima del DAG GenreML
"""
from __future__ import annotations

import logging
from pathlib import Path

import pendulum
from airflow.sdk import Param, dag

log = logging.getLogger(__name__)

OUTPUT_DIR = Path("/usr/local/airflow/include/output/genreML")
BRONZE_DIR = OUTPUT_DIR / "bronze"
SILVER_DIR = OUTPUT_DIR / "silver"


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
    log.info("DAG GenreML definido.")


genreML_hf_ingest()
