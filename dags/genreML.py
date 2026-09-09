"""
### GenreML - capa Plata

Construye el CSV de trabajo del proyecto

Pregunta de investigacion:

> Es posible predecir el genero musical de una cancion utilizando unicamente
> sus caracteristicas de audio?

La unidad de analisis es una cancion o track individual. La fuente es el
dataset publico `maharshipandya/spotify-tracks-dataset` de Hugging Face. El
DAG arranca desde la API Parquet del Hub:
`https://huggingface.co/api/datasets/maharshipandya/spotify-tracks-dataset/parquet/default/train`.


Validaciones principales:

* existencia de `track_id` y `track_genre`;
* presencia de las caracteristicas de audio esperadas;
* `track_id` unico en Plata;
* mas de 1.000 filas;
* al menos 5 columnas utiles;
* mezcla de tipos y reporte de `dtypes`;
* nulos por columna y por feature;
* ninguna columna completamente vacia;
* cantidad y distribucion de generos;
* columnas faltantes o inesperadas respecto del esquema esperado.


"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from typing import Any

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

PROJECT_COLUMNS = [
    "track_id",
    "duration_ms",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
    "track_genre",
]

AUDIO_FEATURES = [
    "duration_ms",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]

CRITICAL_COLUMNS = ["track_id", "track_genre", *AUDIO_FEATURES]
INTEGER_COLUMNS = ["duration_ms", "key", "mode", "time_signature"]
FLOAT_COLUMNS = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

MIN_ROWS = 1_000
MIN_COLUMNS = 5


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


def _silver_dataset_path(resolved_revision: str) -> Path:
    return SILVER_DIR / f"revision={resolved_revision}" / "genreML_tracks.csv"


def _safe_ratio(series):
    return series.fillna(0).sort_values(ascending=False)


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
        """Resuelve la version de la fuente en Hugging Face.

        Entrada: no recibe parametros de usuario; usa `main` como referencia
        de origen.
        Salida: metadatos chicos con repo, split, revision pedida, revision
        resuelta y URLs Parquet de descarga.
        Separacion: fija la identidad del lote antes de descargarlo, para que
        Bronce y Plata queden particionados por la misma version de fuente.
        """
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
        """Descarga o reutiliza los Parquet crudos en la capa Bronce.

        Entrada: metadatos de `resolve_source_revision`.
        Salida: rutas de Bronce y un manifiesto liviano con bytes descargados y
        partes reutilizadas.
        Separacion: es la unica tarea que toca Hugging Face; no limpia ni
        interpreta los datos, solo conserva los Parquet que entrega la fuente.
        """
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

    @task
    def inspect_bronze(bronze: dict) -> dict:
        """Inspecciona los Parquet crudos sin transformarlos.

        Entrada: rutas de archivos Bronce.
        Salida: estadisticas pequenas de filas, columnas, duplicados crudos y
        esquema observado.
        Separacion: deja visible la forma original de la fuente antes de tomar
        decisiones de Plata.
        """
        import pandas as pd

        bronze_paths = bronze["bronze_paths"]
        try:
            df = pd.concat(
                [pd.read_parquet(path) for path in bronze_paths],
                ignore_index=True,
            )
        except ImportError as exc:
            raise ImportError(
                "Para leer Bronce en formato Parquet hace falta instalar "
                "`pyarrow` o `fastparquet` en el entorno de Airflow. "
                "Agregar `pyarrow>=15` a requirements.txt."
            ) from exc
        columns = list(df.columns)
        duplicate_track_ids = (
            int(df["track_id"].duplicated().sum()) if "track_id" in df.columns else None
        )
        stats = {
            "bronze_paths": bronze_paths,
            "raw_rows": int(df.shape[0]),
            "raw_columns": int(df.shape[1]),
            "columns": columns,
            "duplicate_track_ids_before_dedup": duplicate_track_ids,
            "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        }
        log.info("Dataset Bronce: %s filas x %s columnas", df.shape[0], df.shape[1])
        log.info("Columnas Bronce: %s", columns)
        log.info("Dtypes Bronce: %s", stats["dtypes"])
        if duplicate_track_ids is not None:
            log.info(
                "Duplicados por track_id detectados antes de deduplicar: %s",
                duplicate_track_ids,
            )
        return {**bronze, **stats}

    @task
    def build_silver(bronze_stats: dict) -> dict:
        """Construye Plata desde Bronce.

        Entrada: rutas y estadisticas de los Parquet Bronce.
        Salida: CSV Plata con columnas del proyecto y estadisticas de
        deduplicacion.
        Separacion: concentra transformaciones reproducibles sin volver a
        consultar Hugging Face.
        """
        import pandas as pd

        bronze_paths = bronze_stats["bronze_paths"]
        revision = bronze_stats["resolved_revision"]
        try:
            df = pd.concat(
                [pd.read_parquet(path) for path in bronze_paths],
                ignore_index=True,
            )
        except ImportError as exc:
            raise ImportError(
                "Para transformar Bronce en formato Parquet hace falta instalar "
                "`pyarrow` o `fastparquet` en el entorno de Airflow. "
                "Agregar `pyarrow>=15` a requirements.txt."
            ) from exc

        missing_critical = [column for column in CRITICAL_COLUMNS if column not in df.columns]
        if missing_critical:
            raise ValueError(
                "No se puede construir Plata porque faltan columnas criticas: "
                f"{missing_critical}. Esquema observado: {list(df.columns)}"
            )

        available_project_columns = [column for column in PROJECT_COLUMNS if column in df.columns]
        excluded_bronze_columns = sorted(set(df.columns) - set(PROJECT_COLUMNS) - {"Unnamed: 0"})
        if excluded_bronze_columns:
            log.warning(
                "Columnas de Bronce excluidas de Plata por no ser audio features "
                "ni clave/objetivo: %s",
                excluded_bronze_columns,
            )
        if "Unnamed: 0" in df.columns:
            log.info("Se descarta 'Unnamed: 0': es indice exportado, no variable analitica.")

        silver = df[available_project_columns].copy()
        for column in ["track_id", "track_genre"]:
            if column in silver.columns:
                silver[column] = silver[column].astype("string").str.strip()

        for column in INTEGER_COLUMNS:
            if column in silver.columns:
                silver[column] = pd.to_numeric(silver[column], errors="coerce").astype("Int64")
        for column in FLOAT_COLUMNS:
            if column in silver.columns:
                silver[column] = pd.to_numeric(silver[column], errors="coerce")

        before_drop_empty_id = len(silver)
        silver = silver[silver["track_id"].notna() & (silver["track_id"] != "")]
        dropped_empty_id = before_drop_empty_id - len(silver)
        if dropped_empty_id:
            log.warning(
                "Se eliminaron %s filas sin track_id porque no pueden identificar un track.",
                dropped_empty_id,
            )

        duplicates_before = int(silver["track_id"].duplicated().sum())
        rows_before_dedup = len(silver)
        if duplicates_before:
            duplicate_sample = (
                silver.loc[silver["track_id"].duplicated(keep=False), "track_id"]
                .head(10)
                .tolist()
            )
            log.warning(
                "Duplicados por track_id antes de deduplicar: %s. Muestra: %s",
                duplicates_before,
                duplicate_sample,
            )
        silver = silver.drop_duplicates(subset=["track_id"], keep="first").reset_index(drop=True)
        duplicates_removed = rows_before_dedup - len(silver)

        destination = _silver_dataset_path(revision)
        destination.parent.mkdir(parents=True, exist_ok=True)
        silver.to_csv(destination, index=False)

        log.info(
            "Dataset Plata escrito: %s filas x %s columnas -> %s",
            silver.shape[0],
            silver.shape[1],
            destination,
        )
        log.info(
            "Duplicados eliminados en Plata: %s. Filas sin track_id eliminadas: %s",
            duplicates_removed,
            dropped_empty_id,
        )

        return {
            **bronze_stats,
            "silver_path": str(destination),
            "silver_rows": int(silver.shape[0]),
            "silver_columns": int(silver.shape[1]),
            "duplicates_removed": int(duplicates_removed),
            "dropped_empty_track_id": int(dropped_empty_id),
            "excluded_bronze_columns": excluded_bronze_columns,
        }

    @task
    def validate_silver(silver_stats: dict) -> dict:
        """Valida la calidad del dataset Plata.

        Entrada: ruta del CSV Plata y estadisticas de Bronce/transformacion.
        Salida: las mismas rutas con un resumen de validacion.
        Separacion: falla explicitamente ante problemas criticos antes de
        publicar el entregable.
        """
        import pandas as pd

        silver_path = silver_stats["silver_path"]
        df = pd.read_csv(silver_path, low_memory=False)
        problems = []

        missing_columns = [column for column in PROJECT_COLUMNS if column not in df.columns]
        missing_audio = [column for column in AUDIO_FEATURES if column not in df.columns]
        unexpected_columns = sorted(set(df.columns) - set(PROJECT_COLUMNS))
        duplicate_final = int(df["track_id"].duplicated().sum()) if "track_id" in df else None
        fully_empty_columns = df.columns[df.isna().all()].tolist()
        null_ratio = _safe_ratio(df.isna().mean())
        null_count = df.isna().sum().sort_values(ascending=False)
        dtype_counts = df.dtypes.value_counts()
        genre_counts = df["track_genre"].value_counts(dropna=False) if "track_genre" in df else None
        feature_nulls = (
            pd.DataFrame({
                "null_count": df[AUDIO_FEATURES].isna().sum(),
                "null_ratio": df[AUDIO_FEATURES].isna().mean(),
            })
            if not missing_audio
            else None
        )

        if "track_id" not in df.columns:
            problems.append("falta la columna critica track_id")
        elif not df["track_id"].is_unique:
            problems.append(f"track_id no es unico: {duplicate_final} duplicados finales")
        if "track_genre" not in df.columns:
            problems.append("falta la columna critica track_genre")
        elif df["track_genre"].isna().any():
            problems.append(f"track_genre tiene {int(df['track_genre'].isna().sum())} nulos")
        if missing_audio:
            problems.append(f"faltan features de audio esperadas: {missing_audio}")
        if len(df) <= MIN_ROWS:
            problems.append(f"volumen insuficiente: {len(df)} filas, minimo requerido > {MIN_ROWS}")
        if df.shape[1] < MIN_COLUMNS:
            problems.append(
                f"ancho insuficiente: {df.shape[1]} columnas, minimo requerido {MIN_COLUMNS}"
            )
        if fully_empty_columns:
            problems.append(f"columnas completamente vacias: {fully_empty_columns}")
        if unexpected_columns:
            problems.append(f"columnas inesperadas en Plata: {unexpected_columns}")

        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns = df.select_dtypes(include=["object", "string", "bool"]).columns.tolist()
        if not numeric_columns:
            problems.append("no se detectaron columnas numericas en Plata")
        if not categorical_columns:
            problems.append("no se detectaron columnas categoricas o booleanas en Plata")

        log.info("Dataset Bronce: %s filas", silver_stats.get("raw_rows"))
        log.info("Dataset Plata: %s filas x %s columnas", df.shape[0], df.shape[1])
        log.info(
            "Duplicados por track_id detectados antes de deduplicar: %s",
            silver_stats.get("duplicate_track_ids_before_dedup"),
        )
        log.info("Duplicados eliminados: %s", silver_stats.get("duplicates_removed"))
        log.info("track_id unico: %s", "OK" if "track_id" in df and df["track_id"].is_unique else "ERROR")
        if genre_counts is not None:
            log.info("Generos encontrados: %s", int(df["track_genre"].nunique(dropna=True)))
            log.info("Distribucion de track_genre:\n%s", genre_counts.to_string())
        log.info("df.shape: %s", df.shape)
        log.info("df.dtypes.value_counts():\n%s", dtype_counts.to_string())
        log.info("Columnas con nulos:\n%s", null_count[null_count > 0].to_string())
        log.info("Porcentaje de nulos:\n%s", (null_ratio[null_ratio > 0] * 100).to_string())
        if feature_nulls is not None:
            log.info("Nulos por feature de audio:\n%s", feature_nulls.to_string())
        log.info("Columnas completamente vacias: %s", fully_empty_columns)
        log.info("Columnas faltantes respecto del esquema esperado: %s", missing_columns)
        log.info("Columnas inesperadas en Plata: %s", unexpected_columns)
        log.info("Rutas Bronce utilizadas: %s", silver_stats["bronze_paths"])
        log.info("Ruta CSV Plata producido: %s", silver_path)

        if problems:
            raise ValueError("Validacion final fallida:\n  - " + "\n  - ".join(problems))

        log.info("Validacion final: OK")
        return {
            **silver_stats,
            "validation": {
                "status": "OK",
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "genres": int(df["track_genre"].nunique(dropna=True)),
                "duplicate_track_ids_final": int(duplicate_final or 0),
                "fully_empty_columns": fully_empty_columns,
            },
        }

    source = resolve_source_revision()
    bronze = land_bronze(source)
    bronze_stats = inspect_bronze(bronze)
    silver = build_silver(bronze_stats)
    validate_silver(silver)


genreML_hf_ingest()
