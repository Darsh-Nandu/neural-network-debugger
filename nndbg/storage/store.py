from __future__ import annotations

import json
import uuid
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import torch

from nndbg.utils import get_logger

logger = get_logger(__name__)

class ActivationStore:

    def __init__(self, path: str = ":memory:"):
        self.path = path
        self._conn = duckdb.connect(path)
        self._init_schema()
        logger.info(f"ActivationStore ready at '{path}'")

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id      VARCHAR PRIMARY KEY,
                model_name  VARCHAR,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                config      JSON
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS activations (
                run_id      VARCHAR,
                axis_name   VARCHAR,
                group_name  VARCHAR,
                sample_idx  INTEGER,
                layer_name  VARCHAR,
                mean        FLOAT,
                std         FLOAT,
                min_val     FLOAT,
                max_val     FLOAT,
                sparsity    FLOAT,
                l2_norm     FLOAT,
                shape       VARCHAR
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS layer_stats (
                run_id      VARCHAR,
                axis_name   VARCHAR,
                layer_name  VARCHAR,
                group_name  VARCHAR,
                probe_score FLOAT,
                mean_diff   FLOAT
            )
        """)

    def create_run(self, model_name: str, config: dict) -> str:
        run_id = str(uuid.uuid4())[:8]
        self._conn.execute(
            "INSERT INTO runs VALUES (?, ?, CURRENT_TIMESTAMP, ?)",
            [run_id, model_name, json.dumps(config)]
        )
        logger.info(f"Created run: {run_id}")
        return run_id
    
    def store_activation(
            self,
            run_id: str,
            axis_name: str,
            group_name: str,
            sample_idx: int,
            layer_name: str,
            tensor: torch.Tensor,
    ) -> None:
        arr = tensor.float().numpy().flatten()
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        self._conn.execute(
            "INSERT INTO activations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                run_id, axis_name, group_name, sample_idx, layer_name,
                float(arr.mean()),
                float(arr.std()),
                float(arr.min()),
                float(arr.max()),
                float((arr == 0).mean()),
                float(np.linalg.norm(arr)),
                str(list(tensor.shape)),
            ],
        )

    def store_layer_stat(
            self,
            run_id: str,
            axis_name: str,
            layer_name: str,
            group_name: str,
            probe_score: float,
            mean_diff: float,
    ) -> None:
        self._conn.execute(
            "INSERT INTO layer_stats VALUES (?, ?, ?, ?, ?, ?)",
            [run_id, axis_name, layer_name, group_name, probe_score, mean_diff]
        )

    def get_layer_means(
            self, run_id: str, axis_name: str, layer_name: str
    ) -> Dict[str, Dict]:
        rows = self._conn.execute(
            """
            SELECT group_name, AVG(mean) as avg_mean, AVG(std) as avg_std
            FROM activations
            WHERE run_id=? AND axis_name=? AND layer_name=?
            GROUP BY group_name ORDER BY group_name
            """,
            [run_id, axis_name, layer_name]
        ).fetchall()
        return {r[0]: {"mean": r[1], "std": r[2]} for r in rows}

    def get_probe_scores(
            self, run_id: str, axis_name: str
    ) -> List[Tuple[str, float]]:
        return self._conn.execute(
            """
            SELECT layer_name, AVG(probe_score) as score
            FROM layer_status
            WHERE run_id=? and axis_name=?
            GROUP BY layer_name ORDER BY layer_name
            """,
            [run_id, axis_name]
        ).fetchall()

    def get_all_layers(self, run_id: str) -> List[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT layer_name FROM activations WHERE run_id=? ORDER BY layer_name",
            [run_id]
        ).fetchall()
        return [r[0] for r in rows]
        

    def get_activation_matrix(
        self, run_id: str, axis_name: str
    ) -> Tuple[List[str], List[str], np.ndarray]:
        """
        Returns (layers, groups, matrix).
        matrix[i, j] = mean activation of layer i for group j.
        """
        rows = self._conn.execute(
            """
            SELECT layer_name, group_name, AVG(mean) as m
            FROM activations
            WHERE run_id=? AND axis_name=?
            GROUP BY layer_name, group_name
            ORDER BY layer_name, group_name
            """,
            [run_id, axis_name],
        ).fetchall()

        layers = sorted(set(r[0] for r in rows))
        groups = sorted(set(r[1] for r in rows))
        matrix = np.zeros((len(layers), len(groups)))

        l_idx = {l: i for i, l in enumerate(layers)}
        g_idx = {g: i for i, g in enumerate(groups)}

        for layer, group, val in rows:
            matrix[l_idx[layer], g_idx[group]] = val or 0.0

        return layers, groups, matrix

    def get_sample_stats(
            self,
            run_id: str,
            axis_name: str,
            layer_name: str
        ) -> Dict[str, List[Dict]]:

        rows = self._conn.execute(
            """
            SELECT group_name, mean, std, 12_norm, sparsity, min_val, max_val
            FROM activation
            WHERE run_id=? AND axis_name=? AND layer_name=?
            ORDER BY group_name, sample_idx
            """,
            [run_id, axis_name, layer_name]
        ).fetchall()

        result: Dict[str, List[Dict]] = {}
        for row in rows:
            g = row[0]
            if g not in result:
                result[g] = []
            result[g].append({
                "mean":     row[1],
                "std":      row[2],
                "l2_norm":  row[3],
                "sparsity": row[4],
                "min_val":  row[5],
                "max_val":  row[6],
            })
        return result

    # Cleanup
    def close(self) -> None:
        self._conn.close()
    
    def __repr__(self) -> str:
        return f"ActivationStore(path='{self.path}')"