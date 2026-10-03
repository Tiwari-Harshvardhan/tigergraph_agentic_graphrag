import os
import pyTigerGraph as tg
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    host = os.getenv("TIGERGRAPH_HOST") or os.getenv("TG_HOST")
    graphname = os.getenv("TIGERGRAPH_GRAPH") or os.getenv("TG_GRAPHNAME")
    secret = os.getenv("TIGERGRAPH_SECRET") or os.getenv("TG_SECRET")

    if not all([host, graphname, secret]):
        raise RuntimeError(
            "TIGERGRAPH_HOST/TG_HOST, TIGERGRAPH_GRAPH/TG_GRAPHNAME and "
            "TIGERGRAPH_SECRET/TG_SECRET must be configured"
        )

    conn = tg.TigerGraphConnection(
        host=host,
        graphname=graphname,
        gsqlSecret=secret,
    )

    conn.getToken(secret)

    return conn

