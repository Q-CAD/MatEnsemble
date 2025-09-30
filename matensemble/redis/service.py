import subprocess
import time
import socket
import redis
import json
import pandas as pd

class RedisService:

    def __init__(self, host=None, port=None):
        self.host = host
        self.port = port

    def find_free_port(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))  # let OS pick a free port
        _, self.port = s.getsockname()
        s.close()
        return

    def launch(self):

        if self.port is None:
            self.find_free_port()

        # 1. Start server in background
        self.proc = subprocess.Popen(["flux", "run", "-N1", "--setattr=system.keep=true",
        "/bin/bash", "-c", f"echo $(hostname); redis-server --port {self.port} --bind 0.0.0.0 --daemonize yes"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # 2. Wait a few seconds for server to initialize
        time.sleep(2)

        self.host = self.proc.stdout.readline().strip()

        return
    

    @staticmethod
    def make_key(namespace, key):
        """Helper to create a Redis key with namespace."""
        return f"{namespace}:{key}"

    def register_on_stream(self, namespace, key="timeseries", **kwargs):
        """
        Push data under a namespaced key.
        Example stored key: 'case1:xx'
        """
        r = redis.Redis(host=self.host, port=self.port, decode_responses=True)
        full_key = self.make_key(namespace, key)

        # try to load existing dataset
        try:
            raw = r.lrange(full_key, 0, -1)
            series = [json.loads(item) for item in raw]
        except Exception:
            series = []

        new_point = dict(kwargs)
        series.append(new_point)

        r.rpush(full_key, json.dumps(kwargs))

    def extract_from_stream(self, namespace, key="timeseries", sort=True):
        """
        Extract data from a namespaced key into a DataFrame.
        """
        r = redis.Redis(host=self.host, port=self.port, decode_responses=True)
        full_key = self.make_key(namespace, key)
        raw_list = r.lrange(full_key, 0, -1)
        series = [json.loads(item) for item in raw_list]
        df = pd.DataFrame(series)
        if sort and "timestep" in df.columns:
            df = df.sort_values("timestep").reset_index(drop=True)
        return df

    def shutdown(self):

        try:
            r = redis.Redis(host=self.host, port=self.port, decode_responses=True)
            r.shutdown()
            print("Redis server stopped and stream cleared")
        except Exception as e:
            print("Failed to shutdown Redis:", e)



