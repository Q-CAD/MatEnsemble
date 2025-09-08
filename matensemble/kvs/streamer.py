import flux
import flux.kvs as kvs
import json
import pandas as pd


def init_kvs():
        """Initialize KVS storage."""
        h = flux.Flux()
        kvsdir = kvs.KVSDir(h)
        return kvsdir

def register_on_stream(analysis_key="timeseries", **kwargs):
        """Append time-series data under a single KVS key."""
       
        kvsdir = init_kvs()

        # try to load existing dataset
        try:
                raw = kvsdir[analysis_key]
                series = json.loads(raw)
        except Exception:
                series = []

        new_point = dict(kwargs)
        series.append(new_point)

        # write back updated dataset
        kvsdir[analysis_key] = json.dumps(series)
        kvsdir.commit()
        return


def extract_from_stream(analysis_key="timeseries", sort=True, write_dataframe=True, handle=None):
        """Continuously read full series."""
        
        if handle is None:
                kvsdir = init_kvs()
        else:
                kvsdir = kvs.KVSDir(handle)
                
        raw = kvsdir[analysis_key]
        series = json.loads(raw)
        packed_df = pd.DataFrame(series)
        # Create DataFrame and sort by timestep
        if sort:
                try:
                        packed_df = packed_df.sort_values('timestep').reset_index(drop=True)
                except Exception as e:
                        print(f"Error occurred while sorting DataFrame: {e}, Check whether 'timestep' key is present in the data points.")

        # Optionally write to CSV file
        if write_dataframe:
                packed_df.to_csv('consumed_dataframe.csv', index=None, sep=' ')

        return packed_df


