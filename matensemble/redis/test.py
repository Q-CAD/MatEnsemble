from matensemble.redis.service import RedisService

rds = RedisService()
#rds.port = 6379
rds.launch()
rds.register_on_stream('30K', key='xx', timestep=1000, xx=0.0)
rds.register_on_stream('30K', key='xx', timestep=0, xx=0.1)
rds.register_on_stream('20K', key='xx', timestep=0, xx=0.0)
rds.register_on_stream('30K', key='xx', timestep=500, xx=0.5)

print ("30K:", rds.extract_from_stream('30K', key='xx'))
print ("20K:", rds.extract_from_stream('20K', key='xx'))
rds.shutdown()