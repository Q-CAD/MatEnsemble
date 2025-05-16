def compute_order_from_pairs(file, frame_start=0, frame_end='DEFAULT', frame_interval = 1, bond_lo=2.4, bond_hi=2.6, cutoff=3.0, number_of_bins=200):
        
	from ovito.io import import_file
	from ovito.modifiers import CoordinationAnalysisModifier

	pipeline = import_file(file)
	modifier = CoordinationAnalysisModifier(cutoff = cutoff, number_of_bins = number_of_bins)
	pipeline.modifiers.append(modifier)
	order=[]
	rdf_traj = {'frame':[], 'rdf': []}

	if frame_end == 'DEFAULT':
		frame_end = pipeline.source.num_frames

	for frame in range(frame_start,frame_end, frame_interval):
		data = pipeline.compute(frame)
		# coordination_data = data.particles_.coordination[...]
		rdf = data.tables['coordination-rdf'].xy()
		order.append(rdf[:,1][(rdf[:,0]<bond_hi) & ((rdf[:,0]>bond_lo))].sum())
		rdf_traj['frame'].append(frame)
		rdf_traj['rdf'].append(rdf)
		
	return order, rdf_traj
