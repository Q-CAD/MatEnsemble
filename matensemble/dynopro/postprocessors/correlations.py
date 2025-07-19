import numpy as np
import copy
import gc

def compute_rdf(matesnemble_data, cutoff=3.0, number_of_bins=100, z_min=None):

        from ovito.modifiers import CoordinationAnalysisModifier, ExpressionSelectionModifier
        data = copy.deepcopy(matesnemble_data)
        if z_min is not None:
                ## Select particles with Z > z_min ##
                selection_modifier = ExpressionSelectionModifier(expression=f"Position.Z > {z_min}")
                data.data.apply(selection_modifier)

        modifier_coordination = CoordinationAnalysisModifier(cutoff=cutoff, number_of_bins=number_of_bins)
        modifier_coordination.only_selected = z_min is not None
        data.data.apply(modifier_coordination)
        rdf = data.data.tables['coordination-rdf'].xy()

        del data
        gc.collect()

        return rdf

def compute_adf(matesnemble_data, cutoff=4.0, number_of_bins=100, z_min=None):

        from ovito.modifiers import BondAnalysisModifier, CreateBondsModifier, ExpressionSelectionModifier, DeleteSelectedModifier
        data = copy.deepcopy(matesnemble_data)
        if z_min is not None:
                ## Select particles with Z > z_min ##
                selection_modifier = ExpressionSelectionModifier(expression=f"Position.Z <= {z_min}")
                data.data.apply(selection_modifier)
                modifier_delete = DeleteSelectedModifier()
                data.data.apply(modifier_delete)


        modifier_cb = CreateBondsModifier(cutoff = cutoff)
        modifier_ba = BondAnalysisModifier(bins=number_of_bins)

        data.data.apply(modifier_cb)
        data.data.apply(modifier_ba)
        adf = data.data.tables['bond-angle-distr'].xy()

        del data
        gc.collect()

        return adf
        