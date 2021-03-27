
import traits.api as tr
from .wb_fets2d3u1m_fe import FETS2D3U1M
from bmcs_shell.folding.wb_tessellation import WBTessellation
import numpy as np
from ibvpy.mesh.i_fe_uniform_domain import IFEUniformDomain

INPUT = '+cp_input'


@tr.provides(IFEUniformDomain)
class FETriangularMesh(tr.HasStrictTraits):

    X_Id = tr.Array(np.float_, value=[[0,0, 0], [2,0, 0], [2,2,0], [2,0,0], [1,1,0]])
    I_Fi = tr.Array(np.int_, value=[[0,1,4],
                                    [1,2,4],
                                    [2,3,4],
                                    [3,0,4]])

    fets = tr.Instance(FETS2D3U1M, ())

    n_nodal_dofs = tr.DelegatesTo('fets')
    dof_offset = tr.Int(0)

    n_active_elems = tr.Property
    def _get_n_active_elems(self):
        return len(self.I_Fi)


@tr.provides(IFEUniformDomain)
class WBShellFETriangularMesh(WBTessellation, FETriangularMesh):

    X_Id = tr.Property
    def _get_X_Id(self):
        return self.X_Ia


