# For gmsh tutorials see:  https://gitlab.onelab.info/gmsh/gmsh/-/tree/master/tutorials

import gmsh
import sys
import numpy as np
import bmcs_utils.api as bu
from scipy.spatial.transform import Rotation


class PrintTessellGenerator:

    def __init__(self, X_Ia, I_Fi):
        self.X_Ia, self.I_Fi = X_Ia, I_Fi

    def export_geometry_as_step(self):
        # Export geometry as a STEP file
        gmsh.write('wb_shell.step')

    def export_mesh_as_stl(self):
        # Export mesh as an STL file
        gmsh.write('wb_shell.stl')

    def generate(self):
        thickness = 2
        fold_thickness = 0.6

        # Each dimension Dim has Tags refering to objects starting from 1,
        #  (a point is dim=1, a line, surface or more is dim=2 and a volume is dim=3)

        gmsh.initialize()

        gmsh.model.add("t11")
        # meshSize = 0

        # CAD (OpenCASCADE) kernel should be used (gmsh.model.occ) instead of default kernel (gmsh.model.geo)
        # to be able to use boolean operations (cut, fuse, etc..) and CAD objects.

        # Adding tunnels to subtract later: ---------------------------------------------------------------

        creases = []

        X_Lia = X_Ia[I_Li]
        lines_num = X_Lia.shape[0]

        for l in range(lines_num):
            path_2a = X_Lia[l, ...]

            # Pipe cross-section
            cs_points = np.array([[1., 0, 0], [0, thickness - fold_thickness, 0], [-1, 0, 0]])
            cs_points = bu.Extruder.transform_first_contour(path_2a, cs_points, adapt_dimensions=True)

            # Rotate the triangle around its base if it points down 90, 180 (90+90), 270 (180+90)
            # (for a square or circle, this is not needed)
            center_of_rotation = (cs_points[0, :] + cs_points[2, :]) / 2
            for angle in range(3):
                if np.any(cs_points[:, 2] < -0.001):
                    cs_points = rotate_points_around_vector(cs_points, path_2a[1, :] - path_2a[0, :], 90,
                                                            center_of_rotation)
                else:
                    break

            points = []
            for point in cs_points:
                points.append(gmsh.model.occ.addPoint(*point))
            lines = [gmsh.model.occ.addLine(points[k], points[k + 1]) for k in range(len(points) - 1)] + [
                gmsh.model.occ.addLine(points[-1], points[0])]
            cl = gmsh.model.occ.addCurveLoop(lines)
            pl = gmsh.model.occ.addPlaneSurface([cl])

            # Pipe path (wire)
            points = []
            points.append(gmsh.model.occ.addPoint(*path_2a[0, :]))
            points.append(gmsh.model.occ.addPoint(*path_2a[1, :]))
            line = gmsh.model.occ.addLine(points[0], points[1])
            wire = gmsh.model.occ.addWire(curveTags=[line])
            creases.append(gmsh.model.occ.addPipe(dimTags=[(2, pl)], wireTag=wire)[0])

        # vols = gmsh.model.occ.getEntities(dim=3)
        # gmsh.model.occ.fuse(vols, vols)

        # Adding outer area of the pattern with extrusion: ------------------------------------------------

        xpoints = np.array([gmsh.model.occ.addPoint(*X_a) for X_a in X_Ia])

        wb_facets = []

        for I_i in I_Fi:
            xpoints1 = xpoints[I_i]
            curves = [gmsh.model.occ.addLine(xpoints1[k], xpoints1[k + 1]) for k in range(len(xpoints1) - 1)] + [
                gmsh.model.occ.addLine(xpoints1[-1], xpoints1[0])]

            cl = gmsh.model.occ.addCurveLoop(curves)
            pl = gmsh.model.occ.addPlaneSurface([cl])
            wb_facets.append(pl)

            # To generate quadrangles instead of triangles, we can simply add
        #     gmsh.model.mesh.setRecombine(1, pl)

        # pg = gmsh.model.addPhysicalGroup(dim = 3, tags=wb_facets, name='pg')
        # print(pg)

        # Extrude (extrude is already a volume or CAD object)
        for wb_facet in wb_facets:
            ext = gmsh.model.occ.extrude(dimTags=[(2, wb_facet)], dx=0, dy=0, dz=2, numElements=[], heights=[],
                                         recombine=True)

        # sl = gmsh.model.occ.addSurfaceLoop(wb_facets)
        # base_vol = gmsh.model.occ.addVolume([sl])

        # creases_num = 1
        # for vol in np.arange(gmsh.model.occ.getMaxTag(dim=3))[creases_num:-1]:
        # #     gmsh.model.occ.fuse([(3, vol)], [(3, vol + 1)])
        #     gmsh.model.occ.cut([(3, vol)], [(3, creases[0])])

        vols = gmsh.model.occ.getEntities(dim=3)
        tess_block = gmsh.model.occ.fuse(vols[len(creases):], vols[len(creases):])
        print('tess_block=', tess_block[0])
        # gmsh.model.occ.getEntities(dim=3)
        print('creases=', creases)

        # gmsh.model.occ.cut(tess_block[0], [creases[0]])
        print(gmsh.model.occ.cut(tess_block[0], creases))

        gmsh.model.occ.remove(dimTags=gmsh.model.occ.getEntities(dim=2), recursive=True)

        # max3DTag = gmsh.model.occ.getMaxTag(3)
        # arr = np.arange(lines_num + 1, max3DTag + 1)
        # print(gmsh.model.occ.getMaxTag(3))
        # for l in range(lines_num + 1):
        #     print('l=', l)
        #     print(gmsh.model.occ.getMaxTag(3))
        #     for i in arr:
        #         if i == arr[-2]:
        #             gmsh.model.occ.cut([(3, i)], creases[l]) # delete pipe at last
        #             break
        #         else:
        #             gmsh.model.occ.cut([(3, i)], creases[l], removeTool=False) # keep pipe to cut from it later

        # dimTag means tuple of (dimention, tag). Where tag is like an ID

        # Meshing ---------------------------------------------------- ------------------------------------------------

        gmsh.model.occ.synchronize()

        # field = gmsh.model.mesh.field
        # field.add("MathEval", 1)
        # field.setString(1, "F", "1")
        # field.setAsBackgroundMesh(1)

        # # To generate quadrangles instead of triangles, we can simply add
        # gmsh.model.mesh.setRecombine(2, pl)

        # If we'd had several surfaces, we could have used the global option
        # "Mesh.RecombineAll":
        #
        # gmsh.option.setNumber("Mesh.RecombineAll", 1)

        # You can also set the subdivision step alone, with
        #
        # gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)

        # gmsh.model.mesh.generate(2)

        # Note that you could also apply the recombination algorithm and/or the
        # subdivision step explicitly after meshing, as follows:
        #
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.recombine()
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        # gmsh.model.mesh.refine()

        # Launch the GUI to see the results:
        if '-nopopup' not in sys.argv:
            gmsh.fltk.run()

        gmsh.finalize()