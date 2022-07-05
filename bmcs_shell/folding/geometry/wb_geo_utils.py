

class WBGeoUtils:

    @staticmethod
    def export_obj_file(wb_shell, name='wb_3d_print.obj'):
        I_Fi = wb_shell.I_Fi
        X_Ia = wb_shell.X_Ia / 1000

        # Write to obj file
        f = open(name, 'w')
        f.write('# Vertices: (' + str(X_Ia.shape[0]) + ')\n')
        for v in X_Ia:
            f.write('v ' + str(v)[1:-1] + '\n')
        f.write('\n# Tri Facets: (' + str(I_Fi.shape[0]) + ')\n')
        for ind in I_Fi + 1:
            f.write('f ' + str(ind)[1:-1] + '\n')
        f.close()
