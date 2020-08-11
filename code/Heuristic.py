import os
import sys
import copy
import grid2op
import numpy as np
import pandas as pd
import pandapower as pp

class TopologyHeuristic(object):

    def __init__(self, grid=None, path=None):
        if os.path.exists(os.path.join(path, "ptdf_matrix.csv")):
            self.ptdf_matrix = self._load_ptdf_matrix(path)
            self.lodf_matrix = self._load_lodf_matrix(path)
            self.lines_rewired = self._load_lines_rewired(path)
        else:
            if not isinstance(grid, pp.auxiliary.pandapowerNet):
                raise Exception("Wrong grid provided")
            self.grid = grid
            self.lines_rewired = self._rewire()
            self.ptdf_matrix = self._calculate_ptdf(method=2)
            self.lodf_matrix = self._calculate_lodf(method=2)

            self._save_lines_rewired(path)
            self._save_ptdf_matrix(path)
            self._save_lodf_matrix(path)

    def _rewire(self):
        """
        This function adds trafos to lines assigning a fixed (low) reactance
        returns: list of lines
        """

        # lines parameters
        res = pd.DataFrame(self.grid.line[["from_bus", "to_bus", "x_ohm_per_km"]])
        
        trafos = pd.DataFrame(self.grid.trafo[["hv_bus", "lv_bus"]])
        trafos = trafos.rename(columns={"hv_bus": "from_bus", "lv_bus": "to_bus"})
        # default value for transformers reactance
        trafos["x_ohm_per_km"] = 0.00001

        # aggregate trafos and lines
        res = res.append(trafos, ignore_index=True)

        return res

    def _calculate_ptdf(self, method):
        if method == 1:
            return self._calculate_ptdf_method1()
        else:
            return self._calculate_ptdf_method2()

    def _calculate_ptdf_method1(self): # add a state parameter
        """
        This function calculates the power transission distribution factor.
        It is based on the formula: PTDF = Br (lxl) x A (lxn) x X (nxn).
        l: number of lines, n: number of nodes
        """
        grid_lines = self.lines_rewired

        # TODO: why shape is twice the numbers of bus
        n_bus = int(self.grid.bus.shape[0]/2)
        n_line = grid_lines.shape[0]
        #print("n bus: {}".format(n_bus))
        #print("n lines: {}".format(n_line))

        Br = np.identity(n_line) * (1 / np.array(grid_lines["x_ohm_per_km"].sort_index()))
        #print("BR: \n{}".format(Br))
        A = np.zeros((n_line, n_bus)) 
        for i in range(n_line):
            frombus = grid_lines["from_bus"][i]
            tobus = grid_lines["to_bus"][i]
            A[i, frombus] = 1
            A[i, tobus] = -1
        #print("A: \n{}".format(A))
        B = np.zeros((n_bus, n_bus))
        for i in range(n_bus):
            for j in range(i, n_bus):
                if i == j:
                    a = np.array(grid_lines["from_bus"] == i)
                    b = np.array(grid_lines["to_bus"] == i)
                    adm_bus = np.array(grid_lines["x_ohm_per_km"][a+b])
                    B[i,j] = sum(1 / adm_bus)
                else:
                    a = np.array(grid_lines["from_bus"] == i)
                    b = np.array(grid_lines["to_bus"] == j)
                    from_and_to_bool = a * b
                    if not sum(from_and_to_bool) == 0: # if exists a line between nodes
                        B[i,j] = -1 * sum (1 / np.array(grid_lines["x_ohm_per_km"][from_and_to_bool]))
                        B[j,i] = B[i,j]
        #print("B: \n{}".format(B))
        #X = np.linalg.inv(B)
        #X[n_bus-1,:] = 0
        #X[:,n_bus-1] = 0

        a = Br.dot(A)
        # not include reference node. This is not well specify in most studies
        r = pd.DataFrame(a[:,:-1].dot(np.linalg.inv(B[:-1,:-1])))
        r[max(r.columns) + 1] = float(0)
        return r
    
    def _calculate_ptdf_method2(self, to_lodf=False):
        """
        This function calculates ptdf according to:
        PTDF = Bd*It*inv(I*Bd*It)*S
        Bd: diagonal matrix of susceptances (LxL)
        I: node-edge incidence matrix (NxL)
        S: matrix definition for slack bus (NxN)
        """

        # rewire (add transformers as lines) and get dimensions
        grid_lines = self.lines_rewired
        n_bus = int(self.grid.bus.shape[0]/2)
        n_line = grid_lines.shape[0]
        
        # diagonal susceptance matrix
        Bd = np.identity(n_line) * (1 / np.array(grid_lines["x_ohm_per_km"].sort_index()))

        # incidence matrix
        I = np.zeros((n_bus, n_line))
        for i in range(n_line):
            frombus = grid_lines["from_bus"][i]
            tobus = grid_lines["to_bus"][i]
            I[frombus, i] = 1
            I[tobus, i] = -1
        if not to_lodf:
            I = I[:-1,:] # not consider last bus

        S = np.identity(n_bus)
        S[n_bus-1, :] = -1
        S[:, n_bus-1] = 0
        if not to_lodf:
            S = S[:-1, :-1] # not consider last bus
        else:
            S = I

        It = np.transpose(I)
        inverse = np.linalg.inv(I.dot(Bd).dot(It))

        ans = pd.DataFrame(Bd.dot(It).dot(inverse).dot(S))
        if not to_lodf:
            # just add a 0 column
            ans[max(ans.columns) + 1] = float(0)
        return ans
  
    def _calculate_lodf(self, method=1):
        if method == 1:
            return self._calculate_lodf_method1()
        else:
            return self._calculate_lodf_method2()

    def _calculate_lodf_method1(self):
        """
        This function calculate the line outage distribution factors.
        It is based on the precalculation of the ptdf matrix.
        """
        return self.ptdf_matrix.dot(np.transpose(1/(1-self.ptdf_matrix)))

    def _calculate_lodf_method2(self):
        """
        This function calculates LODF matrix according to:
        LODF = PTDF'*inv(1l - diag(PTDF'))
        """
        #ptdf = self._calculate_ptdf_method2(to_lodf=False) #lxn
        ptdf_prim = self._calculate_ptdf_method2(to_lodf=True) #lxl
        a = np.ones(np.diag(ptdf_prim).size)
        b = 1 - np.diag(ptdf_prim)
        inverse = np.diag(np.divide(a, b, out=np.zeros_like(a), where=b!=0))
        return pd.DataFrame(ptdf_prim.dot(inverse))

    def _save_ptdf_matrix(self, path):
        self.ptdf_matrix.to_csv(os.path.join(path, "ptdf_matrix.csv"), index=False)
    
    def _save_lodf_matrix(self, path):
        self.lodf_matrix.to_csv(os.path.join(path, "lodf_matrix.csv"), index=False)
    
    def _save_lines_rewired(self, path):
        self.lines_rewired.to_csv(os.path.join(path, "lines_rewired.csv"), index=False)

    def _load_ptdf_matrix(self, path):
        return pd.read_csv(os.path.join(path, "ptdf_matrix.csv"))

    def _load_lodf_matrix(self, path):
        return pd.read_csv(os.path.join(path, "lodf_matrix.csv"))
    
    def _load_lines_rewired(self, path):
        return pd.read_csv(os.path.join(path, "lines_rewired.csv"))

    def get_ptdf_matrix(self):
        return self.ptdf_matrix
    
    def get_lodf_matrix(self):
        return self.lodf_matrix
    
    def get_lines_rewired(self):
        return self.lines_rewired


class TopologyPTDF(object):

    def __init__(self, env=None, path=None):
        self.TRAFO_REACTANCE_DFLT = 0.00001

        self.grid = copy.deepcopy(env.backend._grid)
        self.path_save = path
        self.lines_rewired = self._get_rewired_lines(self.grid)
        self.ptdfs = self._get_ptdfs()

    def _get_rewired_lines(self, grid):
        """
        Find and load lines_rewired.csv or create it.
        It is usef to extract reactances (x) of lines.
        return pandas.DataFrame
        """
        file_path = os.path.join(self.path_save, "lines_rewired.csv")
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            lines_rewired = self._rewire(grid)
            lines_rewired.to_csv(file_path, index=False)
            return lines_rewired
    
    def _rewire(self, grid):
        """
        This function adds trafos to lines assigning a fixed (low) reactance
        returns: list of lines
        """

        # lines parameters
        res = pd.DataFrame(grid.line[["from_bus", "to_bus", "x_ohm_per_km"]])
        
        trafos = pd.DataFrame(grid.trafo[["hv_bus", "lv_bus"]])
        trafos = trafos.rename(columns={"hv_bus": "from_bus", "lv_bus": "to_bus"})
        # default value for transformers reactance
        trafos["x_ohm_per_km"] = self.TRAFO_REACTANCE_DFLT

        # aggregate trafos and lines
        res = res.append(trafos, ignore_index=True)

        return res

    def _get_ptdfs(self):
        ptdfs = {}
        for f in os.listdir(self.path_save): 
            if f.startswith('ptdf_') and f.endswith('.csv'):
                key = hex(int(f[5:][:-4], 16)) # remove ptdf_ and .csv
                ptdfs[key] = pd.read_csv(os.path.join(self.path_save, f)) # load csv
        return ptdfs

    def check_calculated_matrix(self, line_status):
        """
        Evaluates if key exists. If not it calculate new ptdf matrix.
        Return: ptdf_matrix (pandas.DataFrame)
        """

        key = self._boollist_as_hex(line_status)
        try:
            return self.ptdfs[key]
        except KeyError:
            ptdf = self._calculate_ptdf(line_status, method=2)
            ptdf.to_csv(os.path.join(self.path_save, 'ptdf_{}.csv'.format(key)), index=False)
            self.ptdfs[key] = ptdf
            return self.ptdfs[key]
    
    def _boollist_as_hex(self, line_status):
        return hex(int(''.join(['1' if x else '0' for x in line_status]), 2))

    def _calculate_ptdf(self, line_status, method):
        if method == 1:
            return self._calculate_ptdf_method1(line_status)
        else:
            return self._calculate_ptdf_method2(line_status)

    def _calculate_ptdf_method1(self, line_status): # add a state parameter
        """
        This function calculates the power transission distribution factor.
        It is based on the formula: PTDF = Br (lxl) x A (lxn) x X (nxn).
        l: number of lines, n: number of nodes
        """
        print("Warning: Method 1 is deprecated. Using method 2...")
        return self._calculate_ptdf_method2(line_status)
        grid_lines = self.lines_rewired

        # TODO: why shape is twice the numbers of bus
        n_bus = int(self.grid.bus.shape[0]/2)
        n_line = grid_lines.shape[0]
        #print("n bus: {}".format(n_bus))
        #print("n lines: {}".format(n_line))

        Br = np.identity(n_line) * (1 / np.array(grid_lines["x_ohm_per_km"].sort_index()))
        #print("BR: \n{}".format(Br))
        A = np.zeros((n_line, n_bus)) 
        for i in range(n_line):
            frombus = grid_lines["from_bus"][i]
            tobus = grid_lines["to_bus"][i]
            A[i, frombus] = 1
            A[i, tobus] = -1
        #print("A: \n{}".format(A))
        B = np.zeros((n_bus, n_bus))
        for i in range(n_bus):
            for j in range(i, n_bus):
                if i == j:
                    a = np.array(grid_lines["from_bus"] == i)
                    b = np.array(grid_lines["to_bus"] == i)
                    adm_bus = np.array(grid_lines["x_ohm_per_km"][a+b])
                    B[i,j] = sum(1 / adm_bus)
                else:
                    a = np.array(grid_lines["from_bus"] == i)
                    b = np.array(grid_lines["to_bus"] == j)
                    from_and_to_bool = a * b
                    if not sum(from_and_to_bool) == 0: # if exists a line between nodes
                        B[i,j] = -1 * sum (1 / np.array(grid_lines["x_ohm_per_km"][from_and_to_bool]))
                        B[j,i] = B[i,j]
        #print("B: \n{}".format(B))
        #X = np.linalg.inv(B)
        #X[n_bus-1,:] = 0
        #X[:,n_bus-1] = 0

        a = Br.dot(A)
        # not include reference node. This is not well specify in most studies
        r = pd.DataFrame(a[:,:-1].dot(np.linalg.inv(B[:-1,:-1])))
        r[max(r.columns) + 1] = float(0)
        return r
    
    def _calculate_ptdf_method2(self, line_status, to_lodf=False):
        """
        This function calculates ptdf according to:
        PTDF = Bd*It*inv(I*Bd*It)*S
        Bd: diagonal matrix of susceptances (LxL)
        I: node-edge incidence matrix (NxL)
        S: matrix definition for slack bus (NxN)
        """

        # rewire (add transformers as lines) and get dimensions
        grid_lines = self.lines_rewired
        n_bus = int(self.grid.bus.shape[0]/2)
        n_line = grid_lines.shape[0]
        
        # diagonal susceptance matrix
        Bd = np.identity(n_line) * (1 / np.array(grid_lines["x_ohm_per_km"]))
        lines_out = np.where(np.array(line_status) == False)
        print("### line out: {}".format(lines_out))
        for l in lines_out:
            Bd[l, l] = 0

        # incidence matrix
        I = np.zeros((n_bus, n_line))
        for i in range(n_line):
            frombus = grid_lines["from_bus"][i]
            tobus = grid_lines["to_bus"][i]
            I[frombus, i] = 1
            I[tobus, i] = -1
        if not to_lodf:
            I = I[:-1,:] # not consider last bus

        S = np.identity(n_bus)
        S[n_bus-1, :] = -1
        S[:, n_bus-1] = 0
        if not to_lodf:
            S = S[:-1, :-1] # not consider last bus
        else:
            S = I

        It = np.transpose(I)
        inverse = np.linalg.inv(I.dot(Bd).dot(It))

        ans = pd.DataFrame(Bd.dot(It).dot(inverse).dot(S))
        if not to_lodf:
            # just add a 0 column
            ans[max(ans.columns) + 1] = float(0)
        return ans