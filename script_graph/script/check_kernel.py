import sys,gc
import scipy
from tqdm import tqdm
sys.path.append("../")
from common import *
from collections import defaultdict

import networkx as nx
import rdkit.Chem.Draw
from rdkit import Chem
from rdkit.Chem import AllChem,rdMolTransforms,ChemicalFeatures
from rdkit import RDConfig
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
from rdkit.Chem.rdmolops import SanitizeFlags
from sklearn import preprocessing
from statistics import mode
DrawingOptions.bondLineWidth=1.8


## feature extraction #####################################################

COUPLING_TYPE_STATS=[
    #type   #mean, std, min, max
    '1JHC',  94.9761528641869,   18.27722399839607,   66.6008,   204.8800,
    '2JHC',  -0.2706244378832,    4.52360876732858,  -36.2186,    42.8192,
    '3JHC',   3.6884695895355,    3.07090647005439,  -18.5821,    76.0437,
    '1JHN',  47.4798844844683,   10.92204561670947,   24.3222,    80.4187,
    '2JHN',   3.1247536134185,    3.67345877025737,   -2.6209,    17.7436,
    '3JHN',   0.9907298624944,    1.31538940138001,   -3.1724,    10.9712,
    '2JHH', -10.2866051639817,    3.97960190019757,  -35.1761,    11.8542,
    '3JHH',   4.7710233597359,    3.70498129755812,   -3.0205,    17.4841,
]
NUM_TARGET = len(COUPLING_TYPE_STATS)//5

COUPLING_TYPE_MEAN = [ COUPLING_TYPE_STATS[i*5+1] for i in range(NUM_TARGET)]
COUPLING_TYPE_STD  = [ COUPLING_TYPE_STATS[i*5+2] for i in range(NUM_TARGET)]
COUPLING_TYPE      = [ COUPLING_TYPE_STATS[i*5  ] for i in range(NUM_TARGET)]


#---

SYMBOL=['H', 'C', 'N', 'O', 'F']

BOND_TYPE = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
HYBRIDIZATION=[
    #Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    #Chem.rdchem.HybridizationType.SP3D,
    #Chem.rdchem.HybridizationType.SP3D2,
]


def one_hot_encoding(x, set):
    one_hot = [int(x == s) for s in set]
    if 0:
        if sum(one_hot)==0: print('one_hot_encoding() return NULL!', x, set)
    return one_hot

class WL(object):
    """
    original: @nyanp
    """
    def __init__(self, mol, depth, sep = None):
        self.m = mol
        self.init()
        self.sep = sep
        self.calc(depth)
        
    def init(self):
        self.labels = {}
        for i in range(len(self.m.GetAtoms())):
            self.labels[i] = '(' + self.m.GetAtomWithIdx(i).GetSymbol() + ')'
            
    def calc(self, depth: int):
        for i in range(depth):
            new_labels = {}
            
            sep = self.sep[i] if self.sep else '-'
            
            for atom in self.m.GetAtoms():
                lbl = ''.join(sorted([self.labels[n.GetIdx()] for n in atom.GetNeighbors()]))
                new_labels[atom.GetIdx()] = '(' + self.labels[atom.GetIdx()] + sep + lbl + ')'
                
            self.labels = new_labels
            
    def to_arr(self):
        return [self.labels[i] for i in range(len(self.m.GetAtoms()))]



###############################################################
def make_graph(molecule_name, gb_structure, gb_scalar_coupling):
    #https://stackoverflow.com/questions/14734533/how-to-access-pandas-groupby-dataframe-by-key
    #----
    global unique_wl
    df = gb_scalar_coupling.get_group(molecule_name)
    # ['id', 'molecule_name', 'atom_index_0', 'atom_index_1', 'type',
    #        'scalar_coupling_constant', 'fc', 'sd', 'pso', 'dso'],
    coupling = (
        df.id.values,
        df[['fc', 'sd', 'pso', 'dso']].values,
        df[['atom_index_0', 'atom_index_1']].values,
        #type = np.array([ one_hot_encoding(t,COUPLING_TYPE) for t in df.type.values ], np.uint8)
        np.array([ COUPLING_TYPE.index(t) for t in df.type.values ], np.int32),
        df.scalar_coupling_constant.values,
    )
    #----
    df = gb_structure.get_group(molecule_name)
    df = df.sort_values(['atom_index'], ascending=True)
    # ['molecule_name', 'atom_index', 'atom', 'x', 'y', 'z']
    a   = df.atom.values.tolist()
    xyz = df[['x','y','z']].values
    mol = mol_from_axyz(a, xyz)
    #---
    assert( #check
       a == [ mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
    )
    #---
    wl_depth2 = WL(mol, depth=2).to_arr()    
    unique_wl |= set(wl_depth2)
    return None

## xyz to mol #############################################################
#<todo> check for bug ????
# https://github.com/jensengroup/xyz2mol
def get_atom(atom):
    ATOM = [ x.strip() for x in ['h ','he', \
          'li','be','b ','c ','n ','o ','f ','ne', \
          'na','mg','al','si','p ','s ','cl','ar', \
          'k ','ca','sc','ti','v ','cr','mn','fe','co','ni','cu', \
          'zn','ga','ge','as','se','br','kr', \
          'rb','sr','y ','zr','nb','mo','tc','ru','rh','pd','ag', \
          'cd','in','sn','sb','te','i ','xe', \
          'cs','ba','la','ce','pr','nd','pm','sm','eu','gd','tb','dy', \
          'ho','er','tm','yb','lu','hf','ta','w ','re','os','ir','pt', \
          'au','hg','tl','pb','bi','po','at','rn', \
          'fr','ra','ac','th','pa','u ','np','pu'] ]
    atom = atom.lower()
    return ATOM.index(atom) + 1

def getUA(maxValence_list, valence_list):
    UA = []
    DU = []
    for i, (maxValence,valence) in enumerate(zip(maxValence_list, valence_list)):
        if maxValence - valence > 0:
            UA.append(i)
            DU.append(maxValence - valence)
    return UA,DU


def get_BO(AC,UA,DU,valences,UA_pairs,quick):
    BO = AC.copy()
    DU_save = []
    while DU_save != DU:
        for i,j in UA_pairs:
            BO[i,j] += 1
            BO[j,i] += 1
        BO_valence = list(BO.sum(axis=1))
        DU_save = copy.copy(DU)
        UA, DU = getUA(valences, BO_valence)
        UA_pairs = get_UA_pairs(UA,AC,quick)[0]
    return BO


def valences_not_too_large(BO,valences):
    number_of_bonds_list = BO.sum(axis=1)
    for valence, number_of_bonds in zip(valences,number_of_bonds_list):
        if number_of_bonds > valence:
            return False
    return True




def get_atomic_charge(atom,atomic_valence_electrons,BO_valence):
    if atom == 1:
        charge = 1 - BO_valence
    elif atom == 5:
        charge = 3 - BO_valence
    elif atom == 15 and BO_valence == 5:
        charge = 0
    elif atom == 16 and BO_valence == 6:
        charge = 0
    else:
        charge = atomic_valence_electrons - 8 + BO_valence
    return charge

def BO2mol(mol,BO_matrix, atomicNumList,atomic_valence_electrons,mol_charge,charged_fragments):
# based on code written by Paolo Toscani
    l = len(BO_matrix)
    l2 = len(atomicNumList)
    BO_valences = list(BO_matrix.sum(axis=1))
    if (l != l2):
        raise RuntimeError('sizes of adjMat ({0:d}) and atomicNumList '
            '{1:d} differ'.format(l, l2))
    rwMol = Chem.RWMol(mol)
    bondTypeDict = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE
    }
    for i in range(l):
        for j in range(i + 1, l):
            bo = int(round(BO_matrix[i, j]))
            if (bo == 0):
                continue
            bt = bondTypeDict.get(bo, Chem.BondType.SINGLE)
            rwMol.AddBond(i, j, bt)
    mol = rwMol.GetMol()
    if charged_fragments:
        mol = set_atomic_charges(mol,atomicNumList,atomic_valence_electrons,BO_valences,BO_matrix,mol_charge)
    else:
        mol = set_atomic_radicals(mol,atomicNumList,atomic_valence_electrons,BO_valences)
    return mol

def set_atomic_charges(mol,atomicNumList,atomic_valence_electrons,BO_valences,BO_matrix,mol_charge):
    q = 0
    for i,atom in enumerate(atomicNumList):
        a = mol.GetAtomWithIdx(i)
        charge = get_atomic_charge(atom,atomic_valence_electrons[atom],BO_valences[i])
        q += charge
        if atom == 6:
            number_of_single_bonds_to_C = list(BO_matrix[i,:]).count(1)
            if number_of_single_bonds_to_C == 2 and BO_valences[i] == 2:
                    q += 1
                    charge = 0
            if number_of_single_bonds_to_C == 3 and q + 1 < mol_charge:
                    q += 2
                    charge = 1
        if (abs(charge) > 0):
            a.SetFormalCharge(int(charge))
    # shouldn't be needed anymore bit is kept just in case
    #mol = clean_charges(mol)
    return mol


def set_atomic_radicals(mol,atomicNumList,atomic_valence_electrons,BO_valences):
# The number of radical electrons = absolute atomic charge
    for i,atom in enumerate(atomicNumList):
        a = mol.GetAtomWithIdx(i)
        charge = get_atomic_charge(atom,atomic_valence_electrons[atom],BO_valences[i])
        if (abs(charge) > 0):
            a.SetNumRadicalElectrons(abs(int(charge)))
    return mol

def get_bonds(UA,AC):
    bonds = []
    for k,i in enumerate(UA):
        for j in UA[k+1:]:
            if AC[i,j] == 1:
                bonds.append(tuple(sorted([i,j])))
    return bonds

def get_UA_pairs(UA,AC,quick):
    bonds = get_bonds(UA,AC)
    if len(bonds) == 0:
        return [()]
    if quick:
        G=nx.Graph()
        G.add_edges_from(bonds)
        #dictbond=nx.max_weight_matching(G)
        #lkeys = list(dictbond.keys())
        #lvals = list(dictbond.values())
        #UA_pairs = [((lkeys[i],lvals[i]) for i in range(0,len(lkeys),2) )]
        #return UA_pairs
        UA_pairs = [list(nx.max_weight_matching(G))]
        return UA_pairs
    max_atoms_in_combo = 0
    UA_pairs = [()]
    for combo in list(itertools.combinations(bonds, int(len(UA)/2))):
        flat_list = [item for sublist in combo for item in sublist]
        atoms_in_combo = len(set(flat_list))
        if atoms_in_combo > max_atoms_in_combo:
            max_atoms_in_combo = atoms_in_combo
            UA_pairs = [combo]
 #           if quick and max_atoms_in_combo == 1*int(len(UA)/2):
 #               return UA_pairs
        elif atoms_in_combo == max_atoms_in_combo:
            UA_pairs.append(combo)
    return UA_pairs


""""""""""""""
""""""""""""""

def BO_is_OK(BO,AC,charge,DU,atomic_valence_electrons,atomicNumList,charged_fragments):
    Q = 0 # total charge
    q_list = []
    if charged_fragments:
        BO_valences = list(BO.sum(axis=1))
        for i,atom in enumerate(atomicNumList):
            q = get_atomic_charge(atom,atomic_valence_electrons[atom],BO_valences[i])
            Q += q
            if atom == 6:
                number_of_single_bonds_to_C = list(BO[i,:]).count(1)
                if number_of_single_bonds_to_C == 2 and BO_valences[i] == 2:
                    Q += 1
                    q = 2
                if number_of_single_bonds_to_C == 3 and Q + 1 < charge:
                    Q += 2
                    q = 1
            if q != 0:
                q_list.append(q)
    if (BO-AC).sum() == sum(DU) and charge == Q and len(q_list) <= abs(charge):
        return True
    else:
        return False


"""
hinokky function
"""
def calc_total_electrons(valence_list,atomic_num_list):
    electron_sum=0
    for i0 in range(len(atomic_num_list)):
        atomic_num = atomic_num_list[i0]
        if (atomic_num==1):
            electron_sum +=  2 -valence_list[i0]
        elif (atomic_num<10):
            electron_sum += 8 -valence_list[i0]
    return electron_sum

def ang_vectors(r1,r2):
    dot=np.dot(r1,r2)
    norm_r1 = np.linalg.norm(r1)
    norm_r2 = np.linalg.norm(r2)
    cos = dot /norm_r1/norm_r2
    rad = np.arccos(cos)/2.0/np.pi*360.0
    return rad


def calc_angle_list(atom0,idbond,bond_list,xyz):
    r0 = np.array(xyz[idbond]) - np.array(xyz[atom0])
    # ang_mean = 2.0*np.pi * 109.5/360.0
    ang_mean = 109.5
    ang_list=[]
    for ib in bond_list:
        if (idbond==ib):
            ang_list.append(ang_mean)
            continue
        r1 = np.array(xyz[ib]) - np.array(xyz[atom0])
        ang_list.append(ang_vectors(r0,r1))
    ang_list = [np.abs(ang_list[i] - ang_mean) for i in range(len(bond_list))]
    max_diff_id = bond_list[np.argmax(ang_list)]
    return max_diff_id


def AC2BO(AC,atomicNumList,charge,charged_fragments,xyz_coordinates,quick):
    # TODO
    atomic_valence = defaultdict(list)
    atomic_valence[1] = [1]
    atomic_valence[6] = [4]
    atomic_valence[7] = [4,3]
    atomic_valence[8] = [2,1]
    atomic_valence[9] = [1]
    atomic_valence[14] = [4]
    atomic_valence[15] = [5,4,3]
    atomic_valence[16] = [6,4,2]
    atomic_valence[17] = [1]
    atomic_valence[32] = [4]
    atomic_valence[35] = [1]
    atomic_valence[53] = [1]
    atomic_valence_electrons = {}
    atomic_valence_electrons[1] = 1
    atomic_valence_electrons[6] = 4
    atomic_valence_electrons[7] = 5
    atomic_valence_electrons[8] = 6
    atomic_valence_electrons[9] = 7
    atomic_valence_electrons[14] = 4
    atomic_valence_electrons[15] = 5
    atomic_valence_electrons[16] = 6
    atomic_valence_electrons[17] = 7
    atomic_valence_electrons[32] = 4
    atomic_valence_electrons[35] = 7
    atomic_valence_electrons[53] = 7
    # make a list of valences, e.g. for CO: [[4],[2,1]]
    valences_list_of_lists = []
    max_electrons=0
    for atomicNum in atomicNumList:
        max_electrons+=atomic_valence_electrons[atomicNum]
        valences_list_of_lists.append(atomic_valence[atomicNum])
    # convert [[4],[2,1]] to [[4,2],[4,1]]
    temp_valences_list = list(itertools.product(*valences_list_of_lists))
    valences_list=[]
    for i in range(len(temp_valences_list)):
        valence = temp_valences_list[i]
        if (calc_total_electrons(valence,atomicNumList) == max_electrons):
            valences_list.append(valence)
    best_BO = AC.copy()
    # implemenation of algorithm shown in Figure 2
    # UA: unsaturated atoms
    # DU: degree of unsaturation (u matrix in Figure)
    # best_BO: Bcurr in Figure
    AC_valence = list(AC.sum(axis=1))
    list_over_atom=[]
    max_var = np.max(valences_list,axis=0)
    for i in range(len(AC_valence)):
        var = AC_valence[i]
        if (var>max_var[i]):
            list_over_atom.append(i)
    if (not len(list_over_atom)==0):
        for i0 in list_over_atom:
            if (sum(AC[i0])<5):continue
            tmp_list=AC[i0]
            AClist= [i for i , x in enumerate(tmp_list) if x==1]
            # print(AClist)
            max_diff_id=[]
            for ip in AClist:
                max_id= calc_angle_list(i0,ip,AClist,xyz_coordinates)
                max_diff_id.append(max_id)
                # print(ip,max_id)
            i1 = mode(max_diff_id)
            AC[i0,i1]=0
            AC[i1,i0]=0
   
    AC_valence = list(AC.sum(axis=1))
    for valences in valences_list:
        UA,DU_from_AC = getUA(valences, AC_valence)

        if len(UA) == 0 and BO_is_OK(AC,AC,charge,DU_from_AC,atomic_valence_electrons,atomicNumList,charged_fragments):
            return AC,atomic_valence_electrons
        UA_pairs_list = get_UA_pairs(UA,AC,quick)
        for UA_pairs in UA_pairs_list:
            BO = get_BO(AC,UA,DU_from_AC,valences,UA_pairs,quick)
            if BO_is_OK(BO,AC,charge,DU_from_AC,atomic_valence_electrons,atomicNumList,charged_fragments):
                return BO,atomic_valence_electrons
            elif BO.sum() >= best_BO.sum() and valences_not_too_large(BO,valences):
                best_BO = BO.copy()
    return best_BO,atomic_valence_electrons


def AC2mol(mol,AC,atomicNumList,charge,charged_fragments,xyz_coordinates,quick):
    # convert AC matrix to bond order (BO) matrix
    BO,atomic_valence_electrons = AC2BO(AC,atomicNumList,charge,charged_fragments,xyz_coordinates,quick)
    # add BO connectivity and charge info to mol object
    mol = BO2mol(mol,BO, atomicNumList,atomic_valence_electrons,charge,charged_fragments)
    return mol

def get_proto_mol(atomicNumList):
    mol = Chem.MolFromSmarts("[#"+str(atomicNumList[0])+"]")
    rwMol = Chem.RWMol(mol)
    for i in range(1,len(atomicNumList)):
        a = Chem.Atom(atomicNumList[i])
        rwMol.AddAtom(a)
    mol = rwMol.GetMol()
    return mol


def get_atomicNumList(atomic_symbols):
    atomicNumList = []
    for symbol in atomic_symbols:
        atomicNumList.append(get_atom(symbol))
    return atomicNumList

def xyz2AC(atomicNumList,xyz):
    mol = get_proto_mol(atomicNumList)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i,(xyz[i][0],xyz[i][1],xyz[i][2]))
    mol.AddConformer(conf)
    dMat = Chem.Get3DDistanceMatrix(mol)
    Rcovtable=[0.31,0.28,1.28,0.96,0.85,0.76,0.71,0.66,0.57]
    num_atoms = len(atomicNumList)
    AC = np.zeros((num_atoms,num_atoms)).astype(int)
    for i in range(num_atoms):
        a_i = mol.GetAtomWithIdx(i)
        Rcov_i = Rcovtable[a_i.GetAtomicNum()-1]*1.30
        for j in range(i+1,num_atoms):
            a_j = mol.GetAtomWithIdx(j)
            Rcov_j = Rcovtable[a_j.GetAtomicNum()-1]*1.30
            if dMat[i,j] <= Rcov_i + Rcov_j:
                AC[i,j] = 1
                AC[j,i] = 1
    return AC,mol


def read_xyz_file(filename):
    atomic_symbols  = []
    xyz_coordinates = []
    with open(filename, "r") as file:
        for line_number,line in enumerate(file):
            if line_number == 0:
                num_atoms = int(line)
            elif line_number == 1:
                if "charge=" in line:
                    charge = int(line.split("=")[1])
                else:
                    charge = 0
            else:
                atomic_symbol, x, y, z = line.split()
                atomic_symbols.append(atomic_symbol)
                xyz_coordinates.append([float(x),float(y),float(z)])
    atomicNumList = get_atomicNumList(atomic_symbols)
    return atomicNumList,xyz_coordinates,charge

#-----
## https://www.kaggle.com/sunhwan/using-rdkit-for-atomic-feature-and-visualization
def chiral_stereo_check(mol):
    # avoid sanitization error e.g., dsgdb9nsd_037900.xyz
    Chem.SanitizeMol(mol, SanitizeFlags.SANITIZE_ALL - SanitizeFlags.SANITIZE_PROPERTIES)
    Chem.DetectBondStereochemistry(mol,-1)
    # ignore stereochemistry for now
    Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
    Chem.AssignAtomChiralTagsFromStructure(mol,-1)
    return mol

def xyz2mol(atomicNumList, xyz_coordinates, charge,charged_fragments,quick):
    AC,mol  = xyz2AC(atomicNumList,xyz_coordinates)
    new_mol = AC2mol(mol,AC,atomicNumList,charge,charged_fragments,xyz_coordinates,quick)
    new_mol = chiral_stereo_check(new_mol)
    return new_mol


def MolFromXYZ(filename):
    charged_fragments = True
    quick = True
    atomicNumList,xyz_coordinates,charge = read_xyz_file(filename)
    mol = xyz2mol(atomicNumList, xyz_coordinates, charge, charged_fragments, quick)
    return mol

## champs dataset #############################################################
'''
dsgdb9nsd_000001.xyz

5

C -0.0126981359 1.0858041578 0.0080009958
H 0.0021504160 -0.0060313176 0.0019761204
H 1.0117308433 1.4637511618 0.0002765748
H -0.5408150690 1.4475266138 -0.8766437152
H -0.5238136345 1.4379326443 0.9063972942

'''
def read_champs_xyz(xyz_file):
    line = read_list_from_file(xyz_file, comment=None)
    num_atom = int(line[0])
    xyz=[]
    symbol=[]
    for n in range(num_atom):
        l = line[1+n]
        l = l.replace('\t', ' ').replace('  ', ' ')
        l = l.split(' ')
        symbol.append(l[0])
        xyz.append([float(l[1]),float(l[2]),float(l[3]),])
    return symbol, xyz


def mol_from_axyz(symbol, xyz):
    charged_fragments = True
    quick   =  True
    charge  = 0
    atom_no = get_atomicNumList(symbol)
    mol     = xyz2mol(atom_no, xyz, charge, charged_fragments, quick)
    return mol

def load_csv():
    DATA_DIR = '../..'
    #structure
    df_structure = pd.read_csv(DATA_DIR + '/input/structures.csv')
    #coupling
    df_train = pd.read_csv(DATA_DIR + '/input/train.csv')
    df_test  = pd.read_csv(DATA_DIR + '/input/test.csv')
    df_test['scalar_coupling_constant']=0
    df_scalar_coupling = pd.concat([df_train,df_test])
    df_scalar_coupling_contribution = pd.read_csv(DATA_DIR + '/input/scalar_coupling_contributions.csv')
    df_scalar_coupling = pd.merge(df_scalar_coupling, df_scalar_coupling_contribution,
            how='left', on=['molecule_name','atom_index_0','atom_index_1','atom_index_0','type'])
    gb_scalar_coupling = df_scalar_coupling.groupby('molecule_name')
    gb_structure       = df_structure.groupby('molecule_name')
    return gb_structure, gb_scalar_coupling


def do_one(p):
    i, molecule_name, gb_structure, gb_scalar_coupling, graph_file = p
    g = make_graph(molecule_name, gb_structure, gb_scalar_coupling)

##----
def run_convert_to_graph():
    graph_dir = '../../data/graph_v6'
    os.makedirs(graph_dir, exist_ok=True)
    gb_structure, gb_scalar_coupling = load_csv()
    molecule_names = list(gb_scalar_coupling.groups.keys())
    molecule_names = np.sort(molecule_names)
  
    param=[]
    for i, molecule_name in enumerate(molecule_names):
        graph_file = graph_dir + '/%s.pickle'%molecule_name
        p = (i, molecule_name, gb_structure, 
                gb_scalar_coupling, graph_file)
        param.append(p)
    if 1:
        for p in tqdm(param):
            do_one(p)

# main #################################################################
if __name__ == '__main__':
    unique_wl = set()
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_convert_to_graph()#
    print(unique_wl)
    pd.Series(unique_wl).to_csv("unique_wl.csv")
