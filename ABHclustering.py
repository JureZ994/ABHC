
__author__ = 'Jure'
from cluster import Cluster, Point
import operator
from random import choice, sample, randint, shuffle
import math
from collections import defaultdict
from scipy.cluster.hierarchy import dendrogram
from CT import CT
from operator import itemgetter
from cluster import normalized_mutual_info_score
from cluster import adjusted_rand_score
import numpy as np
from itertools import combinations
import sys


class ABHclustering:
    def __init__(self, points, points2,  clusters, attributes=None, candidates = None):
        #Examples
        self.points = points
        self.clusters = clusters
        #point attributes
        self.attributes = attributes
        #silhouette treshold for critical examples
        self.treshold = 1
        #critical examples
        self.candidates = candidates
        #Argument
        self.condition = []
        #constraints
        self.constraints = []
        #critical point
        self.critical_example = []
        self.condition_history=[]
        self.NMI = 0
        self.ARI = 0
        self.Z = np.array([])
        self.vseSilhuete = {}
        self.n_clusters = None
        self.dim = None



    def __repr__(self):
        return str(self.points)

    def avg(self, l):
        return sum(l) / float(len(l))
    def getAi(self, cluster, primer):  #cluster = kluc, primer je pa vrstica v data
        dist = 0
        for example in self.clusters[cluster].points:
            if primer != example:
                dist += self.points_distance(example, primer)
        return dist / (self.clusters[cluster].n - 1)

    def getBi(self, point):
        razdalje = []
        for cluster in self.clusters:
            if point not in self.clusters[cluster].points:
                sum = 0
                for x in self.clusters[cluster].points:
                    sum += self.points_distance(x, point)
                razdalje.append(sum / self.clusters[cluster].n)
        if len(razdalje) != 0:
            dist = min(razdalje)
        else:
            dist = 0
        return dist

    def metodaSilhuet(self):
        Si = []
        for cluster in self.clusters:
            if self.clusters[cluster].n == 1:
                Si.append(0)
            else:
                for point in self.clusters[cluster].points:
                    ai = self.getAi(cluster, point)
                    bi = self.getBi(point)
                    if ai == 0 and bi == 0:
                        Si.append(0)
                    else:
                        Si.append((bi - ai) / max(ai, bi))
        sum = 0
        for el in Si:
            sum += el
        return sum/len(Si)
    def hierarhicalClustering(self, clusters = None):
        """
        Main hierarhical clustering loop
        """

        self.l.log("Finding clusters...")
        n = len(self.points)  #na začetku je vsak primer svoj cluster
        idZ = 0
        m = len(self.points)
        while(n != 1):
            dist, pair = self.closest_clusters()
            par = list()
            for el in pair:
                par.append(el)

            tocke = []
            tocke.append(self.clusters[par[0]].points)
            tocke.append(self.clusters[par[1]].points)

            #print("tocke: ", len(tocke))
            novCluster = Cluster(m+idZ)
            novCluster.update(par[0], par[1], dist, tocke)
            self.clusters.pop(par[0])
            self.clusters.pop(par[1])
            self.clusters.update({(m+idZ): novCluster})

            if idZ == 0:
                self.Z = [par[0], par[1], dist, novCluster.n]
            else:
                newrow = [par[0], par[1], dist, novCluster.n]
                self.Z = np.vstack([self.Z, newrow])

            n = len(self.clusters)
            #self.vseSilhuete.update({idZ: self.metodaSilhuet()})
            print("par: ", par, ", dist: ", round(dist,2))
            idZ+=1
        print(len(self.Z))

        #vrnil naj bi matriko Z, in rezultate metod, ki nam povejo koliko clustrov je
        #print("Optimalno stevilo clustrov po metodi silhuet: ", len(self.points)-1-max(self.vseSilhuete.items(), key=operator.itemgetter(1))[0])
        return self.clusters
    def rebuildClusters(self, clusters, steviloClustrov):
        self.l.log("Building clusters... ")
        m = len(self.points)
        for id in range(0, m - steviloClustrov):
            idc1 = self.Z[id][0]
            idc2 = self.Z[id][1]
            tocke = []
            tocke.append(clusters[idc1].points)
            tocke.append(clusters[idc2].points)
            novCluster = Cluster(len(self.points)+id)
            novCluster.update(idc1, idc2, self.Z[id][2], tocke)
            clusters.pop(idc1)
            clusters.pop(idc2)
            clusters.update({m+id: novCluster})
        new_keys = []
        for i,cluster in enumerate(clusters):
            points = []
            clusters[cluster].name = "Cluster"+str(i)
            clusters[cluster].clusterId = i
            clusters[cluster].dim = self.dim
            for point in clusters[cluster].points:
                points.append(point)
            clusters[cluster].points = points
            clusters[cluster].centroid = clusters[cluster].calculateCentroid()
            new_keys.append(i)

        for key, n_key in zip(clusters.keys(), new_keys):
            clusters[n_key] = clusters.pop(key)
        return clusters

    def cluster_distance(self, c1, c2):
        """
        Compute distance between two clusters.
        """
        z = [(a,b) for a in self.clusters[c1].points for b in self.clusters[c2].points]   #hrani vse kombinacije c1 z c2
        """
        izračunaj min, max, average razdaljo med posameznimi elementi c1 z c2
        v mojem primeru racunam average linkage
        """
        dist = self.average_linkage(z)
        return dist

    def closest_clusters(self, clusters_checked=None):
        """
        Find a pair of closest clusters and returns the pair of clusters and their distance.
        """
        if clusters_checked is None:
            dis, pair = min((self.cluster_distance(c1, c2), (c1, c2)) for c1, c2 in combinations(self.clusters, 2))
            return dis, pair
        else:
            dis = sys.maxsize
            pair = None
            for c1, c2 in combinations(self.clusters,2):
                dist = self.cluster_distance(c1,c2)
                if dist < dis  and [c1,c2] not in clusters_checked:
                    dis = dist
                    pair = (c1,c2)
            return dis, pair



    def average_linkage(self, x):
        skupna_razdalja = 0
        for a, b in x:
            skupna_razdalja += self.points_distance(a, b)
        return skupna_razdalja/len(x)
    def points_distance(self, p1, p2):
        """
        Euclidean distance between two points.
        """
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1.coords, p2.coords)))
    def getClusterId(self, point):
        for cluster in self.clusters:
            if point in self.clusters[cluster].points:
                return cluster

    def check_cannot_link(self, constraints, points1, points2):
        for x in constraints:
            if "cannot-link" in x:
                if x['cannot-link'][0] in points1 and x['point'][0] in points2:
                    return True
                if x['cannot-link'][0] in points2 and x['point'][0] in points1:
                    return True
        return False
    def check_must_link(self, constraints, points):    #vrne par?
        for x in constraints:
            if "must-link" in x:
                if x['point'][0] in points:
                    if x['must-link'][0] not in points:
                        return [self.getClusterId(x['point'][0]), self.getClusterId(x['must-link'][0])]
                if x['must-link'][0] in points:  #tocka ima ML
                    if x['point'][0] not in points:  #in nista v istem clustru
                        return [self.getClusterId(x['must-link'][0]), self.getClusterId(x['point'][0])]
        return -1


    def ABHclustering(self, constraints,final_n_of_clusters, clusters=None):
        """
        Main hierarhical clustering loop
        """
        self.l.log("Finding clusters...")
        print("omejitve: ", constraints)
        print("clustri: ", clusters)
        self.clusters = clusters

        n = len(self.points)  #na začetku je vsak primer svoj cluster
        idZ = 0
        m = n
        stop_clustering = False
        while(n != final_n_of_clusters):
            condition = True
            clusters_checked = []
            while condition:
                if len(clusters_checked) == len(self.clusters):
                    print("Ni mozno nadaljne zruzevanje, ostalo je ",len(self.clusters)," clustrov.")
                    break
                dist, pair = self.closest_clusters(clusters_checked)
                if(pair is None):
                    stop_clustering = True
                    break
                par = list()
                for el in pair:
                    par.append(el)
                #ali ima katerakoli tocka iz obeh clustrov ML, jo zdruzi in ponovno poisci najblizja clustra
                ML_pair = self.check_must_link(constraints, self.clusters[par[0]].points)
                if ML_pair == -1:
                    ML_pair = self.check_must_link(constraints, self.clusters[par[1]].points)
                if ML_pair != -1:
                    par[0] = ML_pair[0]
                    par[1] = ML_pair[1]
                condition = self.check_cannot_link(constraints, self.clusters[par[0]].points, self.clusters[par[1]].points)
                if condition:
                    clusters_checked.append([par[0], par[1]])
                dist = self.cluster_distance(par[0], par[1])
            if stop_clustering:
                break
            print("par: ", par, ", dist: ", round(dist,2), " ", len(self.clusters))

            tocke = []
            tocke.append(self.clusters[par[0]].points)
            tocke.append(self.clusters[par[1]].points)

            #print("tocke: ", tocke)
            novCluster = Cluster(m+idZ)
            novCluster.update(par[0], par[1], dist, tocke)
            self.clusters.pop(par[0])
            self.clusters.pop(par[1])
            self.clusters.update({m+idZ: novCluster})

            if idZ == 0:
                self.Z = [par[0], par[1], dist, novCluster.n]
            else:
                newrow = [par[0], par[1], dist, novCluster.n]
                self.Z = np.vstack([self.Z, newrow])

            n = len(self.clusters)
            idZ+=1

        return self.clusters

    def silhouette(self):
        """
        Calculating silhouette index for each point and adding it to its attribute
        """
        for cid, cluster in enumerate(self.clusters):
            for point in self.clusters[cluster].points:
                ai = self.getAi(cluster, point)
                bi = self.getBi(point)
                s = float(bi - ai)/ max(ai,bi) if max(ai,bi) > 0 else 0.0
                point.silhuette = s


    def get_candidates(self):
        """
        Finding critical examples by silhouette index

        If the silhouette index is below the limit, then it is considered a critical example
        """
        self.l.log("Finding shilhouette values...")
        self.silhouette()
        list = []
        #Sort each cluster by silhuette values and display lowest value
        self.l.log("Finding critical examples...")
        for i,c in enumerate(self.clusters):
            self.clusters[c].points.sort(key=lambda x: x.silhuette, reverse=True)
            for p in reversed(self.clusters[c].points):
                if p not in [ y['point'][0]  for x in self.condition_history  for y in x]:
                    list.append((p, self.clusters[c].clusterId))
        self.l.log("ordering examples...")

        list.sort(key=lambda x: x[0].silhuette, reverse=False)
        self.candidates = list
        #c.printCases(dataOriginal,i)
    def get_pair(self, critical_point_target):
        cluster = None
        cli=None
        while cluster == None or cluster == critical_point_target:
            cli=randint(0,len(self.clusters)-1)
            cluster = self.clusters[cli]
        return (cluster.points[0], cli)

    def counter_example(self, condition):
        self.l.log("Finding counter examples...")
        #1 counter example per cluster
        critical = condition['point']
        counter_example_array = []

        for cli in range(0,len(self.clusters)):
            #If target cluster, we want counter examples that do not conform with the argument
            cl_counters = 0
            for point in self.clusters[cli].points:
                if  cl_counters >= 1:
                    break
                if point == self.critical_example[-1][0] or cli == self.critical_example[-1][1]:    #tocka je kriticni
                                                                  #ali tocka je v istem clustru
                    continue
                test = False
                for arg in condition["arguments"]:
                    co = self.attributes.index(arg[0])               #index atributa
                    operator = arg[1]                                #operator
                    val = arg[2]                                       #vrednost
                    #IF ARG IS HIGH/LOW SET TO NUMBER WITH 10% +/-
                    if arg[1] == '>>>':
                        operator ='>'
                        val = critical[0].coords[co]*0.9
                        if critical[0].coords[co] == 0:
                            val = -0.2
                    elif arg[1] == '<<<':
                        operator = '<'
                        val = critical[0].coords[co]*1.1
                        if critical[0].coords[co] == 0:
                            val = 0.2
                    op = self.get_operator(operator)
                    val = int(val)
                    if op(point.coords[co], val):
                        test = True
                if(test):
                    self.l.log("COUNTER EXAMPLE")
                    self.l.log("Counter example is in cluster: "+str(self.clusters[cli].name)+" CLUSTER INDEX: "+str(cli))
                    cl_counters+=1
                    counter_example_array.append((point,cli))
                    #break
        return counter_example_array


    def argument_steps(self):
        critical_point = self.critical_example[-1]

        #Informative print of clusters - reference points
        #We present clusters to expert for better understanding
        for cluster in self.clusters:
            self.l.log("Cluster "+str(cluster.name)+": "+str(cluster)+" distance: "+str(critical_point.getDistance(cluster.centroid)))

        #We present the example to the expert
        self.l.log("Critical Example is in cluster: "+str(self.critical_example[1]))#+" with centroid: "+str(clusters[c[1]])
        self.condition.append(self.get_argument())
        self.counter_example()




    def parse_rule(self, rule, critical_point, current_cluster):
        rule_dic = {}
        if_then = rule.split("then")
        s = if_then[0].split("and")
        #point target
        rule_dic["point"] = self.critical_example
        #condition[4] = int(critical_point.reference)
        #current cluster
        rule_dic["current_cluster"] = int(current_cluster)
        #condition[5] = int(current_cluster)
        l = []
        atr = map(str.lower, self.attributes)
        for cond in s:
            conditionSplit = cond.split(" ")
            condition = [0]*3
            for c in conditionSplit:
                if c in atr:
                    attribute = c
                    #atribute index
                    atrIndex = atr.index(attribute)
                    condition[0] = atrIndex
                elif c in self.def_operators():
                    operatorString = c
                    #Operator
                    condition[1] = operatorString
                else:
                    try:
                        #target value
                        value = float(c)
                        condition[2] = value
                    except:
                        pass
            if condition[1] == '>>>' or condition[1] == '<<<':
                condition[2] = ''
            l.append(condition)
        rule_dic["arguments"]=l
        t_cluster = if_then[1].strip()
        target_cluster = None
        for i,c in enumerate(self.clusters):
            if c.name.lower() == t_cluster:
                target_cluster = i
        try:
            rule_dic["target_cluster"] = int(target_cluster)
        except:
            self.l.log("Wrong target cluster.")
            return 0

        if condition[1] not in self.def_operators():
            self.l.log("Wrong operator.")
            return 0
            #We have our condition. find counter examples
        return rule_dic

    def purity(self):
        helper_points=[]
        helper_points_real=[]
        cluster_names=[]

        for i,cluster in enumerate(self.clusters):
            d = defaultdict(lambda: 0)
            for p in self.clusters[cluster].points:
                helper_points.append(i)
                if p.cheat not in cluster_names:
                    cluster_names.append(p.cheat)
                helper_points_real.append( cluster_names.index(p.cheat))

                d[p.cheat] +=1

            self.clusters[cluster].purity = d
        self.NMI = normalized_mutual_info_score(helper_points, helper_points_real)
        self.ARI = adjusted_rand_score(helper_points, helper_points_real)



    def def_operators(self):
        return {
            '>': operator.gt,
            '<': operator.lt,
            '<=': operator.le,
            '>=': operator.ge,
            '==': operator.eq,
            '!=': operator.ne,
            '>>>': '>>>',
            '<<<': '<<<'
            }
    def def_ops(self):
        return {
        '*': operator.mul,
        '+': operator.add,
        '/': operator.truediv,
        }
    def get_operator(self, op):
        return self.def_operators()[op]
    def get_ops(self, op):
        return self.def_ops()[op]





