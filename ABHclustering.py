import functools

__author__ = 'Jure'
from cluster import Cluster, Point
import operator
from random import choice, sample, randint, shuffle
import math
from collections import defaultdict
from scipy.cluster.hierarchy import dendrogram
from CT import CT
from operator import itemgetter
from collections import Counter
import collections
from cluster import normalized_mutual_info_score
from cluster import adjusted_rand_score
import numpy as np
from itertools import combinations
import sys
from scipy import spatial
import pandas as pd
from scipy.spatial import distance_matrix

class ABHclustering:
    def __init__(self, points, points2,  clusters, attributes=None, candidates = None):
        #Examples
        self.points = points
        self.clusters = clusters
        #point attributes
        self.attributes = attributes
        #silhouette treshold for critical examples
        self.treshold = 1
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
        self.distance_type = None
        self.distances={}
        self.linkage = None
        self.prev_dict = {}
        self.diff = []



    def __repr__(self):
        return str(self.points)

    def avg(self, l):
        return sum(l) / float(len(l))
    def getAi(self, cluster, primer):  #cluster = kluc, primer je pa vrstica v data
        dist = 0
        for example in self.clusters[cluster].points:
            if primer != example:
                dist += self.points_distance(example, primer)
        if len(self.clusters[cluster].points) - 1 == 0:
            return 0
        else:
            return dist / (len(self.clusters[cluster].points) - 1)

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
    def izbrisi_razdalje(self, kljuc):
        #print("<<<<<<<<<")
        for cluster in self.clusters:
            if cluster != kljuc:
                key3=str(cluster)+ " "+ str(kljuc)
                key4=str(kljuc)+ " "+str(cluster)
                z = [(a.reference,b.reference) for a in self.clusters[cluster].points for b in self.clusters[kljuc].points]
                for l in z:
                    key1=str(l[0])+" "+str(l[1])
                    key2=str(l[1])+" "+str(l[0])
                    #print(key1)
                    self.distances.pop(key1, None)
                    self.distances.pop(key2, None)
                self.distances.pop(key3, None)
                self.distances.pop(key4, None)
        #print("<<<<<<<<<")
    def dodaj_razdalje(self, kljuc):
        #print("<>>>>>>>>>>>>>>>")
        for cluster in self.clusters:
            if cluster != kljuc:
                #print(cluster, kljuc)
                #print(type(cluster), type(kljuc))

                key = str(cluster)+ " " + str(kljuc)
                if self.linkage == "Average":
                    z = [(a, b) for a in self.clusters[cluster].points for b in self.clusters[kljuc].points]  # hrani vse kombinacije c1 z c2
                    dist = self.average_linkage(z)
                elif self.linkage == "Ward":
                    c=[]
                    u=[]
                    v=[]
                    for p in self.clusters[cluster].points:
                        c.append(p.coords)
                        u.append(p.coords)
                    for r in self.clusters[kljuc].points:
                        c.append(r.coords)
                        v.append(r.coords)
                    centroid_UV = np.average(c, axis=0)
                    centroid_U = np.average(u, axis=0)
                    centroid_V = np.average(v, axis=0)
                    dist1 = 0
                    dist2 = 0
                    dist3 = 0
                    for point in c:
                        if self.distance_type == "Cosine":
                            dist1 += spatial.distance.cosine(centroid_UV, point)
                        elif self.distance_type == "Euclidian":
                            dist1 += spatial.distance.euclidean(centroid_UV, point) ** 2
                    for point in u:
                        if self.distance_type == "Cosine":
                            dist2 += spatial.distance.cosine(centroid_U, point)
                        elif self.distance_type == "Euclidian":
                            dist2 += spatial.distance.euclidean(centroid_U, point) ** 2
                    for point in v:
                        if self.distance_type == "Cosine":
                            dist3 += spatial.distance.cosine(centroid_V, point)
                        elif self.distance_type == "Euclidian":
                            dist3 += spatial.distance.euclidean(centroid_V, point) ** 2
                    dist = dist1-dist2-dist3
                else:
                    print("Napaka mere razdalj, uporabi WARD ali AVERAGE")
                    exit(1)
                #print(key)
                self.distances.update({key: dist})
        #print(">>>>>>>>>>>>>>>>>>")


    def hierarhicalClustering(self, clusters = None):
        """
        Main hierarhical clustering loop
        """
        distanca = 0
        self.l.log("Building distance matrix...")
        n = len(self.points)  #na začetku je vsak primer svoj cluster
        data = []
        for c in self.clusters:
            p = [point.coords for point in self.clusters[c].points]
            data.append(p[0])
        df = pd.DataFrame(data,columns= np.array([a for a in self.attributes]))
        n_df = (df.values)
        self.d_matrix = np.zeros(((df.values).shape[0],(df.values).shape[0]))
        for i in range((df.values).shape[0]):
            for j in range((df.values).shape[0]):
                kljuc1 = str(i)+' '+str(j)
                kljuc2 = str(j)+' '+str(i)
                if i != j:
                    if kljuc1 in self.distances:
                        continue
                    elif kljuc2 in self.distances:
                        continue
                    else:
                        if self.linkage == "Ward":
                            l = []
                            l.append(n_df[i])
                            l.append(n_df[j])
                            centroid = np.average(l,axis=0)
                            dist = 0
                            if self.distance_type == "Cosine":
                                dist += spatial.distance.cosine(centroid, n_df[i])**2
                                dist += spatial.distance.cosine(centroid, n_df[j])**2
                            elif self.distance_type == "Euclidean":
                                dist += spatial.distance.euclidean(centroid, n_df[i])**2
                                dist += spatial.distance.euclidean(centroid, n_df[i])**2
                            self.distances.update({kljuc1: dist})
                        elif self.linkage == "Average":
                            if self.distance_type == "Cosine":
                                dist = spatial.distance.cosine(n_df[i], n_df[j])
                            elif self.distance_type == "Euclidean":
                                dist = spatial.distance.euclidean(n_df[i], n_df[j])
                            self.distances.update({kljuc1: dist})
                        else:
                            print("Error creating distance matrix...")
                            exit(1)

        idZ = 0
        m = len(self.points)
        self.l.log("Finding clusters...")
        while n > 1:
            """
            dist, pair = self.closest_clusters()
            par = list()
            for el in pair:
                par.append(el)
           
            dist = np.amin(self.d_matrix)
            result = np.where(self.d_matrix == dist)

            par = list()
            for el in result[0]:
                par.append(el)
            print("--",par)
            """
            key = min(self.distances, key=self.distances.get)
            par = key.split(' ')
            par = [int(i) for i in par]
            dist = self.distances[key]
            #print("--------------------")
            #print(par[0], par[1])
            self.distances.pop(key, None)
            self.izbrisi_razdalje(par[0])
            self.izbrisi_razdalje(par[1])
            #print("5 238" in self.distances)
            tocke = []
            tocke.append(self.clusters[par[0]].points)
            tocke.append(self.clusters[par[1]].points)

            #print("tocke: ", len(tocke))
            novCluster = Cluster(m+idZ)
            novCluster.update(par[0], par[1], dist, tocke)
            novCluster.centroid = novCluster.calculateCentroid()
            self.clusters.pop(par[0])
            self.clusters.pop(par[1])
            self.clusters.update({(m+idZ): novCluster})
            #print("dodajam razdalje...")
            self.dodaj_razdalje(m+idZ)
            """
            novCluster = Cluster(par[0])
            novCluster.update(par[0], par[1], dist, tocke)
            novCluster.centroid = novCluster.calculateCentroid()
            self.clusters.pop(par[0])
            self.clusters.pop(par[1])
            self.clusters.update({(par[0]): novCluster})
            #TODO: preracunaj razdalje v matriki razdalj
            """
            if idZ == 0:
                self.Z = [par[0], par[1], dist, novCluster.n]
            else:
                newrow = [par[0], par[1], dist, novCluster.n]
                self.Z = np.vstack([self.Z, newrow])

            n = len(self.clusters)
            #self.vseSilhuete.update({idZ: self.metodaSilhuet()})
            print("par: ", par, ", dist: ", '%.08f' % dist)
            #print(idZ, n, m+idZ)
            idZ += 1

        self.l.log("Dendrogram created...")

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
    def calculateCentroid(self, c1, c2):
        allPoints = []
        for point in self.clusters[c1].points:
            allPoints.append(point.coords)
        for point in self.clusters[c2].points:
            allPoints.append(point.coords)
        reduce_coord = lambda i:functools.reduce(lambda x,p : x + p[i],allPoints,0.0)
        centroid_coords = [reduce_coord(i)/len(allPoints) for i in range(len(allPoints[0]))]
        centroid_coords = [round(elem, 2) for elem in centroid_coords]
        return Point(centroid_coords, None, str(len(self.Z) + len(self.points)))
    def distance(self, point, centroid):
        if self.distance_type == "Euclidean":
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(point.coords, centroid.coords)))
        elif self.distance_type == "Cosine":
            return spatial.distance.cosine(point.coords, centroid.coords)
    def cluster_distance(self, c1, c2):
        """
        Compute distance between two clusters.
        """
        if self.linkage == "Average":
            z = [(a,b) for a in self.clusters[c1].points for b in self.clusters[c2].points]   #hrani vse kombinacije c1 z c2
            dist = self.average_linkage(z)
            return dist
        elif self.linkage == "Ward":
            print("distanca1")
            skupniCentorid = self.calculateCentroid(c1, c2)
            print("distanca2")
            sum1 = 0
            sum2 = 0
            sumBoth = 0
            for point in self.clusters[c1].points:
                sum1 += self.distance(point, self.clusters[c1].centroid)
                sumBoth += self.distance(point, skupniCentorid)
            for point in self.clusters[c2].points:
                sum2 += self.distance(point, self.clusters[c2].centroid)
                sumBoth += self.distance(point, skupniCentorid)
            print("distanca3")
            #return (sumBoth - sum1 - sum2)

            return sum1 + sum2 - sumBoth
        else:
            self.log("Potrebno je nastaviti tip povezav, moznosti so: AVERAGE , WARD")
            exit()

    def closest_clusters(self, clusters_checked=None):
        """
        Find a pair of closest clusters and returns the pair of clusters and their distance.
        """
        print("iscem clustre...")
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
        Returns Euclidean distance or cosinus similarity between two points.
        """
        if self.distance_type == "Euclidean":
            dist = spatial.distance.euclidean(p1.coords, p2.coords)
        elif self.distance_type == "Cosine":
            dist = spatial.distance.cosine(p1.coords, p2.coords)
        """
        kljuc1 = str(p1.reference)
        kljuc1+=' '
        kljuc1 += str(p2.reference)

        kljuc2 = str(p2.reference)
        kljuc2+=' '
        kljuc2+= str(p1.reference)
        
        if self.distance_type == "EUCLIDIAN":
            if kljuc1 in self.distances:
                return self.distances[kljuc1]
            elif kljuc2 in self.distances:
                return self.distances[kljuc2]
            else:
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(p1.coords, p2.coords)))
                self.distances.update({kljuc1: dist})
            return self.distances[kljuc1]
        elif self.distance_type == "COSINUS":
            if kljuc1 in self.distances:
                return self.distances[kljuc1]
            elif kljuc2 in self.distances:
                return self.distances[kljuc2]
            else:
                dist = spatial.distance.cosine(p1.coords, p2.coords)
                self.distances.update({kljuc1: dist})
            return self.distances[kljuc1]
        """
        return dist
    def getClusterID(self, point, clusters):
        for cluster in clusters:
            if point in clusters[cluster].points:
                return cluster
        return -1
    def getClusterId(self, point):
        for cluster in self.clusters:
            if point in self.clusters[cluster].points:
                return cluster

    def check_cannot_link(self, constraints, points1, points2):
        for x in constraints:
            if "cannot-link" in x:
                if x['cannot-link'][0] in points1 and x['point'][0] in points2:
                    print("#### ", x['cannot-link'][0], x['point'][0] in points2, x)
                    return True
                if x['cannot-link'][0] in points2 and x['point'][0] in points1:
                    print("#### ", x['cannot-link'][0], x['point'][0], x)
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
    def sort_constraints(self):
        target=[]
        no_target=[]
        for c in self.constraints:
            if 'target_cluster' in c:
                target.append(c)
            else:
                no_target.append(c)
        cannot_first = []
        last=[]
        for c in target:
            if 'cannot-link' in c:
                cannot_first.append(c)
            else:
                last.append(c)
        return cannot_first+last+no_target

    def ABHclustering(self, constraints,final_n_of_clusters, clusters=None):
        """
        Main hierarhical clustering loop
        """
        self.l.log("Creating transitive ML closure...")
        stevec = len(clusters)

        for x in constraints:
            if 'must-link' in x:
                #print("omejitev: ", x)
                kluc1 = self.getClusterID(x['point'][0], clusters)
                kluc2 = self.getClusterID(x['must-link'][0], clusters)
                #print(kluc1, " | " , kluc2," | ", stevec, kluc1 == kluc2)
                if kluc1 != kluc2:
                    tocke = []
                    tocke.append(clusters[kluc1].points)
                    tocke.append(clusters[kluc2].points)
                    clusters.pop(kluc1)
                    clusters.pop(kluc2)
                    nov = Cluster(stevec)
                    nov.update(kluc1, kluc2, 0, tocke)  #TLE DEJ NOT
                    clusters.update({stevec: nov})
                    stevec += 1
        m = stevec
        self.l.log("Creating distance matrix....")
        self.distances = {}
        self.clusters = clusters
        stevec = 0
        for c in self.clusters:
            print(self.clusters[c].points, self.clusters[c].clusterId)
            for p in self.clusters[c].points:
                stevec+=1

        print(len(self.clusters), stevec)

        z = [(clusters[a].clusterId, clusters[b].clusterId) for a in self.clusters for b in self.clusters]
        for l in z:
            kljuc1 = str(l[0])+" "+str(l[1])
            kljuc2 = str(l[1])+" "+str(l[0])
            if l[0] != l[1]:
                if kljuc1 in self.distances:
                    continue
                elif kljuc2 in self.distances:
                    continue
                else:
                    if self.linkage == "Ward":
                        c = []
                        u = []
                        v = []

                        for p in self.clusters[l[0]].points:
                            c.append(p.coords)
                            u.append(p.coords)
                        for r in self.clusters[l[1]].points:
                            c.append(r.coords)
                            v.append(r.coords)
                        centroid_uv = np.average(c, axis=0)
                        centroid_u = np.average(u, axis=0)
                        centroid_v = np.average(v, axis=0)
                        dist1 = 0
                        dist2 = 0
                        dist3 = 0
                        for point in c:
                            if self.distance_type == "Cosine":
                                dist1 += spatial.distance.cosine(centroid_uv, point) ** 2
                            elif self.distance_type == "Euclidean":
                                dist1 += spatial.distance.euclidean(centroid_uv, point) ** 2
                        for point in u:
                            if self.distance_type == "Cosine":
                                dist2 += spatial.distance.cosine(centroid_u, point) ** 2
                            elif self.distance_type == "Euclidean":
                                dist2 += spatial.distance.euclidean(centroid_u, point) ** 2
                        for point in v:
                            if self.distance_type == "Cosine":
                                dist3 += spatial.distance.cosine(centroid_v, point) ** 2
                            elif self.distance_type == "Euclidean":
                                dist3 += spatial.distance.euclidean(centroid_v, point) ** 2
                        dist = dist1 - dist2 - dist3
                        self.distances.update({kljuc1: dist})
                    elif self.linkage == "Average":
                        u = [(a, b) for a in self.clusters[l[0]].points for b in
                             self.clusters[l[1]].points]
                        dist = self.average_linkage(u)
                        self.distances.update({kljuc1: dist})
                    else:
                        print("Error creating distance matrix...")
                        exit(1)


        self.l.log("Finding clusters...")
        '''
        s = sorted(self.distances.items(), key=lambda x: x[1])
        for k, v in s:
            print(k, v)
        '''
        print("st. omejitev: ", len(constraints))
        #print("clustri: ", self.clusters.keys())

        self.Z = np.array([])

        #n = len(self.points)  #na začetku je vsak primer svoj cluster
        n = len(self.clusters)
        idZ = 0
        stop_clustering = False
        while (n != final_n_of_clusters):
            #print("### ",n," ###")
            condition = True
            #clusters_checked = []
            while condition:
                """
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
                self.constraints = self.sort_constraints()
              
                #ali ima katerakoli tocka iz obeh clustrov ML, jo zdruzi in ponovno poisci najblizja clustra
                #ML_pair = self.check_must_link(constraints, self.clusters[par[0]].points)
                
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
            #print("par: ", par, ", dist: ", round(dist,2), " ", len(self.clusters))
                """
                key = min(self.distances, key=self.distances.get)
                kljuc = key
                par = key.split(' ')
                par = [int(i) for i in par]
                dist = self.distances[kljuc]
                #print("   ->",key, " ", self.check_cannot_link(constraints, self.clusters[par[0]].points, self.clusters[par[1]].points))
                if self.check_cannot_link(constraints, self.clusters[par[0]].points, self.clusters[par[1]].points):
                    self.distances[kljuc] = sys.maxsize
                    if dist == sys.maxsize:
                        self.l.log("ABHC cannot find clusters under those constraints...")
                        return self.clusters
                    print("   Cannot link:", par)
                else:
                    break
                # print("--------------------")
                # print(par[0], par[1])
            #print(self.distances.keys())
            self.distances.pop(kljuc, None)
            self.izbrisi_razdalje(par[0])
            self.izbrisi_razdalje(par[1])
            #print(self.distances.keys())
            tocke = []
            tocke.append(self.clusters[par[0]].points)
            tocke.append(self.clusters[par[1]].points)
            #print("tocke: ", len(tocke))
            novCluster = Cluster(m+idZ)
            novCluster.update(par[0], par[1], dist, tocke)
            novCluster.centroid = novCluster.calculateCentroid()
            self.clusters.pop(par[0])
            self.clusters.pop(par[1])
            self.clusters.update({(m+idZ): novCluster})
            #print("clustri:")
            #print(self.clusters.keys())
            #print("dodajam razdalje...")

            #print("NOV:" ,m+idZ)
            self.dodaj_razdalje(m+idZ)

            print("par: ", par, "dist: ",  '%.08f' % dist)
            if idZ == 0:
                self.Z = [par[0], par[1], dist, novCluster.n]
            else:
                newrow = [par[0], par[1], dist, novCluster.n]
                self.Z = np.vstack([self.Z, newrow])

            n = len(self.clusters)
            idZ += 1
        #zapomni si primere, kateri so v drugi skupini kot v prejšni iteraciji.
        self.diff = []
        clusters_checked = set()
        hm = 0

        for cluster in self.clusters:
            val = -1
            for point in self.clusters[cluster].points:
                hm+=1
                if val < 0:
                    val = self.prev_dict[point.reference]
                    if val in clusters_checked:
                        self.diff.append(point.reference)
                else:
                    if val != self.prev_dict[point.reference]:
                        self.diff.append(point.reference)
            clusters_checked.add(val)
        self.prev_dict = self.make_dict()
        print(len(self.diff))
        print(sorted(self.diff))
        print("stevilo primerov: ", hm)
        return self.clusters
    def make_dict(self):
        i = 0
        dict = {}
        for c in self.clusters:
            for p in self.clusters[c].points:
                dict.update({p.reference: i})
            i += 1
        return dict
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
        while cluster is None:
            cli = randint(0,len(self.clusters)-1)
            if cli == critical_point_target:
                continue
            cluster = self.clusters[cli]
            point=None
            val = -2
            for p in cluster.points:
                if p.silhuette > val:
                    val = p.silhuette
                    point = p
            break
        return point, cli

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
        '/': operator.truediv,
        '*': operator.mul,
        '+': operator.add,
        '-': operator.sub
        }
    def get_operator(self, op):
        return self.def_operators()[op]
    def get_ops(self, op):
        return self.def_ops()[op]

    def CAclustering(self, constraints, final_n_of_clusters, clusters=None):

        """
        Main hierarhical clustering loop
        """
        self.l.log("Creating transitive ML closure...")
        stevec = len(clusters)

        for c in clusters:
            print(clusters[c].clusterId, clusters[c].points)

        for x in constraints:
            if 'must-link' in x:
                print("omejitev: ", x)
                kluc1 = self.getClusterID(x['point'][0], clusters)
                kluc2 = self.getClusterID(x['must-link'][0], clusters)
                print(kluc1, " | ", kluc2, " | ", stevec, kluc1 == kluc2)
                if kluc1 != kluc2:
                    tocke = []
                    tocke.append(clusters[kluc1].points)
                    tocke.append(clusters[kluc2].points)
                    clusters.pop(kluc1)
                    clusters.pop(kluc2)
                    nov = Cluster(stevec)
                    nov.update(kluc1, kluc2, 0, tocke)  # TLE DEJ NOT
                    clusters.update({stevec: nov})
                    stevec += 1
        m = stevec
        self.l.log("Creating distance matrix....")
        self.distances = {}
        self.clusters = clusters
        stevec = 0
        for c in self.clusters:
            print(self.clusters[c].points, self.clusters[c].clusterId)
            for p in self.clusters[c].points:
                stevec += 1

        print(len(self.clusters), stevec)

        z = [(clusters[a].clusterId, clusters[b].clusterId) for a in self.clusters for b in self.clusters]
        for l in z:
            kljuc1 = str(l[0]) + " " + str(l[1])
            kljuc2 = str(l[1]) + " " + str(l[0])
            if l[0] != l[1]:
                if kljuc1 in self.distances:
                    continue
                elif kljuc2 in self.distances:
                    continue
                else:
                    if self.linkage == "Ward":
                        c = []
                        u = []
                        v = []

                        for p in self.clusters[l[0]].points:
                            c.append(p.coords)
                            u.append(p.coords)
                        for r in self.clusters[l[1]].points:
                            c.append(r.coords)
                            v.append(r.coords)
                        centroid_uv = np.average(c, axis=0)
                        centroid_u = np.average(u, axis=0)
                        centroid_v = np.average(v, axis=0)
                        dist1 = 0
                        dist2 = 0
                        dist3 = 0
                        for point in c:
                            if self.distance_type == "Cosine":
                                dist1 += spatial.distance.cosine(centroid_uv, point) ** 2
                            elif self.distance_type == "Euclidean":
                                dist1 += spatial.distance.euclidean(centroid_uv, point) ** 2
                        for point in u:
                            if self.distance_type == "Cosine":
                                dist2 += spatial.distance.cosine(centroid_u, point) ** 2
                            elif self.distance_type == "Euclidean":
                                dist2 += spatial.distance.euclidean(centroid_u, point) ** 2
                        for point in v:
                            if self.distance_type == "Cosine":
                                dist3 += spatial.distance.cosine(centroid_v, point) ** 2
                            elif self.distance_type == "Euclidean":
                                dist3 += spatial.distance.euclidean(centroid_v, point) ** 2
                        dist = dist1 - dist2 - dist3
                        self.distances.update({kljuc1: dist})
                    elif self.linkage == "Average":
                        u = [(a, b) for a in self.clusters[l[0]].points for b in
                             self.clusters[l[1]].points]
                        dist = self.average_linkage(u)
                        self.distances.update({kljuc1: dist})
                    else:
                        print("Error creating distance matrix...")
                        exit(1)

        self.l.log("Finding clusters...")
        '''
        s = sorted(self.distances.items(), key=lambda x: x[1])
        for k, v in s:
            print(k, v)
        '''
        print("st. omejitev: ", len(constraints))
        # print("clustri: ", self.clusters.keys())

        self.Z = np.array([])

        # n = len(self.points)  #na začetku je vsak primer svoj cluster
        n = len(self.clusters)
        idZ = 0
        stop_clustering = False
        while (n != final_n_of_clusters):
            # print("### ",n," ###")
            condition = True
            # clusters_checked = []
            while condition:

                key = min(self.distances, key=self.distances.get)
                kljuc = key
                par = key.split(' ')
                par = [int(i) for i in par]
                dist = self.distances[kljuc]
                # print("   ->",key, " ", self.check_cannot_link(constraints, self.clusters[par[0]].points, self.clusters[par[1]].points))
                if self.check_cannot_link(constraints, self.clusters[par[0]].points, self.clusters[par[1]].points):
                    self.distances[kljuc] = sys.maxsize
                    if dist == sys.maxsize:
                        self.l.log("ABHC cannot find clusters under those constraints...")
                        return self.clusters
                    print("   Cannot link:", par)
                else:
                    break
                # print("--------------------")
                # print(par[0], par[1])
            # print(self.distances.keys())
            self.distances.pop(kljuc, None)
            self.izbrisi_razdalje(par[0])
            self.izbrisi_razdalje(par[1])
            # print(self.distances.keys())
            tocke = []
            tocke.append(self.clusters[par[0]].points)
            tocke.append(self.clusters[par[1]].points)
            # print("tocke: ", len(tocke))
            novCluster = Cluster(m + idZ)
            novCluster.update(par[0], par[1], dist, tocke)
            novCluster.centroid = novCluster.calculateCentroid()
            self.clusters.pop(par[0])
            self.clusters.pop(par[1])
            self.clusters.update({(m + idZ): novCluster})
            # print("clustri:")
            # print(self.clusters.keys())
            # print("dodajam razdalje...")

            # print("NOV:" ,m+idZ)
            self.dodaj_razdalje(m + idZ)

            print("par: ", par, "dist: ", '%.08f' % dist)
            if idZ == 0:
                self.Z = [par[0], par[1], dist, novCluster.n]
            else:
                newrow = [par[0], par[1], dist, novCluster.n]
                self.Z = np.vstack([self.Z, newrow])

            n = len(self.clusters)
            idZ += 1
        return self.clusters




