# -*- coding: utf-8 -*-
# __author__ = 'peter'
import functools
import itertools
import os
from tkinter import *

import math

from ABHclustering import ABHclustering
from cluster import Cluster, Point
import sys
import csv
from CT import CT

#RUN_SET = "IRIS"
RUN_SET = "BONITETE"
LOG_NAME = "results.log"
TEST_CONST = [10, 30, 60, 100, 150, 200, 250, 300, 400]

class App:
    def __init__(self, master):
        self.master = master
        self.helper = None
        self.constraint_count = 0
        self.attributes = []
        menu_frame = Frame()
        menu_frame.pack(fill=X, side=TOP)
        self.menu_frame = menu_frame
        self.step = 0

        self.autorun = None
        #elf.autorun = Button(menu_frame, text="Autorun", command=self.bot2)
        #self.autorun.pack(side=LEFT)
        frame = Frame()
        frame.pack(expand=1, fill='both', side=RIGHT)
        self.frame = frame

        cikel_frame = Frame()
        cikel_frame.pack(expand=1, fill=X, side=TOP, anchor=NW)
        self.cikel_frame = cikel_frame

        info_frame = Frame()
        info_frame.pack(expand=1, fill=X, side=TOP)
        self.info_frame = info_frame

        self.cikel_label = StringVar()
        Label(self.cikel_frame, textvariable=self.cikel_label, justify=LEFT).pack(side=TOP, anchor=NW)

        self.info_label = StringVar()
        self.info_label.set("Clusters found:\n")
        Label(self.info_frame, textvariable=self.info_label, font="Helvetica 14 bold italic", justify=LEFT).pack(side=TOP, anchor=SW)

        #self.listbox = Listbox(self.info_frame, width="50")
        #self.listbox.pack(side=BOTTOM, anchor=SW)

        #self.details_label = StringVar()
        #self.details_label.set("")
        #Label(self.info_frame, textvariable=self.details_label, justify=LEFT).pack(side=BOTTOM, anchor=SW)

        # self.output_field = Text(master, height=10, width=80)
        # self.output_field.pack(side=LEFT, fill=BOTH, expand=1)

        self.cluster_output = Text(frame, height=1)
        self.cluster_output.pack(side=LEFT, fill='both', expand=1, anchor=W)
        self.cluster_output_scroll = Scrollbar(frame)
        self.cluster_output_scroll.pack(side=RIGHT, fill=Y)
        self.cluster_output_scroll.config(command=self.cluster_output.yview)
        self.cluster_output.config(yscrollcommand=self.cluster_output_scroll.set)
        self.cluster_output.bind('<<Modified>>', self.showEnd)
        # self.output_field.bind('<<Modified>>',self.showEndOutput)

        master.geometry("{0}x{1}+0+0".format(
            master.winfo_screenwidth()-100, master.winfo_screenheight()-100))
        self.final_n_of_clusters = None
        self.my_clusters = None
        self.n_clusters_index_start = 0
        self.n_clusters_index_end = 5
        self.critical_index_start = 0
        self.critical_index_end = 5
        self.cikel_nbr = 0
        self.args = []
        self.additional_attributes = []
        self.proti = None
        self.pogoj = None
        self.data_filename = None

        if RUN_SET == "IRIS":
            self.init_iris()
        elif RUN_SET == "BONITETE":
            self.init_bonitete()

        self.hc_button = Button(menu_frame, text="Start Hierarchical Clustering", command=self.hierarhicalClustering)
        self.hc_button.pack(side=LEFT)

        self.determine_button = None
        self.plot_button = None
        self.rename_button = None
        self.get_criticals_button = None
        self.argument_button = None
        self.counter_example_button = None
        self.improve_button = None
        self.prev_button = None
        self.next_button = None
        self.plot_clusters_button = None
        self.labels = []
        self.cikel_label.set("ABHC Cikel: " + str(self.cikel_nbr) + "\n"
                                                                    "-----------------------------------------\n"
                                                                    "Initialization: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Hierarchical clustering: Ready\n"
                                                                    "-----------------------------------------\n"
                                                                    "Determine the number of clusters:\n"
                                                                    "-----------------------------------------\n"
                                                                    " Selecting critical example:\n"
                                                                    "-----------------------------------------\n"
                                                                    " Critical example argumentation:\n"
                                                                    "-----------------------------------------\n"
                                                                    " Counter example constraints:\n "
                                                                    "-----------------------------------------\n"
                                                                    "Run ABHC:")

    def set_labels(self):
        self.listbox.delete(0, END)
        self.map_point_name_to_point = {}
        for i, c in enumerate(self.abh.condition):
            self.listbox.insert(END, "Example " + str(c['point'][0].reference))
            self.map_point_name_to_point["Example " + str(c['point'][0].reference)] = c['point']

    def display_critical_data(self, ok):
        now = self.listbox.curselection()[0]
        name = self.listbox.get(ACTIVE)
        o = name + " " + str(self.abh.condition[now]['point'][0]) + "\n"
        o += "Old cluster: " + str(self.abh.clusters[self.abh.condition[now]['current_cluster']]) + "\n"
        #o += "New cluster: " + str(self.abh.clusters[self.abh.condition[now]['target_cluster']]) + "\n"
        o += "Argument:\n"
        cond = self.abh.condition[now]['arguments']
        condn = len(cond)
        for i, arg in enumerate(cond):
            o += arg[0] + " " + arg[1]
            if arg[2]:
                o += arg[2]
            if i + 1 != condn:
                o += " AND "
        self.details_label.set(o)

    def sum_attributes(self, points_list, class_index):
        return_val = [0] * len(points_list[0])
        for p in points_list:
            return_val = [float(x) + float(y) for i, (x, y) in enumerate(zip(return_val, p))]
        return [x / len(points_list) for x in return_val]

    def sum_attributes_max(self, points_list, class_index):
        return_val = [0] * len(points_list[0])
        for p in points_list:
            return_val = [float(y) if float(y) > float(x) else float(x) for i, (x, y) in enumerate(zip(return_val, p))]
        return return_val

    def getMax(self, row):
        max = -sys.maxsize - 1
        for i in row:
            if i.isdigit():
                if i > max:
                    max = i
        return max

    def getMin(self, row):
        min = sys.maxsize
        for i in row:
            if i.isdigit():
                if i < min:
                    min = i
        return min

    def init_iris(self):
        """
        Finding critical examples from the iris dataset
        """
        # read data
        input = open('iris.data', 'r')
        reader = csv.reader(input)
        self.attributes = next(reader)
        points = []
        data = [d[:] for d in reader]

        self.clusters = {}
        self.clustersCopy = {}
        self.updatedClusters = {}
        # Create points from data
        for i, line in enumerate(data):
            points.append(Point([float((line[x])) for x in range(0, len(line[:4]))], cheat=line[4], reference=i))
            cluster = Cluster(i)
            cluster.points.append(
                Point([float((line[x])) for x in range(0, len(line[:4]))], cheat=line[4], reference=i))
            cluster.primeri.append(i)
            self.clusters.update({i: cluster})
            self.clustersCopy.update({i: cluster})
            self.updatedClusters.update({i: cluster})
        self.points = points
        self.my_clusters = self.clusters
        self.abh = ABHclustering(self.points, points, self.clusters, self.attributes, candidates=None)
        self.abh.dim = 4
        self.abh.linkage = "AVERAGE"         #other possibility: AVERAGE
        self.abh.distance_type = "EUCLIDIAN"
        self.abh.l = CT(LOG_NAME, RUN_SET)
        self.log(self.abh.l.dataset(self.abh))
        self.cikel_nbr += 1
        self.data_filename ="IRIS"

    def init_bonitete(self):
        """
        Finding critical examples from the bonitete dataset
        """
        #OUTLIERS = [9, 11, 38, 62, 80, 82, 90, 94, 121, 126, 132, 136, 155, 156, 164, 165, 169, 173, 188]
        OUTLIERS = []
        input = open('bonitete_tutor.tab', 'r')
        reader = csv.reader(input)
        atributi = next(reader)
        atributi = [i.split('\t') for i in atributi]
        atributi = atributi[0]
        self.attributes = [atributi[i] for i in range(2,24)]+[atributi[33]] + [atributi[1]] + [atributi[0]]
        points = []
        self.clusters = {}
        self.clustersCopy = {}
        self.updatedClusters = {}
        data = [d[:] for d in reader]
        reference = 0
        for i, line in enumerate(data):
            if i > 1 and (i-2) not in OUTLIERS:
                vrstica = line[0].split('\t')
                vektor = []
                cheat = None

                for j in range(2, 24):
                    vektor.append(float(vrstica[j]))
                if vrstica[33] == "FALSE":
                    vektor.append(float(0))
                elif vrstica[33] == "TRUE":
                    vektor.append(float(1))
                if vrstica[34] == "A":
                    cheat = "GOOD"
                else:
                    cheat = "BAD"
                if vrstica[1] == "S":
                    vektor.append(float(0))
                elif vrstica[1] == "M":
                    vektor.append(float(1))
                elif vrstica[1] == "L":
                    vektor.append(float(2))
                vektor.append(float(ord(vrstica[0])))
                points.append(Point(vektor, cheat=cheat, reference=int(reference)))
                cluster = Cluster(reference)
                cluster.points.append(Point(vektor, cheat=cheat, reference=int(reference)))
                cluster.primeri.append(reference)
                self.clusters.update({reference: cluster})
                self.clusters[reference].centroid = Point(vektor, None, int(reference))
                self.clustersCopy.update({reference: cluster})
                self.updatedClusters.update({reference: cluster})
                reference += 1
                #print(vrstica[atributi.index('lt.ebit.margin.change')] , " = ")
                #a = float(vrstica[atributi.index('net.sales')]) - float(vrstica[atributi.index('cost.of.goods.and.services')]) - float(vrstica[atributi.index('cost.of.labor')]) - float(vrstica[atributi.index('financial.expenses')])
                #print(float(vrstica[atributi.index('EBIT')] ) / float(vrstica[atributi.index('net.income')]))
                #print(vrstica[atributi.index('lt.ebit.margin.change')] ," = ", )
                #print(vrstica[atributi.index('lt.sales.growth')], " = ", (float(vrstica[atributi.index('lt.assets')]) / float(vrstica[atributi.index('lt.liabilities')])))
                #print(vrstica[atributi.index('net.debt/EBITDA')], " = ", ((float(vrstica[atributi.index('debt')]) + float(vrstica[atributi.index('cash')])   ) / float(vrstica[atributi.index('EBITDA')])))
                """
                print(vrstica[atributi.index('TIE')], " = ")
                a = float(vrstica[atributi.index('EBIT')])
                b = float(vrstica[atributi.index('interest')])
                if a == 0 or b == 0:
                    print("0")
                else:
                    if round(a / b, 2) == float(vrstica[atributi.index('ROA')]):
                        st1 += 1
                    else:
                        print(round(a / b, 2))

                #print(vrstica[atributi.index('ROA')])
                #print(vrstica[atributi.index('net.sales')])
                #print(vrstica[atributi.index('total.oper.liabilities')] + vrstica[atributi.index('assets')])
                #print(vrstica[atributi.index('ROA')], " = ", (
                        #float(vrstica[atributi.index('net.income')]) / (
                        #float(vrstica[atributi.index('assets')]) + float(vrstica[atributi.index('cash')]) + float(vrstica[atributi.index('inventories')]) + float(vrstica[atributi.index('lt.assets')]) )))
                print("-----")
                """

        self.points = points
        self.my_clusters = self.clusters
        self.abh = ABHclustering(self.points, points, self.clusters, self.attributes, candidates=None)
        self.abh.linkage = "WARD"         #other possibilities: AVERAGE , WARD
        self.abh.dim = 25
        self.abh.distance_type = "COSINUS"       #other possibilities: EUCLIDIAN , COSINUS
        self.abh.l = CT(LOG_NAME, RUN_SET)
        self.log(self.abh.l.dataset(self.abh))
        self.cikel_nbr += 1
        self.data_filename="BONITETE"



    def hierarhicalClustering(self):
        if self.step < 1:
            self.step = 1
        self.log("Hierarchical clustering in cikel: " + str(self.cikel_nbr) + " with constraints: " + str(
            self.constraint_count) + "\n")
        self.master.config(cursor="wait")
        self.master.update()
        self.abh.clusters = self.abh.hierarhicalClustering(self.clusters)
        self.master.config(cursor="")
        self.hc_button.destroy()

        if self.plot_button is None:
            self.plot_button = Button(self.menu_frame, text="Plot dendrogram", command=self.plot2D)
            self.plot_button.pack(side=LEFT)
        if self.determine_button is None:
            self.determine_button = Button(self.menu_frame, text="Determine number of clusters",
                                           command=self.determine_clusters)
            self.determine_button.pack(side=LEFT)
            self.cikel_label.set("ABHC Cikel: " + str(self.cikel_nbr) + "\n"
                                                                        "-----------------------------------------\n"
                                                                        "Initialization: Finished\n"
                                                                        "-----------------------------------------\n"
                                                                        "Hierarchical clustering: Finished\n"
                                                                        "-----------------------------------------\n"
                                                                        "Determine the number of clusters: Ready\n"
                                                                        "-----------------------------------------\n"
                                                                        "Selecting critical example:\n"
                                                                        "-----------------------------------------\n"
                                                                        "Critical example argumentation:\n"
                                                                        "-----------------------------------------\n"
                                                                        "Counter example constraints:\n "
                                                                        "-----------------------------------------\n"
                                                                        "Run ABHC:")

    def plot2D(self):
        try:
            import numpy as np
            import math
            import matplotlib.pyplot as plt
            from scipy.cluster.hierarchy import dendrogram

        except ImportError as e:
            self.cluster_output.insert("Missing packages, cluster plots not supported....\n")
            self.cluster_output.insert(e)

            return  # module doesn't exist, deal with it.

        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        dendrogram(
            self.abh.Z,
            truncate_mode='lastp',  # show only the last p merged clusters
            p=12,  # show only the last p merged clusters
            leaf_rotation=90.,
            leaf_font_size=12.,
            show_contracted=True,  # to get a distribution impression in truncated branches
        )
        plt.show(block=True)
        self.plot_button.destroy()
        '''

        self.plot_popup()
        self.master.wait_window(self.top)        
        '''

    def plot_popup(self):
        self.plot_button.destroy()
        #top = self.top = Toplevel(self.master)
        try:
            import numpy as np
            import math
            import matplotlib.pyplot as plt
            from scipy.cluster.hierarchy import dendrogram

        except ImportError as e:
            self.cluster_output.insert("Missing packages, cluster plots not supported....\n")
            self.cluster_output.insert(e)

            return  # module doesn't exist, deal with it.
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        dendrogram(
            self.abh.Z,
            truncate_mode='lastp',  # show only the last p merged clusters
            p=12,  # show only the last p merged clusters
            leaf_rotation=90.,
            leaf_font_size=12.,
            show_contracted=True,  # to get a distribution impression in truncated branches
        )
        plt.show(block=True)

        #b = Button(top, text="OK", command=self.plot_close).pack()


    def plot_close(self, event=None):
        self.top.destroy()

    def determine_clusters(self):
        self.log("Determining number of clusters...\n")
        self.number_of_clusters_popup()
        self.master.wait_window(self.top)
        if self.helper != None:
            self.final_n_of_clusters = int(self.helper)
            self.initialClusters = self.clustersCopy.copy()
            self.abh.clusters = self.abh.rebuildClusters(self.initialClusters, int(self.helper))
            #self.log(self.abh.l.clusters(self.abh))
            self.refresh_cluster_data()
            self.update_display_data()
            self.helper = None
        if self.plot_clusters_button is None:
            self.plot_clusters_button = Button(self.menu_frame, text="Plot clusters", command=self.plot_clusters)
            self.plot_clusters_button.pack(side=LEFT)
        if self.rename_button is None:
            self.rename_button = Button(self.menu_frame, text="Rename clusters", command=self.rename_clusters)
            self.rename_button.pack(side=LEFT)
        if self.get_criticals_button is None:
            self.get_criticals_button = Button(self.menu_frame, text="Select critical example",
                                               command=self.get_criticals)
            self.get_criticals_button.pack(side=LEFT)

        self.cikel_label.set("ABHC Cikel: " + str(self.cikel_nbr) + "\n"
                                                                    "-----------------------------------------\n"
                                                                    "Initialization: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Hierarchical clustering: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Determine the number of clusters: Finished\n"
                                                                    "------------------------------------------\n"
                                                                    "Selecting critical example: Ready\n"
                                                                    "-----------------------------------------\n"
                                                                    "Critical example argumentation: \n"
                                                                    "-----------------------------------------\n"
                                                                    "Counter example constraints:\n "
                                                                    "-----------------------------------------\n"
                                                                    "Run ABHC:")

        self.determine_button.destroy()

    def refresh_cluster_data(self):
        void = self.abh.l.clusters(self.abh)
        self.log("============================================\n")
        self.log("============================================\n")
        self.log("ITERATION NUMBER: "+str(self.cikel_nbr)+"\n")
        self.log("NMI: " +str(round(self.abh.NMI, 4)) +"\n")
        self.log("ARI: " + str(round(self.abh.ARI, 4)) + "\n")
        f = open('points_Cikel' + str(self.cikel_nbr) + ".txt", 'w')
        for cluster in self.abh.clusters:
            f.write("=====" + self.abh.clusters[cluster].name + "=====\n")
            self.abh.clusters[cluster].points.sort(key=lambda x: int(x.reference), reverse=False)
            for p in self.abh.clusters[cluster].points:
                f.write(str(p.reference) + " " + p.cheat + " " + str(p.coords)+"\n")
        f.close()
        self.log("============================================\n")
        self.log("============================================\n")

    def plot_clusters(self):
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import pylab as pl
            import plotly.express as px
        except ImportError as e:
            self.cluster_output.insert("Missing packages, cluster plots not supported....\n")
            self.cluster_output.insert(e)

            return  # module doesn't exist, deal with it.
        #top = self.top = Toplevel(self.master)
        color = ['blue', 'red', 'green', 'black', 'purple', 'orange']
        podatki = np.array([])
        for i, cluster in enumerate(self.abh.clusters):
            if i == 0:
                podatki = [p.coords for p in self.abh.clusters[cluster].points]
            else:
                podatki = np.vstack([podatki, [p.coords for p in self.abh.clusters[cluster].points]])
            #print(np.array([p.coords for p in self.clusters[cluster].points]))
            #print(np.array(self.attributes))


        df = pd.DataFrame(podatki, columns= np.array([a for a in self.abh.attributes]))
        df.head()
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        scaler.fit(df)
        scaled_data = scaler.transform(df)
        pca = PCA(copy=True, n_components=2, whiten=False)
        pca.fit(scaled_data)
        x_pca = pca.transform(scaled_data)

        #fig = Figure(figsize=(6,6))
        #a = fig.add_subplot(111)
        counter = 0
        plt.figure()
        plt.xlim([math.ceil(x_pca[:, 0].min() - .5), math.ceil(x_pca[:, 0].max() + .5)])
        plt.ylim([math.ceil(x_pca[:, 1].min() - .5), math.ceil(x_pca[:, 1].max() + .5)])
        plt.xlabel('First principal component')
        plt.ylabel('Second Principal Component')
        for i,cluster in enumerate(self.abh.clusters):
            for j in self.abh.clusters[cluster].points:

                #plt.annotate(str(j.reference), xy=j.coords, xytext=(0, 0), color="black")
                #a.text(x_pca[counter, 0], x_pca[counter, 1], str(j.reference), color='black', fontsize=7)
                pl.text(x_pca[counter, 0], x_pca[counter, 1], str(j.reference), color='black', fontsize=9)
                #a.plot(x_pca[counter, 0], x_pca[counter, 1], marker=str(j.reference),color='black')
                #a.scatter(x_pca[counter][0], x_pca[counter][1], color=color[i])
                plt.scatter(x_pca[counter, 0], x_pca[counter, 1], marker='o', c=color[i], s=100)
                counter += 1
        #a.scatter(v,x,color='red')

        #a.invert_yaxis()

        #a.set_title ("Estimation Grid", fontsize=16)
        #a.set_ylabel("Second Principal Component", fontsize=14)
        #a.set_xlabel("First principal component", fontsize=14)

        #canvas = FigureCanvasTkAgg(fig, master=top)
        #canvas.get_tk_widget().pack()
        #canvas.draw()

        plt.show(block=True)



    def rename_clusters(self):
        self.log("Renaming clusters...\n")
        for cluster in self.abh.clusters:
            nbr = 5 if len(self.abh.clusters[cluster].points) >= 5 else len(self.abh.clusters[cluster].points)

            print_list = [self.abh.clusters[cluster].centroid] + self.abh.clusters[cluster].points[0:nbr]

            self.rename_popup(self.abh.l.candidates(self.abh, print_list, 0, nbr+1), self.abh.clusters[cluster].name)
            self.master.wait_window(self.top)
            if self.helper != None:
                self.abh.clusters[cluster].name = self.helper
                self.helper = None
        self.refresh_cluster_data()
        self.update_display_data()
        # name = raw_input("Clusters new name:")
        # self.l.log('\n')
        # cluster.name = name

    def get_criticals(self):
        self.abh.prev_dict = self.abh.make_dict()
        if self.abh.candidates == None or len(self.abh.candidates) == 0:
            self.log("Finding critical examples...\n")
            self.abh.get_candidates()
            self.update_display_data()
        self.log("Found " + str(len(self.abh.candidates)) + " critical examples\n")
        start = self.critical_index_start
        self.pick_critical_popup(self.critical_index_start, self.critical_index_end)
        self.master.wait_window(self.top)
        if (self.helper == None and start == self.critical_index_start) or self.helper == 'Choose example index:':

            self.log("No example picked.\n")
            if self.abh.critical_example != None:
                return
        elif start != self.critical_index_start and self.critical_index_start != 0:
            # Example not picked because we want to see more
            self.get_criticals()
            return 0

        if self.helper == None:
            return 0
        if self.step < 2:
            self.step = 2
        self.abh.critical_example.append(self.abh.candidates[int(self.helper)])
        self.abh.candidates.remove(self.abh.critical_example[-1])

        #self.log("We picked Example " + str(self.abh.critical_example[-1][0].reference) + " now we need to argument it.\n")
        #print("KRITICNI PRIMER: ", self.abh.critical_example)

        if self.argument_button is None:
            self.argument_button = Button(self.menu_frame, text="Argument constraint", command=self.argument_steps)
            self.argument_button.pack(side=LEFT)
        if str(self.argument_button['state']) == 'disabled':
            self.argument_button['state'] = 'normal'

        self.cikel_label.set("ABHC Cikel: " + str(self.cikel_nbr) + "\n"
                                                                    "-----------------------------------------\n"
                                                                    "Initialization: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Hierarchical clustering: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Determine the number of clusters: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Selecting critical example: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Critical example argumentation: Ready\n"
                                                                    "-----------------------------------------\n"
                                                                    "Counter example constraints:\n "
                                                                    "-----------------------------------------\n"
                                                                    "Run ABHC:")

    def argument_steps(self):
        print("ARGUMENTIRAJ KRITICNI PRIMER")
        """
        if len(self.listbox.curselection()) != 0:
            critical_point = self.map_point_name_to_point[self.listbox.get(self.listbox.curselection())][0]
        else:
        """
        critical_point = self.abh.critical_example[-1][0]
        # Informative print of clusters - reference points
        # We present clusters to expert for better understanding
        #for cluster in self.abh.clusters:
            #self.log("Cluster " + str(self.abh.clusters[cluster].name) + ":  distance: " + str(critical_point.getDistance(self.abh.clusters[cluster].centroid)) + "\n")

        # We present the example to the expert

        #self.log("Critical Example is in cluster: " + self.abh.clusters[self.abh.critical_example[-1][1]].name + "\n")
        self.get_argument_with_pair_popup()
        self.cikel_label.set("ABHC Cikel: " + str(self.cikel_nbr) + "\n"
                                                                    "-----------------------------------------\n"
                                                                    "Initialization: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Hierarchical clustering: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Determine the number of clusters: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Selecting critical example: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Critical example argumentation: In progress\n"
                                                                    "-----------------------------------------\n"
                                                                    "Counter example constraints:\n "
                                                                    "-----------------------------------------\n"
                                                                    "Run ABHC:")
        self.master.wait_window(self.top)
        if len(self.abh.condition) != 0 and critical_point == self.abh.condition[-1]["point"][0]:
            self.log("We are changing argument for previous critical example.\n")
            self.abh.condition[-1] = self.helper
            self.helper = None
        elif self.step >= 3:
            list = [p['point'][0] == critical_point for p in self.abh.condition]
            if any(list):
                point_index = list.index(True)
                self.log("We are chaning argument for a critical example.\n")
                self.abh.condition[point_index] = self.helper
                self.helper = None
            else:
                self.log("We are adding argument for a new critical example.\n")
                self.abh.condition.append(self.helper)
                self.helper = None
        else:
            if self.helper == 0 or self.helper == '0':
                return
            else:
                self.abh.condition.append(self.helper)
                self.helper = None
        if self.step < 3:
            self.step = 3

        #self.set_labels()
        for i in range(0, len(self.abh.condition)):
            cond = self.abh.condition[i]['arguments']
            arg_str = "IF "
            for k, arg in enumerate(cond):
                arg_str += arg[0] + " " + arg[1] + " "
                if arg[2] != None:
                    arg_str += arg[2]
                if k < len(cond):
                    arg_str += " AND "
            arg_str += "THEN "
            arg_str += self.choice_string[self.abh.condition[i]['counter'][0]['act']] + " "
            arg_str += " WITH Example: " + str(self.abh.condition[i]['counter'][0]['example'][0].reference) + "\n"

            #self.log(arg_str)
        # argument collected, allow fetching counter examples and argumenting them


        if self.counter_example_button is None:
            self.counter_example_button = Button(self.menu_frame, text="Argument counter examples",
                                                 command=self.counter_steps)
            self.counter_example_button.pack(side=LEFT)
        if str(self.counter_example_button['state']) == 'disabled':
            self.counter_example_button['state'] = 'normal'
        self.cikel_label.set("ABHC Cikel: " + str(self.cikel_nbr) + "\n"
                                                                    "-----------------------------------------\n"
                                                                    "Initialization: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Hierarchical clustering: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Determine the number of clusters: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Selecting critical example: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Critical example argumentation: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Counter example constraints: Ready\n "
                                                                    "-----------------------------------------\n"
                                                                    "Run ABHC:")

    def counter_steps(self):
        #print(self.abh.condition)
        for condition in self.abh.condition:
            #self.log("Counter examples for critical example: Example " + str(condition["point"][0].reference) + "\n")

            # open popup with data and ask to argument counter example
            counters = self.abh.counter_example(condition)
            print("STEVILO PROTIPRIMEROV: ", len(counters))
            for counter in counters:
                self.pogoj = condition
                self.proti = counter
                self.counter_argument_popup(condition, counter)
                self.cikel_label.set("ABHC Cikel: " + str(self.cikel_nbr) + "\n"
                                                                            "-----------------------------------------\n"
                                                                            "Initialization: Finished\n"
                                                                            "-----------------------------------------\n"
                                                                            "Hierarchical clustering: Finished\n"
                                                                            "-----------------------------------------\n"
                                                                            "Determine the number of clusters: Finished\n"
                                                                            "-----------------------------------------\n"
                                                                            "Selecting critical example: Finished\n"
                                                                            "-----------------------------------------\n"
                                                                            "Critical example argumentation: Finished\n"
                                                                            "-----------------------------------------\n"
                                                                            "Counter example constraints: In progress\n "
                                                                            "-----------------------------------------\n"
                                                                            "Run ABHC:")
                self.master.wait_window(self.top)

                if self.helper and self.helper['counter']:
                    condition["counter"] = condition["counter"] + self.helper['counter']
                self.helper = None
            if self.step < 4:
                self.step = 4

        if self.improve_button is None:
            self.improve_button = Button(self.menu_frame, text="Hierarchical clustering", command=self.improve_hc)
            self.improve_button.pack(side=LEFT)
        if str(self.improve_button['state']) == 'disabled':
            self.improve_button['state'] = 'normal'
        self.cikel_label.set("ABHC Cikel: " + str(self.cikel_nbr) + "\n"
                                                                    "-----------------------------------------\n"
                                                                    "Initialization: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Hierarchical clustering: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Determine the number of clusters: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Selecting critical example: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Critical example argumentation: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Counter example constraints: Finished\n "
                                                                    "-----------------------------------------\n"
                                                                    "Run ABHC: Ready")

    def rename_popup(self, text, cluster):
        top = self.top = Toplevel(self.master)

        output_field = Text(top, height=30, width=120)
        output_field.insert(END, text)
        output_field.pack(fill=BOTH, expand=1)

        # self.output_field_scroll = Scrollbar(self.master)
        # self.output_field_scroll.pack(side=RIGHT, fill=Y)
        # self.output_field_scroll.config(command=self.output_field.yview)
        # self.output_field.config(yscrollcommand=self.output_field_scroll.set)
        label = Label(top, text="Rename " + cluster + ":", font="Helvetica 16 bold italic")
        label.pack(side=LEFT)
        self.e = Entry(top)
        self.e.pack(side=LEFT)

        b = Button(top, text="OK", command=self.rename_close).pack()
        self.e.bind('<Return>', self.rename_close)

    def rename_close(self, event=None):
        self.critical_index_start = 0
        self.critical_index_end = 5
        self.helper = self.e.get()
        self.top.destroy()

    def number_of_clusters_popup(self):
        top = self.top = Toplevel(self.master)

        label = Label(top, text="Vpisi stevilo clustrov : ", font="Helvetica 16 bold italic")
        label.pack(side=LEFT)
        self.e = Entry(top)
        self.e.pack(side=LEFT)

        b = Button(top, text="OK", command=self.number_close).pack()
        self.e.bind('<Return>', self.rename_close)

    def number_close(self, event=None):
        self.log("Dolocili smo, da je koncno stevilo clustrov enako : " + self.e.get() + "\n")
        self.helper = self.e.get()
        self.top.destroy()

    def pick_critical_popup(self, start=0, end=5):
        end = end if len(self.abh.candidates) >= end else len(self.abh.candidates)
        text = "Showing top " + str(end) + " critical examples. Choose one to argument\n"

        #self.log(text)
        text += self.abh.l.candidates(self.abh, self.abh.candidates[start:end], start, end)
        #self.log(text)

        top = self.top = Toplevel(self.master)

        output_field = Text(top, height=30, width=150)
        output_field.insert(END, text)
        output_field.pack(fill=BOTH, expand=1)

        if len(self.abh.candidates) > 0 and start >= 5:
            self.prev_button = Button(top, text="<<<", command=lambda: self.prev_critical_display(start, end)).pack(
                side=LEFT)
        elif self.prev_button != None:
            self.prev_button.destroy()

        self.e = StringVar(self.master)
        self.e.set("Choose example index:")  # default value
        choice = [str(i) for i in range(start, end)]
        w = OptionMenu(top, self.e, *choice).pack(side=LEFT)

        if len(self.abh.candidates) > end:
            self.next_button = Button(top, text=">>>", command=lambda: self.next_critical_display(start, end)).pack(
                side=LEFT)
        elif self.next_button != None:
            self.next_button.destroy()

        b = Button(top, text="OK", command=self.rename_close).pack(side=LEFT)

    def next_critical_display(self, start, end):
        self.critical_index_start += 5
        self.critical_index_end += 5
        self.top.destroy()

    def prev_critical_display(self, start, end):
        self.critical_index_start = self.critical_index_start - 5
        self.critical_index_end = self.critical_index_end - 5
        self.top.destroy()

    def get_argument_with_pair_popup(self):
        """
        if len(self.listbox.curselection()) != 0:
            critical_point = self.map_point_name_to_point[self.listbox.get(self.listbox.curselection())]
            critical_point_target = self.map_point_name_to_point[self.listbox.get(self.listbox.curselection())][1]
        else:
        """
        critical_point = self.abh.critical_example[-1]
        critical_point_target = self.abh.critical_example[-1][1]

        self.counter = self.abh.get_pair(critical_point_target)
        # print("counter: ", self.counter)
        print_list = [critical_point, self.counter] + [self.abh.clusters[c].represent() for c in self.abh.clusters]
        # Ask if data is correct
        self.m = critical_point

        text = self.abh.l.candidates(self.abh, print_list, 0, len(print_list))


        top = self.top = Toplevel(self.master)

        frame = Frame(self.top)
        frame.pack(fill=X, anchor=W)

        self.test = StringVar()
        self.test.set(text)

        self.output_field = Label(top, textvariable=self.test, anchor="nw", justify=LEFT, font="Consolas 11")
        self.output_field.pack(fill=BOTH, expand=1)

        b = Button(top, text="OK", command=self.argument_pair_close).pack()
        b2 = Button(top, text="AND", command=self.argument_new).pack()
        b3 = Button(top, text="ADD ATTRIBUTE", command=self.attribute_new).pack()
        b4 = Button(top, text="DELETE ATTRIBUTE", command=self.delete_att).pack()

        str_o = "Example " + str(critical_point[0].reference) + " has "
        self.args = [self.create_arg_form(True)]

        self.label1 = Label(self.support_frame, text="THEN", font="Helvetica 14 bold italic")
        self.label1.pack(side=LEFT)

        self.counter_act = StringVar()
        self.choice_string = ['Nothing', 'Cannot-link', 'Must-link']

        self.counter_act.set(self.choice_string[0])  # default value
        self.op2 = OptionMenu(self.support_frame, self.counter_act, *self.choice_string).pack(side=LEFT)

    def argument_new(self):
        self.args.append(self.create_arg_form())

    def attribute_new(self):
        self.create_attribute_form()
    def delete_att(self):
        self.create_delete_form()

    def create_arg_form(self, if_label=False):
        self.support_frame = frame = Frame(self.top)
        frame.pack(fill=X, side=TOP, anchor=S)

        if not if_label:
            label = Label(frame, text="AND", font="Helvetica 14 bold italic")
            label.pack(side=LEFT)
        elif if_label == True:
            label = Label(frame, text="IF", font="Helvetica 14 bold italic")
            label.pack(side=LEFT)
        self.atr = StringVar()
        self.izbira = [i for i in self.abh.attributes]
        self.atr.set(self.izbira[0])  # default value
        self.om1 = OptionMenu(frame, self.atr, *self.izbira)
        self.om1.pack(side=LEFT, anchor=W)

        self.op = StringVar()
        choice = [i for i in self.abh.def_operators()]
        self.op.set(choice[0])  # default value
        om2 = OptionMenu(frame, self.op, *choice).pack(side=LEFT, anchor=W)

        self.e = Entry(frame)
        self.e.pack(side=LEFT, anchor=W)
        callback = lambda *args: (self.e.configure(
            state='disabled') if self.op.get() == '<<<' or self.op.get() == '>>>' else self.e.configure(
            state='normal'))
        self.op.trace("w", callback)

        return [self.atr, self.op, self.e, self.om1]
    def create_delete_form(self):
        self.support_frame = frame = Frame(self.top)
        frame.pack(fill=X, side=TOP, anchor=S)
        self.atribs = StringVar()
        choice0 = [i for i in self.abh.attributes]
        self.atribs.set(choice0[0])  # default value
        w3 = OptionMenu(frame, self.atribs, *choice0).pack(side=LEFT, anchor=E)
        b4 = Button(frame, text="DELETE ATTRIBUTE", command=self.delete_points).pack(side=RIGHT)


    def create_attribute_form(self):
        self.support_frame = frame = Frame(self.top)
        frame.pack(fill=X, side=TOP, anchor=S)
        labela = Label(frame, text="ATTRIBUTE NAME:", font="Helvetica 12 bold italic")
        labela.pack(side=LEFT)

        self.f = Entry(frame)
        self.f.pack(side=LEFT, anchor=W)

        label = Label(frame, text="EQUATION:", font="Helvetica 12 bold italic")
        label.pack(side=LEFT)

        self.atr1 = StringVar()
        choice0 = [i for i in self.abh.attributes]
        self.atr1.set(choice0[0])  # default value
        w3 = OptionMenu(frame, self.atr1, *choice0).pack(side=LEFT, anchor=W)

        self.op1 = StringVar()
        choice1 = [i for i in self.abh.def_ops()]
        self.op1.set(choice1[0])  # default value
        w4 = OptionMenu(frame, self.op1, *choice1).pack(side=LEFT, anchor=W)

        self.atr2 = StringVar()
        choice2 = [i for i in self.abh.attributes]
        self.atr2.set(choice2[0])  # default value
        w5 = OptionMenu(frame, self.atr2, *choice2).pack(side=LEFT, anchor=W)

        self.b4 = Button(frame, text="CREATE NEW ATTRIBUTE", command=self.update_points).pack(side=RIGHT)
    def delete_points(self):
        idAtributa = self.abh.attributes.index(self.atribs.get())
        self.abh.attributes.pop(idAtributa)

        for cluster in self.abh.clusters:
            for point in self.abh.clusters[cluster].points:
                point.coords.pop(idAtributa)
                point.n = point.n-1
            self.abh.clusters[cluster].dim -= 1
            self.abh.clusters[cluster].centroid = self.abh.clusters[cluster].calculateCentroid()
        self.abh.distances = {}
        print_list = [self.m, self.counter] + [self.abh.clusters[c].represent() for c in self.abh.clusters]
        # Ask if data is correct
        text = self.abh.l.candidates(self.abh, print_list, 0, len(print_list))
        self.test.set(text)
        self.izbira = [i for i in self.abh.attributes]

        if len(self.args) > 0:
            for a in self.args:
                if len(a) > 3:
                    a[3].children["menu"].delete(0, "end")
                    for i in self.izbira:
                        a[3].children["menu"].add_command(label=i, command=lambda c=i: a[0].set(c))
                    a[0].set(self.izbira[0])

        self.support_frame.destroy()
        self.update_display_data()

    def update_points(self):
        new_name = self.f.get()
        idAtr1 = self.abh.attributes.index(self.atr1.get())
        idAtr2 = self.abh.attributes.index(self.atr2.get())
        operand = self.op1.get()
        self.abh.attributes.append(new_name)
        for cluster in self.abh.clusters:
            for point in self.abh.clusters[cluster].points:
                if operand == '/':
                    if point.coords[idAtr2] == 0:
                        point.coords.append(0.0)
                    else:
                        point.coords.append(round(point.coords[idAtr1] / point.coords[idAtr2], 2))
                elif operand == '*':
                    point.coords.append(round(point.coords[idAtr1] * point.coords[idAtr2], 2))
                elif operand == '+':
                    point.coords.append(round(point.coords[idAtr1] + point.coords[idAtr2], 2))
                point.n = point.n + 1
            self.abh.clusters[cluster].dim = self.abh.clusters[cluster].dim + 1
            self.abh.clusters[cluster].centroid = self.abh.clusters[cluster].calculateCentroid()
        self.abh.distances = {}

        print_list = [self.m, self.counter] + [self.abh.clusters[c].represent() for c in self.abh.clusters]
        # Ask if data is correct
        text = self.abh.l.candidates(self.abh, print_list, 0, len(print_list))
        self.test.set(text)
        self.izbira = [i for i in self.abh.attributes]

        if len(self.args) > 0:
            for a in self.args:
                if len(a) > 3:
                    a[3].children["menu"].delete(0, "end")
                    for i in self.izbira:
                        a[3].children["menu"].add_command(label=i, command=lambda c=i: a[0].set(c))
                    a[0].set(self.izbira[0])

        self.support_frame.destroy()
        self.update_display_data()

    def argument_pair_close(self):
        rule_dic = {}

        """
        if len(self.listbox.curselection()) != 0:
            critical_point = self.map_point_name_to_point[self.listbox.get(self.listbox.curselection())]
        else:
        """
        critical_point = self.abh.critical_example[-1]

        rule_dic["point"] = critical_point
        # condition[4] = int(critical_point.reference)
        # current cluster
        rule_dic["current_cluster"] = critical_point[1]
        # point target
        condition = []
        for a in self.args:
            val = a[2].get()
            if val == '' and a[1].get() != '>>>' and a[1].get() != '<<<':
                continue
            elif a[1].get() == '>>>' and a[1].get() == '<<<':
                val = ''

            condition.append([a[0].get(), a[1].get(), val])  # condition[5] = int(current_cluster)
        rule_dic["arguments"] = condition

        t = self.counter_act.get().lower()
        if t == 'nothing':
            t = 0
        elif t == 'cannot-link':
            t = 1
        elif t == 'must-link':
            t = 2
        else:
            t = 0
        counter = [{'act': t, 'example': self.counter}]
        rule_dic['counter'] = counter
        #self.log(str(rule_dic))
        # We have our condition. find counter examples
        self.helper = rule_dic
        self.top.destroy()

    def counter_argument_popup(self, condition, counter):
        critical_point = condition["point"][0]
        self.counter = counter
        print_list = [condition["point"], counter] + [self.abh.clusters[c].represent() for i, c in
                                                      enumerate(self.abh.clusters)]

        #self.log("Fetching argument for counter example\n")

        # Ask if data is correct
        self.m = condition["point"]
        self.args = []

        text = self.abh.l.candidates(self.abh, print_list, 0, len(print_list))
        #self.log(text)

        top = self.top = Toplevel(self.master)

        frame = Frame(self.top)
        frame.pack(fill=X, anchor=W)

        self.test = StringVar()
        self.test.set(text)

        self.output_field = Label(top, textvariable=self.test, anchor="nw", justify=LEFT, font="Consolas 11")
        self.output_field.pack(fill=BOTH, expand=1)

        b = Button(top, text="OK", command=self.ce_argument_close).pack()
        b2 = Button(top, text="AND", command=self.argument_new).pack()
        b3 = Button(top, text="ADD ATTRIBUTE", command=self.attribute_new).pack()
        b4 = Button(top, text="DELETE ATTRIBUTE", command=self.delete_att).pack()

        str_o = "Example " + str(counter[0].reference) + " has "
        for x in condition["arguments"]:
            str_o += str(x[0]) + " " + str(x[1])
            if x[2] != None or x[2] != "":
                str_o += str(x[2])
        str_o = "Do these two examples fit in the same cluster?"

        label = Label(frame, text=str_o, font="Helvetica 14 bold italic")
        label.pack(side=LEFT)


        self.support_frame = Frame(self.top)
        self.support_frame.pack(fill=X, side=TOP, anchor=S)
        label = Label(self.support_frame, text="act: ", font="Helvetica 14 bold italic")
        label.pack(side=LEFT)

        self.counter_act = StringVar()
        choice = ['Nothing', 'Cannot-link', 'Must-link']
        self.counter_act.set(choice[0])  # default value
        w2 = OptionMenu(self.support_frame, self.counter_act, *choice).pack(side=LEFT)

    def compare_arg(self, a, b):
        equal = True
        if len(a) <= 0:
            return True
        for x in a:
            for i in x:
                for j in b:
                    if i != b:
                        equal = False
                        break
                if equal == False:
                    break
        return equal

    def ce_argument_close(self):
        rule_dic = {}
        # point target
        condition = []
        t = self.counter_act.get().lower()
        if t == 'nothing':
            t = 0
        elif t == 'cannot-link':
            t = 1
        elif t == 'must-link':
            t = 2
        else:
            t = 0
        counter = [{'act': t, 'example': self.counter}]
        rule_dic['counter'] = counter

        #self.log(str(rule_dic))
        # We have our condition. find counter examples
        self.helper = rule_dic
        self.top.destroy()

    def improve_hc(self):
        # try next critical if none
        self.cikel_label.set("ABHC Cikel: " + str(self.cikel_nbr) + "\n"
                                                                    "-----------------------------------------\n"
                                                                    "Initialization: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Hierarchical clustering: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Determine the number of clusters: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Selecting critical example: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Critical example argumentation: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Counter example constraints: Finished\n "
                                                                    "-----------------------------------------\n"
                                                                    "Run ABHC: In progress")

        #self.log("Writing constraints from arguments...\n")
        constraints = []
        f = open('constraints_Cikel' + str(self.cikel_nbr) + ".txt", 'w')
        for c in self.abh.condition:
            if c['counter'] != None:
                for counter in c['counter']:
                    constraint = {}
                    if c['point'] != None:
                        constraint['point'] = c['point']
                    if counter['act'] == 1 and counter['example'] != None:
                        constraint['cannot-link'] = counter['example']
                    elif counter['act'] == 2 and counter['example'] != None:
                        constraint['must-link'] = counter['example']
                    if constraint not in constraints:
                        constraints.append(constraint)
                        f.write(str(constraint) + '\n')  # python will convert \n to os.linesep
                        self.constraint_count += 1
            f.write("NUMBER OF CONSTRAINTS: " + str(len(constraints)) + '\n')  # python will convert \n to os.linesep
            f.write("NUMBER OF ALL CONSTRAINTS: " + str(self.constraint_count) + '\n')  # python will convert \n to os.linesep
        f.close()  # you can omit in most cases as the destructor will call it
        #self.log("iter: " + str(self.cikel_nbr) + '\n')
        self.log("NUMBER OF ALL CONSTRAINTS: " + str(self.constraint_count) + '\n')


        self.abh.constraints = self.abh.constraints + constraints
        #self.log(str(self.abh.constraints) + '\n')
        self.log(self.abh.l.conditions(self.abh) + '\n')

        clusters = self.updatedClusters.copy()


        self.master.config(cursor="wait")
        self.master.update()
        self.abh.clusters = self.abh.ABHclustering(self.abh.constraints, self.final_n_of_clusters, clusters)
        self.master.config(cursor="")
        new_keys = []
        for i, cluster in enumerate(self.abh.clusters):
            points = []
            self.abh.clusters[cluster].name = "Cluster" + str(i)
            self.abh.clusters[cluster].clusterId = i
            self.abh.clusters[cluster].dim = len(self.abh.attributes)
            for point in self.abh.clusters[cluster].points:
                points.append(point)
            self.abh.clusters[cluster].points = points
            self.abh.clusters[cluster].centroid = self.abh.clusters[cluster].calculateCentroid()
            new_keys.append(i)
        for key, n_key in zip(self.abh.clusters.keys(), new_keys):
            self.abh.clusters[n_key] = self.abh.clusters.pop(key)
        self.abh.condition_history.append(self.abh.condition)
        self.abh.condition = []
        self.abh.candidates = []

        self.step = 1

        self.cikel_nbr += 1
        self.refresh_cluster_data()
        self.update_display_data()
        self.cikel_label.set("ABHC Cikel: " + str(self.cikel_nbr) + "\n"
                                                                    "-----------------------------------------\n"
                                                                    "Initialization: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Hierarchical clustering: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Determine the number of clusters: Finsihed\n"
                                                                    "-----------------------------------------\n"
                                                                    " Selecting critical example: Ready\n"
                                                                    "-----------------------------------------\n"
                                                                    " Critical example argumentation:\n"
                                                                    "-----------------------------------------\n"
                                                                    " Counter example constraints:\n "
                                                                    "-----------------------------------------\n"
                                                                    "Run ABHC:")

        self.argument_button['state'] = 'disabled'
        self.counter_example_button['state'] = 'disabled'
        self.improve_button['state'] = 'disabled'
    def showEnd(self, event):
        self.cluster_output.see(END)
        self.cluster_output.edit_modified(0)  # IMPORTANT - or <<Modified>> will not be called later.

    def showEndOutput(self, event):
        self.output_field.see(END)
        self.output_field.edit_modified(0)  # IMPORTANT - or <<Modified>> will not be called later.

    def log(self, string):
        self.cluster_output.insert(END, string)
        self.abh.l.log(string)

    def update_display_data(self):
        """
        Updates display data
        :return:
        """
        """
        # Show file name
        o = "===========================\n"
        o += "===========" + str(self.data_filename) + "============\n"
        o += "===========================\n"
        details_label = StringVar()
        details_label.set(o)
        Label(self.frame, textvariable=details_label, justify=LEFT).pack(side=TOP, anchor=W)

        # show nmi and ari
        o = ""
        if self.abh.NMI is not None and self.abh.ARI is not None:
            o += "NMI: " + str(self.abh.NMI) + "\nARI: " + str(self.abh.ARI) + "\n"
            self.log(o)
        details_label = StringVar()
        details_label.set(o)
        Label(self.frame, textvariable=details_label, justify=LEFT).pack(side=TOP, anchor=W)
        """
        # show cluster centroids
        if len(self.labels) > 0:
            for label in self.labels:
                label.destroy()

        for cluster in self.abh.clusters:
            if self.abh.clusters[cluster].name is not None:
                o = str(self.abh.clusters[cluster].name)
            else:
                o = "Cluster "+str(self.abh.clusters[cluster])
            self.abh.clusters[cluster].points.sort(key=lambda x: int(x.reference), reverse=False)
            details_label = StringVar()
            details_label.set(o)
            l = Label(self.info_frame, textvariable=details_label, font="Helvetica 10 bold italic", justify=LEFT, cursor='hand1')
            l.pack(side=TOP, anchor=W)
            # On centroid click, open window with points
            l.bind("<Button-1>", lambda e, cluster=cluster: self.open_cluster_window(cluster))
            self.labels.append(l)
    def open_cluster_window(self, cluster):
        if hasattr(self, 'top'):
            self.top.destroy()
        top = self.top = Toplevel(self.master)

        self.top.point_frame = None
        o = "Cluster" + str(self.abh.clusters[cluster])
        f = self.top.list_frame = Frame(self.top)
        f.pack(expand=1, fill='both', side=LEFT, anchor=NW)


        scrollbar = Scrollbar(f)
        scrollbar.pack( side = RIGHT, fill=Y )


        self.cluster_listbox = Listbox(f, width=30, yscrollcommand = scrollbar.set)
        self.cluster_listbox.pack(expand=1, fill='both', side=TOP, anchor=W)
        self.cluster_listbox.bind("<<ListboxSelect>>", lambda e,cluster=cluster:self.display_point_data(e,cluster))

        self.cluster_listbox.delete(0, END)
        self.cluster_listbox.insert(END, o)
        #self.log(self.abh.clusters[cluster].__repr__())

        # dodaj tocke na levo stran
        index = 0
        for p in self.abh.clusters[cluster].points:
            self.cluster_listbox.insert(END, p)
            if int(p.reference) in self.abh.diff:
                self.cluster_listbox.itemconfig((index+1), {'bg':'red'})
            index +=1

            #self.log(p.__repr__())
        scrollbar.config( command = self.cluster_listbox.yview )
        self.cluster_listbox.select_set(0) #This only sets focus on the first item.
        self.cluster_listbox.event_generate("<<ListboxSelect>>")

    def display_point_data(self, e, cluster):
        index = int(self.cluster_listbox.curselection()[0])
                # Frame with cikel info
        if self.top.point_frame:
            self.top.point_frame.pack_forget()
        self.top.point_frame = Frame(self.top)
        self.top.point_frame.pack(fill='both', side=LEFT, anchor="e")

        self.top.point_label = StringVar()

        if index == 0:
            o=self.abh.clusters[cluster].printStats(self.abh)
            print(o)
            #self.log(o)
            self.top.point_label.set(o)
        else:
            value = self.abh.clusters[cluster].points[index-1]
            o='You selected case: "%s" \n' % (value)
            o+=self.abh.l.candidates(self.abh, [value],0,1)
            #self.log(o)
            self.top.point_label.set(o)
        self.top.label = Label(self.top.point_frame, textvariable=self.top.point_label, justify=LEFT,  font="Consolas 11").pack(side=TOP, anchor=W)

    def rename_clusters_auto(self, req_names):
        names = []
        for cluster in self.abh.clusters:
            if self.abh.clusters[cluster].name in req_names:
                break
            num = 0
            name = ""
            print(self.abh.clusters[cluster].purity)
            for c, n in self.abh.clusters[cluster].purity:
                if n > num:
                    num = n
                    name = c
            if name not in names:
                self.abh.clusters[cluster].name = name
                names.append(name)
            else:
                for n in req_names:
                    if n not in names:
                        self.abh.clusters[cluster].name = name
                        names.append(name)
                    else:
                        return self.rename_clusters_auto(req_names)
        for cluster in self.abh.clusters:
            if not self.abh.clusters[cluster].name in req_names:
                self.abh.clusters = self.abh.hierarhicalClustering(self.clustersCopy.copy())
                return self.rename_clusters_auto(req_names)
    def cluster_name_to_int(self, name):
        for i,c in enumerate(self.abh.clusters):
            if self.abh[c].name == name:
                return i
    def bot(self):
        n_clusters = 3
        for t in TEST_CONST:
            self.test_nbr = 0
            while self.test_nbr < 30:
                self.__init__(root)
                self.log("===============|st Testa"+str(self.test_nbr)+"|===========pri st omejitev: "+str(t))
                self.test_nbr +=1
                self.abh.clusters = self.abh.hierarhicalClustering(self.clusters)
                self.initialClusters = self.clustersCopy.copy()
                self.abh.clusters = self.abh.rebuildClusters(self.initialClusters, n_clusters)
                self.refresh_cluster_data()
                first_critical = True
                req_names = ["BAD", "GOOD"]

                while True:
                    print("CIKEL ST: ", self.cikel_nbr, " |", len(self.abh.clusters))
                    if self.cikel_nbr >= 300 or self.constraint_count > t:
                        break
                    if len(self.abh.clusters) > n_clusters:
                        break
                    self.abh.get_candidates()
                    if len(self.abh.candidates) == 0:
                        break
                    if first_critical:
                        test = False
                        self.abh.critical_example.append(self.abh.candidates[0])
                    else:
                        test = True
                        for cand in self.abh.candidates:
                            if self.cluster_name_to_int(cand[0].cheat) != cand[1]:  # ni v pravem clustru
                                self.abh.critical_example.append(cand)
                                test = False
                                break
                    print("Found critical Example",)
                    print(test)
                    print(self.abh.critical_example[-1])
                    if test or self.abh.NMI >= 1:
                        #self.refresh_cluster_data()
                        break
                    self.counter = self.abh.get_pair(self.abh.critical_example[-1][1])
                    counter_list = []
                    print(self.counter[0].cheat, self.counter[0].reference , self.counter[0].silhuette)
                    print(self.abh.critical_example[-1][0].cheat, self.abh.critical_example[-1][0].reference, self.abh.critical_example[-1][0].silhuette)

                    if self.counter[0].cheat == self.abh.critical_example[-1][0].cheat:  #protiprimer je enak kriticnemu primeru
                        counter_list.append({'example': self.counter, 'act': 2})
                    else:
                        counter_list.append({'example': self.counter, 'act': 1})
                    arguments = []
                    """
                    if self.abh.critical_example[-1][0].cheat == 'Iris-setosa':
                        arguments.append(['PedalWidth', '<<<', ''])

                    if self.abh.critical_example[-1][0].cheat == 'Iris-virginica':
                        arguments.append(['PedalWidth', '>>>', ''])

                    if self.abh.critical_example[-1][0].cheat == 'Iris-versicolor':
                        arguments.append(['PedalLength', '>>>', '']) 
                    """
                    if self.abh.critical_example[-1][0].cheat == 'BAD':
                        arguments.append(['net.income', '<<<', ''])

                    if self.abh.critical_example[-1][0].cheat == 'GOOD':
                        arguments.append(['net.income', '>>>', ''])

                    arg_dic = {'point': self.abh.critical_example[-1], 'counter': counter_list,
                               'current_cluster': self.abh.critical_example[-1][1], 'arguments': arguments}
                    print(arg_dic)
                    exit
                    self.abh.condition.append(arg_dic)

                    # get counters
                    for condition in self.abh.condition:
                        #self.log("Counter examples for critical example: Example " + str(condition["point"][0].reference) + "\n")

                        # open popup with data and ask to argument counter example
                        counters = self.abh.counter_example(condition)
                        for counter in counters:
                            #print(counter[0].cheat)
                            #print(self.abh.critical_example[-1][0].cheat)
                            if self.abh.critical_example[-1][0].cheat == counter[0].cheat:
                                act = 2
                            else:
                                act = 1
                            counter_list = [{'example': counter, 'act': act}]
                            condition["counter"] = condition["counter"] + counter_list
                    self.improve_hc()
                    self.initialClusters = self.clustersCopy.copy()
                    self.abh.clusters = self.abh.rebuildClusters(self.initialClusters, n_clusters)
                    self.refresh_cluster_data()
                # Write report
                f = open('report1.txt', 'a')
                o = str(self.test_nbr)+': Cikles: '+str(self.cikel_nbr)+' constraints: '+str((self.constraint_count))+' ARI: '+str(self.abh.ARI)+' NMI: '+str(self.abh.NMI)
                o=o+" first_critical: "+str(first_critical)+" test: "+str(t)+"\n"

                f.write(o)  # python will convert \n to os.linesep
                f.close()  # you can omit in most cases as the destructor will call it

    def bot2(self):
        n_clusters = 2
        first = True
        for t in TEST_CONST:
            self.test_nbr = 0
            while self.test_nbr < 30:
                self.__init__(root)
                self.log("===============|st Testa" + str(self.test_nbr) + "|===========pri st omejitev: " + str(t))
                self.log("st atributov: " +str(len(self.abh.attributes)))
                self.test_nbr += 1
                self.abh.clusters = self.abh.hierarhicalClustering(self.clusters)
                self.initialClusters = self.clustersCopy.copy()
                self.abh.clusters = self.abh.rebuildClusters(self.initialClusters, n_clusters)
                self.abh.attributes.append('equity.ratio')
                self.abh.attributes.append('current.ratio')
                self.abh.attributes.append('debt.to.total.assets.ratio')
                self.abh.attributes.append('net.debt/EBITDA')
                self.abh.attributes.append('ROA')
                self.abh.attributes.append('TIE')
                self.abh.dim = 31
                self.abh.distances = {}
                for cluster in self.abh.clusters:
                    for point in self.abh.clusters[cluster].points:
                        point.coords.append(round(point.coords[self.abh.attributes.index('equity')] /
                                                  (point.coords[self.abh.attributes.index('lt.assets')]+point.coords[self.abh.attributes.index('st.assets')]), 2)) #equity ratio
                        point.coords.append(round(point.coords[self.abh.attributes.index('st.assets')] /
                                                  point.coords[self.abh.attributes.index('st.liabilities')], 2)) #current ratio
                        point.coords.append(round(point.coords[self.abh.attributes.index('total.oper.liabilities')] /
                                                  (point.coords[self.abh.attributes.index('lt.assets')]+point.coords[self.abh.attributes.index('st.assets')]), 2)) #debt to total assets ratio
                        point.coords.append(round((point.coords[self.abh.attributes.index('debt')] - point.coords[self.abh.attributes.index('cash')])/
                                                  point.coords[self.abh.attributes.index('EBITDA')], 2)) #net.debt/EBITDA
                        EBIT = point.coords[self.abh.attributes.index('EBIT')]
                        assets = point.coords[self.abh.attributes.index('assets')]
                        interest = point.coords[self.abh.attributes.index('interest')]
                        if EBIT == 0 or assets == 0:
                            point.coords.append(0)
                        else:
                            point.coords.append(round(point.coords[self.abh.attributes.index('EBIT')] /
                                                  point.coords[self.abh.attributes.index('assets')], 2) * 100) #ROA
                        if EBIT == 0 or interest == 0:
                            point.coords.append(0)
                        else:
                            point.coords.append(round(point.coords[self.abh.attributes.index('EBIT')] /
                                                  point.coords[self.abh.attributes.index('interest')], 2)) #TIE



                        point.n = 31
                    self.abh.clusters[cluster].dim = 31
                    self.abh.clusters[cluster].centroid = self.abh.clusters[cluster].calculateCentroid()
                self.refresh_cluster_data()
                first_critical = True
                req_names = ["BAD", "GOOD"]
                while True:
                    print("CIKEL ST: ", self.cikel_nbr, " |", len(self.abh.clusters))
                    if self.cikel_nbr >= 300 or self.constraint_count > t:
                        print("++++++ stevilo omejitev je cez mejo", self.constraint_count, t)
                        break
                    if len(self.abh.clusters) > n_clusters:
                        print("+++++++ st clustrov > n_ clusters", len(self.abh.clusters))
                        break
                    self.abh.get_candidates()
                    print("ST KRITICNIH PRIMEROV: ", len(self.abh.candidates))
                    if len(self.abh.candidates) == 0:
                        print("+++++ zmankalo je kandidatov")
                        break
                    if first_critical:
                        test = False
                        self.abh.critical_example.append(self.abh.candidates[0])
                    else:
                        test = True
                        print("+++++++", self.abh.candidates)
                        for cand in self.abh.candidates:
                            print(cand[0].cheat, self.cluster_name_to_int(cand[0].cheat) , cand[1])
                            if self.cluster_name_to_int(cand[0].cheat) != cand[1]:  # ni v pravem clustru
                                self.abh.critical_example.append(cand)
                                test = False
                                break
                    print("Found critical Example", )
                    print(test)
                    if test or self.abh.NMI >= 1:
                        # self.refresh_cluster_data()
                        print("+++++KONEC: ",test, self.abh.NMI)
                        break
                    self.counter = self.abh.get_pair(self.abh.critical_example[-1][1])
                    counter_list = []

                    if self.counter[0].cheat == self.abh.critical_example[-1][0].cheat:  # protiprimer je enak kriticnemu primeru
                        counter_list.append({'example': self.counter, 'act': 2})
                    else:
                        counter_list.append({'example': self.counter, 'act': 1})
                    arguments = []
                    if self.abh.critical_example[-1][0].cheat == 'BAD':
                        arguments.append(['net.income', '<', '122'])
                        arguments.append(['equity.ratio' , '<' ,'0'])
                        arguments.append(['current.ratio' , '<' ,'0'])

                    if self.abh.critical_example[-1][0].cheat == 'GOOD':
                        arguments.append(['net.income', '>', '122'])
                        arguments.append(['equity.ratio','>','0'])
                        arguments.append(['current.ratio','>','0'])

                    arg_dic = {'point': self.abh.critical_example[-1], 'counter': counter_list,
                               'current_cluster': self.abh.critical_example[-1][1], 'arguments': arguments}
                    self.abh.condition.append(arg_dic)
                    # get counters
                    for condition in self.abh.condition:
                        self.log(
                            "Counter examples for critical example: Example " + str(
                                condition["point"][0].reference) + "\n")

                        counters = self.abh.counter_example(condition)
                        for counter in counters:
                            if self.abh.critical_example[-1][0].cheat == counter[0].cheat:
                                act = 2
                            else:
                                act = 1
                            counter_list = [{'example': counter, 'act': act}]
                            condition["counter"] = condition["counter"] + counter_list

                    self.improve_hc()

                    self.abh.clusters = self.abh.rebuildClusters(self.updatedClusters.copy(), n_clusters)

                    self.refresh_cluster_data()
                # Write report
                f = open('report_150cons.txt', 'a')
                o = str(t) + '| '+ str(self.test_nbr) + ': Cikles: ' + str(self.cikel_nbr) + ' constraints: ' + str(
                    (self.constraint_count)) + ' ARI: ' + str(self.abh.ARI) + ' NMI: ' + str(self.abh.NMI)
                o = o + " first_critical: " + str(first_critical) + " test: " + str(test) + "\n"

                f.write(o)  # python will convert \n to os.linesep
                f.close()  # you can omit in most cases as the destructor will call it



root = Tk()
app = App(root)
root.mainloop()
