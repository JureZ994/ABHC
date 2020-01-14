#-*- coding: utf-8 -*-
# __author__ = 'peter'
from tkinter import *
from ABHclustering import ABHclustering
from cluster import Cluster, Point
import sys
import csv
from CT import CT
RUN_SET = "IRIS"
#RUN_SET = "BONITETE"
LOG_NAME = "ABHC.log"

class App:
    def __init__(self, master):
        self.master = master
        self.helper = None
        self.constraint_count = 0
        self.attributes = []
        menu_frame = Frame()
        menu_frame.pack(fill=X,side=TOP)
        self.menu_frame = menu_frame
        self.step = 0

        self.autorun = None

        frame = Frame(bg="red")
        frame.pack(expand=1,fill='both',side=RIGHT)
        self.frame = frame

        cikel_frame = Frame()
        cikel_frame.pack(expand=1,fill=X,side=TOP)
        self.cikel_frame = cikel_frame

        info_frame = Frame()
        info_frame.pack(expand=1,fill=X,side=TOP)
        self.info_frame = info_frame


        self.cikel_label = StringVar()
        Label(self.cikel_frame, textvariable=self.cikel_label, justify=LEFT).pack(side=TOP, anchor=W)

        self.info_label = StringVar()
        self.info_label.set("===================================\nCritical Examples:\n")
        Label(self.info_frame, textvariable=self.info_label, justify=LEFT).pack(side=TOP, anchor=W)

        self.listbox = Listbox(self.info_frame, width="50")
        self.listbox.pack(side=TOP, anchor=W)
        self.listbox.bind("<Double-Button-1>", self.display_critical_data)

        self.details_label = StringVar()
        self.details_label.set("")
        Label(self.info_frame, textvariable=self.details_label, justify=LEFT).pack(side=TOP, anchor=W)

        #self.output_field = Text(master, height=10, width=80)
        #self.output_field.pack(side=LEFT, fill=BOTH, expand=1)

        self.cluster_output = Text(frame, height=1)
        self.cluster_output.pack(side=LEFT, fill='both', expand=1, anchor=W)
        self.cluster_output_scroll = Scrollbar(frame)
        self.cluster_output_scroll.pack(side=RIGHT, fill=Y)
        self.cluster_output_scroll.config(command=self.cluster_output.yview)
        self.cluster_output.config(yscrollcommand=self.cluster_output_scroll.set)
        self.cluster_output.bind('<<Modified>>',self.showEnd)
        #self.output_field.bind('<<Modified>>',self.showEndOutput)

        #master.geometry('800x550')
        self.final_n_of_clusters = None
        self.my_clusters = None
        self.n_clusters_index_start=0
        self.n_clusters_index_end=5
        self.critical_index_start=0
        self.critical_index_end=5
        self.cikel_nbr = 0
        self.args = []
        self.additional_attributes=[]

        if RUN_SET == "IRIS":
            self.init_iris()
        elif RUN_SET == "BONITETE":
            self.init_bonitete()


        self.hc_button = Button(menu_frame, text="Start Hierarhical Clustering", command=self.hierarhicalClustering)
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
        self.cikel_label.set("ABHC Cikel: "+str(self.cikel_nbr)+"\n"
                             "-----------------------------------------\n"
                             "Initialization: Finished\n"
                             "-----------------------------------------\n"
                             "Hierarhical clustering: Ready\n"
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
        for i,c in enumerate(self.abh.condition):
            self.listbox.insert(END, "Example "+str(c['point'][0].reference))
            self.map_point_name_to_point["Example "+str(c['point'][0].reference)] = c['point']

    def display_critical_data(self, ok):
        now = self.listbox.curselection()[0]
        name = self.listbox.get(ACTIVE)
        o=name+" "+str(self.abh.condition[now]['point'][0])+"\n"
        o+="Old cluster: "+str(self.abh.clusters[self.abh.condition[now]['current_cluster']])+"\n"
        o+="New cluster: "+str(self.abh.clusters[self.abh.condition[now]['target_cluster']])+"\n"
        o+="Argument:\n"
        cond=self.abh.condition[now]['arguments']
        condn=len(cond)
        for i,arg in enumerate(cond):
            o+=arg[0]+" "+arg[1]
            if arg[2]:
                o+=arg[2]
            if i+1 != condn:
                o+=" AND "
        self.details_label.set(o)


    def sum_attributes(self, points_list, class_index):
        return_val = [0]*len(points_list[0])
        for p in points_list:
            return_val = [float(x) + float(y) for i, (x, y) in enumerate(zip(return_val, p)) ]
        return [x / len(points_list) for x in return_val]

    def sum_attributes_max(self, points_list, class_index):
        return_val = [0]*len(points_list[0])
        for p in points_list:
            return_val = [float(y) if float(y) > float(x)  else float(x) for i, (x, y) in enumerate(zip(return_val, p)) ]
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
        #read data
        input = open('iris.data', 'r')
        reader = csv.reader(input)
        self.attributes = next(reader)
        points = []
        data = [d[:] for d in reader]

        self.clusters = {}
        self.clustersCopy = {}
        #Create points from data
        for i, line in enumerate(data):
            points.append(Point([float((line[x])) for x in range(0, len(line[:4]))], cheat=line[4], reference=i))
            cluster = Cluster(i)
            cluster.points.append(Point([float((line[x])) for x in range(0, len(line[:4]))], cheat=line[4], reference=i))
            cluster.primeri.append(i)
            self.clusters.update({i: cluster})
            self.clustersCopy.update({i: cluster})
        self.points = points
        self.my_clusters = self.clusters
        self.abh = ABHclustering(self.points, points, self.clusters, self.attributes, candidates=None)
        self.abh.dim = 4
        self.abh.l = CT(LOG_NAME, RUN_SET)
        self.log(self.abh.l.dataset(self.abh))
        self.cikel_nbr +=1
    def init_bonitete(self):
        """
        Finding critical examples from the bonitete dataset
        """

    def hierarhicalClustering(self):
        if self.step < 1:
            self.step = 1
        self.log("Hierarhical clustering in cikel: " + str(self.cikel_nbr) + " with constraints: " + str(
                self.constraint_count) + "\n")
        self.master.config(cursor="wait")
        self.master.update()
        self.abh.clusters = self.abh.hierarhicalClustering(self.clusters)
        self.master.config(cursor="")
        self.hc_button.destroy()


        if self.plot_button is None:
            self.plot_button = Button(self.menu_frame, text="Plot", command=self.plot2D)
            self.plot_button.pack(side=LEFT)
        if self.determine_button is None:
            self.determine_button = Button(self.menu_frame, text="Determine number of clusters", command=self.determine_clusters)
            self.determine_button.pack(side=LEFT)
            self.cikel_label.set("ABHC Cikel: " + str(self.cikel_nbr) + "\n"
                                                                        "-----------------------------------------\n"
                                                                        "Initialization: Finished\n"
                                                                        "-----------------------------------------\n"
                                                                        "Hierarhical clustering: Finished\n"
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

            return # module doesn't exist, deal with it.

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
        '''
        
        self.plot_popup()
        self.master.wait_window(self.top)        
        '''

    def plot_popup(self):
        top = self.top = Toplevel(self.master)
        try:
            import numpy as np
            import math
            import matplotlib.pyplot as plt
            from scipy.cluster.hierarchy import dendrogram

        except ImportError as e:
            self.cluster_output.insert("Missing packages, cluster plots not supported....\n")
            self.cluster_output.insert(e)

            return # module doesn't exist, deal with it.

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

        b = Button(top, text="OK", command=self.plot_close).pack()

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
            self.log(self.abh.l.clusters(self.abh))
            self.helper = None
        if self.rename_button is None:
            self.rename_button = Button(self.menu_frame, text="Rename clusters", command=self.rename_clusters)
            self.rename_button.pack(side=LEFT)
        if self.get_criticals_button is None:
            self.get_criticals_button = Button(self.menu_frame, text="Select critical example", command=self.get_criticals)
            self.get_criticals_button.pack(side=LEFT)

        self.cikel_label.set("ABHC Cikel: " + str(self.cikel_nbr) + "\n"
                                                                    "-----------------------------------------\n"
                                                                    "Initialization: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Hierarhical clustering: Finished\n"
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
        self.log("============================================\n")
        self.log("============================================\n")
        self.log("============================================\n")
        self.log(self.abh.l.clusters(self.abh))
        f = open('points_Cikel' + str(self.cikel_nbr) + ".txt", 'w')
        for cluster in self.abh.clusters:
            f.write(str(cluster) + " " + self.abh.clusters[cluster].name + "\n")
            for p in self.abh.clusters[cluster].points:
                f.write(str(p) + " " + p.cheat + "\n")
        f.close()

    def rename_clusters(self):
        self.log("Renaming clusters...\n")
        for cluster in self.abh.clusters:
            nbr = 5 if len(self.abh.clusters[cluster].points) >= 5 else len(self.abh.clusters[cluster].points)

            print_list = [self.abh.clusters[cluster].centroid] + self.abh.clusters[cluster].points[0:nbr]

            self.rename_popup(self.abh.l.candidates(self.abh, print_list, 0, nbr), self.abh.clusters[cluster].name)
            self.master.wait_window(self.top)
            if self.helper != None:
                self.abh.clusters[cluster].name = self.helper
                self.helper = None
        self.refresh_cluster_data()
        # name = raw_input("Clusters new name:")
        # self.l.log('\n')
        # cluster.name = name

    def get_criticals(self):
        if self.abh.candidates == None or len(self.abh.candidates) == 0:
            self.log("Finding critical examples...\n")
            self.abh.get_candidates()
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

        self.log(
            "We picked Example " + str(self.abh.critical_example[-1][0].reference) + " now we need to argument it.\n")
        if self.argument_button is None:
            self.argument_button = Button(self.menu_frame, text="Argument constraint", command=self.argument_steps)
            self.argument_button.pack(side=LEFT)

        self.cikel_label.set("ABHC Cikel: " + str(self.cikel_nbr) + "\n"
                                                                    "-----------------------------------------\n"
                                                                    "Initialization: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Hierarhical clustering: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Determine the number of clusters\n"
                                                                    "-----------------------------------------\n"
                                                                    "Selecting critical example: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Critical example argumentation: Ready\n"
                                                                    "-----------------------------------------\n"
                                                                    "Counter example constraints:\n "
                                                                    "-----------------------------------------\n"
                                                                    "Run ABHC:")

    def argument_steps(self):
        if len(self.listbox.curselection()) != 0:
            critical_point = self.map_point_name_to_point[self.listbox.get(self.listbox.curselection())][0]
        else:
            critical_point = self.abh.critical_example[-1][0]
        # Informative print of clusters - reference points
        # We present clusters to expert for better understanding
        for cluster in self.abh.clusters:
            self.log("Cluster " + str(self.abh.clusters[cluster].centroid) + ":  distance: " + str(
                critical_point.getDistance(self.abh.clusters[cluster].centroid)) + "\n")

        # We present the example to the expert

        self.log("Critical Example is in cluster: " + self.abh.clusters[self.abh.critical_example[-1][1]].name + "\n")
        self.get_argument_with_pair_popup()
        self.cikel_label.set("ABHC Cikel: " + str(self.cikel_nbr) + "\n"
                                                                    "-----------------------------------------\n"
                                                                    "Initialization: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Hierarhical clustering: Finished\n"
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
        self.set_labels()
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

            self.log(arg_str)
        # argument collected, allow fetching counter examples and argumenting them
        if self.counter_example_button is None:
            self.counter_example_button = Button(self.menu_frame, text="Argument counter examples",
                                                 command=self.counter_steps)
            self.counter_example_button.pack(side=LEFT)

        self.cikel_label.set("ABHC Cikel: " + str(self.cikel_nbr) + "\n"
                                                                    "-----------------------------------------\n"
                                                                    "Initialization: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Hierarhical clustering: Finished\n"
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
        print(self.abh.condition)
        for condition in self.abh.condition:
            self.log("Counter examples for critical example: Example " + str(condition["point"][0].reference) + "\n")

            # open popup with data and ask to argument counter example
            counters = self.abh.counter_example(condition)
            print("STEVILO PROTIPRIMEROV: " , len(counters))
            for counter in counters:
                #TODO: dodaj gumb za dodajanje novega atributa tudi tukaj
                self.counter_argument_popup(condition, counter)
                self.cikel_label.set("ABHC Cikel: " + str(self.cikel_nbr) + "\n"
                                                                            "-----------------------------------------\n"
                                                                            "Initialization: Finished\n"
                                                                            "-----------------------------------------\n"
                                                                            "Hierarhical clustering: Finished\n"
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
                """
                if self.helper and  self.helper['arguments']:
                    condition["arguments"] = condition["arguments"] + self.helper['arguments']
                """

                if self.helper and self.helper['counter']:
                    condition["counter"] = condition["counter"] + self.helper['counter']
                self.helper = None
            if self.step < 4:
                self.step = 4

        if self.improve_button is None:
            self.improve_button = Button(self.menu_frame, text="Hierarhical clustering", command=self.improve_hc)
            self.improve_button.pack(side=LEFT)
        self.cikel_label.set("ABHC Cikel: " + str(self.cikel_nbr) + "\n"
                                                                    "-----------------------------------------\n"
                                                                    "Initialization: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Hierarhical clustering: Finished\n"
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

        self.log(text)
        text += self.abh.l.candidates(self.abh, self.abh.candidates[start:end], start, end)
        self.log(text)

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

    def get_argument_with_pair_popup(self, argumenti=None, link=None):
        if len(self.listbox.curselection()) != 0:
            critical_point = self.map_point_name_to_point[self.listbox.get(self.listbox.curselection())]
            critical_point_target = self.map_point_name_to_point[self.listbox.get(self.listbox.curselection())][1]
        else:
            critical_point = self.abh.critical_example[-1]
            critical_point_target = self.abh.critical_example[-1][1]

        self.counter = self.abh.get_pair(critical_point_target)
       # print("counter: ", self.counter)
        print_list = [critical_point, self.counter] + [self.abh.clusters[c].represent() for c in self.abh.clusters]
        # Ask if data is correct


        text = self.abh.l.candidates(self.abh, print_list, 0, len(print_list))

        top = self.top = Toplevel(self.master)

        frame = Frame(self.top)
        frame.pack(fill=X, anchor=W)

        output_field = Text(top, height=30, width=120)
        output_field.insert(END, text)
        output_field.pack(fill=BOTH, expand=1)

        b = Button(top, text="OK", command=self.argument_pair_close).pack()
        b2 = Button(top, text="AND", command=self.argument_new).pack()
        b3 = Button(top, text="ADD ATTRIBUTE", command=self.attribute_new).pack()

        str_o = "Example " + str(critical_point[0].reference) + " has "

        self.args = [self.create_arg_form(True, argumenti)]

        label = Label(self.support_frame, text="THEN", font="Helvetica 14 bold italic")
        label.pack(side=LEFT)

        self.counter_act = StringVar()
        self.choice_string = ['Nothing', 'Cannot-link', 'Must-link']
        if link is None:
            self.counter_act.set(self.choice_string[0])  # default value
        else:
            self.counter_act.set(self.choice_string[self.choice_string.index(link)])
        w2 = OptionMenu(self.support_frame, self.counter_act, *self.choice_string).pack(side=LEFT)

    def argument_new(self):
        self.args.append(self.create_arg_form())
    def attribute_new(self):
        self.create_attribute_form()

    def create_arg_form(self, if_label=False, argumenti = None):
        self.support_frame = frame = Frame(self.top)
        frame.pack(fill=X, side=TOP, anchor=S)
        if argumenti is not None:
            for i in range(0, len(argumenti)):
                if i== 0:
                    label = Label(frame, text="IF", font="Helvetica 14 bold italic")
                    label.pack(side=LEFT)
                else:
                    label = Label(frame, text="AND", font="Helvetica 14 bold italic")
                    label.pack(side=LEFT)
                self.atr = StringVar()
                choice = [i for i in self.abh.attributes]
                if argumenti[i][0] is not None:
                    self.atr.set(choice[choice.index(argumenti[i][0])])
                else:
                    self.atr.set(choice[0])  # default value
                w1 = OptionMenu(frame, self.atr, *choice).pack(side=LEFT, anchor=W)

                self.op = StringVar()
                choice = [i for i in self.abh.def_operators()]
                if argumenti[i][1] is not None:
                    self.op.set(choice[choice.index(argumenti[i][1])])
                else:
                    self.op.set(choice[0])  # default value
                w2 = OptionMenu(frame, self.op, *choice).pack(side=LEFT, anchor=W)

                self.e = Entry(frame)
                self.e.pack(side=LEFT, anchor=W)
                callback = lambda *args: (self.e.configure(
                    state='disabled') if self.op.get() == '<<<' or self.op.get() == '>>>' else self.e.configure(
                    state='normal'))
                if argumenti[i][2] is not None:
                    self.e.insert(0, argumenti[i][2])
                self.op.trace("w", callback)
        else:
            if not if_label:
                label = Label(frame, text="AND", font="Helvetica 14 bold italic")
                label.pack(side=LEFT)
            elif if_label == True:
                label = Label(frame, text="IF", font="Helvetica 14 bold italic")
                label.pack(side=LEFT)
            self.atr = StringVar()
            choice = [i for i in self.abh.attributes]
            self.atr.set(choice[0])  # default value
            w1 = OptionMenu(frame, self.atr, *choice).pack(side=LEFT, anchor=W)

            self.op = StringVar()
            choice = [i for i in self.abh.def_operators()]
            self.op.set(choice[0])  # default value
            w2 = OptionMenu(frame, self.op, *choice).pack(side=LEFT, anchor=W)

            self.e = Entry(frame)
            self.e.pack(side=LEFT, anchor=W)
            callback = lambda *args: (self.e.configure(state='disabled') if self.op.get() == '<<<' or self.op.get() == '>>>' else self.e.configure(state='normal'))
            self.op.trace("w", callback)

        return [self.atr, self.op, self.e]
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

        self.b4 = Button(frame, text="CREATE ATTRIBUTE", command=self.update_points).pack(side=RIGHT)

    def update_points(self):
        new_name = self.f.get()
        idAtr1 = self.abh.attributes.index(self.atr1.get())
        idAtr2 = self.abh.attributes.index(self.atr2.get())
        operand = self.op1.get()
        self.abh.attributes.append(new_name)
        for cluster in self.abh.clusters:
            for point in self.abh.clusters[cluster].points:
                if operand == '/':
                    point.coords.append(round(point.coords[idAtr1]/point.coords[idAtr2],2))
                elif operand == '*':
                    point.coords.append(round(point.coords[idAtr1]* point.coords[idAtr2], 2))
                elif operand == '+':
                    point.coords.append(round(point.coords[idAtr1] + point.coords[idAtr2], 2))
                point.n = point.n + 1
            self.abh.clusters[cluster].dim = self.abh.clusters[cluster].dim + 1
            self.abh.clusters[cluster].centroid = self.abh.clusters[cluster].calculateCentroid()

        argumenti = []
        for i in range(0, len(self.args)):
            argumenti.append([self.args[i][0].get(), self.args[i][1].get(), self.args[i][2].get()])
        link = self.counter_act.get()
        self.top.destroy()
        self.get_argument_with_pair_popup(argumenti,link)
        self.top.destroy()


    def argument_pair_close(self):
        rule_dic = {}
        if len(self.listbox.curselection()) != 0:
            critical_point = self.map_point_name_to_point[self.listbox.get(self.listbox.curselection())]
        else:
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
        self.log(str(rule_dic))
        # We have our condition. find counter examples
        self.helper = rule_dic
        self.top.destroy()

    def counter_argument_popup(self, condition, counter):
        critical_point = condition["point"][0]
        self.counter = counter
        print_list = [condition["point"], counter] + [self.abh.clusters[c].represent() for i, c in enumerate(self.abh.clusters)]

        self.log("Fetching argument for counter example\n")

        # Ask if data is correct

        text = self.abh.l.candidates(self.abh, print_list, 0, len(print_list))
        self.log(text)

        top = self.top = Toplevel(self.master)

        frame = Frame(self.top)
        frame.pack(fill=X, anchor=W)

        output_field = Text(top, height=30, width=120)
        output_field.insert(END, text)
        output_field.pack(fill=BOTH, expand=1)

        b = Button(top, text="OK", command=self.ce_argument_close).pack()
        b2 = Button(top, text="AND", command=self.argument_new).pack()

        if "target_cluster" in condition:
            str_o = "Example " + str(counter[0].reference) + " has "

            for x in condition["arguments"]:
                str_o += str(x[0]) + " " + str(x[1])
                if x[2] != None or x[2] != "":
                    str_o += str(x[2])

            str_o += " and is not in cluster: " + str(self.abh.clusters[int(condition["target_cluster"])].name) + "\n"
        else:
            str_o = "Example " + str(counter[0].reference) + " has "
            for x in condition["arguments"]:
                str_o += str(x[0]) + " " + str(x[1])
                if x[2] != None or x[2] != "":
                    str_o += str(x[2])
            str_o = "Do these two examples fit in the same cluster?"

        if "target_cluster" in condition:
            str_o += "We want example " + str(critical_point.reference) + "  in cluster: " + str(
                self.abh.clusters[int(condition["target_cluster"])].name) + " because it has"
        label = Label(frame, text=str_o, font="Helvetica 14 bold italic")
        label.pack(side=LEFT)
        # self.args=[]
        # self.args = [self.create_arg_form(False)]
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

        self.log(str(rule_dic))
        # We have our condition. find counter examples
        self.helper = rule_dic
        self.top.destroy()

    def improve_hc(self):
        # try next critical if none
        self.cikel_label.set("ABHC Cikel: " + str(self.cikel_nbr) + "\n"
                                                                    "-----------------------------------------\n"
                                                                    "Initialization: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Hierarhical clustering: Finished\n"
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

        self.log("Writing constraints from arguments...\n")
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
                        constraints.append((constraint))
                    elif counter['act'] == 2 and counter['example'] != None:
                        constraint['must-link'] = counter['example']
                        constraints.append((constraint))
                    f.write(str(constraint) + '\n')  # python will convert \n to os.linesep
                    self.constraint_count += 1
            f.write("NUMBER OF CONSTRAINTS: " + str(len(constraints)) + '\n')  # python will convert \n to os.linesep
            f.write("NUMBER OF ALL CONSTRAINTS: " + str(
                self.constraint_count) + '\n')  # python will convert \n to os.linesep
        f.close()  # you can omit in most cases as the destructor will call it
        #self.log("iter: " + str(self.cikel_nbr) + '\n')
        self.log("NUMBER OF ALL CONSTRAINTS: " + str(self.constraint_count) + '\n')
        self.abh.constraints = self.abh.constraints + constraints
        self.log(str(self.abh.constraints)+'\n')
        self.log(self.abh.l.conditions(self.abh))

        clusters = self.clustersCopy.copy()
        self.abh.clusters = self.abh.ABHclustering(self.abh.constraints,self.final_n_of_clusters, clusters )
        new_keys = []
        for i,cluster in enumerate(self.abh.clusters):
            points = []
            self.abh.clusters[cluster].name = "Cluster"+str(i)
            self.abh.clusters[cluster].clusterId = i
            self.abh.clusters[cluster].dim = len(self.abh.attributes)
            for point in self.abh.clusters[cluster].points:
                points.append(point)
            self.abh.clusters[cluster].points = points
            self.abh.clusters[cluster].centroid = self.abh.clusters[cluster].calculateCentroid()
            new_keys.append(i)
        for key, n_key in zip(self.abh.clusters.keys(), new_keys):
            self.abh.clusters[n_key] = self.abh.clusters.pop(key)
        self.refresh_cluster_data()
        self.abh.condition_history.append(self.abh.condition)
        self.abh.condition = []
        self.abh.candidates = []
        self.improve_button = None

        self.step = 1

        self.cikel_nbr += 1
        self.cikel_label.set("ABHC Cikel: " + str(self.cikel_nbr) + "\n"
                                                                    "-----------------------------------------\n"
                                                                    "Initialization: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Hierarhical clustering: Finished\n"
                                                                    "-----------------------------------------\n"
                                                                    "Determine the number of clusters: Ready\n"
                                                                    "-----------------------------------------\n"
                                                                    " Selecting critical example:\n"
                                                                    "-----------------------------------------\n"
                                                                    " Critical example argumentation:\n"
                                                                    "-----------------------------------------\n"
                                                                    " Counter example constraints:\n "
                                                                    "-----------------------------------------\n"
                                                                    "Run ABHC:")
        # self.abk.plot_clusters2d()

    def showEnd(self, event):
        self.cluster_output.see(END)
        self.cluster_output.edit_modified(0)  # IMPORTANT - or <<Modified>> will not be called later.

    def showEndOutput(self, event):
        self.output_field.see(END)
        self.output_field.edit_modified(0)  # IMPORTANT - or <<Modified>> will not be called later.

    def log(self, string):
        self.cluster_output.insert(END, string)
        self.abh.l.log(string)

    def cluster_name_to_int(self, name):
        for i, c in enumerate(self.abh.clusters):
            if c.name == name:
                return i

    def rename_clusters_auto(self, req_names):
        names = []
        for cluster in self.abh.clusters:
            if self.abh.clusters[cluster].name in req_names:
                break
            num = 0
            name = ""
            for c, n in self.abh.clusters[cluster].purity.items():
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
                self.hierarhicalClustering()
                return self.rename_clusters_auto(req_names)



    def apply_additional_attributes(self):
        new_points = []
        attr_to_add = []
        for add_atr in self.additional_attributes:
            if add_atr.name in  self.attributes:
                self.log("Attribute already set")
                continue
            attr_to_add.append(add_atr)
            self.attributes.append(add_atr.name)

        for point in self.points:
            new_coords = point.coords
            for add_atr in attr_to_add:
                new_coords.append(add_atr.get_value_for_point(point.coords))
            new_points.append(Point(new_coords,cheat = point.cheat, reference = point.reference))
        self.points = new_points
        self.abh.points = new_points


root = Tk()
app = App(root)
root.mainloop()