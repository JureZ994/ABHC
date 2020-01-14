#from __future__ import print_function
from tabulate import tabulate
from cluster import Point
from collections import defaultdict
class CT:
    def __init__(self, file="new.log", name=None):
        self.file = file
        #self.f = open(file,"a")
        self.name = name
        f1=open(self.file, 'w+')
        f1.write(self.name+"\n")
        f1.close()
        #logging.basicConfig(filename='new.log',level=print)
        #logging.getLogger().addHandler(logging.StreamHandler())

    def log(self, string, end="\n"):
        f1=open(self.file, 'a')
        f1.write(string+"\n")
        f1.close()
        print(string)


    def dataset(self, abk):

        o = "=================================\n"
        o = o + "=========="+str(self.name)+"============\n"
        o = o + "=================================\n"
        o = o + "Number of cases: "+str(len(abk.points))+"\n"
        o = o + "Number of attributes: "+str(len(abk.attributes))+"\n"
        f1=open(self.file, 'a')
        f1.write(o+"\n")
        f1.close()
        return o


    def clusters(self, abh):

        abh.purity()
        o = "NMI: " + str(abh.NMI) + "\nARI: " + str(abh.ARI) + "\n"

        for cluster in abh.clusters:
            o += "======"+str(abh.clusters[cluster].name)+"=======\n"
            o += "Atributes: "
            table = []
            print(len(abh.attributes))
            p = abh.clusters[cluster].centroid
            print(p)
            for i,label in enumerate(abh.attributes):
                row = []
                row.append(label)
                #log(label+"", "")
                p = abh.clusters[cluster].centroid
                #indent = 11 if inx == 0 else 15
                #log(str(p.coords[i]).rjust(indent), "  ")
                if isinstance(p, tuple) :
                    pr = str(p[0].coords[i])
                else:
                    try:
                        pr = str(p.centroid.coords[i])
                    except:
                        pr = str(p.coords[i])

                row.append(pr)
                table.append(row)
            o += tabulate(table)+"\n"


            o += "Number of cases:"+str(len(abh.clusters[cluster].points))+"\n"
            s = float(len(abh.points))
            if len(abh.clusters[cluster].purity.values()) != 0:
                o += "Purity: "+str(max(abh.clusters[cluster].purity.values()))+"/"+ str(len(abh.clusters[cluster].points))+ "\n"
            #self.log(str(cluster.purity))
            #o+="NMI: "+str(cluster.nmi)+"\n"
            abh.clusters[cluster].stats()
            #self.log("Purity: "+str(max(cluster.purity.values())/s)+"\n")

            o += "Radius: "+str(round(abh.clusters[cluster].max_distance, 2))+"\n"
            o += "Average distance: "+str(round(abh.clusters[cluster].avrage_distance, 2))+"\n"
            h=[[],[]]
            for c in abh.clusters:
                h[0].append(abh.clusters[c].name)
                h[1].append(abh.clusters[c].centroid.getDistance(abh.clusters[cluster].centroid))
            o += tabulate([h[1]], headers=h[0])+"\n"
        f1=open(self.file, 'a')
        f1.write(o+"\n")
        f1.close()
        return o

    def conditions(self, abk):
        save = "############CONDITIONS################\n"
        dict_constraints = {"must-link":0, "cannot-link":0, "no-link":0}
        dict_attr_arguments = defaultdict(lambda: defaultdict(int))
        for condition in abk.condition:
            for counter in condition["counter"]:
                if "act" in counter:
                    if counter['act'] == 1:
                        dict_constraints["cannot-link"]+=1
                    if counter['act'] == 2:
                        dict_constraints["must-link"]+=1
                    if counter['act'] == 0:
                        dict_constraints["no-link"]+=1
            for arg in condition['arguments']:
                dict_attr_arguments[arg[0]][arg[1]+" "+arg[2]] +=1


        for key, c in dict_constraints.items():
            save+=str(key)+": "+str(c)+"\n"
        for key, c in dict_attr_arguments.items():
            save+=str(key)+": "+"\n"
            for o_key, o in c.items():
                save+="\t"+str(o_key)+": "+str(o)+"\n"

            #save+=str(dict_attr_arguments)
            #save+=str(dict_constraints)
        return save

    def candidates(self, abk, points, start,end):
        o = ""
        #WE ADD THE HEADER
        h = ["Attribute"]
        for i in range(0,len(points)):
            #log(str(i)+" Example "+str(points[i].reference)+" ".rjust(5), "")
            if isinstance(points[i], tuple):
                h.append(" (Example "+str(points[i][0].reference)+")")
            else:
                try:
                    h.append(" ("+str(points[i].name)+")")
                except:
                    h.append(" (Example "+str(points[i].reference)+")")

        max_string_len = 0
        for i,label in enumerate(abk.attributes):
            if len(label) > max_string_len:
                max_string_len = len(label)
        if len("Silhuette") > max_string_len:
            max_string_len = len("Silhuette")
        if len("Attribute") > max_string_len:
            max_string_len = len("Attribute")
        # WE ADD THE POINTS ATTRIBUTES
        table = [['Index']+list(range(start,end))]
        for i,label in enumerate(abk.attributes):
            row = []
            add_spaces=""
            if len(label) < max_string_len:
                add_spaces = " "*(max_string_len-len(label))
            row.append(label+add_spaces)
            #log(label+"", "")
            fp=points[0]


            for inx ,p in enumerate(points):
                if isinstance(p, tuple) :
                    if len(p[0].coords) <= i:
                        pr = "NAN"
                        atr_dif = "NAN"
                        row.append(pr+" ("+atr_dif+")")
                        continue
                    pr = str(round(p[0].coords[i] , 2))
                    if isinstance(fp, tuple):
                        atr_dif = str(round(p[0].coords[i]-fp[0].coords[i], 2))
                    else:
                        atr_dif = str(round(p[0].coords[i]-fp.coords[i]), 2)

                else:
                    try:
                        if len(p.coords) <= i:
                            pr = "NAN"
                            atr_dif = "NAN"
                            row.append(pr+" ("+atr_dif+")")
                            continue
                        pr = str(p.coords[i])
                        if isinstance(fp, tuple):
                            atr_dif = str(round(p.coords[i]-fp[0].coords[i], 2))
                        else:
                            atr_dif = str(round(p.coords[i]-fp[0].coords[i], 2))
                    except:
                        if len(p.coords) <= i:
                            pr = "NAN"
                            atr_dif = "NAN"
                            row.append(pr+" ("+atr_dif+")")
                            continue
                        pr = str(round(p.coords[i], 2))
                        atr_dif = str(round(p.coords[i]-fp.coords[i], 2))

                row.append(pr+" ("+atr_dif+")")
            table.append(row)

        table2=[]

        #WE ADD THE POINTS CURRENT CLUSTER
        row = ["CLUSTER"]
        for inx, p in enumerate(points):
            if isinstance(p, tuple):
                pr = str(abk.clusters[p[1]].name)
            else:
                try:
                    pr = str(p.cheat)
                except:
                    pr = 'is a cluster'
                if(pr=="None"):
                    pr = 'is a cluster'
            row.append(pr)
        table2.append(row)
        #WE ADD THE POINTS SILHUETTE VALUE
        row = ["Silhuette"]
        for inx, p in enumerate(points):

            if isinstance(p, tuple) and p[0].silhuette:
                row.append(str(round(p[0].silhuette,2)))
            else:
                row.append("0")

        table2.append(row)




        """
        row = ["cheat"]
        for inx, p in enumerate(points):

            if isinstance(p, tuple) and p[0].cheat:
                row.append(str(p[0].cheat))
            else:
                row.append("--")

        table2.append(row)
        """
        for i, cluster in enumerate(abk.clusters):
            row = [str(abk.clusters[cluster].name)+" D"]
            for inx, p in enumerate(points):
                if isinstance(p, tuple) and p[0].distances and len(p[0].distances) > i:
                    row.append(round(p[0].getDistance(abk.clusters[cluster].centroid),2))
                elif isinstance(p, tuple):
                    row.append(round(p[0].getDistance(abk.clusters[cluster].centroid),2))
                elif isinstance(p, Point):
                    row.append(round(p.getDistance(abk.clusters[cluster].centroid),2))
                else:
                    row.append(round(p.centroid.getDistance(abk.clusters[cluster].centroid),2))


            table2.append(row)





        o += tabulate(table, headers=h)+"\n"
        o += tabulate(table2, headers=h)+"\n"
        f1=open(self.file, 'a')
        f1.write(o+"\n")
        f1.close()
        return o
