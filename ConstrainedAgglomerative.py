import csv
from random import randint
import re

from cluster import normalized_mutual_info_score
from cluster import adjusted_rand_score

stevilo_primerov = 150
stevilo_skupin = 2

def generate_link(link_limit, tip):
    link = []
    while True:
        pair = (randint(0, stevilo_primerov - 1), randint(0, stevilo_primerov - 1))
        if pair[0] != pair[1] and pair not in link:
            link.append(pair)
        if len(link) >= link_limit:
            return link

def create_random_constraints(N):
    N_CL = randint(0, N-1)
    N_ML = N-N_CL
    cannot_link = generate_link(N_CL, "CL")
    must_link = generate_link(N_ML, "ML")
    print(cannot_link, len(cannot_link))
    print(must_link, len(must_link))


create_random_constraints(10)
#create_random_must_link("must.csv", 10)
    #create_random_cannot_link("cannot.csv", 10)