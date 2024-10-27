from pymnet import *

mplex = MultiplexNetwork(
    couplings = "none"
    )

people = [
    "John", "Max", "Kevin",
    "Hannah", "Rachel", "Jennifer",
    "Daniel", "Lydia", "Sarah", "Liz"
    ]
mplex.add_layer("Friendship")
mplex.add_layer("Professional")
mplex.add_layer("Recreational")
mplex['John','Friendship']['Max','Friendship'] = 1
friends = [
    ["John", "Max"],
    ["Hannah", "Max"],
    ["Lydia", "John"],
    ["Sarah", "Rachel"],
    ["Lydia", "Daniel"],
    ["Hannah", "Jennifer"],
    ["Liz", "Max"],
    ["Sarah", "Kevin"],
    ["Kevin", "Max"],
    ["Daniel", "Max"],
    ["Hannah", "Liz"]
    ]
for x in range(len(friends)):
    mplex[friends[x][0],'Friendship'][friends[x][1], 'Friendship'] = 1

coworkers = [
    ["John", "Max"],
    ["Jennifer", "John"],
    ["Jennifer", "Max"],
    ["Jennifer", "Liz"],
    ["John", "Liz"],
    ["Max", "Liz"],
    ["Sarah", "Lydia"],
    ["Hannah", "Daniel"],
    ["Daniel", "Kevin"],
    ["Kevin", "Hannah"],
    ["Kevin", "John"]
    ]
for x in range(len(coworkers)):
    mplex[coworkers[x][0], 'Professional'][coworkers[x][1], 'Professional'] = 1

teammates = [
    ["John", "Max"],
    ["Daniel", "Max"],
    ["Sarah", "Max"],
    ["Sarah", "Daniel"],
    ["Sarah", "John"],
    ["Sarah", "Daniel"],
    ["Sarah", "Lydia"],
    ["Lydia", "Hannah"],
    ["Lydia", "Liz"],
    ["Lydia", "Rachel"],
    ["Hannah", "Liz"],
    ["Hannah", "Rachel"],
    ["Liz", "Rachel"]
    ]
for x in range(len(teammates)):
    mplex[teammates[x][0],'Recreational'][teammates[x][1], 'Recreational'] = 1

fig = draw(
    mplex, show = True, layout = "spring",
    figsize = (10,10), defaultLayerLabelLoc = (1,1.04)
    )