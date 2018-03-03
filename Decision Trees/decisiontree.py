from math import log
from operator import itemgetter
from copy import deepcopy

def buildtree(ipdata, labels):
    oplabels = [row[-1] for row in ipdata]
    if len(oplabels) == oplabels.count(oplabels[0]):
        return oplabels[0]
    if len(ipdata[0]) == 1:
        label_count = {}
        for i in oplabels:
            if i not in label_count:
                label_count[i] = 1
            else:
                label_count[i] += 1
        sorted_dict = sorted(label_count.items(), key=itemgetter(1), reverse=True)
        return sorted_dict[0][0]
    attrtosplit = pickattribute(ipdata)
    attrlabel = labels[attrtosplit]
    decisiontree = {attrlabel: {}}
    labels.remove(attrlabel)
    attrvals = [row[attrtosplit] for row in ipdata]
    uniqattrvals = set(attrvals)
    for i in uniqattrvals:
        subLabels = deepcopy(labels)
        decisiontree[attrlabel][i] = buildtree(calcsubset(ipdata, attrtosplit, i), subLabels)
    return decisiontree

def calcentropy(ipdata):
    n = len(ipdata)
    label_count = {}
    for row in ipdata:
        label = row[-1]
        if label not in label_count:
            label_count[label] = 1
        else:
            label_count[label] += 1
    entropy = 0.0
    for i in label_count:
        p = float(label_count[i]) / n
        entropy -= p * log(p, 2)
    return entropy

def calcsubset(ipdata, attrno, attrval):
    opdata = []
    for row in ipdata:
        if row[attrno] == attrval:
            newrow = row[:attrno]
            newrow.extend(row[attrno + 1:])
            opdata.append(newrow)
    return opdata

def pickattribute(ipdata):
    totalattr = len(ipdata[0]) - 1
    base_entropy = calcentropy(ipdata)
    max_ig = 0.0;
    selectedattr = -1
    for i in range(totalattr):
        attrvals = [row[i] for row in ipdata]
        uniqattrvals = set(attrvals)
        attrentropy = 0.0
        for j in uniqattrvals:
            subset = calcsubset(ipdata, i, j)
            p = len(subset) / float(len(ipdata))
            attrentropy += p * calcentropy(subset)

        ig = base_entropy - attrentropy
        if (max_ig < ig):
            max_ig = ig
            selectedattr = i
    return selectedattr

def printtree(decisiontree,depth):
    tab = "  " * depth
    if type(decisiontree) is dict:
        for i in decisiontree:
            if depth % 2 == 0:
                print(i)
            else:
                print(tab,i,": ",end="")
            printtree(decisiontree[i],depth+1)
    else:
        print(decisiontree)

def predict(decisiontree, labels, testdata):
    initattr = list(decisiontree.keys())[0]
    if initattr not in decisiontree:
        return None
    subdict = decisiontree[initattr]
    attrindex = labels.index(initattr)
    key = testdata[attrindex]
    if key not in subdict:
        return None
    subvalue = subdict[key]
    if type(subvalue) is not dict:
        classLabel = subvalue
    else:
        classLabel = predict(subvalue, labels, testdata)
    return classLabel

f = open('dt-data.txt', 'r')
colheaders = f.readline().strip()
rowdata = []
labels = []
for line in f:
    line = line.rstrip('\n')
    if len(line) != 0:
        second = [x.strip() for x in line[4:len(line)-1].split(',')]
        rowdata.append(second)

attributes = [x.strip() for x in colheaders.split(',')]

for i in range(len(attributes)):
    if "(" in attributes[i]:
        i1 = attributes[i].replace('(','')
        attributes[i] = i1
    if ")" in attributes[i]:
        i2 = attributes[i].replace(")","")
        attributes[i] = i2
attributes = attributes[:6]

decisiontree = buildtree(rowdata, attributes)

print("Decision Tree:\n")
printtree(decisiontree,0)
print("")

result = predict(decisiontree, ['Occupied', 'Price', 'Music', 'Location', 'VIP', 'Favorite Beer'], ['Moderate','Cheap','Loud','City-Center','No','No'])
if result == None:
    print("The Decision Tree cannot handle this case as the training data is insufficient.")
else:
    print("The predicted value is "+ result)
