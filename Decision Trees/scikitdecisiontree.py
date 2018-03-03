from sklearn import tree
from sklearn import preprocessing


f = open('dt-data.txt', 'r')
colheaders = f.readline().strip()
rowdata = []
target = []

for line in f:
    line = line.rstrip('\n')
    if len(line) != 0:
        second = [x.strip() for x in line[4:len(line)-1].split(',')]
        rowdata.append(second[:6])
        target.append(second[6])

le = preprocessing.LabelEncoder()

test_data = ['Moderate','Cheap','Loud','City-Center','No','No']
converted_test = []

col1 = list(list(zip(*rowdata))[0])
res1 = le.fit_transform(col1).tolist()
converted_test += le.transform([test_data[0]]).tolist()

col2 = list(list(zip(*rowdata))[1])
res2 = le.fit_transform(col2).tolist()
converted_test += le.transform([test_data[1]]).tolist()

col3 = list(list(zip(*rowdata))[2])
res3 = le.fit_transform(col3).tolist()
converted_test += le.transform([test_data[2]]).tolist()

col4 = list(list(zip(*rowdata))[3])
res4 = le.fit_transform(col4).tolist()
converted_test += le.transform([test_data[3]]).tolist()

col5 = list(list(zip(*rowdata))[4])
res5 = le.fit_transform(col5).tolist()
converted_test += le.transform([test_data[4]]).tolist()

col6 = list(list(zip(*rowdata))[5])
res6 = le.fit_transform(col6).tolist()
converted_test += le.transform([test_data[5]]).tolist()

full = []
for i in zip(res1,res2,res3,res4,res5,res6):
    full.append(list(i))

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(full,target)

print(clf.predict([converted_test]))