import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from matplotlib import pyplot as plt










#First Requirment
accArray1=[] #array to store the total acc
TreSizeArray1=[] #array to store the total sizes
for r in range(5):
    data = pd.read_csv(r'BankNote_Authentication.csv')
    df=pd.DataFrame(data)
    dataset=df[['variance','skewness','curtosis','entropy']]
    x=dataset.copy()
    y=data['class']
    clf=DecisionTreeClassifier()
    x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.75)
    cld=clf.fit(x_train,y_train)
    predections=clf.predict(x_test)
    print("accuracy for Experiment " ,r," : ", accuracy_score(y_test,predections))
    accArray1.append(accuracy_score(y_test,predections))    

    treeSize = clf.tree_.node_count
    TreSizeArray1.append(treeSize)    

    print("size Experiment " ,r," : ", treeSize)
    print(" ")
    
    
    
    
    
    
    
    
    
#Second Requirment   
arr=[0.7,0.6,0.5,0.4,0.3]
MeanaccArray=[]
MeanTreSizeArray=[]
MinaccArray=[]
MinTreSizeArray=[]
MaxaccArray=[]
MaxTreSizeArray=[]
for r in range(5):
    accArray=[]
    TreSizeArray=[]
    for exp in range(5): #if the range is 50 or more will represent the idea perfictly
        data = pd.read_csv(r'BankNote_Authentication.csv')
        df=pd.DataFrame(data)
        dataset=df[['variance','skewness','curtosis','entropy']]
        x=dataset.copy()
        y=data['class']
        clf=DecisionTreeClassifier()
        x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=arr[r])
        cld=clf.fit(x_train,y_train)
        predections=clf.predict(x_test)
        accArray.append(accuracy_score(y_test,predections))    
        treeSize = clf.tree_.node_count
        TreSizeArray.append(treeSize)    

    AccMean=np.mean(accArray)
    AccMax=np.max(accArray)
    AccMin=np.min(accArray)
    print("Mean accuracy of Experiment " ,r," : ", AccMean)
    print("Min accuracy of Experiment " ,r," : ", AccMin)
    print("Max accuracy of Experiment " ,r," : ", AccMax)

    MeanaccArray.append(AccMean)
    MaxaccArray.append(AccMax)
    MinaccArray.append(AccMin)
    
    SizeMean=np.mean(TreSizeArray)
    SizeMax=np.max(TreSizeArray)
    SizeMin=np.min(TreSizeArray)
    print("Mean Tree Size of Experiment " ,r," : ", SizeMean)
    print("Min Tree Size of Experiment " ,r," : ", SizeMin)
    print("Max Tree Size of Experiment " ,r," : ", SizeMax)
    print(" ")

    MeanTreSizeArray.append(SizeMean)
    MinTreSizeArray.append(SizeMin)
    MaxTreSizeArray.append(SizeMax)

    









#drawing the plots
plt.figure("Accuracy against  training set size")
plt.plot([0.3,0.4,0.5,0.6,0.7], MeanaccArray)
plt.title('Accuracy against  training set size ')
plt.xlabel('Train training set Size')
plt.ylabel('Accuracy')
plt.show()

plt.figure('Number of nodes in the final tree against training set size')
plt.plot([0.3,0.4,0.5,0.6,0.7], MeanTreSizeArray)
plt.title('Number of nodes in the final tree against training set size')
plt.xlabel('Training set size')
plt.ylabel('Number of nodes in the final tree')
plt.show()

















#Genrating the pdf and adding the report for the Second Requirment
ch = 20
m = 10 
pw = 210 - 2 

pdf = FPDF()
pdf.add_page()

pdf.set_font('Arial', 'B', 24)
pdf.cell(w=0, h=20, txt="Report for the Second Requirment", ln=1)
pdf.set_font('Arial', '', 5)

pdf.cell(w=0, h=ch, txt="Experiment 1"+"    training set size : 30%"+"    Mean Accuracy : "+str(MeanaccArray[0]) +"    Max Accuracy :"+str(MaxaccArray[0]) +"    Min Accuracy :"+str(MinaccArray[0]) +"    Mean Tree Size :"+str(MeanTreSizeArray[0]) +"    Min Tree Size :"+str(MinTreSizeArray[0]) +"    Max Tree Size :"+str(MaxTreSizeArray[0]) , border=1, ln=1)
pdf.cell(w=0, h=ch, txt="Experiment 2"+"    training set size : 40%"+"    Mean Accuracy : "+str(MeanaccArray[1]) +"    Max Accuracy :"+str(MaxaccArray[1]) +"    Min Accuracy :"+str(MinaccArray[1]) +"    Mean Tree Size :"+str(MeanTreSizeArray[1]) +"    Min Tree Size :"+str(MinTreSizeArray[1]) +"    Max Tree Size :"+str(MaxTreSizeArray[1]) , border=1, ln=1)
pdf.cell(w=0, h=ch, txt="Experiment 3"+"    training set size : 50%"+"    Mean Accuracy : "+str(MeanaccArray[2]) +"    Max Accuracy :"+str(MaxaccArray[2]) +"    Min Accuracy :"+str(MinaccArray[2]) +"    Mean Tree Size :"+str(MeanTreSizeArray[2]) +"    Min Tree Size :"+str(MinTreSizeArray[2]) +"    Max Tree Size :"+str(MaxTreSizeArray[2]) , border=1, ln=1)
pdf.cell(w=0, h=ch, txt="Experiment 4"+"    training set size : 60%"+"    Mean Accuracy : "+str(MeanaccArray[3]) +"    Max Accuracy :"+str(MaxaccArray[3]) +"    Min Accuracy :"+str(MinaccArray[3]) +"    Mean Tree Size :"+str(MeanTreSizeArray[3]) +"    Min Tree Size :"+str(MinTreSizeArray[3]) +"    Max Tree Size :"+str(MaxTreSizeArray[3]) , border=1, ln=1)
pdf.cell(w=0, h=ch, txt="Experiment 5"+"    training set size : 70%"+"    Mean Accuracy : "+str(MeanaccArray[4]) +"    Max Accuracy :"+str(MaxaccArray[4]) +"    Min Accuracy :"+str(MinaccArray[4]) +"    Mean Tree Size :"+str(MeanTreSizeArray[4]) +"    Min Tree Size :"+str(MinTreSizeArray[4]) +"    Max Tree Size :"+str(MaxTreSizeArray[4]) , border=1, ln=1)

pdf.ln(ch)
pdf.multi_cell(w=0, h=10)


pdf.output(f'./ML-Report for Second Req.pdf', 'F')




















#Genrating the pdf and adding the report for the First Requirment
pdf = FPDF()
pdf.add_page()

pdf.set_font('Arial', 'B', 24)
pdf.cell(w=0, h=20, txt="Report for the First Requirment", ln=1)
pdf.set_font('Arial', '', 10)

pdf.cell(w=0, h=ch, txt="Experiment 1"+"    training set size : 25%"+"    Accuracy : "+str(accArray1[0]) +"    Tree Size :"+str(TreSizeArray1[0]) , border=1, ln=1)
pdf.cell(w=0, h=ch, txt="Experiment 2"+"    training set size : 25%"+"    Accuracy : "+str(accArray1[1]) +"    Tree Size :"+str(TreSizeArray1[1]) , border=1, ln=1)
pdf.cell(w=0, h=ch, txt="Experiment 3"+"    training set size : 25%"+"    Accuracy : "+str(accArray1[2]) +"    Tree Size :"+str(TreSizeArray1[2]) , border=1, ln=1)
pdf.cell(w=0, h=ch, txt="Experiment 4"+"    training set size : 25%"+"    Accuracy : "+str(accArray1[3]) +"    Tree Size :"+str(TreSizeArray1[3]) , border=1, ln=1)
pdf.cell(w=0, h=ch, txt="Experiment 5"+"    training set size : 25%"+"    Accuracy : "+str(accArray1[4]) +"    Tree Size :"+str(TreSizeArray1[4]) , border=1, ln=1)

pdf.ln(ch)
pdf.multi_cell(w=0, h=10)


pdf.output(f'./ML-Report for First Req.pdf', 'F')