#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

## Sklearn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc #plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler

## Plotting
import matplotlib.pyplot as plt
import seaborn as sns

## Pandas
import pandas as pd 
## Numpy
import numpy as np

## Miscellaneous
import warnings
import os
import joblib
from time import time
from scipy import stats
import itertools
import subprocess
import json
from elasticsearch import Elasticsearch


search_object_pir = {"size": 400, "query": {"match_all": {}}}
es = Elasticsearch(['http://192.168.1.35:9200'])
res = es.search(index='eventi_postprocessing_eee', body=json.dumps(search_object_pir))


lista = []
for e in res['hits']['hits']:
    lista.append(e['_source'])
    
for e in lista:
    e['Pir_Read_Value'] = int(e['Pir_Read_Value'])
    e['AvgScore'] = int(e['AvgScore'])
    e['AlarmFrames'] = int(e['AlarmFrames'])
    e['Microphone'] = int(e['Microphone'])
    
df = pd.DataFrame(lista)
del df['Date']
df.head(326)


# In[2]:


warnings.filterwarnings("ignore")
pd.set_option('mode.chained_assignment', None)


# In[3]:


currentDirectory=os.getcwd()
print(currentDirectory)


# In[4]:


def folder_path(path):
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
        return(path)
    else:
        print ("Successfully created the directory %s " % path)
        return(path)
    return(path)


# In[5]:


# OUTPUTS: Folder for storing OUTPUTS
OUTPUT_path=folder_path(currentDirectory)

# Models: Folder for storing Models
models_path=folder_path(OUTPUT_path+'/Models')
# Figures: Folder for storing Figures
figures_path=folder_path(OUTPUT_path+'/Figures')


# In[6]:


properties = list(df.columns.values)
properties.remove('Alert')
print(properties)
X = df[properties]
y = df['Alert']


# In[7]:


print(X.head())


# In[8]:


print(y.head())


# In[9]:


def print_results(results,name,filename_pr):
    with open(filename_pr, mode='w') as file_object:
        print(name,file=file_object)
        print(name)
        print('BEST PARAMS: {}\n'.format(results.best_params_),file=file_object)
        print('BEST PARAMS: {}\n'.format(results.best_params_))
        means = results.cv_results_['mean_test_score']
        stds = results.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, results.cv_results_['params']):
            print('{} {} (+/-{}) for {}'.format(name,round(mean, 3), round(std * 2, 3), params),file=file_object)
            print('{} {} (+/-{}) for {}'.format(name,round(mean, 3), round(std * 2, 3), params))


# ### Visualize the data

# In[10]:


DropList=['Alert']
sqrt_length=round(len(list(df.drop(DropList,axis=1)))**(1/2))
ncols=sqrt_length
nrows=sqrt_length
fig, axes = plt.subplots(ncols=ncols,nrows=nrows,figsize=[30,20])
fig.suptitle('Histogram Plot Raw Data', y=1.05, fontsize=24)
fig.tight_layout()

#axes[-1,(i+1)*-1].axis('off')
slog_train = df['Alert'] == 1
for name, ax in zip(list(df.drop(['Alert'],axis=1)),axes.flat):
    x1=df[~slog_train][name]
    x2=df[slog_train][name]
    x3=df[name]
    print(x3.unique())
    labels=['No Alert','Alert','Combined']
    ax.hist([x1,x2,x3],label=labels)
    ax.legend()
    ax.set_xlabel(name)
    ax.set_ylabel('Count #')
    plt.tight_layout()
histplot_fig = os.path.join(figures_path,'Histogram Plot Raw Data.png')
plt.savefig(histplot_fig,dpi=300,bbox_inches='tight')


# ## Scale the data

# In[11]:


X = df.drop(['Alert'], axis= 1)
y= pd.DataFrame(df['Alert'])

#Scale Data
scaler = MinMaxScaler()
X=MinMaxScaler().fit_transform(X.values)
X = pd.DataFrame(X)
X.columns=(df.drop(['Alert'], axis= 1)).columns


# In[12]:


Xy=pd.concat([y,X],axis=1)


# ## Pearson Coefficient

# In[13]:


fix,ax = plt.subplots(figsize=(22,22))
heatmap_data = Xy
sns.heatmap(heatmap_data.corr(),vmax=1,linewidths=0.01,
            square=True,annot=True,linecolor="white", cmap='Greens')
bottom,top=ax.get_ylim()
ax.set_ylim(bottom+0.5,top-0.5)
heatmap_title='Heatmap of Pearson Correlation Coefficient Matrix for Features'
ax.set_title(heatmap_title)
heatmap_fig=os.path.join(figures_path,'Heatmap.png')
plt.savefig(heatmap_fig,dpi=300,bbox_inches='tight')
plt.show()


# ## Split the dataset

# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=432)


# In[15]:


X_train=pd.DataFrame(X_train)
y_train=pd.DataFrame(y_train)

X_test=pd.DataFrame(X_test)
y_test=pd.DataFrame(y_test)


# In[16]:


print('X_train - length:',len(X_train), 'y_train - length:',len(y_train))
print('X_test - length:',len(X_test),'y_test - length:',len(y_test))
print('Percent heldout for testing:', round(100*(len(X_test)/len(df)),0),'%')


# In[17]:


path=folder_path(OUTPUT_path+'/Models/GB')
GB_model_dir=os.path.join(path,'GB_model.pkl')
gb = GradientBoostingClassifier()   
parameters = {
    'loss': ['deviance'],
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [500],
    'max_depth': [1,3],
    'max_features': ['log2']
        }
cv=GridSearchCV(gb, parameters, cv=10)
cv.fit(X_train, y_train.values.ravel())
print_results(cv,'Gradient Boost (GB)',os.path.join(path,'GR_GridSearchCV_results.txt'))
cv.best_estimator_
joblib.dump(cv.best_estimator_,GB_model_dir)


# In[18]:


path=folder_path(OUTPUT_path+'/Models/SVM')
SVM_model_dir=os.path.join(path,'SVM_model.pkl')
svc = SVC(probability = True)    
parameters = {
        'C': [0.01, 1, 10],
        'coef0': [10.0],
        'degree': [3],
        'gamma': [0.1],
        'kernel': ['poly','linear','rbf'],
        }
cv=GridSearchCV(svc,parameters, cv=10)
cv.fit(X_train, y_train.values.ravel())
print_results(cv,'Support Vector Machine (SVM)',os.path.join(path,'SVM_GridSearchCV_results.txt'))
cv.best_estimator_
joblib.dump(cv.best_estimator_,SVM_model_dir)


# In[19]:


path=folder_path(OUTPUT_path+'/Models/ETC')
ETC_model_dir=os.path.join(path,'ETC_model.pkl')

etc = ExtraTreesClassifier(oob_score=True,bootstrap=True)    
parameters = {
        'criterion': ['entropy'],
        'max_features': ['log2', 0.25],
        'n_estimators': [1000]
        }
cv=GridSearchCV(etc, parameters, cv=10)
cv.fit(X_train, y_train.values.ravel())
print_results(cv,'Extra Trees Classifier (ETC)',os.path.join(path,'ETC_GridSearchCV_results.txt'))
cv.best_estimator_
joblib.dump(cv.best_estimator_,ETC_model_dir)


# In[20]:


path=folder_path(OUTPUT_path+'/Models/LR')
LR_model_dir=os.path.join(path,'LR_model.pkl')

lr = LogisticRegression(solver='liblinear')
parameters = {
        'C': [0.001, 0.01, 0.1, 1, 1.5, 10, 100, 1000],
        'fit_intercept':[True],
        'penalty': ['l2']
        }
cv=GridSearchCV(lr, parameters, cv=10)
cv.fit(X_train, y_train.values.ravel())
print_results(cv,'Logistic Regression (LR)',os.path.join(path,'LR_GridSearchCV_results.txt'))
cv.best_estimator_
joblib.dump(cv.best_estimator_,LR_model_dir)


# In[21]:


path=folder_path(OUTPUT_path+'/Models/MLP')
MLP_model_dir=os.path.join(path,'MLP_model.pkl')

mlp = MLPClassifier()
parameters = {
        'hidden_layer_sizes': [(10,),(100,)],
        'activation': ['logistic'],
        'learning_rate': ['constant','invscaling','adaptive']
        }
cv=GridSearchCV(mlp, parameters, cv=10)
cv.fit(X_train, y_train.values.ravel())
print_results(cv,'Neural Network (MLP)',os.path.join(path,'MLP_GridSearchCV_results.txt'))
cv.best_estimator_
joblib.dump(cv.best_estimator_,MLP_model_dir)


# In[22]:


path=folder_path(OUTPUT_path+'/Models/RF')
RF_model_dir=os.path.join(path,'RF_model.pkl')

rf = RandomForestClassifier(oob_score = True)    
parameters = {
        'criterion': ['entropy','gini'],
        'max_features': [0.25],
        'n_estimators': [500,1000]
        }
cv = GridSearchCV(rf, parameters, cv=10)
cv.fit(X_train, y_train.values.ravel())
print_results(cv,'Random Forest (RF)',os.path.join(path,'RF_GridSearchCV_results.txt'))
cv.best_estimator_
joblib.dump(cv.best_estimator_,RF_model_dir)


# In[23]:


models = {}

for mdl in ['GB', 
            'RF',
            'SVM',
            'ETC',
            'LR',
            'MLP']:
    model_path=os.path.join(OUTPUT_path,'Models/{}/{}_model.pkl')
    models[mdl] = joblib.load(model_path.format(mdl,mdl))


# In[35]:


def evaluate_model(fig_path,name, model, features, labels, y_test_ev, fc):
        CM_fig=os.path.join(fig_path,'Figure{}.A_{}_Confusion_Matrix.png'.format(fc,name))
        VI_fig=os.path.join(fig_path,'Figure{}.B_{}_Variable_Importance_Plot.png'.format(fc,name))
        
        start = time()
        pred = model.predict(features)
        print(pred)
        end = time()
        y_truth=y_test_ev
        accuracy = round(accuracy_score(labels, pred), 3)
        precision = round(precision_score(labels, pred), 3)
        recall = round(recall_score(labels, pred), 3)
        print('{} -- Accuracy: {} / Precision: {} / Recall: {} / Latency: {}ms'.format(name,
                                                                                       accuracy,
                                                                                       precision,
                                                                                       recall,
                                                                                       round((end - start)*1000, 1)))
        
        
        pred=pd.DataFrame(pred)
        pred.columns=['Alert']
        # Convert from Binary to Categorical
        Binary_name={0:'NOAlarm',1:'Alarm'}
        y_truth['Alert']=y_truth['Alert'].map(Binary_name)
        pred['Alert']=pred['Alert'].map(Binary_name)
        class_names = ['NOAlarm','Alarm']        
        cm = confusion_matrix(y_test_ev, pred, class_names)
        
        FP_L='False Positive'
        FP = cm[0][1]
        FN_L='False Negative'
        FN = cm[1][0]
        TP_L='True Positive'
        TP = cm[1][1]
        TN_L='True Negative'
        TN = cm[0][0]

        #TPR_L= 'Sensitivity, hit rate, recall, or true positive rate'
        TPR_L= 'Sensitivity'
        TPR = round(TP/(TP+FN),3)
        #TNR_L= 'Specificity or true negative rate'
        TNR_L= 'Specificity'
        TNR = round(TN/(TN+FP),3) 
        #PPV_L= 'Precision or positive predictive value'
        PPV_L= 'Precision'
        PPV = round(TP/(TP+FP),3)
        #NPV_L= 'Negative predictive value'
        NPV_L= 'NPV'
        NPV = round(TN/(TN+FN),3)
        #FPR_L= 'Fall out or false positive rate'
        FPR_L= 'FPR'
        FPR = round(FP/(FP+TN),3)
        #FNR_L= 'False negative rate'
        FNR_L= 'FNR'
        FNR = round(FN/(TP+FN),3)
        #FDR_L= 'False discovery rate'
        FDR_L= 'FDR'
        FDR = round(FP/(TP+FP),3)

        ACC_L= 'Accuracy'
        ACC = round((TP+TN)/(TP+FP+FN+TN),3)
        
        stats_data = {'Name':name,
                     ACC_L:ACC,
                     FP_L:FP,
                     FN_L:FN,
                     TP_L:TP,
                     TN_L:TN,
                     TPR_L:TPR,
                     TNR_L:TNR,
                     PPV_L:PPV,
                     NPV_L:NPV,
                     FPR_L:FPR,
                     FNR_L:FDR}
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm,cmap=plt.cm.gray_r)
        plt.title('{} {} Confusion Matrix on Unseen Test Data'.format(' ',name),y=1.08)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + class_names)
        ax.set_yticklabels([''] + class_names)
        # Loop over data dimensions and create text annotations.
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text = ax.text(j, i, cm[i, j],
                               ha="center", va="center", color="r")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(CM_fig,dpi=400,bbox_inches='tight')
        #plt.show()
        
        
        return accuracy,name, model, stats_data
        


# In[36]:


def plot_roc_cur(fper, tper,mdl_i,ax): 
    roc_auc=str(round(auc(fper,tper),3))
    label_i='ROC-'+mdl_i+'  (AUC = '+roc_auc+')'
    auc
    ax.plot(fper, tper,label=label_i)
    ax.set_xlabel('False Positive Rate', size=15, labelpad=20)
    ax.set_ylabel('True Positive Rate', size=15, labelpad=20)
    ax.set_xlim([-0.05,1.05])
    ax.set_ylim([-0.05,1.05])


# In[37]:


def sort_list(list1, list2): 
    zipped_pairs = zip(list2, list1)   
    z = [x for _, x in sorted(zipped_pairs)]       
    return z 


# In[38]:


ev_accuracy=[None]*len(models)
ev_name=[None]*len(models)
ev_model=[None]*len(models)
ev_stats=[None]*len(models)
count=1
countincr=count+1
f, ax2 = plt.subplots(figsize=(15,15))
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate',size=15,labelpad=20)
plt.ylabel('True Positive Rate',size=15,labelpad=20)
for name, mdl in models.items():
        y_test_ev=y_test
        fper,tper,thresholds = [],[],[]
        ev_accuracy[count-1],ev_name[count-1],ev_model[count-1], ev_stats[count-1] = evaluate_model(figures_path,
                                                                                                    name,
                                                                                                    mdl,
                                                                                                    X_test,
                                                                                                    y_test,
                                                                                                    y_test_ev,
                                                                                                    count+1)
        Binary_name={'NOAlarm':0,'Alarm':1}
        y_test['Alert']=y_test['Alert'].map(Binary_name)
        y_pred=pd.DataFrame(mdl.predict_proba(pd.DataFrame(X_test))[:,1])
        fper,tper,thresholds = roc_curve(y_test,y_pred)
        plot_disp = plot_roc_cur(fper,tper,name,ax=ax2)
            
        count=count+1        
ax2.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='Coin Flip (AUC = 0.5)')
ax2.legend()
f.savefig(os.path.join(figures_path,'Figure.png'),dpi=300,bbox_inches='tight')


# In[28]:


ev_stats=pd.DataFrame(ev_stats)
ev_stats.head(12)




