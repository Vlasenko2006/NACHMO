from tensorflow.python.summary.summary_iterator import summary_iterator
import numpy as np
import pickle

def TBreader(path,species):
    train_loss = np.zeros([1,1])
    validation_loss = np.zeros([1,1])
    L1_norm = np.zeros([len(species),1])
    aux = np.zeros([len(species),1])
    aux1 = np.zeros([1,1])
    aux2 = np.zeros([1,1])



    previous_epoch_value = -1 
    epoch = -1

    for e in summary_iterator(path):
        for v in e.summary.value:
            if v.tag == "epoch": epoch = v.simple_value
            if v.tag == "train_loss":
                aux1[0,0] = v.simple_value
            if v.tag == "validation_loss":
                aux2[0,0] = v.simple_value

            for i in range(0,len(species)):
                L1 = "hp/L1 " + species[i]
                if v.tag == L1:
                    aux[i] = v.simple_value
            if epoch > previous_epoch_value:
                L1_norm = np.append(L1_norm, aux, axis=1)
                train_loss = np.append(train_loss, aux1, axis = 1)
                validation_loss = np.append(validation_loss, aux2, axis=1)
                previous_epoch_value = epoch



    species = species + ["train_loss"] + ["validation_loss"]

    summary = np.append(np.append(L1_norm, train_loss, axis=0), validation_loss, axis=0)

    summary = list(summary)
    Summary = []
    for i in range(0,len(summary)):
        Summary.append ([species[i]] + [summary[i]])
    return Summary







species = ["CO","HNO3","SO4","XO2","O1D","SO2","O3P","ALD2","PAN","CH3O","N2O5","NO3","HCHO","O3","C2O3","HO2","NO2","NO","CH3O2","OH"]


for i in range(100,400,100):
    Summary = TBreader("event_" + str(i), species)
    with open("List_of_events_" + str(i), "wb") as fp:   #Pickling
        pickle.dump(Summary, fp)


print(Summary)
