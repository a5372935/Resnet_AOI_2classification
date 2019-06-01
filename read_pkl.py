import pickle
import numpy as np
import matplotlib.pyplot as plt

alpha_0 = pickle.load(open("resnet18_flatten_200e_V3.pkl", 'rb'), encoding='utf-8')
#alpha = 0
loss0 = str(alpha_0).split(", '", 1)
loss0 = str(loss0[0]).split(": ", 1)
loss0 = str(loss0[1]).split("[", 1)
loss0 = str(loss0[1]).split("]", 1)
loss0 = str(loss0[0]).split(", ")
loss0 = np.array(loss0, dtype= float)
#----------------------------------------- 
acc0 = str(alpha_0).split(", '", 1)
acc0 = str(acc0[1]).split(": ", 1)
acc0 = str(acc0[1]).split(", '", 1)
acc0 = str(acc0[0]).split("[", 1)
acc0 = str(acc0[1]).split("]", 1)
acc0 = str(acc0[0]).split(", ")
acc0 = np.array(acc0, dtype= float)
#-----------------------------------------
val_acc0 = str(alpha_0).split(", '", 1)
val_acc0 = str(val_acc0[1]).split(": ", 1)
val_acc0 = str(val_acc0[1]).split(", '", 1)
val_acc0 = str(val_acc0[1]).split(", '", 1)
val_acc0 = str(val_acc0[1]).split(", '", 1)
val_acc0 = str(val_acc0[0]).split(": ", 1)
val_acc0 = str(val_acc0[1]).split("[", 1)
val_acc0 = str(val_acc0[1]).split("]", 1)
val_acc0 = str(val_acc0[0]).split(", ")
val_acc0 = np.array(val_acc0, dtype= float)
#-----------------------------------------

alpha_25 = pickle.load(open(".pkl", 'rb'), encoding='utf-8')
#alpha = 0.25
val_loss25 = str(alpha_25).split(", '", 1)
val_loss25 = str(val_loss25[0]).split(": ", 1)
val_loss25 = str(val_loss25[1]).split("[", 1)
val_loss25 = str(val_loss25[1]).split("]", 1)
val_loss25 = str(val_loss25[0]).split(", ")
val_loss25 = np.array(val_loss25, dtype= float)
#----------------------------------------- 
val_acc25 = str(alpha_25).split(", '", 1)
val_acc25 = str(val_acc25[1]).split(": ", 1)
val_acc25 = str(val_acc25[1]).split(", '", 1)
val_acc25 = str(val_acc25[0]).split("[", 1)
val_acc25 = str(val_acc25[1]).split("]", 1)
val_acc25 = str(val_acc25[0]).split(", ")
val_acc25 = np.array(val_acc25, dtype= float)
#-----------------------------------------
acc25 = str(alpha_25).split(", '", 1)
acc25 = str(acc25[1]).split(": ", 1)
acc25 = str(acc25[1]).split(", '", 1)
acc25 = str(acc25[1]).split(", '", 1)
acc25 = str(acc25[1]).split(", '", 1)
acc25 = str(acc25[0]).split(": ", 1)
acc25 = str(acc25[1]).split("[", 1)
acc25 = str(acc25[1]).split("]", 1)
acc25 = str(acc25[0]).split(", ")
acc25 = np.array(acc25, dtype= float)
#-----------------------------------------
loss25 = str(alpha_25).split(", '", 1)
loss25 = str(loss25[1]).split(": ", 1)
loss25 = str(loss25[1]).split(", '", 1)
loss25 = str(loss25[1]).split(", '", 1)
loss25 = str(loss25[0]).split(": ", 1)
loss25 = str(loss25[1]).split("[", 1)
loss25 = str(loss25[1]).split("]", 1)
loss25 = str(loss25[0]).split(", ")
loss25 = np.array(loss25, dtype= float)
#-----------------------------------------

plt.figure()
x = np.arange(0, 200, 1)
#plt.plot(x, acc0, '-', color = "blue", lw = 1)
plt.plot(x, acc25, '-', color = "blue", lw = 1)
#plt.plot(x, val_acc0, '--', color = "green", lw = 1)
plt.plot(x, val_acc25, '--', color = "green", lw = 1)
#plt.plot(x, loss0, '-', color = "red", lw = 1)
plt.plot(x, val_loss25, '-', color = "red", lw = 1)
plt.plot(x, loss25, '-', color = "black", lw = 1)

plt.ylabel("acc / loss")
plt.xlabel("epoch")

plt.show()

print(alpha_25)