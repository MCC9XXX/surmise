fig, axs = plt.subplots(4, 4, figsize=(14, 14))

for id_i in range(3):
    for id_j in range(id_i+1,4):
        
        for idi in range(4):
            ml_sample = np.ones((len(flat_samples_ml), 1)) * np.median(flat_samples_ml, axis=0)
        
        #ml_sample[:, id_i] = flat_samples_ml[:, id_i]
        #ml_sample[:, id_j] = flat_samples_ml[:, id_j]

        ml_sample[:, id_i] = prior_covid.rnd(len(flat_samples_ml))[:, id_i] #flat_samples_ml[:, id_i]
        ml_sample[:, id_j] = prior_covid.rnd(len(flat_samples_ml))[:, id_j] 
           
        for i in range(0, 4):
            for j in range(i, 4):
                ml_sample = np.concatenate([ml_sample, np.reshape(ml_sample[:, i] * ml_sample[:, j], (len(ml_sample), 1))], axis = 1)
                    
        yp = classification_model.predict_proba(ml_sample)[:, 1] 
           
        scatter = axs[id_j,id_i].scatter(ml_sample[:,id_i], ml_sample[:,id_j], c=yp, vmin=0, vmax=1, cmap="Spectral")
        legend1 = axs[id_j,id_i].legend(*scatter.legend_elements(num=5),
                            loc="upper left", title="Ranking")
        axs[id_j,id_i].add_artist(legend1)

labels = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$', r'$\theta_4$']
axs[0, 0].set_ylabel(labels[0], fontsize=12)
axs[1, 0].set_ylabel(labels[1], fontsize=12)
axs[2, 0].set_ylabel(labels[2], fontsize=12)
axs[3, 0].set_ylabel(labels[3], fontsize=12)

axs[3, 0].set_xlabel(labels[0], fontsize=12)
axs[3, 1].set_xlabel(labels[1], fontsize=12)
axs[3, 2].set_xlabel(labels[2], fontsize=12)
axs[3, 3].set_xlabel(labels[3], fontsize=12)
plt.show()





fig, axs = plt.subplots(4, 4, figsize=(14, 14))

for id_i in range(3):
    for id_j in range(id_i+1,4):
        
        for idi in range(4):
            ml_sample = np.ones((len(flat_samples_ml), 1)) * np.median(flat_samples_ml, axis=0)

        ml_sample[:, id_i] = prior_covid.rnd(len(flat_samples_ml))[:, id_i] 
        ml_sample[:, id_j] = prior_covid.rnd(len(flat_samples_ml))[:, id_j] 
           
        yp = np.zeros((len(flat_samples_ml)))
        for i in range(len(flat_samples_ml)):
            yp[i] = log_likelihood(ml_sample[i,:], obsvar, emulator_f_PCGPwM, np.sqrt(real_data_tr), xtr)
           
        yp = np.max(yp)/yp
        scatter = axs[id_j,id_i].scatter(ml_sample[:,id_i], ml_sample[:,id_j], c=yp, cmap="Spectral")
        legend1 = axs[id_j,id_i].legend(*scatter.legend_elements(num=5),
                            loc="upper left", title="Ranking")
        axs[id_j,id_i].add_artist(legend1)

labels = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$', r'$\theta_4$']
axs[0, 0].set_ylabel(labels[0], fontsize=12)
axs[1, 0].set_ylabel(labels[1], fontsize=12)
axs[2, 0].set_ylabel(labels[2], fontsize=12)
axs[3, 0].set_ylabel(labels[3], fontsize=12)

axs[3, 0].set_xlabel(labels[0], fontsize=12)
axs[3, 1].set_xlabel(labels[1], fontsize=12)
axs[3, 2].set_xlabel(labels[2], fontsize=12)
axs[3, 3].set_xlabel(labels[3], fontsize=12)
plt.show()




fig, axs = plt.subplots(4, 4, figsize=(14, 14))

for id_i in range(3):
    for id_j in range(id_i+1,4):
        
        for idi in range(4):
            ml_sample = np.ones((len(flat_samples_ml), 1)) * np.median(flat_samples_ml, axis=0)

        ml_sample[:, id_i] = prior_covid.rnd(len(flat_samples_ml))[:, id_i] 
        ml_sample[:, id_j] = prior_covid.rnd(len(flat_samples_ml))[:, id_j] 
           
        yp = np.zeros((len(flat_samples_ml)))
        for i in range(len(flat_samples_ml)):
            yp[i] = log_likelihood(ml_sample[i,:], obsvar, emulator_f_PCGPwM, np.sqrt(real_data_tr), xtr)
        
        yp = np.max(yp)/yp
        
        for i in range(0, 4):
            for j in range(i, 4):
                ml_sample = np.concatenate([ml_sample, np.reshape(ml_sample[:, i] * ml_sample[:, j], (len(ml_sample), 1))], axis = 1)
                    
        ypc = classification_model.predict_proba(ml_sample)[:, 1]             
        pp = yp*ypc
        #pp1 = pp[pp>0.5]
        scatter = axs[id_j,id_i].scatter(ml_sample[:,id_i], ml_sample[:,id_j], c=pp, cmap="Spectral")
        legend1 = axs[id_j,id_i].legend(*scatter.legend_elements(num=5),
                            loc="upper left", title="Ranking")
        axs[id_j,id_i].add_artist(legend1)

labels = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$', r'$\theta_4$']
axs[0, 0].set_ylabel(labels[0], fontsize=12)
axs[1, 0].set_ylabel(labels[1], fontsize=12)
axs[2, 0].set_ylabel(labels[2], fontsize=12)
axs[3, 0].set_ylabel(labels[3], fontsize=12)

axs[3, 0].set_xlabel(labels[0], fontsize=12)
axs[3, 1].set_xlabel(labels[1], fontsize=12)
axs[3, 2].set_xlabel(labels[2], fontsize=12)
axs[3, 3].set_xlabel(labels[3], fontsize=12)
plt.show()