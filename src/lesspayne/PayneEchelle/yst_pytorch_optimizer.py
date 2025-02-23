import os
import numpy as np
import torch

# insert results
results_filename =  os.path.join(data_dir,'results_' + str(start_index) + '.npy')
# restore previous results
popt1 = np.load("../popt_array_transferred_model_synthetic.npy")
popt1 = torch.from_numpy(popt1[90000:,:25].astype("float32")).cuda()
#=======================================================================================
# assumes:
#  - weight ~ 1/sigma (and not 1/sigma**2)
def least_squares_fit(model, data, weight, params):
    loss_fn = torch.nn.MSELoss(reduction='mean')
    data *= weight
    params_found = False
    count_nans = 0
    orig_params = params.clone()
#----------------------------------------------------------------------------------------
    while params_found is False:
        optimizer = torch.optim.LBFGS([params], lr=0.01, max_iter=100,\
                                      line_search_fn='strong_wolfe')
        n_epochs = 10
#----------------------------------------------------------------------------------------
        for epoch in range(n_epochs):
            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                fit = weight*model(params)
                loss = loss_fn(fit, data)
                if loss.requires_grad:
                    loss.backward()
                return loss
            optimizer.step(closure)
            min_loss = closure().item()
#----------------------------------------------------------------------------------------
        if np.isnan(min_loss):
            params.data = orig_params.data
            count_nans+=1
        else:
            params_found=True
        if count_nans>10:
            params.data = orig_params.data
            params_found=True
    return min_loss
#=======================================================================================
# define loss
mse = torch.nn.MSELoss(reduction='mean')
cp_start_time = time.time()
#----------------------------------------------------------------------------------------
# Loop over all spectra
for cur_indx in range(start_index*2500,start_index*2500+2500):
    print("Spectrum ", cur_indx)
    obs_batch = obs_dataset.__getitem__(cur_indx) 
    if use_cuda:
        obs_batch = batch_to_cuda(obs_batch)
#----------------------------------------------------------------------------------------
    # Create z_sp
    with torch.no_grad():
        zsh_obs, zsp_obs = model.obs_to_z(obs_batch['x'].unsqueeze(0))
    model.cur_z_sp = zsp_obs
    x_weight = obs_batch['x_msk'].unsqueeze(0)/obs_batch['x_err'].unsqueeze(0)
    #x_weight = 1./obs_batch['x_err'].unsqueeze(0)
#----------------------------------------------------------------------------------------
    # initialize stellar parameters with ThePayne guess
    #y_payne = (obs_batch['y'].unsqueeze(0) - model.y_min) / (model.y_max - model.y_min) - 0.5
    y_payne = (popt1[cur_indx:cur_indx+1] - model.y_min) / (model.y_max - model.y_min) - 0.5
    # no initialization
    #y_payne = torch.from_numpy(np.zeros((1,25)).astype("float32")).cuda()
    params = torch.nn.Parameter(y_payne, requires_grad=True)
    print("  init loss full  : ", mse(x_weight*model.y_to_obs(params), obs_batch['x'].unsqueeze(0)*x_weight).item())
    loss_full = least_squares_fit(model.y_to_obs, obs_batch['x'].unsqueeze(0), x_weight, params)
    print("  final loss full : ", loss_full)
    y_pred = (params + 0.5) * (model.y_max - model.y_min) + model.y_min 
    if cur_indx==start_index*2500:
        y_preds = np.array(y_pred.data.cpu().numpy())
    else:
        y_preds = np.vstack((y_preds, y_pred.data.cpu().numpy()))
#----------------------------------------------------------------------------------------
    # Save every 15 minutes
    if time.time() - cp_start_time >= 15*60:
        np.save(results_filename, y_preds)
        cp_start_time = time.time()
np.save(results_filename, y_preds)
