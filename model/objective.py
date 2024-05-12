import torch
import torch.nn.functional as F
def get_intention_objective(output, train_batch):
    targets_trajectory = train_batch["trajectory"]
    
    intention_trajs = output['intention_trajs']
    batch_size = output['pred_list'][0][0].shape[0]

    pdm_intention_trajs = train_batch['pdm_nonreactive_trajectory'].data[...,:2].permute(1,0,2,3)
    intention_loss = (pdm_intention_trajs - intention_trajs).norm(2,-1).mean(-1).mean()
    return intention_loss

def get_imitation_loss(output, train_batch, future_steps,learning_query):
    targets_trajectory = train_batch["trajectory"]
    
    intention_trajs = output['intention_trajs']
    batch_size = output['pred_list'][0][0].shape[0]

    total_loss = 0
    if learning_query:
        pdm_intention_trajs = train_batch['pdm_nonreactive_trajectory'].data[...,:2].permute(1,0,2,3)
        intention_points = pdm_intention_trajs[:,:,-1].permute(1,0,2)
        
        intentions = intention_points
        target_scores = train_batch['pdm_nonreactive_trajectory'].metrics[:,-3]
        aug_scores = 0.1*torch.ones((batch_size,1),device=intention_trajs.device)
        aug_scores= torch.cat((target_scores,aug_scores),dim=-1)
        soft_target2 = F.softmax(2*aug_scores, dim=-1)[:,:-1].detach()
    else:
        intention_points = output["intention_points"].permute(1,0,2)

    gt = targets_trajectory.xy[:,:future_steps]
    gt_heading = targets_trajectory.heading[:,:future_steps]
    dist = (gt[:,-1].unsqueeze(1) - intention_points).norm(dim=-1)  # (num_center_objects, num_query)
    center_gt_positive_idx = dist.argmin(dim=-1).detach()
    
    all_layer_loss = 0
    for pred in output['pred_list']:
        score, trajs = pred[0], pred[1]
        target_trajs = trajs[torch.arange(batch_size),center_gt_positive_idx]
        l2_norm = torch.norm(
            gt - target_trajs[...,:2],
            p=2,dim=-1
            ).mean(-1)
        sin_loss = (gt_heading.cos() - target_trajs[...,2].cos()).abs().mean(-1)
        cos_loss = (gt_heading.sin() - target_trajs[...,2].sin()).abs().mean(-1)
        
        soft_target = F.softmax(-0.5*dist, dim=-1).detach()
        if intention_trajs is not None:
            soft_target = 0.5*soft_target+(1-0.5)*soft_target2
            
        conf_loss = torch.sum(-soft_target * F.log_softmax(score, dim=-1), dim=-1)
        layer_loss = 1.0*(l2_norm+sin_loss+cos_loss).mean()+1.0*conf_loss.mean()
        all_layer_loss += layer_loss
    all_layer_loss = all_layer_loss/len(output['pred_list'])
    total_loss+= all_layer_loss
    return total_loss

def get_non_reactive_loss(output, train_batch,future_steps):
    non_reactive_traj = output['non_reactive_traj']

    # targets_trajectory = cast(Trajectory, targets["trajectory"])
    # pdm_nonreactive_trajectory
    targets_trajectory = train_batch["trajectory"]
    # targets_trajectory = cast(Trajectory, targets['pdm_nonreactive_trajectory'])
    l2_norm = torch.norm(targets_trajectory.xy[:,:future_steps,:2]
                            - non_reactive_traj,p=2,dim=-1)
    fde = l2_norm[:,-1].mean()
    ade = l2_norm[:,:].mean(-1).mean()
    return fde + ade