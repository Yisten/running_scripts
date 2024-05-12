from typing import List, Optional, cast

import torch
from torch import nn
import torch.nn.functional as F
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from model.hivt_utils import (
    LocalEncoder,
    GlobalInteractor,
    MTRDecoder
)
from torch_geometric.utils import subgraph
import pytorch_lightning as pl
from model.objective import get_intention_objective, get_imitation_loss, get_non_reactive_loss

class HiVT(pl.LightningModule):

    def __init__(
        self,
        num_modes: int, rotate: bool,
        node_dim: int, edge_dim: int,
        embed_dim: int, num_heads: int,
        dropout: float, num_temporal_layers: int,
        num_global_layers: int, local_radius: float,
        past_trajectory_sampling: TrajectorySampling,
        future_trajectory_sampling: TrajectorySampling,
        input_window: int = 20, step_size:int = 5,
        alpha_collision:float = 0.01,
        sigma:float = 0.1, opt_collision:bool=False,
        if_train:bool=False,
        num_groups:int=4,
        learning_query_points:bool=False,
        num_learning_queries:int=15,
        pdm_builder:bool=False,
        num_decoder_layers:int=4,
        max_epochs:int=32,
        **args
    )->None:
        self.save_hyperparameters()
        if learning_query_points:
            assert pdm_builder, "currently, learn queries from pdm."
        self.if_train = if_train
        self.pdm_builder = pdm_builder
        self.learning_query_points = learning_query_points
        
        super().__init__()
        
        self.historical_steps = past_trajectory_sampling.num_poses+1
        self.future_steps = future_trajectory_sampling.num_poses
        self.num_modes = num_modes
        self.rotate = rotate

        self.step_size = step_size
        self.input_window = input_window
        self.horizon = future_trajectory_sampling.time_horizon
        self.timesteps = float(self.horizon/self.future_steps)
        self.alpha_collision=alpha_collision
        self.sigma = sigma

        self.local_encoder = LocalEncoder(
            historical_steps=self.historical_steps,
            node_dim=node_dim,edge_dim=edge_dim,
            embed_dim=embed_dim,num_heads=num_heads,
            dropout=dropout,num_temporal_layers=num_temporal_layers,
            local_radius=local_radius,input_window = input_window,
            if_train=if_train
            )
        
        self.global_interactor = GlobalInteractor(
            historical_steps=self.historical_steps,
            embed_dim=embed_dim, edge_dim=edge_dim,
            num_modes=num_modes, num_heads=num_heads,
            num_layers=num_global_layers, dropout=dropout,
            rotate=rotate, num_groups = num_groups
            )
        self.nonreactive_decode = True
        if self.nonreactive_decode:
            self.nonreactive_decoder = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU(inplace=True),
                nn.Linear(embed_dim, self.future_steps * 2)
                )
            
        self.decoder = MTRDecoder(
                input_dim=embed_dim, d_model=embed_dim//2,
                future_steps=self.future_steps, num_modes=num_modes,
                historical_steps=self.historical_steps,
                 num_modes_pred = 1,
                 attention_heads = 8,
                 dropout = 0.1,
                 num_decoder_layers=num_decoder_layers,
                 learning_query_points=learning_query_points,
                 num_learning_queries=num_learning_queries,
                )

        self.ADE_scale = 2
        self.lam = 0.4
    def forward(self, feature_dict):

        data = feature_dict['hivt_pyg'].pyg_data
        device = data.x.device
        nearby_mask = data['nearby_mask']
        # make ego history be observed by nearby agents
        data.obs_mask[~nearby_mask] = 0
        data = self.data_process(data, feature_dict)

        local_embed = self.local_encoder(data=data)
        agent_global_embed, ego_global_embed =\
            self.global_interactor(data=data, 
                local_embed=local_embed,
                nearby_mask=nearby_mask
            )
        prediction, decision = self.decoder(
            data=data, local_embed=local_embed, 
            agent_global_embed=agent_global_embed,
            ego_global_embed=ego_global_embed,
            nearby_mask=nearby_mask
            )

        if self.nonreactive_decode:
            non_reactive_traj = self.nonreactive_decoder(
                local_embed[~nearby_mask]
                ).reshape(-1,self.future_steps,2)
        else:
            non_reactive_traj = None

        if self.if_train:
            # get prediction loss
            predtraj_norm, predhead_norm, predconf_norm=\
                self.get_prediction_loss(data,prediction)
            trajectory = torch.zeros((1,3,3),device=device)
            ego_polygon = torch.zeros((1,3,3),device=device)
            scores = torch.zeros((1,3,3),device=device)
        else:
            predtraj_norm = torch.zeros(1,device=device)
            predconf_norm = torch.zeros(1,device=device)
            predhead_norm = torch.zeros(1,device=device)

            pred_scores_final, pred_trajs_final = self.decoder.generate_final_prediction(decision['pred_list'])
            trajectory = pred_trajs_final
            # trajectory = decision['pred_list'][-1][1]
            scores = pred_scores_final
            # scores = decision['pred_list'][-1][0]
            ego_polygon = torch.zeros((1,3,3),device=device)
        return {
            "data" : trajectory,
            "polygon": ego_polygon, 
            "score": scores, 
            "prediction":prediction["trajs_global"],
            "pred_list":decision['pred_list'],
            "decision_trajectories":decision['trajs'],
            "decision_headings":decision['heads'],
            "decision_condfidence":decision['confs'],
            "intention_points":decision['intention_points'],
            "intention_trajs":decision['intention_trajs'],
            "predtraj_norm": predtraj_norm,
            "predconf_norm" :predconf_norm,
            "predhead_norm": predhead_norm,
            "batch": data['batch'],
            "importance": torch.tensor(1).to(device=device),
            "non_reactive_traj":non_reactive_traj,
            }
    
    def data_process(self,data,feature_dict):

        if self.if_train:
            data.y = data.y[:,:self.future_steps]
            data.y_ang = data.y_ang[:,:self.future_steps]
        data.positions = data.positions[:,:self.historical_steps+self.future_steps]
        data.padding_mask = data.padding_mask[:,:self.historical_steps+self.future_steps]
        data.rotate_angles = data.rotate_angles[:,:self.historical_steps+self.future_steps]
        
        data['route_edge'] = feature_dict['hivt_pyg'].route_data.x[:,:2]
        data['route_node'] = feature_dict['hivt_pyg'].route_data.x[:,2:]
        data['route_batch'] = feature_dict['hivt_pyg'].route_data.batch

        #lane downsample
        if True:
            data['lane_traffic'] = data['lane_traffic'][::2]
            data['lane_route'] = data['lane_route'][::2]
            data['lane_vectors'] = data['lane_vectors'][::2]
            idx = data['lane_actor_index'][0]%2 == 0
            data['lane_actor_index'] = data['lane_actor_index'][:,idx]
            data['lane_actor_index'][0] = data['lane_actor_index'][0] // 2
            data['lane_actor_vectors'] = data['lane_actor_vectors'][idx]
        return data
    
    def get_prediction_loss(self,data,prediction):
        nearby_mask = data['nearby_mask']
        fut_mask = ~data['padding_mask'][
            nearby_mask,self.historical_steps:
            ]
        valid_mask = (fut_mask.sum(1) >4)
        l2_norm = (
                prediction['trajs'] - data.y.unsqueeze(1)
                ).norm(2,-1)[valid_mask]
        fut_mask = fut_mask[valid_mask]
        ADE_norm = (
            l2_norm.masked_fill(~fut_mask.unsqueeze(1),0)
            ).sum(-1) / fut_mask.sum(1).unsqueeze(1)
        minADE_idx = torch.argmin(ADE_norm,dim=-1).detach()
        predtraj_norm = l2_norm[
            torch.arange(l2_norm.shape[0]),minADE_idx
            ].masked_fill(~fut_mask,0)
        heading_ = prediction['heads'][valid_mask][
            torch.arange(l2_norm.shape[0]),minADE_idx
            ]
        predhead_norm = fut_mask*\
            torch.abs(heading_ - data.y_ang[valid_mask])
        predtraj_norm = predtraj_norm.sum(-1) / fut_mask.sum(1)
        predhead_norm = predhead_norm.sum(-1) / fut_mask.sum(1)
        predconf_norm = -F.softmax(-self.ADE_scale*ADE_norm,dim=1)*\
            torch.log(
                F.softmax(prediction['confs'].squeeze(-1),dim=1)
                )[valid_mask]
        predconf_norm = predconf_norm.sum(-1)
        return predtraj_norm, predhead_norm, predconf_norm
    
    def training_step(self, train_batch, batch_idx):
        output = self.forward(train_batch)
        loss = 0
        logs = {}
        if self.learning_query_points:
            intention_loss = get_intention_objective(output, train_batch)
            loss += intention_loss
        imitation_loss = get_imitation_loss(
            output, train_batch, self.future_steps, self.learning_query_points
            )
        non_reactive_loss = get_non_reactive_loss(output, train_batch,self.future_steps)
        prediction_loss = output['predtraj_norm']+\
            output['predhead_norm']+output['predconf_norm']
        
        loss = loss + 0.5*non_reactive_loss + imitation_loss + prediction_loss
        self.log("train_loss",loss,on_step=True,on_epoch=True,prog_bar=True,logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = 0
        logs = {}
        if self.learning_query_points:
            intention_loss = get_intention_objective(output, batch)
            logs['intention_loss'] = intention_loss
            loss += intention_loss

        imitation_loss = get_imitation_loss(
            output, batch, self.future_steps, self.learning_query_points
            )
        
        non_reactive_loss = get_non_reactive_loss(output, batch,self.future_steps)
        prediction_loss = output['predtraj_norm']+\
            output['predhead_norm']+output['predconf_norm']
        loss = loss + 0.5*non_reactive_loss + imitation_loss + prediction_loss
        self.log("val_loss",loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-4, weight_decay=5e-4,
            betas=[0.9,0.999]
            )
        return optimizer
    
    def lr_schedulers(self):
        lr_schedulers = torch.optim.lr_scheduler.CosineAnnealingLR(
            T_max=self.hparams.max_epochs,
            eta_min=5e-6
            )
        return lr_schedulers