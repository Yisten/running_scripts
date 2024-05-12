from typing import Optional,List,Tuple
import time
import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptTensor
from torch_geometric.typing import Size
from torch_geometric.utils import softmax
from torch_geometric.utils import subgraph
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import Data
from nuplan.planning.training.modeling.models.mha_lib import MultiheadAttention,TransformerEncoder
# from torch_geometric.nn.pool import global_mean_pool
from torch_scatter import scatter
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_bicycle_layer_geometric_center import KinematicBicycleLayerGeometricCenter
from nuplan.planning.training.modeling.models.dynamics_layers.deep_dynamical_system_layer import DeepDynamicalSystemLayer

from nuplan.planning.training.modeling.models.transformer import position_encoding_utils
import copy

def build_mlps(c_in, mlp_channels=None, ret_before_act=False, without_norm=False):
    layers = []
    num_layers = len(mlp_channels)

    for k in range(num_layers):
        if k + 1 == num_layers and ret_before_act:
            layers.append(nn.Linear(c_in, mlp_channels[k], bias=True))
        else:
            if without_norm:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=True), nn.ReLU()]) 
            else:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=False), nn.BatchNorm1d(mlp_channels[k]), nn.ReLU()])
            c_in = mlp_channels[k]

    return nn.Sequential(*layers)

def get_batch(batch,index):
    phase = 0
    for vanish_node in torch.where(torch.eq(index,False))[0]:
        batch[batch>vanish_node-phase] = batch[batch>vanish_node-phase] - 1
        phase = phase + 1
    return batch

class TemporalData(Data):

    def __init__(self,
                 x: Optional[torch.Tensor] = None,
                 positions: Optional[torch.Tensor] = None,
                 edge_index: Optional[torch.Tensor] = None,
                 edge_attrs: Optional[List[torch.Tensor]] = None,
                 y: Optional[torch.Tensor] = None,
                 num_nodes: Optional[int] = None,
                 padding_mask: Optional[torch.Tensor] = None,
                 bos_mask: Optional[torch.Tensor] = None,
                 rotate_angles: Optional[torch.Tensor] = None,
                 lane_vectors: Optional[torch.Tensor] = None,
                 lane_actor_index: Optional[torch.Tensor] = None,
                 lane_actor_vectors: Optional[torch.Tensor] = None,
                 seq_id: Optional[int] = None,
                 **kwargs) -> None:
        if x is None:
            super(TemporalData, self).__init__()
            return
        super(TemporalData, self).__init__(x=x, positions=positions, edge_index=edge_index, y=y, num_nodes=num_nodes,
                                           padding_mask=padding_mask, bos_mask=bos_mask, rotate_angles=rotate_angles,
                                           lane_vectors=lane_vectors,
                                           lane_actor_index=lane_actor_index, lane_actor_vectors=lane_actor_vectors,
                                           seq_id=seq_id, **kwargs)
        if edge_attrs is not None:
            for t in range(self.x.size(1)):
                self[f'edge_attr_{t}'] = edge_attrs[t]
def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)

class GlobalInteractor(nn.Module):

    def __init__(self,
                 historical_steps: int, embed_dim: int,
                 edge_dim: int, num_modes: int = 6,
                 num_heads: int = 8, num_layers: int = 3,
                 num_modes_pred: int = 6, dropout: float = 0.1,
                 rotate: bool = True,
                 num_groups:int=3,
                 ) -> None:
        super(GlobalInteractor, self).__init__()
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_modes = num_modes
        self.num_modes_pred = num_modes_pred
        if rotate:
            self.rel_embed = MultipleInputEmbedding(in_channels=[edge_dim, edge_dim], out_channel=embed_dim)
        else:
            self.rel_embed = SingleInputEmbedding(in_channel=edge_dim, out_channel=embed_dim)
        self.global_interactor_layers = nn.ModuleList(
            [GlobalInteractorLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
             for _ in range(num_layers)]
             )
 
        self.apply(init_weights)

    def forward(self,
                data: TemporalData,
                local_embed: torch.Tensor,
                nearby_mask:torch.Tensor) -> torch.Tensor:
        device = local_embed.device
        if data.edge_index.shape[0]>0:
            edge_index, _ = subgraph(
                subset=~data['padding_mask'][:, self.historical_steps - 1],
                edge_index=data.edge_index
                )
            rel_pos = (
                data['positions'][edge_index[0], self.historical_steps - 1] -\
                data['positions'][edge_index[1], self.historical_steps - 1]
            ).to(torch.float32)
        else:
            num_edge = 0
            edge_index = torch.zeros((2,num_edge),
                                     dtype=torch.long,device=device)
            rel_pos = torch.zeros((num_edge,2),
                                  dtype=torch.float32,device=device)

        rel_pos = torch.bmm(
            rel_pos.unsqueeze(-2),
            data['rotate_mat'][edge_index[1]]
        ).squeeze(-2)
        rel_theta = data['rotate_angles'][edge_index[0],self.historical_steps-1] - \
            data['rotate_angles'][edge_index[1],self.historical_steps-1]
        rel_theta_cos = torch.cos(rel_theta).unsqueeze(-1)
        rel_theta_sin = torch.sin(rel_theta).unsqueeze(-1)
        rel_embed = self.rel_embed(
            [rel_pos, torch.cat((rel_theta_cos, rel_theta_sin), dim=-1)]
            )
        x = local_embed
        for layer in self.global_interactor_layers:
            x = layer(x, edge_index, rel_embed)

        return x[nearby_mask], x[~nearby_mask]


class GlobalInteractorLayer(MessagePassing):

    def __init__(self,
                embed_dim: int,
                num_heads: int = 8,
                dropout: float = 0.1,
                 **kwargs) -> None:
        super(GlobalInteractorLayer, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.lin_q_node = nn.Linear(embed_dim, embed_dim)
        self.lin_k_node = nn.Linear(embed_dim, embed_dim)
        self.lin_k_edge = nn.Linear(embed_dim, embed_dim)
        self.lin_v_node = nn.Linear(embed_dim, embed_dim)
        self.lin_v_edge = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))

    def forward(self,
                x: torch.Tensor,
                edge_index: Adj,
                edge_attr: torch.Tensor,
                size: Size = None,
                ) -> torch.Tensor:
        x = x + self._mha_block(self.norm1(x), edge_index, edge_attr, size)
        x = x + self._ff_block(self.norm2(x))
        return x
    
    def message(self,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:
        query = self.lin_q_node(x_i).view(
            -1, self.num_heads, self.embed_dim // self.num_heads
            )
        key_node = self.lin_k_node(x_j).view(
            -1, self.num_heads, self.embed_dim // self.num_heads
            )
        key_edge = self.lin_k_edge(edge_attr).view(
            -1, self.num_heads, self.embed_dim // self.num_heads
            )
        value_node = self.lin_v_node(x_j).view(
            -1, self.num_heads, self.embed_dim // self.num_heads
            )
        value_edge = self.lin_v_edge(edge_attr).view(
            -1, self.num_heads, self.embed_dim // self.num_heads
            )
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * (key_node + key_edge)).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return (value_node + value_edge) * alpha.unsqueeze(-1)

    def update(self,
               inputs: torch.Tensor,
               x: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x))
        return inputs + gate * (self.lin_self(x) - inputs)

    def _mha_block(self,
                   x: torch.Tensor,
                   edge_index: Adj,
                   edge_attr: torch.Tensor,
                   size: Size) -> torch.Tensor:
        x = self.out_proj(self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=size))
        return self.proj_drop(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
    
class SingleInputEmbedding(nn.Module):

    def __init__(self,
                 in_channel: int,
                 out_channel: int) -> None:
        super(SingleInputEmbedding, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_channel, out_channel // 4),
            nn.LayerNorm(out_channel // 4),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel // 4, out_channel // 2),
            nn.LayerNorm(out_channel // 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel // 2, out_channel),
            nn.LayerNorm(out_channel))
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class MultipleInputEmbedding(nn.Module):

    def __init__(self,
                 in_channels: List[int],
                 out_channel: int) -> None:
        super(MultipleInputEmbedding, self).__init__()
        self.module_list = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_channel, out_channel // 2),
                           nn.LayerNorm(out_channel // 2),
                           nn.ReLU(inplace=True),
                           nn.Linear(out_channel // 2, out_channel))
             for in_channel in in_channels])
        self.aggr_embed = nn.Sequential(
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel))
        self.apply(init_weights)

    def forward(self,
                continuous_inputs: List[torch.Tensor],
                # categorical_inputs: Optional[List[torch.Tensor]] = None
                ) -> torch.Tensor:
        for i in range(len(self.module_list)):
            continuous_inputs[i] = self.module_list[i](continuous_inputs[i])
        output = torch.stack(continuous_inputs).sum(dim=0)

        return self.aggr_embed(output)

class SimpleLinear(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=[64, 32], scale=None):
        super(SimpleLinear, self).__init__()
        self.hidden_dim = hidden_dim
        self.hidden_layers = len(hidden_dim)
        self.fhidden = nn.ModuleList([None] * (self.hidden_layers - 1))

        if isinstance(scale, torch.Tensor):
            self.scale = scale
        else:
            self.scale = scale

        for i in range(1, self.hidden_layers):
            self.fhidden[i - 1] = nn.Linear(hidden_dim[i - 1], hidden_dim[i])
        self.f1 = nn.Linear(input_dim, hidden_dim[0])
        self.f2 = nn.Linear(hidden_dim[-1], output_dim)

    def forward(self, x):
        hidden = self.f1(x)
        for i in range(1, self.hidden_layers):
            hidden = self.fhidden[i - 1](F.relu(hidden))
        if not self.scale is None:
            return torch.tanh(self.f2(F.relu(hidden))) * self.scale
        else:
            return self.f2(F.relu(hidden))
    
class AttDest(nn.Module):
    def __init__(self, hidden_size: int, dropout=0.1):
        super(AttDest, self).__init__()
        self.hidden_size = hidden_size
        self.dist = nn.Sequential(
            nn.Linear(3, self.hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size // 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
        )

        self.agt = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size // 2, 1))
        self.dropout = nn.Dropout(dropout)
    def forward(self, agts: torch.Tensor, dest_ctrs: torch.Tensor) -> torch.Tensor:
        # n_agt = agts.size(1)
        # num_mods = dest_ctrs.size(1)
        dist = self.dist(dest_ctrs)
        agts = agts + self.dropout(dist)
        conf = self.agt(agts)
        return conf
    
def get_nearby_mask(batch,device):
        #near_mask:0 means ego,1 means nearby agent. As shown in preprocessing, the last agent in one sample is ego.
        num_nodes = batch.shape[0]
        near_mask = torch.ones(num_nodes,dtype=torch.bool,device=device)
        t = (batch[1:]-batch[:-1]).bool()
        near_mask[:-1][t] = False
        near_mask[-1] = False
        near_mask = near_mask.detach()
        return near_mask

class MLPDecoder(MessagePassing):

    def __init__(self,
                 local_channels: int, global_channels: int,
                 future_steps: int, num_modes: int,
                 historical_steps:int, min_scale: float = 1e-3,
                 num_modes_pred:int = 6,
                 avoid_collision: bool=False,
                 alert_dis:float=3, step_size:int=5,
                 infer_plan:bool = True, num_groups:int=3,
                 ) -> None:
        super(MLPDecoder, self).__init__()
        self.input_size = global_channels
        self.hidden_size = local_channels
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.min_scale = min_scale
        self.historical_steps = historical_steps
        self.step_size = step_size
        self.num_heads = 8
        self.avoid_collision = avoid_collision
        self.alert_dis = alert_dis
        # decision decoder
        self.num_modes_pred = num_modes_pred
        self.infer_plan = infer_plan

        self.state_dim = 3
        self.input_dim = 2
        self.num_groups = num_groups
        self.modes_per_group = self.num_modes//self.num_groups
        self.if_rule_scores = False

        self.max_acceleration = 3.0 #(m/s^2)
        self.max_steering_rate = 0.5 #(rad/s)
        self.dropout1 = nn.Dropout(0.1)
        self.apply(init_weights)

    def get_batch(self,batch,index):
        phase = 0
        for vanish_node in torch.where(index==False)[0]:
            batch[batch>vanish_node-phase] = batch[batch>vanish_node-phase] - 1
            phase = phase + 1
        return batch
    
    def align(self,theta,batch,velocity,device):
        ego_velocity = torch.zeros_like(velocity,device=device)
        ego_velocity[:,0] = velocity.norm(2,-1)
        ego_velocity[:,1] = 0
        return ego_velocity
    
    def forward(self,
                data,
                local_embed: torch.Tensor, agent_global_embed: torch.Tensor,
                ego_global_embed:torch.Tensor, nearby_mask:torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = local_embed.device
        batch_size = ego_global_embed.shape[1]
        decision = {}
        ego_local_embed = local_embed[~nearby_mask]
        agent_local_embed = local_embed[nearby_mask]
        prediction = {}

        obs_out = self.aggr_embed1(
            torch.cat((agent_global_embed, agent_local_embed.expand(self.num_modes_pred, *agent_local_embed.shape)), dim=-1)
            ).permute(1,0,2)
        
        # get prediction
        prediction_trajs = self.pred(obs_out).view(-1, self.num_modes_pred,  self.future_steps, 2)
        prediction_headings = (self.pred_headings(obs_out).sigmoid()* 2 -1)* torch.pi
        prediction_confs = self.pred_conf(obs_out, 
                                torch.cat((prediction_trajs[:,:,-1], prediction_headings[:,:,-1].unsqueeze(-1)), dim=2)
                                )
        
        #recover coordinates from agent local frame
        prediction['trajs'], prediction['heads'], prediction['conf'] = prediction_trajs, prediction_headings ,prediction_confs
        prediction_trajs = torch.matmul(prediction_trajs.flatten(1,2),data['inv_rotate_mat'][nearby_mask]).reshape(-1,self.num_modes_pred,self.future_steps, 2)
        prediction_trajs = prediction_trajs + data['positions'][nearby_mask,self.historical_steps-1].unsqueeze(1).unsqueeze(1)
        prediction_headings = prediction_headings + data['rotate_angles'][nearby_mask, self.historical_steps-1].unsqueeze(-1).unsqueeze(-1)
        prediction['rec_trajs'],prediction['rec_heads'] = prediction_trajs.detach(), prediction_headings.detach()
        
        multimode_ego_feature = ego_local_embed.unsqueeze(1) + ego_global_embed.permute(1,0,2)
        multimode_ego_feature = self.aggr_embed2(multimode_ego_feature)
        
        # pre ego
        planning_trajs_pos = self.plan_pos(multimode_ego_feature).view(-1, self.num_modes_pred,  self.future_steps, 2)
        planning_headings_pos = (self.plan_pos_headings(multimode_ego_feature).sigmoid()* 2 -1)* torch.pi
        planning_confs_pos = self.plan_pos_conf(multimode_ego_feature, 
                                torch.cat((planning_trajs_pos[:,:,-1], planning_headings_pos[:,:,-1].unsqueeze(-1)), dim=2)
                                )


        edge_feature = self.route_edge_encoder(data['route_edge'])
        node_feature = self.route_node_encoder(data['route_node'])
        temp = torch.zeros_like(multimode_ego_feature,device=device,dtype=multimode_ego_feature.dtype)
        unique_index = torch.unique(data['route_batch'])
        index = torch.zeros(multimode_ego_feature.size(0),dtype=torch.bool,device=device)
        index[unique_index] = True

        # route exits
        batch = get_batch(data['route_batch'],index)
        temp[index] = multimode_ego_feature[index]+ \
            self.route_aggregator(batch, multimode_ego_feature[index], [edge_feature.unsqueeze(1),node_feature.unsqueeze(1)])
        temp[~index] = multimode_ego_feature[~index]
        multimode_ego_feature = temp

        if self.infer_plan:
            initial_state = torch.zeros((data.ego_state.shape[0],self.num_modes,3),dtype=torch.float,device=device)
            self.modes_per_group = self.num_modes // self.num_groups
            decision_trajs, decision_headings, decision_confs, rule_scores =\
                [None]*self.num_groups, [None]*self.num_groups, [None]*self.num_groups, [None]*self.num_groups
            
            # for j in range(self.num_groups):
            j=0
            batch_state_pred = [None]*(self.future_steps // self.step_size)
            batch_input_pred = [None]*(self.future_steps // self.step_size)
            current_state = initial_state[:, j*self.modes_per_group:(j+1)*self.modes_per_group]
            state_lstm_h = self.state_lstm_h0[j](current_state).flatten(0,1).unsqueeze(0) 
            state_lstm_c = self.state_lstm_c0[j](current_state).flatten(0,1).unsqueeze(0)
            
            scene_feat = self.scene_linear[j](multimode_ego_feature[:,j*self.modes_per_group:(j+1)*self.modes_per_group])
            for t in range(self.future_steps // self.step_size):

                cs_feat = self.cs_linear[j](current_state).to(dtype=torch.float)

                direction = torch.stack((torch.cos(current_state[data['route_batch']][...,2]),
                                        torch.sin(current_state[data['route_batch']][...,2]))).permute(1,2,0)
                dot_pdct = (data['route_node'].unsqueeze(1)*direction).sum(-1)/((data['route_node'].unsqueeze(1).norm(2,-1)+1e-6)*direction.norm(2,-1))
                local_edge_feature = self.local_scope_route_edge_encoder[j](data['route_edge'].unsqueeze(1) - current_state[data['route_batch']][...,:2])
                node = torch.cat((dot_pdct.unsqueeze(-1),
                                data['route_node'].unsqueeze(1).repeat(1,self.modes_per_group,1))
                                ,dim=-1)
                local_node_feature = self.local_scope_route_node_encoder[j](node)
                temp = torch.zeros_like(cs_feat,device=device,dtype=cs_feat.dtype)
                temp[index] = cs_feat[index]+ \
                    self.local_scope_route[j](batch, cs_feat[index], [local_edge_feature,local_node_feature])
                temp[~index] = cs_feat[~index]
                cs_feat = temp

                state_lstm_input = cs_feat.flatten(0,1).unsqueeze(0)
                state_lstm_out,(state_lstm_h,state_lstm_c) = self.state_lstm[j](state_lstm_input, (state_lstm_h, state_lstm_c))
                state_lstm_out = state_lstm_out.reshape(data.ego_state.shape[0], self.modes_per_group,-1)
                # batch_input_pred[t] = self.action_net(torch.cat((current_state, state_lstm_out, obs_lstm_out, scene_feat),dim=-1))
                action1 = self.action_net1[j](torch.cat(
                    (current_state, state_lstm_out, scene_feat),dim=-1)).unsqueeze(-1)
                action2 = self.action_net2[j](torch.cat(
                    (current_state, state_lstm_out, scene_feat),dim=-1)).unsqueeze(-1)
                action3 = self.action_net3[j](torch.cat(
                    (current_state, state_lstm_out, scene_feat),dim=-1)).unsqueeze(-1)
                batch_input_pred[t] = torch.cat((action1,action2),dim=-1)

                batch_u = batch_input_pred[t].detach()
                if True:
                    xy_state = batch_input_pred[t].cumsum(-2) + current_state[...,:2].unsqueeze(-2)
                else:
                    xy_state = batch_input_pred[t]
                batch_state_pred[t] = torch.cat((xy_state,action3),dim=-1)
                
                current_state = batch_state_pred[t][:,:,-1].detach()
            decision_states = torch.cat(batch_state_pred,dim=-2)
            decision_trajs[j] = decision_states[...,:2]
            decision_headings[j] = decision_states[...,2]
            decision_confs[j] = self.decision_conf[j](multimode_ego_feature[:,j*self.modes_per_group:(j+1)*self.modes_per_group],
                            torch.cat((decision_trajs[j][:,:,-1], decision_headings[j][:,:,-1].unsqueeze(-1)), dim=2))
            # forend
            decision_trajs = torch.cat(decision_trajs,dim=1)
            decision_headings = torch.cat(decision_headings,dim=1)
            decision_confs = torch.cat(decision_confs,dim=1)
            rule_scores = torch.zeros_like(decision_confs,device=device)
        else:
            decision_trajs = torch.zeros((batch_size, self.num_modes,self.future_steps,2),device=device)
            decision_headings = torch.zeros((batch_size, self.num_modes,self.future_steps),device=device)
            decision_confs = torch.zeros((batch_size, self.num_modes,1),device=device)
        decision['action1'] = torch.zeros(1,device=device)
        decision['action2'] = torch.zeros(1,device=device)
        prediction['obstacles'] = torch.zeros(1,device=device)
        prediction['obstacles_batch'] = torch.zeros(1,device=device)
        decision['trajs'], decision['heads'], decision['conf'] =\
              decision_trajs, decision_headings, decision_confs.squeeze(-1)
        decision['trajs_pos'], decision['heads_pos'], decision['conf_pos'] =\
                planning_trajs_pos, planning_headings_pos, planning_confs_pos.squeeze(-1)
        
        return prediction, decision


class MultiModeEncoder(nn.Module):
    def __init__(self,hidden_dim,num_layers,num_modes,num_heads,dropout=0.1):
        super(MultiModeEncoder, self).__init__()
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.num_modes = num_modes
        self.num_heads = num_heads
        self.lin_k = nn.ModuleList(
            [nn.Linear(self.hidden_size // 2,self.hidden_size) for i in range(self.num_layers)]
            )
        self.lin_q = nn.ModuleList(
            [nn.Linear(self.hidden_size,self.hidden_size) for i in range(self.num_layers)]
            )
        self.lin_v1 = nn.ModuleList(
            [nn.Linear(self.hidden_size // 2,self.hidden_size) for i in range(self.num_layers)]
            )
        self.lin_v2 = nn.ModuleList(
            [nn.Linear(self.hidden_size // 2,self.hidden_size) for i in range(self.num_layers)]
            )
        self.norm1 = nn.ModuleList(
            [nn.LayerNorm(self.hidden_size)for i in range(self.num_layers)]
            )
        self.norm2 = nn.ModuleList(
            [nn.LayerNorm(self.hidden_size)for i in range(self.num_layers)]
            )
        self.linear1 = nn.ModuleList(
            [nn.Linear(self.hidden_size,self.hidden_size) for i in range(self.num_layers)]
            )
        self.linear2 = nn.ModuleList(
            [nn.Linear(self.hidden_size,self.hidden_size) for i in range(self.num_layers)]
            )
        self.dropout = nn.ModuleList(
            [nn.Linear(self.hidden_size,self.hidden_size) for i in range(self.num_layers)]
            )
        self.apply(init_weights)

    def _sa_block(self,x,edge_feat,node_feat,batch,num_nodes,i,nf):
        k = self.lin_k[i](edge_feat).view(-1, nf, self.num_heads,self.hidden_size // self.num_heads)
        q = self.lin_q[i](x).view(-1, self.num_modes, self.num_heads,self.hidden_size // self.num_heads)
        v1 = self.lin_v1[i](edge_feat).view(-1, nf, self.num_heads,self.hidden_size // self.num_heads)
        v2 = self.lin_v2[i](node_feat).view(-1, nf, self.num_heads,self.hidden_size // self.num_heads)
        v = v1 + v2
        alpha = softmax((k* q[batch]).sum(-1),index = batch,dim=0,num_nodes=num_nodes)
        return scatter((v*alpha.unsqueeze(-1)).flatten(2,3),index=batch,dim=-3,reduce="sum")

    def _encode_layer(self,ego_feature,edge_feat,node_feat,batch,num_nodes,i):
        ego_feature = ego_feature + self._sa_block(self.norm1[i](ego_feature),
                                        edge_feat,node_feat,batch,num_nodes,i,nf=node_feat.shape[1])
        ego_feature = ego_feature + self.linear2[i](self.dropout[i](F.relu(self.linear1[i](self.norm2[i](ego_feature)))))
        return ego_feature
    
    def forward(self,batch,ego_feature,obs_out):
        edge_feat, node_feat = obs_out[0], obs_out[1]
        num_nodes = ego_feature.shape[0]
        for i in range(self.num_layers):
            ego_feature = self._encode_layer(
                ego_feature,edge_feat,node_feat,batch,num_nodes,i
                )
        return ego_feature
    
class RouteEncoderLayer(nn.Module):
    def __init__(self,embed_dim,num_modes,num_heads,dropout=0.1):
        super(RouteEncoderLayer, self).__init__()
        self.num_modes = num_modes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # self-attention 
        self.sa_qcontent_proj = nn.Linear(embed_dim, embed_dim)
        self.sa_qpos_proj = nn.Linear(embed_dim, embed_dim)
        self.sa_kcontent_proj = nn.Linear(embed_dim, embed_dim)
        self.sa_kpos_proj = nn.Linear(embed_dim, embed_dim)
        self.sa_v_proj = nn.Linear(embed_dim, embed_dim)
        self.self_attn = MultiheadAttention(
            embed_dim, num_heads, dropout=dropout
            )
        self.sa_dropout = nn.Dropout(dropout)
        self.sa_norm = nn.LayerNorm(embed_dim)
        #cross-attention
        self.lin_v1 = nn.Linear(embed_dim,embed_dim)
        self.lin_v2 = nn.Linear(embed_dim,embed_dim)
        self.norm1 = nn.Linear(embed_dim,embed_dim)
        self.norm2 = nn.Linear(embed_dim,embed_dim)
        self.norm_v1 = nn.LayerNorm(embed_dim)
        self.norm_v2 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim,embed_dim)
        self.linear2 = nn.Linear(embed_dim,embed_dim)
        self.ca_q_proj = nn.Linear(embed_dim,embed_dim)
        self.ca_qpos_proj = nn.Linear(embed_dim,embed_dim)
        self.ca_qsinpos_proj = nn.Linear(embed_dim,embed_dim)
        self.ca_k_proj = nn.Linear(embed_dim,embed_dim)
        self.ca_k_pos_proj=nn.Linear(embed_dim,embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.apply(init_weights)

    def _ca_block(self,q,query_sine_embed,k,k_pos,v,batch,num_nodes):
        k = k.view(-1, self.num_modes, self.num_heads,self.embed_dim // self.num_heads)
        k_pos = k_pos.view(-1, 1, self.num_heads,self.embed_dim // self.num_heads)
        k_pos = k_pos.repeat(1,self.num_modes,1,1)
        k = torch.cat((k,k_pos),dim=-1)

        q = q.view(-1, self.num_modes, self.num_heads,self.embed_dim // self.num_heads)
        query_sine_embed = query_sine_embed.view(-1, self.num_modes, self.num_heads,self.embed_dim // self.num_heads)
        q = torch.cat((q,query_sine_embed),dim=-1)

        alpha = softmax((k* q[batch]).sum(-1),index = batch,dim=0,num_nodes=num_nodes)
        return scatter((v*alpha.unsqueeze(-1)).flatten(2,3),index=batch,dim=-3,reduce="sum")
    
    def _encode_layer(self,ego_feature,query_pos,
                      edge_feat,node_feat,batch,
                      query_sine_embed, kv_pos, num_nodes):
        q_pos = self.ca_qpos_proj(query_pos)
        query_sine_embed = self.ca_qsinpos_proj(query_sine_embed)
        q = self.ca_q_proj(ego_feature + q_pos)
        
        k = self.ca_k_proj(edge_feat)
        k_pos =self.ca_k_pos_proj(kv_pos)

        v1 = self.lin_v1(edge_feat).view(-1, self.num_modes, self.num_heads,self.embed_dim // self.num_heads)
        v2 = self.lin_v2(node_feat).view(-1, 1, self.num_heads,self.embed_dim // self.num_heads)
        v = v1 + v2

        ego_feature = ego_feature + self._ca_block(q, query_sine_embed,
                                        k,k_pos,v,batch,num_nodes)
        ego_feature = ego_feature + self.linear2(self.dropout(F.relu(self.linear1(self.norm2(ego_feature)))))
        return ego_feature
    
    def forward(self,batch,ego_feature,query_pos,query_sine_embed,kv_pos,obs_out):
        # attention among proposals/queries
        ego_feature = ego_feature +\
              self.sa_dropout(self._sa_block(self.sa_norm(ego_feature), query_pos))#(num_query, batch_size,-1)
        ego_feature = ego_feature.permute(1,0,2)
        query_pos = query_pos.permute(1,0,2)
        
        # cross attention
        edge_feat, node_feat = obs_out[0], obs_out[1]
        num_nodes = ego_feature.shape[0]
        
        ego_feature = self._encode_layer(
            self.norm1(ego_feature),query_pos,self.norm_v1(edge_feat),
            self.norm_v2(node_feat),
            batch,query_sine_embed,kv_pos,num_nodes
        )
        return ego_feature.permute(1,0,2)
    
    def _sa_block(self,tgt, query_pos):
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.sa_qcontent_proj(tgt)      # target is the input of the first decoder layer. zero by default.
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(tgt)

        q = q_content + q_pos
        k = k_content + k_pos

        return self.self_attn(q, k, value=v, attn_mask=None,
                                key_padding_mask=None)[0]

class DistanceDropEdge(object):

    def __init__(self, max_distance: Optional[float] = None) -> None:
        self.max_distance = max_distance

    def __call__(self,
                 edge_index: torch.Tensor,
                 edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index
        mask = torch.norm(edge_attr, p=2, dim=-1) < self.max_distance
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        edge_attr = edge_attr[mask]
        return edge_index, edge_attr
    
class AgentEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim,num_heads=8,if_train=False,vel_dropout=0.75) -> None:
        super().__init__()
        self.if_train = if_train
        self.linear = nn.ModuleList([])
        for i in range(input_dim):
            self.linear.append(nn.Linear(1,embed_dim))
        self.input_dim = input_dim
        self.pos_embed = nn.Parameter(torch.Tensor(1, input_dim, embed_dim))
        self.query = nn.Parameter(torch.Tensor(1, 1, embed_dim))

        self.linear_q,self.linear_k,self.linear_v =\
            nn.Linear(embed_dim,embed_dim),nn.Linear(embed_dim,embed_dim),nn.Linear(embed_dim,embed_dim)
        self.num_heads = num_heads
        self.vel_dropout = vel_dropout
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, agent_input):
        
        x_embed = []
        for i in range(self.input_dim):
            x_embed.append(
                self.linear[i](agent_input[...,i].unsqueeze(-1))
            )
        x_embed = torch.stack(x_embed, dim=-2)
        pos_embed = self.pos_embed.repeat(
            x_embed.shape[0],x_embed.shape[1], 1, 1
            )
        x_embed += pos_embed
        return self.MHA(q=self.query, k=x_embed, v=x_embed)

    def MHA(self,q,k,v):
        q,k,v = self.linear_q(q),self.linear_k(k),self.linear_v(v)
        d1,d2 = k.shape[0],k.shape[1]

        mask = torch.zeros((d1,d2,self.input_dim),device=k.device)
        if self.if_train:
            mask[...,-2:] =\
                torch.rand((d1,d2,2),device=k.device) < self.vel_dropout

        q = q.reshape(1,1,1,self.num_heads,-1)
        k = k.reshape(d1,d2,self.input_dim,self.num_heads,-1)
        v = v.reshape(d1,d2,self.input_dim,self.num_heads,-1)
        alpha = (q * k).sum(-1).masked_fill(
            mask.bool().unsqueeze(-1),-1e6
        )
        alpha = alpha.softmax(-2)
        v = (alpha.unsqueeze(-1)*v).sum(-3).reshape(d1,d2,-1)
        return v
    
class LocalEncoder(nn.Module):

    def __init__(self,
                 historical_steps: int, node_dim: int,
                 edge_dim: int, embed_dim: int,
                 num_heads: int = 8, dropout: float = 0.1,
                 num_temporal_layers: int = 4,
                 local_radius: float = 50,
                 input_window: int = 20,
                 parallel: bool = False,
                 abvehicle: bool = False,
                 if_train: bool = False) -> None:
        super(LocalEncoder, self).__init__()
        self.historical_steps = historical_steps
        self.input_window = input_window
        self.parallel = parallel
        self.embed_dim = embed_dim
        self.drop_edge = DistanceDropEdge(local_radius)
        self.agent_input_dim = 9
        self.agent_encoder = AgentEncoder(input_dim = self.agent_input_dim,
                                    embed_dim = embed_dim,
                                    if_train = if_train,
                                    vel_dropout=0)
        self.ego_encoder = AgentEncoder(input_dim = self.agent_input_dim,
                                    embed_dim = embed_dim,
                                    if_train = if_train,
                                    vel_dropout=0.75)
        self.ego_mlp = nn.Sequential(
            nn.Linear(embed_dim, 2*embed_dim, bias=True),
            # nn.LayerNorm(2*embed_dim),
            nn.ReLU(),
            nn.Linear(2*embed_dim,2*embed_dim,bias=True),
            nn.ReLU(),
            nn.Linear(2*embed_dim,embed_dim,bias=True)
        )
        self.abvehicle = abvehicle
        self.temporal_encoder = TemporalEncoder(historical_steps=input_window,
                                                embed_dim=embed_dim,
                                                num_heads=num_heads,
                                                dropout=dropout,
                                                num_layers=num_temporal_layers)
        self.al_encoder = ALEncoder(node_dim=node_dim,
                                    edge_dim=edge_dim,
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    dropout=dropout)

    def forward(self, data: TemporalData) -> torch.Tensor:

        trajs_input = self.get_traj_feats(data)

        out = self.encode_trajs(trajs_input, data)
        
        out = self.fuse_lane(out, data)
        
        return out
    
    def get_traj_feats(self, data):

        v = data.x #[20,5,2]
        x = data.positions[:,:self.historical_steps] - \
            data.positions[:,self.historical_steps-1].unsqueeze(1)#[20,5,2]
        x = torch.matmul(x,data.rotate_mat)
        lw = data.agent_box[:,:self.historical_steps]

        theta = data.rotate_angles[:,:self.historical_steps] -\
            data.rotate_angles[:,self.historical_steps-1].unsqueeze(1)
        theta = theta.unsqueeze(-1)

        return torch.cat(
            (lw, x, torch.cos(theta), torch.sin(theta), theta, v),
            dim=-1)

    def encode_trajs(self, input, data):

        #encode agents
        agent_input = input[data['nearby_mask']]
        agent_out = self.agent_encoder(
            agent_input
        ).permute(1,0,2)
        padding_mask = data['padding_mask'][
            data['nearby_mask'],
            self.historical_steps - self.input_window: self.historical_steps
            ]
        agent_out = self.temporal_encoder(
            x=agent_out, 
            padding_mask=padding_mask,
            )

        #encode ego
        ego_input = input[~data['nearby_mask']]
        ego_out = self.ego_encoder(
            ego_input[:,-1][:,None]
        ).squeeze(1)
        ego_out = self.ego_mlp(ego_out).to(torch.float32)

        temp = torch.zeros(
            (input.shape[0],self.embed_dim),
            device=ego_out.device
            )
        temp[data['nearby_mask']] = agent_out
        temp[~data['nearby_mask']] = ego_out

        return temp 
    
    def fuse_lane(self,feat, data):
        edge_index, edge_attr = self.drop_edge(
            data['lane_actor_index'], 
            data['lane_actor_vectors']
            )
        
        out = self.al_encoder(
            x=(data['lane_vectors'], feat), 
            edge_index=edge_index, edge_attr=edge_attr,
            traffic_light=data['lane_traffic'],
            is_on_route=data['lane_route'],
            rotate_mat=data['rotate_mat']
            )
        return out
    
class TemporalEncoder(nn.Module):

    def __init__(self,
                 historical_steps: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 no_atten_mask:bool=False,
                 **kwargs) -> None:
        super(TemporalEncoder, self).__init__()
        
        encoder_layer = TemporalEncoderLayer(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
            )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer, 
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim)
            )
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.Tensor(historical_steps + 1, 1, embed_dim))

        if no_atten_mask:
            self.attn_mask = None
        else:
            self.padding_token = nn.Parameter(
                torch.Tensor(historical_steps, 1, embed_dim)
                )
            attn_mask = self.generate_square_subsequent_mask(historical_steps + 1)
            self.register_buffer('attn_mask', attn_mask)
            nn.init.normal_(self.padding_token, mean=0., std=.02)

        nn.init.normal_(self.cls_token, mean=0., std=.02)
        nn.init.normal_(self.pos_embed, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                x: torch.Tensor,
                padding_mask: torch.Tensor,) -> torch.Tensor:
        x = torch.where(
            padding_mask.t().unsqueeze(-1), self.padding_token, x
            ) if padding_mask is not None else x
        expand_cls_token = self.cls_token.expand(-1, x.shape[1], -1)
        x = torch.cat((x, expand_cls_token), dim=0)
        x = x + self.pos_embed
        out = self.transformer_encoder(
            src=x, mask=self.attn_mask, src_key_padding_mask=None
            )
        return out[-1]  # [N, D]

    @staticmethod
    def generate_square_subsequent_mask(seq_len: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class TemporalEncoderLayer(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 **kargs) -> None:
        super(TemporalEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
            )
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                **kargs) -> torch.Tensor:
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self,
                  x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor],
                  key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.self_attn(
            x, x, x, attn_mask=attn_mask, 
            key_padding_mask=key_padding_mask, need_weights=False
            )[0]
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(F.relu_(self.linear1(x))))
        return self.dropout2(x)


class ALEncoder(MessagePassing):

    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 route_dim:int=2,
                 traffic_dim:int=4,
                 **kwargs) -> None:
        super(ALEncoder, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.lane_embed = MultipleInputEmbedding(
            in_channels=[node_dim, edge_dim], out_channel=embed_dim
            )
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))
        self.traffic_light_embed = SingleInputEmbedding(traffic_dim,embed_dim)
        self.is_on_route = SingleInputEmbedding(route_dim, embed_dim)
        self.apply(init_weights)

    def forward(self,
                x: Tuple[torch.Tensor, torch.Tensor],
                edge_index: Adj,
                edge_attr: torch.Tensor,
                traffic_light: torch.Tensor,
                is_on_route: torch.Tensor,
                rotate_mat: Optional[torch.Tensor] = None,
                size: Size = None) -> torch.Tensor:
        x_lane, x_actor = x
        x_actor = x_actor +\
              self._mha_block(
                self.norm1(x_actor), x_lane, edge_index,
                edge_attr, traffic_light, is_on_route,
                rotate_mat, size
                )
        x_actor = x_actor + self._ff_block(self.norm2(x_actor))
        return x_actor

    def message(self,
                edge_index: Adj,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                traffic_light_j,
                is_on_route_j,
                rotate_mat: Optional[torch.Tensor],
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:
        red_light_mask = torch.eq(traffic_light_j[:,2],1)
        rotate_mat = rotate_mat[edge_index[1]]
        x_j = self.lane_embed(
            [torch.bmm(x_j.unsqueeze(-2), rotate_mat).squeeze(-2),
            # torch.bmm(edge_attr.masked_fill(red_light_mask.unsqueeze(-1),1e-6).unsqueeze(-2), rotate_mat).squeeze(-2)],
            torch.bmm(edge_attr.unsqueeze(-2), rotate_mat).squeeze(-2)],
            ) + self.traffic_light_embed(traffic_light_j) +\
                self.is_on_route(is_on_route_j)
        query = self.lin_q(x_i).view(
            -1, self.num_heads, self.embed_dim // self.num_heads
            )
        key = self.lin_k(x_j).view(
            -1, self.num_heads, self.embed_dim // self.num_heads
            )
        value = self.lin_v(x_j).view(
            -1, self.num_heads, self.embed_dim // self.num_heads
            )
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale
        alpha = alpha.masked_fill(red_light_mask.unsqueeze(-1),-1e6)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return value * alpha.unsqueeze(-1)

    def update(self,
               inputs: torch.Tensor,
               x: torch.Tensor) -> torch.Tensor:
        x_actor = x[1]
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x_actor))
        return inputs + gate * (self.lin_self(x_actor) - inputs)

    def _mha_block(self,
                   x_actor: torch.Tensor,
                   x_lane: torch.Tensor,
                   edge_index: Adj,
                   edge_attr: torch.Tensor,
                   traffic_light: torch.Tensor,
                   is_on_route: torch.Tensor,
                   rotate_mat: Optional[torch.Tensor],
                   size: Size) -> torch.Tensor:
        x_actor = self.out_proj(
            self.propagate(edge_index=edge_index, 
                        x=(x_lane, x_actor), edge_attr=edge_attr,
                        traffic_light=traffic_light, is_on_route=is_on_route,
                        rotate_mat=rotate_mat, size=size)
                        )
        return self.proj_drop(x_actor)

    def _ff_block(self, x_actor: torch.Tensor) -> torch.Tensor:
        return self.mlp(x_actor)
    
class QADecoderLayer(MessagePassing):

    def __init__(self,
                embed_dim: int,
                num_heads: int = 8,
                dropout: float = 0.1,
                self_atten: bool = False,
                 **kwargs) -> None:
        super(QADecoderLayer, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.self_atten = self_atten
        # self-attention 
        self.sa_qcontent_proj = nn.Linear(embed_dim, embed_dim)
        self.sa_qpos_proj = nn.Linear(embed_dim, embed_dim)
        self.sa_kcontent_proj = nn.Linear(embed_dim, embed_dim)
        self.sa_kpos_proj = nn.Linear(embed_dim, embed_dim)
        self.sa_v_proj = nn.Linear(embed_dim, embed_dim)
        self.self_attn = MultiheadAttention(
            embed_dim, num_heads, dropout=dropout
            )
        self.sa_dropout = nn.Dropout(dropout)

        #cross-attention
        # self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm0 = nn.LayerNorm(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm_memory = nn.LayerNorm(embed_dim)
        self.ca_kcontent_proj = nn.Linear(embed_dim,embed_dim)
        self.ca_v_proj = nn.Linear(embed_dim,embed_dim)
        self.ca_kpos_proj = nn.Linear(embed_dim,embed_dim)
        self.ca_qcontent_proj = nn.Linear(embed_dim,embed_dim)
        self.ca_qpos_proj = nn.Linear(embed_dim,embed_dim)
        self.ca_qpos_sine_proj = nn.Linear(embed_dim,embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout))

    def forward(self,
                tgt: torch.Tensor,
                memory:torch.Tensor,
                edge_index: Adj,
                size: Size = None,
                pos:torch.Tensor=None,
                query_pos:torch.Tensor=None,
                query_sine_embed:torch.Tensor=None,
                is_first:bool=False) -> torch.Tensor:
        num_query, batch_size, _ = tgt.shape
        # attention among proposals/queries
        tgt = tgt + self.sa_dropout(self._sa_block(self.norm0(tgt), query_pos))#(num_query, batch_size,-1)
        tgt = tgt.permute(1,0,2).flatten(0,1)
        query_pos = query_pos.permute(1,0,2).flatten(0,1)
        # cross attention from nearby agents to ego agents
        tgt = tgt + self._mha_block(self.norm1(tgt),self.norm_memory(memory),
                                edge_index, size,
                                pos, query_pos, query_sine_embed, is_first)

        tgt = tgt + self._ff_block(self.norm2(tgt))

        return tgt.reshape(batch_size,num_query,-1).permute(1,0,2)
    
    def _sa_block(self,tgt, query_pos):
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.sa_qcontent_proj(tgt)      # target is the input of the first decoder layer. zero by default.
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(tgt)

        q = q_content + q_pos
        k = k_content + k_pos

        return self.self_attn(q, k, value=v, attn_mask=None,
                                key_padding_mask=None)[0]
    
    def message(self,
                q_i: torch.Tensor,
                k_j: torch.Tensor,
                v_j:torch.Tensor,
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int],
                ) -> torch.Tensor:
        q = q_i.view(
            -1, self.num_heads, 2*self.embed_dim // self.num_heads
            )
        k = k_j.view(
            -1, self.num_heads, 2*self.embed_dim // self.num_heads
            )
        v = v_j.view(
            -1, self.num_heads, self.embed_dim // self.num_heads
            )
        scale = (2*self.embed_dim // self.num_heads) ** 0.5
        alpha = (k * q).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        # alpha = self.attn_drop(alpha)
        return v * alpha.unsqueeze(-1)

    def update(self,
               inputs: torch.Tensor) -> torch.Tensor:
        return inputs.flatten(1,2)

    def _mha_block( self,
                    tgt: torch.Tensor,
                    memory: torch.Tensor,
                    edge_index: Adj,
                    size: Size,
                    pos:torch.Tensor,
                    query_pos:torch.Tensor,
                    query_sine_embed:torch.Tensor,
                    is_first:torch.Tensor
                   ) -> torch.Tensor:
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)
        k_pos = self.ca_kpos_proj(pos)
        q_content = self.ca_qcontent_proj(tgt)
        
        q_pos = self.ca_qpos_proj(query_pos)
        q = q_content + q_pos
        k = k_content + k_pos

        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)

        num_q, n_model = q_content.shape
        
        q = q.view(num_q, self.num_heads, n_model//self.num_heads)
        query_sine_embed = query_sine_embed.view(num_q, self.num_heads, n_model//self.num_heads)
        q = torch.cat([q, query_sine_embed], dim=-1).view(num_q, n_model * 2)

        num_k, _ = k_content.shape
        k = k.view(num_k, self.num_heads, n_model//self.num_heads)
        k_pos = k_pos.view(num_k, self.num_heads, n_model//self.num_heads)
        k = torch.cat([k, k_pos], dim=-1).view(num_k, n_model * 2)
        x = self.out_proj(self.propagate(
                edge_index=edge_index, q = q, k=k, v=v)
            )
        return self.proj_drop(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
    
class MTRDecoder(MessagePassing):

    def __init__(self,
                 input_dim: int, d_model: int,
                 future_steps: int, num_modes: int,
                 historical_steps:int, 
                 num_modes_pred:int = 1,
                 attention_heads:int = 8,
                 dropout:float = 0.1,
                 num_decoder_layers:int=4,
                 learning_query_points:bool=False,
                 num_learning_queries:int=15,
                 ) -> None:
        super(MTRDecoder, self).__init__()
        self.input_size = input_dim
        self.d_model = d_model
        self.num_future_frames = future_steps
        self.num_modes = num_modes
        self.historical_steps = historical_steps
        self.num_heads = 8
        
        self.dropout = dropout
        self.num_decoder_layers = num_decoder_layers
        self.learning_query_points = learning_query_points
        self.num_learning_queries = num_learning_queries

        in_channels = self.input_size *2
        # define the cross-attn layers
        self.in_proj_center_obj = nn.Sequential(
            nn.Linear(in_channels, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        # define the motion query
        self.intention_points, self.intention_query, self.intention_query_mlps, self.learning_query_points_mlps =\
              self.build_motion_query(self.d_model, learning_query_points, num_learning_queries)

        self.in_proj_obj, self.obj_decoder_layers = self.build_transformer_decoder(
            in_channels=in_channels,
            d_model=self.d_model,
            nhead=attention_heads,
            dropout=dropout,
            num_decoder_layers=num_decoder_layers,
            use_local_attn=False
        )

        map_d_model = self.d_model
        self.route_node_encoder, self.route_edge_encoder, self.map_decoder_layers =\
            self.build_route_decoder(
            in_channels=in_channels,
            d_model=map_d_model,
            num_modes = self.intention_points.shape[0] if not learning_query_points else num_learning_queries,
            nhead=attention_heads,
            dropout=dropout,
            num_decoder_layers=num_decoder_layers,
        )
        
        if map_d_model != self.d_model:
            temp_layer = nn.Linear(self.d_model, map_d_model)
            self.map_query_content_mlps = nn.ModuleList([copy.deepcopy(temp_layer) for _ in range(self.num_decoder_layers)])
            self.map_query_embed_mlps = nn.Linear(self.d_model, map_d_model)
        else:
            self.map_query_content_mlps = [None for _ in range(self.num_decoder_layers)]
            self.map_query_embed_mlps = None

        # define the dense future prediction layers
        self.build_dense_future_prediction_layers(
            hidden_dim=self.d_model, num_future_frames=self.num_future_frames
        )

        # define the motion head
        # self.norm_before_fusion = nn.BatchNorm1d(self.d_model * 2+map_d_model)
        temp_layer = build_mlps(c_in=self.d_model * 2+map_d_model, mlp_channels=[self.d_model, self.d_model], ret_before_act=True)
        self.query_feature_fusion_layers = nn.ModuleList([copy.deepcopy(temp_layer) for _ in range(self.num_decoder_layers)])

        self.motion_reg_heads, self.motion_cls_heads, self.motion_vel_heads = self.build_motion_head(
            in_channels=self.d_model, hidden_size=self.d_model, num_decoder_layers=self.num_decoder_layers
        )
        self.forward_ret_dict = {}
        self.apply(init_weights)

    def get_batch(self,batch,index):
        phase = 0
        for vanish_node in torch.where(torch.eq(index,False))[0]:
            batch[batch>vanish_node-phase] = batch[batch>vanish_node-phase] - 1
            phase = phase + 1
        return batch
    
    def forward(self,
                data,
                local_embed: torch.Tensor, agent_global_embed: torch.Tensor,
                ego_global_embed:torch.Tensor, nearby_mask:torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = local_embed.device
        batch_size = ego_global_embed.shape[1]
        decision = {}
        ego_local_embed = local_embed[~nearby_mask]
        agent_local_embed = local_embed[nearby_mask]
        prediction = {}

        # input_dict = batch_dict['input_dict']
        # obj_feature, obj_mask, obj_pos = batch_dict['obj_feature'], batch_dict['obj_mask'], batch_dict['obj_pos']
        # map_feature, map_mask, map_pos = batch_dict['map_feature'], batch_dict['map_mask'], batch_dict['map_pos']
        obj_feature = torch.cat(
            (agent_local_embed,agent_global_embed),dim=-1
            )
        center_objects_feature = torch.cat(
            (ego_global_embed,ego_local_embed),dim=-1
            )

        # input projection
        center_objects_feature = self.in_proj_center_obj(center_objects_feature)
        obj_feature = self.in_proj_obj(obj_feature)

        # dense future prediction
        obj_feature, pred_dense_future_trajs, pred_dense_future_trajs_global=\
            self.apply_dense_future_prediction(
            obj_feature=obj_feature, 
            obj_pos=data['positions'][nearby_mask,self.historical_steps-1], 
            obj_head=data['rotate_angles'][nearby_mask, self.historical_steps-1].unsqueeze(-1), 
            inv_rotate_mat = data['inv_rotate_mat'][nearby_mask],
        )

        # decoder layers
        pred_list, intention = self.apply_transformer_decoder(
            center_objects_feature=center_objects_feature,
            obj_feature=obj_feature, 
            obj_pos=data['positions'][nearby_mask,self.historical_steps-1],
            obj_head=data['rotate_angles'][nearby_mask, self.historical_steps-1].unsqueeze(-1),
            data=data
        )

        self.forward_ret_dict['pred_list'] = pred_list

        decision = {}
        prediction = {}

        decision['confs'] = torch.zeros(1,device=device)
        decision['trajs'] = torch.zeros(1,device=device)
        decision['heads'] = torch.zeros(1,device=device)

        decision['pred_list'] = pred_list
        decision['intention_points'] = intention['points']
        decision['intention_trajs'] = intention['trajs']
        prediction['trajs'] = pred_dense_future_trajs[...,:2].unsqueeze(1)
        prediction['heads'] = pred_dense_future_trajs[...,2:]
        prediction['trajs_global'] = pred_dense_future_trajs_global.unsqueeze(1)

        prediction['confs'] = torch.zeros((pred_dense_future_trajs.shape[0],1,1),device=device)

        # decision['action1'] = torch.zeros(1,device=device)
        # decision['action2'] = torch.zeros(1,device=device)
        # prediction['obstacles'] = torch.zeros(1,device=device)
        # prediction['obstacles_batch'] = torch.zeros(1,device=device)
        # decision['trajs'], decision['heads'], decision['conf'] =\
        #       decision_trajs, decision_headings, decision_confs.squeeze(-1)
        # decision['trajs_pos'], decision['heads_pos'], decision['conf_pos'] =\
        #         planning_trajs_pos, planning_headings_pos, planning_confs_pos.squeeze(-1)
        
        return prediction, decision

    def apply_transformer_decoder(
            self, center_objects_feature, obj_feature, 
            obj_pos, obj_head, data
            ):
        intention_query, intention_points, intention_trajs = self.get_motion_query(
            center_objects_feature , len(center_objects_feature)
            )
        query_content = torch.zeros_like(intention_query,device=center_objects_feature.device)
        
        # use intention points as queries 
        self.forward_ret_dict['intention_points'] = intention_points.detach().permute(1, 0, 2)  # (num_center_objects, num_query, 2)
        pred_waypoints = intention_points.detach().permute(1, 0, 2)[:, :, None, :]  # (num_center_objects, num_query, 1, 2)
        dynamic_query_center = intention_points.detach()

        num_center_objects = query_content.shape[1]
        num_query = query_content.shape[0]
        device = query_content.device
        center_objects_feature = center_objects_feature[None, :, :].repeat(num_query, 1, 1)  # (num_query, num_center_objects, C)

        pred_list = []
        
        # route exits
        unique_index = torch.unique(data['route_batch'])
        index = torch.zeros(num_center_objects,dtype=torch.bool,device=device)
        index[unique_index] = True
        batch = get_batch(data['route_batch'],index)

        for layer_idx in range(self.num_decoder_layers):
            # query object feature
            # query(ego) head and pos are defaultly 0
            # intention_query dot kv_pos
            # intention_query dot dense_prediction
            obj_query_feature = self.apply_qa_attention(
                kv_feature=obj_feature, kv_pos=obj_pos,kv_head=obj_head,
                query_content=query_content, query_embed=intention_query,
                # qk_edge=data['qa_idx'].cuda(),
                qk_edge=data['qa_idx'].to(device),
                attention_layer=self.obj_decoder_layers[layer_idx],
                dynamic_query_center=dynamic_query_center,
                layer_idx=layer_idx
            ) 

            # query route feature
            map_query_feature = self.apply_route_attention(
                index=index,batch=batch,data=data,
                query_content=query_content, query_embed=intention_query,
                route_edge_encoder=self.route_edge_encoder,
                route_node_encoder=self.route_node_encoder,
                local_scope_route=self.map_decoder_layers[layer_idx],
                dynamic_query_center=dynamic_query_center.permute(1,0,2), device=device
            ) 

            query_feature = torch.cat([center_objects_feature, obj_query_feature, map_query_feature], dim=-1)
            # query_feature = self.norm_before_fusion(
            #     query_feature.flatten(start_dim=0, end_dim=1)
            #     ).view(num_query, num_center_objects, -1) 
            # query_feature = torch.cat([center_objects_feature, obj_query_feature], dim=-1)
            query_content = self.query_feature_fusion_layers[layer_idx](
                query_feature.flatten(start_dim=0, end_dim=1)
            ).view(num_query, num_center_objects, -1) 

            # motion prediction
            query_content_t = query_content.permute(1, 0, 2).contiguous().view(num_center_objects * num_query, -1)
            pred_scores = self.motion_cls_heads[layer_idx](query_content_t).view(num_center_objects, num_query)
            if self.motion_vel_heads is not None:
                pred_trajs = self.motion_reg_heads[layer_idx](query_content_t).view(num_center_objects, num_query, self.num_future_frames, 5)
                pred_vel = self.motion_vel_heads[layer_idx](query_content_t).view(num_center_objects, num_query, self.num_future_frames, 2)
                pred_trajs = torch.cat((pred_trajs, pred_vel), dim=-1)
            else:
                pred_trajs = self.motion_reg_heads[layer_idx](query_content_t).view(num_center_objects, num_query, self.num_future_frames, 3)

            pred_list.append([pred_scores, pred_trajs])

            # update
            pred_waypoints = pred_trajs[:, :, :, 0:2]
            dynamic_query_center = pred_trajs[:, :, -1, 0:2].contiguous().permute(1, 0, 2).detach()  # (num_query, num_center_objects, 2)

        intention = {
            "trajs": intention_trajs,
            "points": intention_points,
        }

        return pred_list, intention
    
    def build_transformer_decoder(self, in_channels, d_model, nhead, dropout=0.1, num_decoder_layers=1, use_local_attn=False):
        in_proj_layer = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        decoder_layer = QADecoderLayer(
            d_model,
            nhead,
            dropout,
            self_atten=True,
            # d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout,
            # activation="relu", normalize_before=False, keep_query_pos=False,
        )
        decoder_layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_decoder_layers)])
        return in_proj_layer, decoder_layers
    
    def build_route_decoder(self, in_channels,
                            d_model, nhead, num_modes,
                            dropout=0.1, num_decoder_layers=1):
        route_node_encoder = build_mlps(
            c_in=4, 
            mlp_channels=[d_model, d_model, d_model], 
            ret_before_act=True, without_norm=True
        )

        route_edge_encoder = build_mlps(
            c_in=2, 
            mlp_channels=[d_model, d_model, d_model], 
            ret_before_act=True, without_norm=True
        )

        decoder_layer = RouteEncoderLayer(
            d_model,
            num_modes,
            nhead,
            dropout=dropout
        )
        decoder_layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_decoder_layers)])
        return route_node_encoder, route_edge_encoder, decoder_layers
    
    def apply_qa_attention(self, kv_feature, kv_pos, kv_head,
                              query_content, query_embed,
                              attention_layer,qk_edge, qk_attr=None, 
                              dynamic_query_center=None,
                              layer_idx=0,
                              query_content_pre_mlp=None, query_embed_pre_mlp=None):
        """
        Args:
            kv_feature (B, N, C):
            kv_mask (B, N):
            kv_pos (B, N, 3):
            query_tgt (M, B, C):
            query_embed (M, B, C):
            dynamic_query_center (M, B, 2): . Defaults to None.
            attention_layer (layer):

            query_index_pair (B, M, K)

        Returns:
            attended_features: (B, M, C)
            attn_weights:
        """
        if query_content_pre_mlp is not None:
            query_content = query_content_pre_mlp(query_content)
        if query_embed_pre_mlp is not None:
            query_embed = query_embed_pre_mlp(query_embed)

        num_q, batch_size, d_model = query_content.shape
        searching_query = position_encoding_utils.gen_sineembed_for_position(dynamic_query_center, hidden_dim=d_model)
        kv_pos = kv_pos[:, 0:2].unsqueeze(1)
        kv_pos_embed = position_encoding_utils.gen_sineembed_for_position(kv_pos, hidden_dim=d_model)
        kv_pos_embed = kv_pos_embed.squeeze(1)

        query_feature = attention_layer(
            tgt=query_content,
            query_pos=query_embed,
            query_sine_embed=searching_query,
            edge_index = qk_edge, 
            memory=kv_feature,
            pos=kv_pos_embed,
            is_first=(layer_idx == 0)
        )  # (M, B, C)

        return query_feature
    
    def apply_route_attention(self, index, batch, data,
                              query_content, query_embed,
                              route_edge_encoder,
                              route_node_encoder,
                              local_scope_route,
                              dynamic_query_center, device):
        #data['route_edge'] is position, data['route_node'] is direction 
        local_edge_feature = route_edge_encoder(
            data['route_edge'].unsqueeze(1) -\
            dynamic_query_center[data['route_batch']][...,:2]
            )
        node_feat = torch.cat((data['route_edge'],data['route_node']),dim=-1)
        local_node_feature = route_node_encoder(node_feat)
        # num_queries, batch_size, dim
        temp = torch.zeros_like(query_content,device=device,dtype=torch.float32)

        query_sine_embed = position_encoding_utils.gen_sineembed_for_position(
            dynamic_query_center, hidden_dim=self.d_model
            )
        kv_pos = position_encoding_utils.gen_sineembed_for_position(
            data['route_edge'].unsqueeze(1), hidden_dim=self.d_model
            )
        route_feat = local_scope_route(
            batch, query_content[:,index], query_embed[:,index],query_sine_embed,kv_pos,
            [local_edge_feature,local_node_feature]
            )
        
        temp[:,index] = query_content[:,index] + route_feat
            
        temp[:,~index] = query_content[:,~index].to(torch.float32)
        query_content = temp
        return query_content

    def build_dense_future_prediction_layers(self, hidden_dim, num_future_frames):
        
        self.dense_future_head = build_mlps(
            c_in=hidden_dim,
            mlp_channels=[hidden_dim, hidden_dim, num_future_frames * 3],
            ret_before_act=True
        )
        self.obj_pos_encoding_layer = build_mlps(
            c_in=4,
            mlp_channels=[hidden_dim, hidden_dim, hidden_dim], 
            ret_before_act=True, without_norm=True
        )
        self.history_mlp = build_mlps(
            c_in=hidden_dim*2, 
            mlp_channels=[hidden_dim, hidden_dim, hidden_dim], 
            ret_before_act=True, without_norm=True
        )
        self.future_traj_mlps = build_mlps(
            c_in=3 * self.num_future_frames, 
            mlp_channels=[hidden_dim, hidden_dim, hidden_dim], 
            ret_before_act=True, without_norm=True
        )
        self.traj_fusion_mlps = build_mlps(
            c_in=hidden_dim * 2, 
            mlp_channels=[hidden_dim, hidden_dim, hidden_dim], 
            ret_before_act=True, without_norm=False
        )

    def apply_dense_future_prediction(self, obj_feature, obj_pos, obj_head, inv_rotate_mat):
        # dense future prediction

        pred_dense_trajs = self.dense_future_head(obj_feature).view(
            obj_feature.shape[0], self.num_future_frames, 3
            )
        prediction_trajs, prediction_headings =\
            pred_dense_trajs[...,:2], pred_dense_trajs[...,2:]
            
        #recover coordinates from agent local frame
        # prediction_trajs = torch.matmul(
        #     prediction_trajs,
        #     data['inv_rotate_mat'][nearby_mask]
        #     ).reshape(-1,self.future_steps, 2)
        # prediction_trajs = prediction_trajs +\
        #         data['positions'][nearby_mask,self.historical_steps-1].unsqueeze(1).unsqueeze(1)
        # prediction_headings = prediction_headings +\
        #         data['rotate_angles'][nearby_mask, self.historical_steps-1].unsqueeze(-1).unsqueeze(-1)
        
        prediction_trajs = torch.matmul(
            prediction_trajs,
            inv_rotate_mat
        ).reshape(-1,self.num_future_frames,2)

        prediction_trajs = prediction_trajs + obj_pos.unsqueeze(1)
        prediction_headings = prediction_headings + obj_head.unsqueeze(-1)
        pred_dense_trajs_global =torch.cat((
            prediction_trajs, prediction_headings
        ),dim=-1)
        #encoding history feature and SE2 feature
        se2_feature = self.obj_pos_encoding_layer(
            torch.cat((obj_pos,torch.sin(obj_head),torch.cos(obj_head)),dim=-1)
        )
        obj_history_feature = self.history_mlp(torch.cat((obj_feature,se2_feature),dim=-1))
        # future feature encoding and fuse to past obj_feature
        obj_future_input = pred_dense_trajs_global.flatten(start_dim=1, end_dim=2) 
        obj_future_feature = self.future_traj_mlps(obj_future_input)
        obj_full_trajs_feature = torch.cat(
            (obj_history_feature, obj_future_feature
             ), dim=-1
             )
        obj_feature = self.traj_fusion_mlps(obj_full_trajs_feature)

        self.forward_ret_dict['pred_dense_trajs'] = pred_dense_trajs

        return obj_feature, pred_dense_trajs, pred_dense_trajs_global.detach()
    
    def build_motion_head(self, in_channels, hidden_size, num_decoder_layers):
        motion_reg_head =  build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_size, hidden_size, self.num_future_frames * 3], ret_before_act=True
        )
        motion_cls_head =  build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_size, hidden_size, 1], ret_before_act=True
        )

        motion_reg_heads = nn.ModuleList([copy.deepcopy(motion_reg_head) for _ in range(num_decoder_layers)])
        motion_cls_heads = nn.ModuleList([copy.deepcopy(motion_cls_head) for _ in range(num_decoder_layers)])
        motion_vel_heads = None 
        return motion_reg_heads, motion_cls_heads, motion_vel_heads
    
    def generate_final_prediction(
        self, pred_list,
    ):
        pred_scores, pred_trajs = pred_list[-1]
        pred_scores = torch.softmax(pred_scores, dim=-1)  # (num_center_objects, num_query)

        num_center_objects, num_query, num_future_timestamps, num_feat = pred_trajs.shape
        if self.num_modes != num_query:
            assert num_query > self.num_modes
            pred_trajs_final, pred_scores_final, selected_idxs = batch_nms(
                pred_trajs=pred_trajs, pred_scores=pred_scores,
                dist_thresh=2.5,
                num_ret_modes=self.num_modes
            )
        else:
            pred_trajs_final = pred_trajs
            pred_scores_final = pred_scores

        return pred_scores_final, pred_trajs_final
    
    def get_motion_query(self, ego_feature, num_ego):
        if self.learning_query_points:
            intention_traj = []
            for i in range(self.num_learning_queries):
                intention_traj.append(
                    self.learning_query_points_mlps[i](ego_feature).view(num_ego,-1,2)
                    )

            intention_trajs = torch.stack(intention_traj)
            intention_points = intention_trajs[...,-1,:]
            
        else:
            intention_trajs = None
            intention_points = self.intention_points[:,None,:].repeat(1,num_ego,1).to(ego_feature.device)  # (num_query, num_center_objects, 2)

        intention_query = position_encoding_utils.gen_sineembed_for_position(intention_points, hidden_dim=self.d_model)
        intention_query = self.intention_query_mlps(
            intention_query.view(-1, self.d_model)
            ).view(-1, num_ego, self.d_model)  # (num_query, num_center_objects, C)
        return intention_query, intention_points, intention_trajs
    
    def build_motion_query(self, d_model, learning_query_points=False,num_learning_queries=1):
        if learning_query_points:
            learning_query_points_mlp = build_mlps(
                c_in=d_model, mlp_channels=[d_model, d_model, 2*25], ret_before_act=True
                )
            learning_query_points_mlps = nn.ModuleList(
                [copy.deepcopy(learning_query_points_mlp) for _ in range(num_learning_queries)]
                )
            intention_points = intention_query = None
        else:
            intention_points = intention_query = intention_query_mlps = learning_query_points_mlps = None
            intention_points_file  = "./cluster_center_32.npy"
            intention_points = torch.load(intention_points_file).float()
        intention_query_mlps = build_mlps(
            c_in=d_model, mlp_channels=[d_model, d_model], ret_before_act=True
        )
        return intention_points, intention_query, intention_query_mlps, learning_query_points_mlps
    
def batch_nms(pred_trajs, pred_scores, dist_thresh, num_ret_modes=6):
    """

    Args:
        pred_trajs (batch_size, num_modes, num_timestamps, 7)
        pred_scores (batch_size, num_modes):
        dist_thresh (float):
        num_ret_modes (int, optional): Defaults to 6.

    Returns:
        ret_trajs (batch_size, num_ret_modes, num_timestamps, 5)
        ret_scores (batch_size, num_ret_modes)
        ret_idxs (batch_size, num_ret_modes)
    """
    batch_size, num_modes, num_timestamps, num_feat_dim = pred_trajs.shape

    sorted_idxs = pred_scores.argsort(dim=-1, descending=True)
    bs_idxs_full = torch.arange(batch_size).type_as(sorted_idxs)[:, None].repeat(1, num_modes)
    sorted_pred_scores = pred_scores[bs_idxs_full, sorted_idxs]
    sorted_pred_trajs = pred_trajs[bs_idxs_full, sorted_idxs]  # (batch_size, num_modes, num_timestamps, 7)
    sorted_pred_goals = sorted_pred_trajs[:, :, -1, :]  # (batch_size, num_modes, 7)

    dist = (sorted_pred_goals[:, :, None, 0:2] - sorted_pred_goals[:, None, :, 0:2]).norm(dim=-1)
    point_cover_mask = (dist < dist_thresh)

    point_val = sorted_pred_scores.clone()  # (batch_size, N)
    point_val_selected = torch.zeros_like(point_val)  # (batch_size, N)

    ret_idxs = sorted_idxs.new_zeros(batch_size, num_ret_modes).long()
    ret_trajs = sorted_pred_trajs.new_zeros(batch_size, num_ret_modes, num_timestamps, num_feat_dim)
    ret_scores = sorted_pred_trajs.new_zeros(batch_size, num_ret_modes)
    bs_idxs = torch.arange(batch_size).type_as(ret_idxs)

    for k in range(num_ret_modes):
        cur_idx = point_val.argmax(dim=-1) # (batch_size)
        ret_idxs[:, k] = cur_idx

        new_cover_mask = point_cover_mask[bs_idxs, cur_idx]  # (batch_size, N)
        point_val = point_val * (~new_cover_mask).float()  # (batch_size, N)
        point_val_selected[bs_idxs, cur_idx] = -1
        point_val += point_val_selected

        ret_trajs[:, k] = sorted_pred_trajs[bs_idxs, cur_idx]
        ret_scores[:, k] = sorted_pred_scores[bs_idxs, cur_idx]

    bs_idxs = torch.arange(batch_size).type_as(sorted_idxs)[:, None].repeat(1, num_ret_modes)

    ret_idxs = sorted_idxs[bs_idxs, ret_idxs]
    return ret_trajs, ret_scores, ret_idxs