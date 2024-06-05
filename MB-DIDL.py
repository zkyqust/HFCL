# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import sys
import copy
from utility.helper import *
from utility.batch_test import *
import multiprocessing
import torch.multiprocessing
import random


class DisentanglementLayer(nn.Module):
    def __init__(self, args, hidden_layer_dim=64, intents_units=64, num_behaviors=3, dropout_rate=0.2):

        super(DisentanglementLayer, self).__init__()
        num_intents = int(args.num_intents)
        self.num_behaviors = num_behaviors
        self.dropout_rate = dropout_rate

        self.intents = self.create_intents_layer(hidden_layer_dim, intents_units, num_intents)

        self.gating_networks = nn.ModuleList([
            self.create_gating_layer(hidden_layer_dim, num_intents) for _ in
            range(num_behaviors)
        ])
        self.initialize_parameters()

    def create_intents_layer(self, input_dim, output_dim, num_intents):
        return nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LeakyReLU(),
                nn.Dropout(p=self.dropout_rate)
            ) for _ in range(num_intents)
        ])

    def create_gating_layer(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Softmax(dim=1)
        )

    def initialize_parameters(self):
        for intent in self.intents:
            for layer in intent:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.normal_(layer.bias, mean=0, std=0.01)

        for gating_network in self.gating_networks:
            for layer in gating_network:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.normal_(layer.bias, mean=0, std=0.01)

    def forward(self, node_embeddings, behavior_embeddings):

        intents_output = torch.stack([intent(node_embeddings) for intent in self.intents], dim=2)

        combined_outputs_list = []
        for behavior_idx in range(self.num_behaviors):
            node_embeddings_behavior = node_embeddings * behavior_embeddings[behavior_idx]

            gate_output = self.gating_networks[behavior_idx](node_embeddings_behavior)

            combined_output = torch.matmul(intents_output, gate_output.unsqueeze(2))

            combined_output = combined_output.squeeze(2)
            combined_outputs_list.append(combined_output)

        combined_outputs = torch.stack(combined_outputs_list, dim=1)
        return combined_outputs


class MB_DIDL(nn.Module):

    def __init__(self, max_item_list, data_config, args):
        super(MB_DIDL, self).__init__()

        self.max_item_list = max_item_list
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.num_nodes = self.n_users + self.n_items
        self.pre_adjs = data_config['pre_adjs']
        self.pre_adjs_tensor = [self._convert_sp_mat_to_sp_tensor(adj).to(device) for adj in self.pre_adjs]
        self.behs = data_config['behs']
        self.n_relations = len(self.behs)
        self.coefficient = torch.tensor(eval(args.coefficient)).view(1, -1).to(device)
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.mess_dropout = eval(args.mess_dropout)
        self.aug_type = args.aug_type
        self.all_weights = {}

        self.all_weights['user_embedding'] = Parameter(torch.FloatTensor(self.n_users, self.emb_dim))
        self.all_weights['item_embedding'] = Parameter(torch.FloatTensor(self.n_items, self.emb_dim))
        self.all_weights['behavior_embedding'] = Parameter(torch.FloatTensor(self.n_relations, self.emb_dim))
        self.weight_size_list = [self.emb_dim] + self.weight_size
        for k in range(self.n_layers):
            self.all_weights['W_gcn_%d' % k] = Parameter(
                torch.FloatTensor(self.weight_size_list[k], self.weight_size_list[k + 1]))
            self.all_weights['W_beh_%d' % k] = Parameter(
                torch.FloatTensor(self.weight_size_list[k], self.weight_size_list[k + 1]))
        self.all_weights['W_att1'] = Parameter(torch.FloatTensor(self.emb_dim, self.emb_dim))
        self.all_weights['W_att2'] = Parameter(torch.FloatTensor(self.emb_dim, self.emb_dim))
        self.reset_parameters()
        self.all_weights = nn.ParameterDict(self.all_weights)
        self.intent_dis_model = DisentanglementLayer(args, self.emb_dim, self.emb_dim, self.n_relations,
                                                     self.mess_dropout[0])
        self.dropout = nn.Dropout(self.mess_dropout[0])
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.all_weights['user_embedding'])
        nn.init.xavier_uniform_(self.all_weights['item_embedding'])
        nn.init.xavier_uniform_(self.all_weights['behavior_embedding'])
        nn.init.xavier_uniform_(self.all_weights['W_att1'])
        nn.init.xavier_uniform_(self.all_weights['W_att2'])
        for k in range(self.n_layers):
            nn.init.xavier_uniform_(self.all_weights['W_gcn_%d' % k])
            nn.init.xavier_uniform_(self.all_weights['W_beh_%d' % k])

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        shape = coo.shape
        return torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape))

    def forward(self, device):
        ego_embeddings = torch.cat((self.all_weights['user_embedding'], self.all_weights['item_embedding']),
                                   dim=0)
        global_embedding = ego_embeddings
        ego_embeddings = self.intent_dis_model(ego_embeddings, self.all_weights['behavior_embedding'])
        all_embeddings = ego_embeddings
        all_global_embeddings = global_embedding
        all_rela_embs = {}
        for i in range(self.n_relations):
            beh = self.behs[i]
            rela_emb = self.all_weights['behavior_embedding'][i]
            rela_emb = torch.reshape(rela_emb, (-1, self.emb_dim))
            all_rela_embs[beh] = [rela_emb]
        total_mm_time = 0.

        for k in range(0, self.n_layers):
            global_embeddings_list = []
            for i in range(self.n_relations):
                st = time()
                embeddings_ = torch.matmul(self.pre_adjs_tensor[i], global_embedding)
                total_mm_time += time() - st
                rela_emb = all_rela_embs[self.behs[i]][k]
                embeddings_ = self.leaky_relu(
                    torch.matmul(torch.mul(embeddings_, rela_emb), self.all_weights['W_gcn_%d' % k]))
                global_embeddings_list.append(embeddings_)
            weighted_tensors = [self.dropout(b_tensor) * weight for b_tensor, weight in
                                zip(global_embeddings_list, self.coefficient.squeeze(0))]
            weighted_sum = torch.stack(weighted_tensors)
            global_embedding = torch.sum(weighted_sum, dim=0)
            all_global_embeddings = all_global_embeddings + global_embedding
            embeddings_list = []

            for i in range(self.n_relations):
                st = time()
                embeddings_ = torch.matmul(self.pre_adjs_tensor[i], ego_embeddings[:, i, :])
                total_mm_time += time() - st
                rela_emb = all_rela_embs[self.behs[i]][k]
                embeddings_ = self.leaky_relu(
                    torch.matmul(torch.mul(embeddings_, rela_emb), self.all_weights['W_gcn_%d' % k]))
                embeddings_list.append(embeddings_)
            ego_embeddings = torch.stack(embeddings_list, dim=1)
            ego_embeddings = self.dropout(ego_embeddings)
            all_embeddings = all_embeddings + ego_embeddings

            for i in range(self.n_relations):
                rela_emb = torch.matmul(all_rela_embs[self.behs[i]][k],
                                        self.all_weights['W_beh_%d' % k])
                all_rela_embs[self.behs[i]].append(rela_emb)
        all_global_embeddings /= self.n_layers + 1
        all_embeddings /= self.n_layers + 1

        attention_weights = torch.matmul(
            torch.matmul(all_global_embeddings, self.all_weights['W_att1']).unsqueeze(1),
            torch.transpose(torch.matmul(all_embeddings, self.all_weights['W_att2']), 1, 2))
        attention_weights_normalized = F.softmax(attention_weights.squeeze(1), dim=1)
        p1_result = torch.einsum('ij,ik->ikj', all_global_embeddings, attention_weights_normalized)
        p2_result = torch.einsum('ij,ijk->ijk', (1 - attention_weights_normalized), all_embeddings)
        all_embeddings = p1_result + p2_result

        all_embeddings = torch.cat((all_embeddings, all_global_embeddings.unsqueeze(1)), dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], 0)
        token_embedding = torch.zeros([1, self.n_relations + 1, self.emb_dim], device=device)
        i_g_embeddings = torch.cat((i_g_embeddings, token_embedding), dim=0)

        for i in range(self.n_relations):
            all_rela_embs[self.behs[i]] = torch.mean(torch.stack(all_rela_embs[self.behs[i]], 0), 0)
        return u_g_embeddings, i_g_embeddings, all_rela_embs


class RecLoss(nn.Module):
    def __init__(self, data_config, args):
        super(RecLoss, self).__init__()
        self.behs = data_config['behs']
        self.n_relations = len(self.behs)
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.emb_dim = args.embed_size
        self.coefficient = eval(args.coefficient)
        self.wid = eval(args.wid)

    def forward(self, input_u, label_phs, ua_embeddings, ia_embeddings, rela_embeddings):
        uid = ua_embeddings[input_u]
        uid = torch.reshape(uid, (-1, self.n_relations, self.emb_dim))
        pos_r_list = []
        for i in range(self.n_relations):
            beh = self.behs[i]
            pos_beh = ia_embeddings[:, i, :][label_phs[i]]
            pos_num_beh = torch.ne(label_phs[i], self.n_items).float()
            pos_beh = torch.einsum('ab,abc->abc', pos_num_beh, pos_beh)
            pos_r = torch.einsum('ac,abc->abc', uid[:, i, :], pos_beh)
            pos_r = torch.einsum('ajk,lk->aj', pos_r, rela_embeddings[beh])
            pos_r_list.append(pos_r)

        loss_rec = []
        for i in range(self.n_relations):
            beh = self.behs[i]
            temp = torch.einsum('ab,ac->bc', ia_embeddings[:, i, :], ia_embeddings[:, i, :]) \
                   * torch.einsum('ab,ac->bc', uid[:, i, :], uid[:, i, :])
            tmp_loss = self.wid[i] * torch.sum(
                temp * torch.matmul(rela_embeddings[beh].T, rela_embeddings[beh]))

            tmp_loss += torch.sum(
                (1.0 - self.wid[i]) * torch.square(pos_r_list[i]) - 2.0 * pos_r_list[i])

            loss_rec.append(self.coefficient[i] * tmp_loss)

        regularizer = torch.sum(torch.square(uid)) * 0.5 + torch.sum(torch.square(ia_embeddings)) * 0.5
        emb_loss = args.decay * regularizer

        return loss_rec, emb_loss


class LossClB(nn.Module):
    def __init__(self, data_config, args):
        super(LossClB, self).__init__()
        self.config = data_config
        self.ssl_temp = args.ssl_temp
        self.ssl_reg_inter = eval(args.ssl_reg_inter)
        self.ssl_mode_inter = args.ssl_inter_mode
        self.user_indices_remove, self.item_indices_remove = None, None

    def forward(self, input_u_list, input_i_list, ua_embeddings, ia_embeddings, aux_beh):
        sslb_loss = 0.

        if self.ssl_mode_inter in ['user_side', 'both_side']:
            emb_tgt = ua_embeddings[input_u_list, -1, :]  # [B, d]
            normalize_emb_tgt = F.normalize(emb_tgt, dim=1)
            emb_aux = ua_embeddings[input_u_list, aux_beh, :]  # [B, d]
            normalize_emb_aux = F.normalize(emb_aux, dim=1)  # [B, dim]
            normalize_all_emb_aux = F.normalize(ua_embeddings[:, aux_beh, :], dim=1)
            pos_score = torch.sum(torch.mul(normalize_emb_tgt, normalize_emb_aux),
                                  dim=1)  # [B, ]
            ttl_score = torch.matmul(normalize_emb_tgt, normalize_all_emb_aux.T)  # [B, N]

            pos_score = torch.exp(pos_score / self.ssl_temp)
            ttl_score = torch.sum(torch.exp(ttl_score / self.ssl_temp), dim=1)

            sslb_loss += -torch.sum(torch.log(pos_score / ttl_score)) * self.ssl_reg_inter[aux_beh]

        if self.ssl_mode_inter in ['item_side', 'both_side']:
            emb_tgt = ia_embeddings[input_i_list, -1, :]
            normalize_emb_tgt = F.normalize(emb_tgt, dim=1)
            emb_aux = ia_embeddings[input_i_list, aux_beh, :]
            normalize_emb_aux = F.normalize(emb_aux, dim=1)  # [B, dim]
            normalize_all_emb_aux = F.normalize(ia_embeddings[:, aux_beh, :], dim=1)  # [N, dim]
            pos_score = torch.sum(torch.mul(normalize_emb_tgt, normalize_emb_aux),
                                  dim=1)
            ttl_score = torch.matmul(normalize_emb_tgt, normalize_all_emb_aux.T)

            pos_score = torch.exp(pos_score / self.ssl_temp)  # ssl_temp温度
            ttl_score = torch.sum(torch.exp(ttl_score / self.ssl_temp), dim=1)
            sslb_loss += -torch.sum(torch.log(pos_score / ttl_score)) * self.ssl_reg_inter[aux_beh]

        return sslb_loss


def get_lables(temp_set, k=0.9999):
    max_item = 0
    item_lenth = []
    for i in temp_set:
        item_lenth.append(len(temp_set[i]))
        if len(temp_set[i]) > max_item:
            max_item = len(temp_set[i])
    item_lenth.sort()

    max_item = item_lenth[int(len(
        item_lenth) * k) - 1]

    print(max_item)
    for i in temp_set:
        if len(temp_set[i]) > max_item:
            temp_set[i] = temp_set[i][0:max_item]
        while len(temp_set[i]) < max_item:
            temp_set[i].append(n_items)
    return max_item, temp_set


def get_train_instances1(max_item_list, beh_label_list):
    user_train = []
    beh_item_list = [list() for i in range(n_behs)]  #

    for i in beh_label_list[-1].keys():
        user_train.append(i)
        beh_item_list[-1].append(beh_label_list[-1][i])
        for j in range(n_behs - 1):
            if not i in beh_label_list[j].keys():
                beh_item_list[j].append([n_items] * max_item_list[j])
            else:
                beh_item_list[j].append(beh_label_list[j][i])

    user_train = np.array(user_train)
    beh_item_list = [np.array(beh_item) for beh_item in beh_item_list]
    user_train = user_train[:, np.newaxis]
    return user_train, beh_item_list


def get_train_pairs(user_train_batch,
                    beh_item_tgt_batch):
    input_u_list, input_i_list = [], []
    for i in range(len(user_train_batch)):
        pos_items = beh_item_tgt_batch[i][np.where(beh_item_tgt_batch[i] != n_items)]  # ndarray [x,]
        uid = user_train_batch[i][0]
        input_u_list += [uid] * len(pos_items)
        input_i_list += pos_items.tolist()

    return np.array(input_u_list).reshape([-1]), np.array(input_i_list).reshape([-1])


def test_torch(ua_embeddings, ia_embeddings, rela_embedding, users_to_test, batch_test_flag=False):
    def get_score_np(ua_embeddings, ia_embeddings, rela_embedding, users, items):
        ug_embeddings = ua_embeddings[users]  # []
        pos_ig_embeddings = ia_embeddings[items]
        dot = np.multiply(pos_ig_embeddings, rela_embedding)  # [I, dim] * [1, dim]-> [I, dim]
        batch_ratings = np.matmul(ug_embeddings, dot.T)  # [U, dim] * [dim, I] -> [U, I]
        return batch_ratings

    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}
    test_users = users_to_test
    n_test_users = len(test_users)

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0
    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        item_batch = range(ITEM_NUM)
        rate_batch = get_score_np(ua_embeddings, ia_embeddings, rela_embedding, user_batch, item_batch)

        user_batch_rating_uid = zip(rate_batch, user_batch)

        batch_result = pool.map(test_one_user, user_batch_rating_uid)

        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision'] / n_test_users
            result['recall'] += re['recall'] / n_test_users
            result['ndcg'] += re['ndcg'] / n_test_users
            result['hit_ratio'] += re['hit_ratio'] / n_test_users
            result['auc'] += re['auc'] / n_test_users
    assert count == n_test_users

    pool.close()
    return result


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = dict()
    set_seed(2020)

    config['device'] = device
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['behs'] = data_generator.behs
    config['trn_mat'] = data_generator.trnMats[-1]

    pre_adj_list = data_generator.get_adj_mat()
    config['pre_adjs'] = pre_adj_list

    n_users, n_items = data_generator.n_users, data_generator.n_items
    behs = data_generator.behs
    n_behs = data_generator.beh_num
    trnDicts = copy.deepcopy(data_generator.trnDicts)

    max_item_list = []
    beh_label_list = []

    for i in range(n_behs):
        max_item, beh_label = get_lables(trnDicts[i])
        max_item_list.append(max_item)
        beh_label_list.append(beh_label)

    t0 = time()
    model = MB_DIDL(max_item_list, data_config=config, args=args).to(device)

    recloss = RecLoss(data_config=config, args=args).to(device)
    loss_cl_b = LossClB(data_config=config, args=args).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_gamma)
    cur_best_pre_0 = 0.
    print('without pretraining.')
    run_time = 1
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []

    stopping_step = 0
    should_stop = False
    user_train1, beh_item_list = get_train_instances1(max_item_list, beh_label_list)

    for epoch in range(args.epoch):
        model.train()
        shuffle_indices = np.random.permutation(np.arange(len(user_train1)))
        user_train1 = user_train1[shuffle_indices]
        beh_item_list = [beh_item[shuffle_indices] for beh_item in beh_item_list]

        t1 = time()
        loss, rec_loss, emb_loss, sslb_loss, ssli_loss = 0., 0., 0., 0., 0.
        n_batch = int(len(user_train1) / args.batch_size)

        iter_time = time()

        for idx in range(n_batch):
            optimizer.zero_grad()
            start_index = idx * args.batch_size
            end_index = min((idx + 1) * args.batch_size, len(user_train1))
            u_batch = user_train1[start_index:end_index]
            beh_batch = [beh_item[start_index:end_index] for beh_item in beh_item_list]

            u_batch_list, i_batch_list = get_train_pairs(user_train_batch=u_batch, beh_item_tgt_batch=beh_batch[-1])

            u_batch = torch.from_numpy(u_batch).to(device)
            beh_batch = [torch.from_numpy(beh_item).to(device) for beh_item in beh_batch]
            u_batch_list = torch.from_numpy(u_batch_list).to(device)
            i_batch_list = torch.from_numpy(i_batch_list).to(device)
            ua_embeddings, ia_embeddings, rela_embeddings = model(device)
            batch_rec_loss_list, batch_emb_loss = recloss(u_batch, beh_batch, ua_embeddings[:, :n_behs, :],
                                                          ia_embeddings[:, :n_behs, :], rela_embeddings)

            batch_sslb_loss_list = []
            for aux_beh in range(n_behs):
                aux_beh_sslb_loss = loss_cl_b(u_batch_list, i_batch_list, ua_embeddings, ia_embeddings, aux_beh)
                batch_sslb_loss_list.append(aux_beh_sslb_loss)


            batch_sslb_loss_list = [loss for coeff, loss in zip(eval(args.coefficient), batch_sslb_loss_list) if
                                    coeff != 0.0]
            batch_rec_loss_list = [loss for coeff, loss in zip(eval(args.coefficient), batch_rec_loss_list) if
                                   coeff != 0.0]

            batch_sslb_loss = sum(batch_sslb_loss_list)

            batch_rec_loss = sum(batch_rec_loss_list)





            batch_sslb_loss = batch_sslb_loss * args.cl_coefficient

            batch_loss = batch_rec_loss + batch_emb_loss + batch_sslb_loss

            batch_loss.backward()

            optimizer.step()

            loss += batch_loss.item() / n_batch
            rec_loss += batch_rec_loss.item() / n_batch
            emb_loss += batch_emb_loss.item() / n_batch
            sslb_loss += batch_sslb_loss.item() / n_batch

        if args.lr_decay:
            scheduler.step()

        torch.cuda.empty_cache()

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        if (epoch + 1) % args.test_epoch != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = (
                    f'Epoch {epoch} [{time() - t1:.1f}s]: '
                    f'train==[{loss:.5f}={rec_loss:.5f} + {emb_loss:.5f} + {sslb_loss:.5f}]'
                )
                print(perf_str)

            continue

        t2 = time()
        model.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings, rela_embeddings = model(device)
            users_to_test = list(data_generator.test_set.keys())
            ret = test_torch(ua_embeddings[:, -2, :].detach().cpu().numpy(),
                             ia_embeddings[:, -2, :].detach().cpu().numpy(),
                             rela_embeddings[behs[-1]].detach().cpu().numpy(), users_to_test)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = (
                f'Epoch {epoch} [{t2 - t1:.1f}s + {t3 - t2:.1f}s]:, '
                f'recall=[{ret["recall"][0]:.5f}, {ret["recall"][1]:.5f}], '
                f'precision=[{ret["precision"][0]:.5f}, {ret["precision"][1]:.5f}], '
                f'hit=[{ret["hit_ratio"][0]:.5f}, {ret["hit_ratio"][1]:.5f}], '
                f'ndcg=[{ret["ndcg"][0]:.5f}, {ret["ndcg"][1]:.5f}]'
            )
            print(perf_str)
