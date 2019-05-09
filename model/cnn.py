"""
A rnn model for relation extraction, written in pytorch.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from utils import constant, torch_utils
from model import layers
from model import cnn

class RelationModel_CNN(object):
    """ A wrapper class for the training and evaluation of models. """
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.model = PositionAwareCNN(opt, emb_matrix)
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])
    
    def update(self, batch):
        """ Run a step of forward and backward model update. """
        if self.opt['cuda']:
            inputs = [b.cuda() for b in batch[:7]]
            labels = batch[7].cuda()
        else:
            inputs = [b for b in batch[:7]]
            labels = batch[7]

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, _ = self.model(inputs)
        loss = self.criterion(logits, labels)
        
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.data.item()
        return loss_val

    def predict(self, batch, unsort=True):
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """
        if self.opt['cuda']:
            inputs = [b.cuda() for b in batch[:7]]
            labels = batch[7].cuda()
        else:
            inputs = [b for b in batch[:7]]
            labels = batch[7]

        orig_idx = batch[8]

        # forward
        self.model.eval()
        logits, _ = self.model(inputs)
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, dim=1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
# =============================================================================
#         predictions = logits.data.cpu().numpy()
# =============================================================================
        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    predictions, probs)))]
        return predictions, probs, loss.data.item()

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                'epoch': epoch
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']



class DecisionLevelAttention(nn.Module):
    """
    A decision-level attention layer where the attention weight is
    a = T' . tanh(Wx + b), then softmax 
    where x is the input, T is context vector
    """

    def __init__(self, input_size, attn_size, opt):
        super(DecisionLevelAttention, self).__init__()

        self.opt = opt
        self.input_size = input_size
        self.attn_size = attn_size
#        self.attn_size = 300
#        self.ulinear = nn.Linear(input_size, attn_size, bias=True)
#        self.vlinear = nn.Linear(input_size, attn_size, bias=True)
        self.ulinear = nn.Linear(opt['hidden_dim_cnn'], attn_size, bias=True)
        self.vlinear = nn.Linear(opt['hidden_dim'], attn_size, bias=True)
#        self.mlinear = nn.Linear(opt['hidden_dim_cnn'], attn_size, bias=True)
#        self.nlinear = nn.Linear(opt['hidden_dim_cnn'], attn_size, bias=True)
        # if opt['ner_dim_subj_obj'] > 0:
        #     self.w1linear = nn.Linear(opt['ner_dim_subj_obj'], attn_size, bias=True)
        #     self.w2linear = nn.Linear(opt['ner_dim_subj_obj'], attn_size, bias=True)
        self.tlinear = nn.Linear(attn_size, 1, bias=False)
        self.init_weights()

    def init_weights(self):
        self.ulinear.weight.data.normal_(std=0.001).to("cuda")
        self.vlinear.weight.data.normal_(std=0.001).to("cuda")
#        self.mlinear.weight.data.normal_(std=0.001).to("cuda")
#        self.nlinear.weight.data.normal_(std=0.001).to("cuda")
        # if self.opt['ner_dim_subj_obj'] > 0:
        #     self.w1linear.weight.data.normal_(std=0.001).to("cuda")
        #     self.w2linear.weight.data.normal_(std=0.001).to("cuda")
        self.tlinear.weight.data.zero_().to("cuda")  # use zero to give uniform attention at the beginning

    def forward(self, inputs):
        """
        inputs = [CNN channel output, RNN channel output]
        each input element has shape: batch_size X feature_dim
        """
        batch_size = inputs[0].size()[0]

        # if self.opt['ner_dim_subj_obj'] > 0:
        #     x1, x2,subj_ner, obj_ner = inputs
        #     x1_proj = torch.tanh(self.ulinear(x1))
        #     x2_proj = torch.tanh(self.vlinear(x2))
        #     subj_ner_proj = torch.tanh(self.w1linear(subj_ner))
        #     obj_ner_proj = torch.tanh(self.w2linear(obj_ner))

        #     hiddens = torch.cat((x1_proj.unsqueeze(1), x2_proj.unsqueeze(1),
        #         subj_ner_proj.unsqueeze(1), obj_ner_proj.unsqueeze(1)), dim=1)  # batch_size X 4 X attn_size
        #     scores = [self.tlinear(x1_proj), self.tlinear(x2_proj),
        #             self.tlinear(subj_ner_proj),self.tlinear(obj_ner_proj)]
        # else:

        x1, x2, x3 = inputs
#        x1_proj = torch.tanh(self.ulinear(x1))
#        x2_proj = torch.tanh(self.vlinear(x2))
#        x3_proj = torch.tanh(self.mlinear(x3))
        x1_proj = F.relu(self.ulinear(x1))
        x2_proj = F.relu(self.ulinear(x2))
        x3_proj = F.relu(self.vlinear(x3))
#        x4_proj = F.relu(self.vlinear(x4))
        hiddens = torch.cat((x1_proj.unsqueeze(1), x2_proj.unsqueeze(1), x3_proj.unsqueeze(1)), dim=1)
#        hiddens = torch.cat((x1.unsqueeze(1), x2.unsqueeze(1)), dim=1)
        scores = [self.tlinear(x1_proj), self.tlinear(x2_proj), self.tlinear(x3_proj)]

        scores = torch.cat(scores, dim=1)

        weights = F.softmax(scores, dim=-1)  # batch_size X 4

        outputs = weights.unsqueeze(1).bmm(hiddens).squeeze(1)   # batch_size X attn_size
        # add activation function
#        outputs = torch.tanh(outputs)
#        outputs = F.relu(outputs)
        return outputs


class PositionAwareCNN(nn.Module):
    """ A sequence model for relation extraction. """

    def __init__(self, opt, emb_matrix=None):
        super(PositionAwareCNN, self).__init__()
        self.drop = nn.Dropout(opt['dropout'])
#        self.drop_cnn = nn.Dropout(opt['dropout_cnn'])
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        if opt['pos_dim'] > 0:
            self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim'],
                    padding_idx=constant.PAD_ID)
        if opt['ner_dim'] > 0:
            self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim'],
                    padding_idx=constant.PAD_ID)
        
        input_size = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']
        self.rnn = nn.LSTM(input_size, opt['hidden_dim'], opt['num_layers'], batch_first=True,\
                dropout=opt['dropout'])#, bidirectional=True)
# =============================================================================
#         cnn
# =============================================================================
        self.conv1 = nn.Conv1d(input_size, opt['hidden_dim_cnn'], opt['k_size'])
        self.conv2 = nn.Conv1d(input_size, opt['hidden_dim_cnn'], opt['k_size2'])
#        self.conv3 = nn.Conv1d(input_size, opt['hidden_dim_cnn'], opt['k_size3'])
        self.activation = nn.ReLU()
#        self.activation = nn.Tanh()
#        self.norm = nn.LayerNorm(opt['hidden_dim_cnn']*2)
#        dlattn_dim = 200
        self.linear = nn.Linear(opt['hidden_dim_cnn']*2+opt['hidden_dim'], opt['num_class'])
#        self.linear2 = nn.Linear(opt['hidden_dim']+opt['hidden_dim_cnn'], 50)
#        self.bilinear = nn.Bilinear(opt['hidden_dim_cnn'], opt['hidden_dim']*2, opt['num_class'])
        
#        self.combine = DecisionLevelAttention(300, dlattn_dim, opt)

        if opt['attn']:
            self.attn_layer = layers.PositionAwareAttention(opt['hidden_dim'],
                    opt['hidden_dim'], 2*opt['pe_dim'], opt['attn_dim'])
            self.attn_layer2 = layers.PositionAwareAttention(opt['hidden_dim_cnn'],
                    opt['hidden_dim_cnn'], 2*opt['pe_dim'], opt['attn_dim_cnn'])
            self.attn_layer3 = layers.PositionAwareAttention(opt['hidden_dim_cnn'],
                    opt['hidden_dim_cnn'], 2*opt['pe_dim'], opt['attn_dim_cnn'])
#            self.attn_layer4 = layers.PositionAwareAttention(opt['hidden_dim_cnn'],
#                    opt['hidden_dim_cnn'], 2*opt['pe_dim'], opt['attn_dim'])
            self.pe_emb = nn.Embedding(constant.MAX_LEN * 2 + 1, opt['pe_dim'])

        self.opt = opt
        self.topn = self.opt.get('topn', 1e10)
        self.use_cuda = opt['cuda']
        self.emb_matrix = emb_matrix
        self.init_weights()
    
    def init_weights(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:,:].uniform_(-1.0, 1.0) # keep padding dimension to be 0
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        if self.opt['pos_dim'] > 0:
            self.pos_emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        if self.opt['ner_dim'] > 0:
            self.ner_emb.weight.data[1:,:].uniform_(-1.0, 1.0)

        self.linear.bias.data.fill_(0)
        init.xavier_uniform_(self.linear.weight, gain=1) # initialize linear layer
        
#        self.linear2.bias.data.fill_(0)
#        init.xavier_uniform_(self.linear2.weight, gain=1)
# =============================================================================
#         self.bilinear.bias.data.fill_(0)
#         init.xavier_uniform_(self.bilinear.weight, gain=1)
# =============================================================================
        
        if self.opt['attn']:
            self.pe_emb.weight.data.uniform_(-1.0, 1.0)

        # decide finetuning
        if self.topn <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.topn < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.topn))
            self.emb.weight.register_hook(lambda x: \
                    torch_utils.keep_partial_grad(x, self.topn))
        else:
            print("Finetune all embeddings.")

    def zero_state(self, batch_size): 
        state_shape = (self.opt['num_layers'], batch_size, self.opt['hidden_dim'])
        h0 = c0 = torch.zeros(*state_shape, requires_grad=False)
        if self.use_cuda:
            return h0.cuda(), c0.cuda()
        else:
            return h0, c0
    
    def forward(self, inputs):
        words, masks, pos, ner, deprel, subj_pos, obj_pos = inputs # unpack
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        batch_size = words.size()[0]
        
        # embedding lookup
        word_inputs = self.emb(words)
        inputs = [word_inputs]
        if self.opt['pos_dim'] > 0:
            inputs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            inputs += [self.ner_emb(ner)]
        inputs = self.drop(torch.cat(inputs, dim=2)) # add dropout to input
        input_size = inputs.size(2)
        
        # rnn
        h0, c0 = self.zero_state(batch_size)
        inputs_rnn = nn.utils.rnn.pack_padded_sequence(inputs, seq_lens, batch_first=True)
        outputs, (ht, ct) = self.rnn(inputs_rnn, (h0, c0))
#        outputs, ht = self.rnn(inputs_rnn, h0)
        outputs, output_lens = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        hidden = self.drop(ht[-1,:,:]) # get the outmost layer h_n
#        hidden2 = self.drop(ht[0,:,:])
#        hidden = torch.cat((hidden,hidden),1)
        outputs = self.drop(outputs)
        
# =============================================================================
#         #cnn
# =============================================================================
#        inputs1 = inputs.transpose(1,2)
        inputs1 = F.pad(inputs.transpose(1,2), (0,1))
        inputs2 = F.pad(inputs.transpose(1,2), (0,2))
#        inputs3 = F.pad(inputs.transpose(1,2), (0,4))
#        outputs_cnn = self.conv1(inputs.transpose(1,2))
        outputs_cnn = self.conv1(inputs1)
        outputs_cnn2 = self.conv2(inputs2)
#        outputs_cnn3 = self.conv3(inputs3)
        outputs_cnn = self.activation(outputs_cnn)
        outputs_cnn2 = self.activation(outputs_cnn2)
#        outputs_cnn3 = self.activation(outputs_cnn3)
#        outputs_cnn = self.drop_cnn(outputs_cnn)
#        outputs_cnn2 = self.drop_cnn(outputs_cnn2)
        
        query = F.max_pool1d(outputs_cnn, max(seq_lens).item())
        query2 = F.max_pool1d(outputs_cnn2, max(seq_lens).item())
#        query3 = F.max_pool1d(outputs_cnn3, max(seq_lens).item())
        outputs_cnn = torch.transpose(outputs_cnn, 1, 2)
        outputs_cnn2 = torch.transpose(outputs_cnn2, 1, 2)
#        outputs_cnn3 = torch.transpose(outputs_cnn3, 1, 2)
        
#        outputs_cnn = torch.cat((outputs_cnn, outputs_cnn2), dim=2)
#        query = torch.cat((query, query2), dim=1)
        
#        outputs = torch.cat((outputs_cnn, outputs), dim=2)
#        hidden = torch.cat((query.squeeze(), hidden), dim=1)
        
        # attention
        if self.opt['attn']:
            # convert all negative PE numbers to positive indices
            # e.g., -2 -1 0 1 will be mapped to 98 99 100 101
            subj_pe_inputs = self.pe_emb(subj_pos + constant.MAX_LEN)
            obj_pe_inputs = self.pe_emb(obj_pos + constant.MAX_LEN)
            pe_features = torch.cat((subj_pe_inputs, obj_pe_inputs), dim=2)
            final_hidden = self.attn_layer(outputs, masks, hidden, pe_features)
            final_query = self.attn_layer2(outputs_cnn, masks, query, pe_features)
#            final_query = torch.squeeze(query)
            final_query2 = self.attn_layer3(outputs_cnn2, masks, query2, pe_features)
#            final_query3 = self.attn_layer4(outputs_cnn3, masks, query3, pe_features)
#            final_query3 = self.attn_layer2(outputs_cnn3, masks, query3, pe_features)
        else:
#            final_hidden = hidden
            final_query = query
            final_query2 = query2
#            final_query3 = query3

#        final_query = self.norm(final_query)
        final = torch.cat((final_query, final_query2, final_hidden), 1)
# =============================================================================
#         final = torch.cat((query.squeeze(), query2.squeeze()), 1)
# =============================================================================
#        final = self.combine([final_query,final_query2, final_hidden])
#        logits = self.linear2(final)
        logits = self.linear(final)
#        logits = self.bilinear(final_query, final_hidden)
        return logits, final
    