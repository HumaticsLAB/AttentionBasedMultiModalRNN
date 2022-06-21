import torch
import torch.nn as nn
import random
import config
from Inception import Inception

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.inception = Inception()
        self.inception.fc = nn.Linear(2048, 1024)
        self.inception.lastconv = nn.Conv2d(3, 2, kernel_size=3)
        self.inception.fc1 = nn.Linear(1024, 300)

    def forward(self, x):

        x = self.inception(x)
        x = x.view(-1, 64, 2048)
        x = self.inception.fc(x)
        x = self.inception.fc1(x)
        
        return x

class Attention(nn.Module):

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_dim = encoder_dim
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden).squeeze(0)  # (batch_size, attention_dim)
        # latent fusion

        att = self.full_att(self.tanh(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)

        # att1 al posto di encoder_out, perchÃ© serve passare da 600 a 300
        # Applicare invece linear successivamente ad attention_w_encoding?
        attention_weighted_encoding = (att1 * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size=input_size + 1, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x_input, decoder_hidden):
        gru_out, self.hidden = self.gru(x_input, decoder_hidden)
        output = self.linear(gru_out)

        return output, self.hidden


class EncoderDecoder(nn.Module):
    def __init__(self, l_seq, device, use_teacher_forcing=True):
        super(EncoderDecoder, self).__init__()
        self.l_seq = l_seq
        self.device = device

        self.D = config.HIDDEN_SIZE
        self.gtrend_size = config.EXOG_LEN*config.EXOG_NUM

        self.category_embeds = nn.Embedding(32, self.D)
        self.color_embeds = nn.Embedding(10, self.D)
        self.fabric_embeds = nn.Embedding(59, self.D)

        self.encoder = Encoder()
        # self.shape_embeds = nn.Embedding(23, self.D)

        self.proj_img_att_concat = nn.Linear(self.D * 2, self.D)
        self.img_attention = Attention(self.D, self.D, self.D)
        self.img_gate_linear = nn.Linear(self.D, self.D)
        self.sigmoid = nn.Sigmoid()

        self.tempo_embeds = nn.Linear(4, self.D)

        self.gtrend_embed = nn.Linear(self.gtrend_size, self.D)
        self.att_t = Attention(self.D, self.D, self.gtrend_size)
        self.t_gate_linear = nn.Linear(self.D, self.gtrend_size)
        self.temporal_embedding = nn.Linear(self.gtrend_size, self.D)

        self.proj_encoder_input = nn.Linear(self.D, self.D)
        self.att_m = Attention(self.D, self.D, self.D)
        self.m_gate_linear = nn.Linear(self.D, self.D)

        self.decoder = DecoderRNN(self.D, self.D, 1)
        self.teacher_forcing_enabled = use_teacher_forcing
        self.teacher_forcing_ratio = 0.5

    def forward(self, input_batch: torch.FloatTensor, categ: torch.LongTensor, color: torch.LongTensor, fabric:  torch.LongTensor, temporal_info: torch.FloatTensor, exogeneous_params: torch.LongTensor,
                target: torch.FloatTensor = None, feats: torch.FloatTensor = None) -> torch.Tensor:

        bs = input_batch.size(0)
        if target is not None:
            target = target.t()

        # Image Embedding
        if feats is None:
            inc_feats = self.encoder(input_batch)
        else:
            inc_feats = feats

        # Word Embedding
        categ_embed = self.category_embeds(categ)
        color_embed = self.color_embeds(color)
        fabric_embed = self.fabric_embeds(fabric.long())

        attr_embedding = torch.cat([categ_embed.unsqueeze(-1), color_embed.unsqueeze(-1), fabric_embed.unsqueeze(-1)], dim=-1).mean(-1)

        decoder_hidden = torch.zeros(1, bs, self.D).to(self.device)
        outputs = torch.zeros(self.l_seq, bs, 1).to(self.device)
        decoder_output = torch.zeros(bs, 1, 1).to(self.device)

        # Temporal Embeddings
        tempo_embed = self.tempo_embeds(temporal_info)


        if config.USE_EXOG:
            E = exogeneous_params

        for t in range(self.l_seq):
            attention_weighted_encoding, _ = self.img_attention(
                self.proj_img_att_concat(torch.cat([inc_feats, attr_embedding.unsqueeze(1).repeat(1, 64, 1)], dim=2)),
                decoder_hidden)
            attention_img_embedding = self.sigmoid(self.img_gate_linear(decoder_hidden)).squeeze() * attention_weighted_encoding


            if config.USE_EXOG:
                attention_weighted_it, _ = self.att_t(self.gtrend_embed(E), decoder_hidden)
                It = self.sigmoid(self.t_gate_linear(decoder_hidden)).squeeze() * attention_weighted_it

                gtrend_embedding = self.temporal_embedding(It)

                
                J = torch.cat(
                    [attention_img_embedding.unsqueeze(1), attr_embedding.unsqueeze(1), tempo_embed.unsqueeze(1), gtrend_embedding.unsqueeze(1)],
                    dim=1)
            
            else:
                J = torch.cat([attention_img_embedding.unsqueeze(1), attr_embedding.unsqueeze(1),], dim=1)




            attention_weighted_eps, _ = self.att_m(self.proj_encoder_input(J), decoder_hidden)

            eps = self.sigmoid(self.m_gate_linear(decoder_hidden)).squeeze() * attention_weighted_eps
            x_input = torch.cat([eps.unsqueeze(1), decoder_output], dim=2)

            decoder_output, decoder_hidden = self.decoder(x_input, decoder_hidden)
            outputs[t] = decoder_output.squeeze(1)

            # Teacher forcing
            teach_forcing = True if random.random() < self.teacher_forcing_ratio else False

            if self.teacher_forcing_enabled and teach_forcing and target is not None:
                decoder_output = target[t].unsqueeze(1).unsqueeze(2)

        outputs = outputs.transpose(0, 1)
        return outputs.squeeze()
