from utils import resize2d
import torch
import numpy as np
from torch.cuda import init
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from Inception import Inception
import random
import config
from PIL import Image

device = torch.device(config.DEVICE)

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
        
        att1 = self.encoder_att(encoder_out) # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden).squeeze(0) # (batch_size, attention_dim)
        #latent fusion
        
        att = self.full_att(self.tanh(att1 + att2.unsqueeze(1))).squeeze(2) # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)

        # att1 al posto di encoder_out, perché serve passare da 600 a 300
        # Applicare invece linear successivamente ad attention_w_encoding?
        attention_weighted_encoding = (att1 * alpha.unsqueeze(2)).sum(dim=1) # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers = 1):
        
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size = input_size + 1, hidden_size = hidden_size,
                            num_layers = num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x_input, decoder_hidden):

        gru_out, self.hidden = self.gru(x_input, decoder_hidden)
        output = self.linear(gru_out)

        return output, self.hidden


class EncoderDecoder(nn.Module):

    def __init__(self, attention_dim, image_feature_size, hidden_size, encoder, decoder, out_len = 12, use_teacher_forcing = False):
        super().__init__()

        self.teacher_forcing_ratio = config.TF_RATE
        self.use_teacher_forcing = use_teacher_forcing
        self.out_len = out_len
        self.encoder = encoder
        self.decoder = decoder
        self.image_feature_size = image_feature_size
        self.hidden_size = hidden_size
        self.attr_emb_size = image_feature_size # input_size / numero attributi (ora 2)
        self.category_embeds = nn.Embedding(32, self.attr_emb_size)
        self.color_embeds = nn.Embedding(10, self.attr_emb_size)
        self.fabric_embeds = nn.Embedding(59, self.attr_emb_size)

        self.tempo_embeds = nn.Linear(4, self.attr_emb_size)
        self.gate_linear = nn.Linear(hidden_size, image_feature_size)
        self.attention = Attention(hidden_size, hidden_size, attention_dim)
        self.sigmoid = nn.Sigmoid()
        if config.model_types[config.MODEL] == "residual":
            self.fc_reduction = nn.Linear(hidden_size, image_feature_size)

    def forward(self, input_batch, category, color, fabric, temporal_info, exogeneous_params, target=None, img_feature=None):

        if target is not None:
            target = target.t()

        batch_size = input_batch.size(0)

        if img_feature is None:
            inception_feat = self.encoder(input_batch)
        else:
            inception_feat = img_feature

        # attribute embeddings
        categ_embed = self.category_embeds(category.long())
        color_embed = self.color_embeds(color.long())
        fabric_embed = self.fabric_embeds(fabric.long())


        # Avg between attributes
        np_mean = np.mean([categ_embed.cpu().detach().numpy(), fabric_embed.cpu().detach().numpy(), color_embed.cpu().detach().numpy()], axis=0)
        attributes_embedding = torch.tensor(np_mean).unsqueeze(1).to(device)

        # Temporal Embeddings
        tempo_embed = self.tempo_embeds(temporal_info)

        # Creating first decoder_hidden_state = 0
        decoder_hidden = torch.zeros(1, batch_size, self.hidden_size).to(device)

        # Initializing predictions vector
        outputs = torch.zeros(self.out_len, batch_size, 1)

        # Initializing first prediction
        decoder_output = torch.zeros(batch_size, 1, 1).to(device)

        # List of alphas, for attention check
        attn_list = []

        for t in range(self.out_len):

            # Attention
            attention_weighted_encoding, alpha = self.attention(torch.cat([inception_feat, attributes_embedding.repeat(1, 64, 1)], dim=2), decoder_hidden)
            # Saving alpha (for visualization purpose)
            attn_list.append(alpha)
            # add Linear
            gate = self.sigmoid(self.gate_linear(decoder_hidden))

            attention_weighted_encoding = gate * attention_weighted_encoding
            attention_weighted_encoding = attention_weighted_encoding.transpose(0,1)

            # Reshape to (batch_size, 1, input_size)
            attention_weighted_encoding = attention_weighted_encoding.sum(1)
            attention_weighted_encoding = attention_weighted_encoding.view(-1, 1, self.image_feature_size)

            #Residual
            if config.model_types[config.MODEL] == "residual":
                weight_cat_embedding = torch.cat([attributes_embedding, attention_weighted_encoding], dim=2)
                weight_cat_embedding = self.fc_reduction(weight_cat_embedding)

                residual_block = attributes_embedding + weight_cat_embedding + attention_weighted_encoding + tempo_embed.unsqueeze(1)

                x_input = [residual_block, decoder_output]
            
            #Concat
            else:
                x_input = [attention_weighted_encoding, attributes_embedding, tempo_embed.unsqueeze(1), decoder_output]
            
            if config.USE_EXOG:
                x_input.append(exogeneous_params.unsqueeze(1))

            # Concatenating last predicition to attention_weighted_encoding + attributes + exogeneous(optional)
            x_input = torch.cat(x_input, dim=2)


            # GRU
            decoder_output, decoder_hidden = self.decoder(x_input, decoder_hidden)
            outputs[t] = decoder_output.squeeze(1)
            # Setting to zero negative outs
            #outputs[t] = torch.clamp(outputs[t], min=0)
            
            # Teacher forcing
            teach_forcing = True if random.random() < self.teacher_forcing_ratio else False

            if self.use_teacher_forcing and teach_forcing and target is not None:
                decoder_output = target[t].unsqueeze(1).unsqueeze(2)

        # Scambio le due dimensioni per avere corrispondenza con il target
        outputs = outputs.transpose(0,1)

        return outputs.squeeze(), attn_list

