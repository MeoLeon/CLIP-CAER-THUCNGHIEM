from torch import nn
from models.Temporal_Model import *
#from models.Prompt_Learner import *
#import copy
import numpy as np
from self_attention import SelfAttention # Nếu bạn lưu file self_attention.py
# Hoặc paste class SelfAttention trực tiếp vào file này

class GenerateModel(nn.Module):
    def __init__(self, clip_model, args):
        super().__init__()
        self.args = args
        self.dtype = next(clip_model.parameters()).dtype
        self.image_encoder = clip_model
        self.temporal_net = Temporal_Transformer_Cls(num_patches=16,
                                                     input_dim=512,
                                                     depth=args.temporal_layers,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)
        
        self.temporal_net_body = Temporal_Transformer_Cls(num_patches=16,
                                                     input_dim=512,
                                                     depth=args.temporal_layers,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)
        self.clip_model_ = clip_model
        self.my_attention_face = SelfAttention(in_dim=512) 
        self.my_attention_body = SelfAttention(in_dim=512)
        self.project_fc = nn.Linear(1024, 512)
        
    def forward(self, image_face,image_body):
        ################# Visual Part #################
        # Face Part
        n, t, c, h, w = image_face.shape
        image_face = image_face.contiguous().view(-1, c, h, w)
        image_face_features = self.image_encoder.encode_image(image_face.type(self.dtype))
        image_face_features = image_face_features.contiguous().view(n, t, -1)
        # --- THÊM ĐOẠN XỬ LÝ ATTENTION FACE ---
        # Đổi chiều thành [N, 512, T] để khớp với Conv1d của SelfAttention
        feat_face_in = image_face_features.permute(0, 2, 1) 
        feat_face_out, attn_map_face = self.my_attention_face(feat_face_in)
        # Ví dụ: Chỉ lưu khi đang ở chế độ eval (không train) để tránh ghi đè liên tục
        if not self.training:
            # Lưu về CPU dưới dạng file numpy
            np.save('attention_face_output.npy', attn_map_face.detach().cpu().numpy())
        # Lưu attention map ra file (Ví dụ: lưu batch đầu tiên dưới dạng .npy)
        # Lưu ý: Việc lưu file mỗi lần forward sẽ làm chậm training, nên chỉ dùng khi test hoặc debug.
        # np.save('attention_map_face.npy', attn_map_face.detach().cpu().numpy())

        # Đổi lại chiều cũ [N, T, 512]
        image_face_features = feat_face_out.permute(0, 2, 1)
        video_face_features = self.temporal_net(image_face_features)  # (4*512)
        
        # Body Part
        n, t, c, h, w = image_body.shape
        image_body = image_body.contiguous().view(-1, c, h, w)
        image_body_features = self.image_encoder.encode_image(image_body.type(self.dtype))
        image_body_features = image_body_features.contiguous().view(n, t, -1)
        # --- THÊM ĐOẠN XỬ LÝ ATTENTION BODY (Tương tự) ---
        feat_body_in = image_body_features.permute(0, 2, 1)
        feat_body_out, attn_map_body = self.my_attention_body(feat_body_in)
        
        # np.save('attention_map_body.npy', attn_map_body.detach().cpu().numpy())
        
        image_body_features = feat_body_out.permute(0, 2, 1)
        # -------------------------------------------------
        video_body_features = self.temporal_net_body(image_body_features)

        # Concatenate the two parts
        video_features = torch.cat((video_face_features, video_body_features), dim=-1)
        video_features = self.project_fc(video_features)
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)

        ################# Text Part ###################
        #prompts = self.prompt_learner()
        #tokenized_prompts = self.tokenized_prompts
        #text_features = self.text_encoder(prompts, tokenized_prompts)
        #text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        ###############################################

        output = video_features  / 0.01
        return output