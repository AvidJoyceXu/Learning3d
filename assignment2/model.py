from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d
import ipdb

class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        self.args_type = args.type
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True) # NOTE: use resnet18 pretrained encoder
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            # TODO:
            # self.decoder = nn.Sequential(
            #     nn.Linear(512, 1024), # [b, 1024]
            #     nn.ReLU(),
            #     nn.Linear(1024, 2048), # [b, 2048]
            #     nn.ReLU(),
            #     nn.Linear(2048, 32 * 32 * 32), # [b, 32*32*32]
            #     nn.Sigmoid() # NOTE: Sigmoid is used to ensure the output is between 0 and 1
            # )   
            # Project and reshape features
            self.voxel_projection = nn.Sequential(
                nn.Linear(512, 4*4*4*512),
                nn.ReLU()
            )
            
            # 3D Deconvolution layers
            deconv1 = nn.Sequential(
                nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(256),
                nn.ReLU()
            )
            
            deconv2 = nn.Sequential(
                nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU()
            )
            
            deconv3 = nn.Sequential(
                nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU()
            )
            
            # Final layer to get binary voxel predictions
            final_conv = nn.Conv3d(64, 1, kernel_size=3, padding=1)
            
            self.decoder = nn.Sequential( # Input: [b, 512]
                deconv1, # [b, 256, 8, 8, 8]
                deconv2, # [b, 128, 16, 16, 16]
                deconv3, # [b, 64, 32, 32, 32]
                final_conv, # [b, 1, 32, 32, 32]
                nn.Sigmoid() # NOTE: Sigmoid is used to ensure the output is between 0 and 1
            )

        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            # TODO:
            self.decoder = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.n_point * 3),
                nn.Tanh()  # NOTE: bound the output to [-1, 1]
            )            
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            # TODO:
            # Get number of vertices in the mesh
            num_vertices = self.mesh_pred.verts_packed().shape[0] // args.batch_size
            
            self.decoder = nn.Sequential(
                nn.Linear(512, 1024), # [b, 1024]
                nn.ReLU(),
                nn.Linear(1024, 2048), # [b, 2048]
                nn.ReLU(),
                nn.Linear(2048, num_vertices * 3), # [b, num_vertices * 3]
                nn.Tanh()  # bound the vertex offsets
            )          

    def init_weights(self):
        if self.args_type == "vox":
            self.voxel_projection[0].weight.data.normal_(0, 0.02)
            self.voxel_projection[0].bias.data.fill_(0)
        for m in self.decoder: # NOTE: Only initialize the decoder
            if isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                print(m)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Sequential):
                for sub_m in m:
                    if isinstance(sub_m, nn.ConvTranspose3d) or isinstance(sub_m, nn.Conv3d) or isinstance(sub_m, nn.Linear):
                        print(sub_m)
                        nn.init.xavier_uniform_(sub_m.weight)
                        if sub_m.bias is not None:
                            nn.init.zeros_(sub_m.bias)

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            # TODO:
            voxel_feature = self.voxel_projection(encoded_feat).reshape(-1, 512, 4, 4, 4) # [b, 512, 4, 4, 4]
            voxels_pred = self.decoder(voxel_feature)
            # voxels_pred = voxels_pred.reshape(-1, 1, 32, 32, 32) # [b, 1, 32, 32, 32]
            # ("[forward]voxel debug: ", voxels_pred.shape)
            return voxels_pred

        elif args.type == "point":
            # TODO:
            pointclouds_pred = self.decoder(encoded_feat)
            pointclouds_pred = pointclouds_pred.reshape(-1, self.n_point, 3)
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            deform_vertices_pred = self.decoder(encoded_feat)
            # print(self.mesh_pred.verts_packed().shape)
            # print(deform_vertices_pred.reshape([-1,3]).shape)
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return mesh_pred         