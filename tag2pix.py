import itertools, time, pickle, pprint
from pathlib import Path
import torch_fidelity
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from PIL import Image
from tqdm import tqdm
import os
import utils
from loader.dataloader import get_dataset, get_tag_dict, ColorSpace2RGB
from model.convnet_dev_CBAM_Att_CARAFE import Discriminator
from model.se_resnet import BottleneckX, SEResNeXt
from model.pretrained import se_resnext_half
# from network import Discriminator



class tag2pix(object):
    def __init__(self, args):
        if args.model == 'tag2pix':
            from network import Generator
        elif args.model == 'senet':
            from model.GD_senet import Generator
        elif args.model == 'resnext':
            from model.GD_resnext import Generator
        elif args.model == 'catconv':
            from model.GD_cat_conv import Generator
        elif args.model == 'catall':
            from model.GD_cat_all import Generator
        elif args.model == 'adain':
            from model.GD_adain import Generator
        elif args.model == 'seadain':
            from model.GD_seadain import Generator
        elif args.model == 'mynet':
            from mynet import Generator
        elif args.model == 'mynet_1':
            from mynet_1 import Generator
        elif args.model == 'mynet_2':
            from mynet_2 import Generator
        elif args.model == 'mynet_3':
            from mynet_3 import Generator
        elif args.model == 'mynet_4':
            from mynet_4 import Generator
        elif args.model == 'mynet_5':
            from mynet_5 import Generator
        elif args.model == 'mynet_6':
            from mynet_6 import Generator
        elif args.model == 'mynet_7':
            from mynet_7 import Generator
        elif args.model == 'mynet_8':
            from mynet_8 import Generator
        elif args.model == 'AFF_cat':
            from model.AFF_cat import Generator
        elif args.model == 'NAM_conv':
            from model.NAM_conv import Generator
        elif args.model == 'CBAM_cat':
            from model.CBAM_cat import Generator  
        elif args.model == 'convnet_dev':
            from model.convnet_dev import Generator  
        elif args.model == 'convnet_dev_changeD':
            from model.convnet_dev_changeD import Generator  
        elif args.model == 'convnet_dev_MSCAM':
            from model.convnet_dev_MSCAM import Generator    
        elif args.model == 'convnet_dev_NAM':
            from model.convnet_dev_NAM import Generator  
        elif args.model == 'convnet_dev_EPSA':
            from model.convnet_dev_EPSA import Generator    
        elif args.model == 'convnet_dev_skeleton_noadain':
            from model.convnet_dev_skeleton_noadain import Generator     
        elif args.model == 'convnet_dev_CA_Att':
            from model.convnet_dev_CA_Att import Generator
        elif args.model == 'convnet_dev_CARAFE':
            from model.convnet_dev_CARAFE import Generator
        elif args.model == 'convnet_dev_CBAM_skeleton_adain_block':
            from model.convnet_dev_CBAM_skeleton_adain_block import Generator
        elif args.model == 'convnet_dev_CBAM_Att_CARAFE':
            from model.convnet_dev_CBAM_Att_CARAFE import Generator    
        elif args.model == 'convnet_dev_skeleton_adain_noatt':
            from model.convnet_dev_skeleton_adain_noatt import Generator    
        elif args.model == 'convnet_dev_CBAM_skeleton_adain':
            from model.convnet_dev_CBAM_skeleton_adain import Generator    
        elif args.model == 'convnet_dev_nose':
            from model.convnet_dev_nose import Generator  
        elif args.model == 'convnet_dev_SGA':
            from model.convnet_dev_SGA import Generator 
        elif args.model == 'convnet_dev_skeleton':
            from model.convnet_dev_skeleton import Generator 
        elif args.model == 'convnet_dev_skeleton_Adain':
            from model.convnet_dev_skeleton_Adain import Generator   
        elif args.model == 'convnet_dev_skeleton_Adain_NAM':
            from model.convnet_dev_skeleton_Adain_NAM import Generator    
        elif args.model == 'convnet_dev_skeleton_Adain_NAM_gate':
            from model.convnet_dev_skeleton_Adain_NAM_gate import Generator    
        elif args.model == 'convnet_dev_skeleton_NAM_gate':
            from model.convnet_dev_skeleton_NAM_gate import Generator       
        elif args.model == 'convnet_dev_skeleton_NAMo_gate':
            from model.convnet_dev_skeleton_NAMo_gate import Generator      
        elif args.model == 'convnet_dev_CBAM_skeleton_adain':
            from model.convnet_dev_CBAM_skeleton_adain import Generator  
        elif args.model == 'convnet_dev_CBAM_skeleton':
            from model.convnet_dev_CBAM_skeleton import Generator  
        elif args.model == 'network_skeleton':
            from model.network_skeleton import Generator 
        else:
            raise Exception('invalid model name: {}'.format(args.model))

        self.args = args
        self.epoch = args.epoch
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size

        self.gpu_mode = not args.cpu
        self.input_size = args.input_size
        self.color_revert = ColorSpace2RGB(args.color_space)
        self.layers = args.layers
        [self.cit_weight, self.cvt_weight ,self.person_weight] = args.cit_cvt_weight

        self.load_dump = (args.load != "")

        self.load_path = Path(args.load)

        self.l1_lambda = args.l1_lambda
        self.guide_beta = args.guide_beta
        self.adv_lambda = args.adv_lambda
        self.save_freq = args.save_freq

        self.two_step_epoch = args.two_step_epoch
        self.brightness_epoch = args.brightness_epoch
        self.save_all_epoch = args.save_all_epoch

        self.iv_dict, self.cv_dict, self.id_to_name = get_tag_dict(args.tag_dump)

        cvt_class_num = len(self.cv_dict.keys())
        cit_class_num = len(self.iv_dict.keys())
        self.class_num = cvt_class_num + cit_class_num

        self.start_epoch = 1

        #### load dataset
        if not args.test:
            self.train_data_loader, self.test_data_loader = get_dataset(args)
            self.result_path = Path(args.result_dir) / time.strftime('%y%m%d-%H%M%S', time.localtime())

            if not self.result_path.exists():
                self.result_path.mkdir()

            self.test_images = self.get_test_data(self.test_data_loader, args.test_image_count)
        else:
            self.test_data_loader = get_dataset(args)
            self.result_path = Path(args.result_dir)


        ##### initialize network
        self.net_opt = {
            'guide': not args.no_guide,
            'relu': args.use_relu,
            'bn': not args.no_bn,
            'cit': not args.no_cit
        }

        if self.net_opt['cit']:
            self.Pretrain_ResNeXT = se_resnext_half(dump_path=args.pretrain_dump, num_classes=cit_class_num, input_channels=1)
        else:
            self.Pretrain_ResNeXT = nn.Sequential()

        self.G = Generator(input_size=args.input_size, layers=args.layers,
                cv_class_num=cvt_class_num, iv_class_num=cit_class_num, net_opt=self.net_opt)
        self.D = Discriminator(input_dim=3, output_dim=1, input_size=self.input_size,
                cv_class_num=cvt_class_num, iv_class_num=cit_class_num)

        for param in self.Pretrain_ResNeXT.parameters():
            param.requires_grad = False
        if args.test:
            for param in self.G.parameters():
                param.requires_grad = False
            for param in self.D.parameters():
                param.requires_grad = False

        self.Pretrain_ResNeXT = nn.DataParallel(self.Pretrain_ResNeXT)
        self.G = nn.DataParallel(self.G)
        self.D = nn.DataParallel(self.D)

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        self.BCE_loss = nn.BCELoss()
        self.CE_loss = nn.CrossEntropyLoss()
        self.L1Loss = nn.L1Loss()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("gpu mode: ", self.gpu_mode)
        print("device: ", self.device)
        print(torch.cuda.device_count(), "GPUS!")

        if self.gpu_mode:
            self.Pretrain_ResNeXT.to(self.device)
            self.G.to(self.device) 
            self.D.to(self.device)
            self.BCE_loss.to(self.device)
            self.CE_loss.to(self.device)
            self.L1Loss.to(self.device)
            
    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        
        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.to(self.device), self.y_fake_.to(self.device)

        if self.load_dump:
            self.load(self.load_path)
            print("continue training!!!!")
        else:
            self.end_epoch = self.epoch

        self.print_params()

        self.D.train()
        print('training start!!')
        start_time = time.time()

        for epoch in range(self.start_epoch, self.end_epoch + 1):
            print("EPOCH: {}".format(epoch))

            self.G.train()
            epoch_start_time = time.time()

            if epoch == self.brightness_epoch:
                print('changing brightness ...')
                self.train_data_loader.dataset.enhance_brightness(self.input_size)

            max_iter = self.train_data_loader.dataset.__len__() // self.batch_size
            
# skeleton_,多加了一个骨架进来
            for iter, (original_, sketch_, person_num, skeleton_, iv_tag_, cv_tag_) in enumerate(tqdm(self.train_data_loader, ncols=80)):
                if iter >= max_iter:
                    break

                if self.gpu_mode:
                    sketch_, original_, person_num, skeleton_, iv_tag_, cv_tag_ = sketch_.to(self.device), original_.to(self.device), person_num.to(self.device), skeleton_.to(self.device), iv_tag_.to(self.device), cv_tag_.to(self.device)

                # update D network
                self.D_optimizer.zero_grad()
# z这里修改了 用skeleton作为补充图 进入预训练好的resnext 然后在融合阶段
                with torch.no_grad():
                    feature_tensor = self.Pretrain_ResNeXT(skeleton_)
                if self.gpu_mode:
                    feature_tensor = feature_tensor.to(self.device)
                #     用原本的特征图
                # with torch.no_grad():
                #     feature_tensor = self.Pretrain_ResNeXT(sketch_)
                # if self.gpu_mode:
                #     feature_tensor = feature_tensor.to(self.device)

                D_real, CIT_real, CVT_real, person_real= self.D(original_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)

                G_f, _,= self.G(sketch_, feature_tensor, cv_tag_)
                if self.gpu_mode:
                    G_f = G_f.to(self.device)

#                     networkskeleton
                # G_f, _, _ = self.G(sketch_, feature_tensor, cv_tag_)
                # if self.gpu_mode:
                #     G_f = G_f.to(self.device)

                    
                D_f_fake, CIT_f_fake, CVT_f_fake, person_f_fake = self.D(G_f)
                D_f_fake_loss = self.BCE_loss(D_f_fake, self.y_fake_)

                if self.two_step_epoch == 0 or epoch >= self.two_step_epoch:
                    CIT_real_loss = self.BCE_loss(CIT_real, iv_tag_) if self.net_opt['cit'] else 0
                    CVT_real_loss = self.BCE_loss(CVT_real, cv_tag_)
                    person_real_loss = self.BCE_loss(person_real, person_num)
                    
                    C_real_loss = self.cvt_weight * CVT_real_loss + self.cit_weight * CIT_real_loss + self.person_weight * person_real_loss

                    CIT_f_fake_loss = self.BCE_loss(CIT_f_fake, iv_tag_) if self.net_opt['cit'] else 0
                    CVT_f_fake_loss = self.BCE_loss(CVT_f_fake, cv_tag_)
                    person_f_fake_loss = self.BCE_loss(person_f_fake, person_num)

                    C_f_fake_loss = self.cvt_weight * CVT_f_fake_loss + self.cit_weight * CIT_f_fake_loss + self.person_weight * person_f_fake_loss
                    
                else:
                    C_real_loss = 0
                    C_f_fake_loss = 0
                    person_f_fake_loss = 0
#                     Ladv  Lrec
                D_loss = self.adv_lambda * (D_real_loss + D_f_fake_loss) + (C_real_loss + C_f_fake_loss)

                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_f, G_g= self.G(sketch_, feature_tensor, cv_tag_)
                if self.gpu_mode:
                    G_f, G_g = G_f.to(self.device), G_g.to(self.device)
                
# #                 network_skeleton
#                 G_f, G_g, G_s = self.G(sketch_, feature_tensor, cv_tag_)
#                 if self.gpu_mode:
#                     G_f,G_g, G_s = G_f.to(self.device), G_g.to(self.device), G_s.to(self.device)
            
                

                D_f_fake, CIT_f_fake, CVT_f_fake , person_f_fake= self.D(G_f)
                D_f_fake_loss = self.BCE_loss(D_f_fake, self.y_real_)

                if self.two_step_epoch == 0 or epoch >= self.two_step_epoch:
                    CIT_f_fake_loss = self.BCE_loss(CIT_f_fake, iv_tag_) if self.net_opt['cit'] else 0
                    CVT_f_fake_loss = self.BCE_loss(CVT_f_fake, cv_tag_)

                    C_f_fake_loss = self.cvt_weight * CVT_f_fake_loss + self.cit_weight * CIT_f_fake_loss
                else:
                    C_f_fake_loss = 0

                L1_D_f_fake_loss = self.L1Loss(G_f, original_)
                
#                 这里才加了引导编码器的损失进来
                L1_D_g_fake_loss = self.L1Loss(G_g, original_) if self.net_opt['guide'] else 0
                
# #                 这是加了骨架的loss
#                 L1_D_s_fake_loss = self.L1Loss(G_s, skeleton_)

#                 G_loss = (D_f_fake_loss + C_f_fake_loss) + \
#                         (L1_D_f_fake_loss + L1_D_g_fake_loss * self.guide_beta + L1_D_s_fake_loss * self.guide_beta) * self.l1_lambda

                G_loss = (D_f_fake_loss + C_f_fake_loss) + \
                         (L1_D_f_fake_loss + L1_D_g_fake_loss * self.guide_beta) * self.l1_lambda

                self.train_hist['G_loss'].append(G_loss.item())           

                G_loss.backward()
                self.G_optimizer.step()

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [{:2d}] [{:4d}/{:4d}] D_loss: {:.8f}, G_loss: {:.8f}".format(
                        epoch, (iter + 1), max_iter, D_loss.item(), G_loss.item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

            # with torch.no_grad():
                # self.visualize_results(epoch)
                # utils.loss_plot(self.train_hist, self.result_path, epoch)

            if epoch >= self.save_all_epoch > 0:
                self.save(epoch)
            elif self.save_freq > 0 and epoch % self.save_freq == 0:
                self.save(epoch)

        print("Training finish!... save training results")

        if self.save_freq == 0 or epoch % self.save_freq != 0:
            if self.save_all_epoch <= 0 or epoch < self.save_all_epoch:
                self.save(epoch)

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: {:.2f}, total {} epochs time: {:.2f}".format(
            np.mean(self.train_hist['per_epoch_time']), self.epoch, self.train_hist['total_time'][0]))


    def test(self):
        # self.epoche=200
        
        for epoche in range(60,200):

            
            # print("convnet_dev_CBAM_Att_CARAFE的测试") 
#             230331-223109  166-200epoch
            # # loadurl=self.result_path /'230301-221217'/ 'tag2pix_{}_epoch.pkl'.format(epoche)
            # loadurl=self.result_path /'230331-223109'/ 'tag2pix_{}_epoch.pkl'.format(epoche)
            # result_path = self.result_path /'convnet_dev_CBAM_Att_CARAFE_result'/ 'tag2pix_{}_epoch'.format(epoche)
            # self.load_test(loadurl)
            
            # print("convnet_dev（cit)的测试") 
            # loadurl=self.result_path /'230207-221257'/ 'tag2pix_{}_epoch.pkl'.format(epoche)
            # result_path = self.result_path /'convnet_dev_new_result'/ 'tag2pix_{}_epoch'.format(epoche)
            # self.load_test(loadurl)
            
            
#             results/230306-194037
            # print("adain的测试") 
            # loadurl=self.result_path /'230306-194037'/ 'tag2pix_{}_epoch.pkl'.format(epoche)
            # result_path = self.result_path /'adain_result'/ 'tag2pix_{}_epoch'.format(epoche)
            # self.load_test(loadurl)
    
    
    #          230310-213847
            # print("convnet_dev_skeleton_NAMo_gate的测试") 
            # loadurl=self.result_path /'230310-213847'/ 'tag2pix_{}_epoch.pkl'.format(epoche)
            # result_path = self.result_path /'convnet_dev_skeleton_NAMo_gate_result'/ 'tag2pix_{}_epoch'.format(epoche)
            
            
            
            # 230307-183630
            # print("convnet_dev_skeleton_Adain_NAM_gate的测试") 
            # loadurl=self.result_path /'230307-183630'/ 'tag2pix_{}_epoch.pkl'.format(epoche)
            # result_path = self.result_path /'convnet_dev_skeleton_Adain_NAM_gate_result'/ 'tag2pix_{}_epoch'.format(epoche)
            
            
#             230310-145447
            # print("convnet_dev_skeleton_NAM_gate的测试") 
            # loadurl=self.result_path /'230310-145447'/ 'tag2pix_{}_epoch.pkl'.format(epoche)
            # result_path = self.result_path /'convnet_dev_skeleton_NAM_gate_result'/ 'tag2pix_{}_epoch'.format(epoche)
            
#             convnet_dev_skeleton_Adain
# 230314-213037  230416-190627
            # print("convnet_dev_skeleton_Adain的测试") 
            # loadurl=self.result_path /'230416-190627'/ 'tag2pix_{}_epoch.pkl'.format(epoche)
            # result_path = self.result_path /'convnet_dev_skeleton_Adain_sketch_result'/ 'tag2pix_{}_epoch'.format(epoche)
            # result_path = self.result_path /'convnet_dev_skeleton_Adain_single_result'/ 'tag2pix_{}_epoch'.format(epoche)
            
# /230327-002238
            # print("network_inputskeleton的测试") 
            # loadurl=self.result_path /'230310-213847'/ 'tag2pix_{}_epoch.pkl'.format(epoche)
            # result_path = self.result_path /'network_inputskeleton_single_result'/ 'tag2pix_{}_epoch'.format(epoche)

            # 230418-232708
            # print("convnet_dev_skeleton_NAM_gate(不加骨架图的)的测试") 
            # loadurl=self.result_path /'230418-232708'/ 'tag2pix_{}_epoch.pkl'.format(epoche)
            # result_path = self.result_path /'convnet_dev_skeleton_NAM_gate_result'/ 'tag2pix_{}_epoch'.format(epoche)
            
#             #       230415-110413  convnet_dev_skeleton_noadain  230424-100141    230629-094515
            print("convnet_dev_skeleton_noadain(不加adain的)的测试") 
            loadurl=self.result_path /'230629-094515'/ 'tag2pix_{}_epoch.pkl'.format(epoche)
            result_path = self.result_path /'convnet_dev_skeleton_noadain_new_result'/ 'tag2pix_{}_epoch'.format(epoche)
            
#             230331-223523  Tag2Pix
            # print("network的测试") 
            # loadurl=self.result_path /'230331-223523'/ 'tag2pix_{}_epoch.pkl'.format(epoche)
            # result_path = self.result_path /'network_result'/ 'tag2pix_{}_epoch'.format(epoche)

            # print("network的测试") 
            # loadurl=self.result_path /'230331-223523'/ 'tag2pix_{}_epoch.pkl'.format(epoche)
            # result_path = self.result_path /'network_single_result'/ 'tag2pix_{}_epoch'.format(epoche)
            
            # print("network_skeleton的测试") 
            # loadurl=self.result_path /'230409-190219'/ 'tag2pix_{}_epoch.pkl'.format(epoche)
            # result_path = self.result_path /'network_skeleton_real__result'/ 'tag2pix_{}_epoch'.format(epoche)  
            # result_path = self.result_path /'network_skeleton_real_single1_result'/ 'tag2pix_{}_epoch'.format(epoche)
            # result_path = self.result_path /'change_result'/ 'tag2pix_{}_epoch'.format(epoche)
            



# 230409-192330
# 230416-185919
            # print("convnet_dev_CBAM_skeleton_adain的测试") 
            # loadurl=self.result_path /'230416-185919'/ 'tag2pix_{}_epoch.pkl'.format(epoche)
            # result_path = self.result_path /'convnet_dev_CBAM_skeleton_adain_result'/ 'tag2pix_{}_epoch'.format(epoche)

            if not result_path.exists():
                result_path.mkdir()
                
            

            self.load_test(loadurl)
            self.D.eval()
            self.G.eval()

            load_path = self.load_path
        
            
            # print(test_path)
        
            # print(loadurl)
            
# 测试skeleton时要这样改！！！
            with torch.no_grad():
                for sketch_, index_,skeleton_, _, cv_tag_ in tqdm(self.test_data_loader, ncols=80):
                    if self.gpu_mode: 
                        sketch_, skeleton_,cv_tag_ = sketch_.to(self.device), skeleton_.to(self.device), cv_tag_.to(self.device)
#                 输入的是什么！看清楚

                    with torch.no_grad():
                        feature_tensor = self.Pretrain_ResNeXT(skeleton_)
                        # feature_tensor = self.Pretrain_ResNeXT(sketch_)

                    if self.gpu_mode:
                        feature_tensor = feature_tensor.to(self.device)

                    # D_real, CIT_real, CVT_real = self.D(original_)
                    G_f, _ = self.G(sketch_, feature_tensor, cv_tag_)
                    G_f = self.color_revert(G_f.cpu())
                    
#            用来测试 tagpix+skeleton
                    # G_f, _, G_s = self.G(sketch_, feature_tensor, cv_tag_)
                    # G_f = self.color_revert(G_f.cpu())


                    for ind, result in zip(index_.cpu().numpy(), G_f):
                        save_path = result_path / f'{ind}.png'
                        if save_path.exists():
                            for i in range(100):
                                save_path = result_path / f'{ind}_{i}.png'
                                if not save_path.exists():
                                    break
                        img = Image.fromarray(result)
                        img.save(save_path)
                        
                        
#         图片保存完了  在result_path里

            # result_path=self.result_path / 'tag2pix_{}_epoch'.format(epoche)
    
            test_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.data_dir,'benchmark')
            # print(str(test_path))
            # print(str(result_path))
            metrics_dict = torch_fidelity.calculate_metrics(
                input1= str(test_path), 
                input2= str(result_path),
                cuda=True, 
                isc=True, 
                fid=True, 
                kid=True, 
                verbose=False,
            )
            print('{}_{}_epoch'.format(self.args.model,epoche))
            print(metrics_dict)
            
            
    def visualize_results(self, epoch, fix=True):
        if not self.result_path.exists():
            self.result_path.mkdir()

        self.G.eval()

        # test_data_loader
        original_, sketch_, _, _, cv_tag_ = self.test_images
        image_frame_dim = int(np.ceil(np.sqrt(len(original_))))

        # iv_tag_ to feature tensor 16 * 16 * 256 by pre-reained Sketch.
        with torch.no_grad():
            feature_tensor = self.Pretrain_ResNeXT(sketch_)

            if self.gpu_mode:
                original_, sketch_, cv_tag_, feature_tensor = original_.to(self.device), sketch_.to(self.device), cv_tag_.to(self.device), feature_tensor.to(self.device)

            # G_f, G_g,= self.G(sketch_, feature_tensor, cv_tag_)
            G_f, G_g, G_s = self.G(sketch_, feature_tensor, cv_tag_)
            if self.gpu_mode:
                G_f = G_f.cpu()
                G_g = G_g.cpu()
                # G_s = G_s.cpu()

            G_f = self.color_revert(G_f)
            G_g = self.color_revert(G_g)
            # G_s = self.color_revert(G_s)

        utils.save_images(G_f[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_path / 'tag2pix_epoch{:03d}_G_f.png'.format(epoch))
        utils.save_images(G_g[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_path / 'tag2pix_epoch{:03d}_G_g.png'.format(epoch))
        # utils.save_images(G_s[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
        #                   self.result_path / 'tag2pix_epoch{:03d}_G_s.png'.format(epoch))

    def save(self, save_epoch):
        if not self.result_path.exists():
            self.result_path.mkdir()

        with (self.result_path / 'arguments.txt').open('w') as f:
            f.write(pprint.pformat(self.args.__dict__))

        save_dir = self.result_path

        torch.save({
            'G' : self.G.state_dict(),
            'D' : self.D.state_dict(),
            'G_optimizer' : self.G_optimizer.state_dict(),
            'D_optimizer' : self.D_optimizer.state_dict(),
            'finish_epoch' : save_epoch,
            'result_path' : str(save_dir)
            }, str(save_dir / 'tag2pix_{}_epoch.pkl'.format(save_epoch)))

        with (save_dir / 'tag2pix_{}_history.pkl'.format(save_epoch)).open('wb') as f:
            pickle.dump(self.train_hist, f)

        print("============= save success =============")
        print("epoch from {} to {}".format(self.start_epoch, save_epoch))
        print("save result path is {}".format(str(self.result_path)))

    def load_test(self, checkpoint_path):
        checkpoint = torch.load(str(checkpoint_path))
        self.G.load_state_dict(checkpoint['G'])

    def load(self, checkpoint_path):
        checkpoint = torch.load(str(checkpoint_path))
        self.G.load_state_dict(checkpoint['G'])
        self.D.load_state_dict(checkpoint['D'])
        self.G_optimizer.load_state_dict(checkpoint['G_optimizer'])
        self.D_optimizer.load_state_dict(checkpoint['D_optimizer'])
        self.start_epoch = checkpoint['finish_epoch'] + 1

        self.finish_epoch = self.args.epoch + self.start_epoch - 1
        self.end_epoch = self.args.epoch

        print("============= load success =============")
        print("epoch start from {} to {}".format(self.start_epoch, self.finish_epoch))
        print("previous result path is {}".format(checkpoint['result_path']))


    def get_test_data(self, test_data_loader, count):
        test_count = 0
        original_, sketch_, skeleton_,_ , iv_tag_, cv_tag_ = [], [], [], [], [], []
        for orig, sket, person_num, skeleton, ivt, cvt in test_data_loader:
            original_.append(orig)
            sketch_.append(sket)
            skeleton_.append(skeleton)
            iv_tag_.append(ivt)
            cv_tag_.append(cvt)

            test_count += len(orig)
            if test_count >= count:
                break

        original_ = torch.cat(original_, 0)
        sketch_ = torch.cat(sketch_, 0)
        skeleton_ = torch.cat(skeleton_, 0)
        iv_tag_ = torch.cat(iv_tag_, 0)
        cv_tag_ = torch.cat(cv_tag_, 0)

        self.save_tag_tensor_name(iv_tag_, cv_tag_, self.result_path / "test_image_tags.txt")

        image_frame_dim = int(np.ceil(np.sqrt(len(original_))))

        if self.gpu_mode:
            original_ = original_.cpu()
        sketch_np = sketch_.data.numpy().transpose(0, 2, 3, 1)
        original_np = self.color_revert(original_)

        utils.save_images(original_np[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        self.result_path / 'tag2pix_original.png')
        utils.save_images(sketch_np[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        self.result_path / 'tag2pix_sketch.png')

        return original_, sketch_, skeleton_, iv_tag_, cv_tag_


    def save_tag_tensor_name(self, iv_tensor, cv_tensor, save_file_path):
        '''iv_tensor, cv_tensor: batched one-hot tag tensors'''
        iv_dict_inverse = {tag_index: tag_id for (tag_id, tag_index) in self.iv_dict.items()}
        cv_dict_inverse = {tag_index: tag_id for (tag_id, tag_index) in self.cv_dict.items()}

        with open(save_file_path, 'w') as f:
            f.write("CIT tags\n")

            for tensor_i, batch_unit in enumerate(iv_tensor):
                tag_list = []
                f.write(f'{tensor_i} : ')

                for i, is_tag in enumerate(batch_unit):
                    if is_tag:
                        tag_name = self.id_to_name[iv_dict_inverse[i]]
                        tag_list.append(tag_name)
                        f.write(f"{tag_name}, ")
                f.write("\n")

            f.write("\nCVT tags\n")

            for tensor_i, batch_unit in enumerate(cv_tensor):
                tag_list = []
                f.write(f'{tensor_i} : ')

                for i, is_tag in enumerate(batch_unit):
                    if is_tag:
                        tag_name = self.id_to_name[cv_dict_inverse[i]]
                        tag_list.append(self.id_to_name[cv_dict_inverse[i]])
                        f.write(f"{tag_name}, ")
                f.write("\n")

    def print_params(self):
        params_cnt = [0, 0, 0]
        for param in self.G.parameters():
            params_cnt[0] += param.numel()
        for param in self.D.parameters():
            params_cnt[1] += param.numel()
        for param in self.Pretrain_ResNeXT.parameters():
            params_cnt[2] += param.numel()
        print(f'Parameter #: G - {params_cnt[0]} / D - {params_cnt[1]} / Pretrain - {params_cnt[2]}')
