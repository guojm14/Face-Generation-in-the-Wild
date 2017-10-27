import load_data as ld
import argparse
import os
import torch
from torch.autograd import Variable as Vb
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models
import torch.optim as optim
#import load_data as ld
import logging
import torchvision.utils  as tov
import pdb
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--lr_vae', type=float, default=0.0002, help='vae learning rate, default=0.001')
parser.add_argument('--lr_gan', type=float, default=0.0002, help='gan learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--maskregu', type=bool, default=True, help='mask_regu')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
opt = parser.parse_args()
print(opt)
LAMBDA = 10 # Gradient penalty lambda hyperparameter
class encoder(nn.Module):
    def __init__(self):
        super(encoder,self).__init__()
        self.main = nn.Sequential(
            # input is 3 x 64 x 64
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 64 x 32 x 32
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.mask=nn.Sequential(

            nn.Conv2d(128, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128,3,1,1,0,bias=False),
            nn.BatchNorm2d(3),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.Sigmoid()
            )
        self.texture=nn.Sequential(
            nn.Conv2d(128, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.AvgPool2d(4)
            )
        self.fc1=nn.Linear(128,128)
        self.fc2=nn.Linear(128,128)
        self.fc3=nn.Linear(256*3,256)
        self.fc4=nn.Linear(256*3,256)
    def forward(self,x):
        feature=self.main(x)
        outmask= self.mask(feature)
        mu1=self.fc3(outmask.view(-1,256*3))
        logvar1=self.fc4(outmask.view(-1,256*3))
        temp=self.texture(feature)
        mu=self.fc1(temp.view(-1,128))
        logvar=self.fc2(temp.view(-1,128))
        return mu,logvar,mu1,logvar1,outmask
class decoder_meta(nn.Module):
    def __init__(self):
        super(decoder,self).__init__()
        self.fc1=nn.Linear(128,128*2)
        self.fc2=nn.Linear(128*2,128*4)

        self.conv2=nn.Sequential(
            nn.ConvTranspose2d(128*4+3, 128 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128 * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(128 * 2, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. (128) x 32 x 32
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Sigmoid()

            # state size. (nc) x 64 x 64
            )
    def forward(self,mask,code):
        #print (code.size())
        x=self.fc1(code)
        x=self.fc2(x)
        #print (x.size())
        x=x.view(-1,128*4,1,1)
        #print x.size()
        temp=x.repeat(1,1,8,8)
        #print temp.size()
        #print mask.size()
        #print temp.size()
        temp1=torch.cat([temp,mask],1)
        out=self.conv2(temp1)
        return out
class upsample_deconv(nn.Module):
    def __init__(self):
        super(upsample_deconv,self).__init__()
        self.upconv=nn.Sequential(
            nn.Conv2d(3+128*2, 128*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128*2, 128*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128*2, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (128) x 32 x 32
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Sigmoid()
            )
    def forward(self,x):
        return self.upconv(x)
class upsample_pixel_shuffle(nn.Module):
    def __init__(self):
        super(upsample_pixel_shuffle,self).__init__()
        self.upconv=nn.Sequential(
            nn.Conv2d(1+128*2, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 64*2*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.PixelShuffle(2),

            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 3*2*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(12),
            nn.LeakyReLU(0.2, inplace=True),

            nn.PixelShuffle(2),
            nn.Sigmoid()
            )
    def forward(self,x):
        return self.upconv(x)

class decoder(nn.Module):
    def __init__(self):
        super(decoder,self).__init__()
        self.fc1=nn.Linear(128,128*2)
        self.fc2=nn.Linear(128*2,128*2)
        self.fc3=nn.Linear(256,256*3)
        self.fc4=nn.Linear(256*3,256*3)
        self.deconv=upsample_deconv()
    def forward(self,mask,code):
        #print code.size()
        code=F.leaky_relu(self.fc1(code))
        code=F.leaky_relu(self.fc2(code))
        #print x.size()
        code=code.view(-1,128*2,1,1)
        #print x.size()
        code=code.repeat(1,1,16,16)
        mask=F.leaky_relu(self.fc3(mask))
        mask=F.sigmoid(self.fc4(mask))
        mask=mask.view(-1,3,16,16)
        #print temp.size()
        #print mask.size()
        #print temp.size()
        temp1=torch.cat([code,mask],1)
        #print temp1.size()
        out=self.deconv(temp1)
        return out,mask

def loss_like(recon_x, x,mask0=None,mask1=None):
    MSE = F.mse_loss(recon_x, x)
    if opt.maskregu:
        MSE1=F.mse_loss(mask1,mask0)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    if opt.maskregu:
        return MSE+MSE1
    else:
        return MSE
def loss_prior(mu,logvar,mu1,logvar1):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD1= -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= opt.batch_size*64*64*3
    KLD1/= opt.batch_size*64*64*3

    return KLD + KLD1
def entropy_loss(x):
    x1=torch.squeeze(torch.sum(torch.sum(x,2),3))
    return -torch.sum(torch.mul(x,torch.log(x)))+torch.sum(torch.mul(x1,torch.log(x1)))
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.enco=encoder()
        self.deco=decoder()
    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Vb(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu
    def forward(self,x):
        mu,logvar,mu1,logvar1,mask0=self.enco(x)
        maskcode=self.reparameterize(mu1,logvar1)
        code=self.reparameterize(mu, logvar)
        x_re,mask1=self.deco(maskcode,code)
        #mask1,mu1,logvar1=self.enco(x_re)
        return x_re,mu,logvar,mu1,logvar1,mask0,mask1
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)
def calc_gradient_penalty(netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(opt.batch_size, real_data.nelement()/opt.batch_size).contiguous().view(opt.batch_size, 3, 64, 64)
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)


    interpolates = interpolates.cuda()
    interpolates = Vb(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty
criterion = nn.BCELoss()
ngpu = 1
nc = 3
ndf = 64
likeweight=1
netD = _netD(ngpu)
netD.apply(weights_init)
real_label = 1
fake_label = 0
one = torch.FloatTensor([1])
mone = one * -1
one = one.cuda()
mone = mone.cuda()
netD.cuda()
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_gan, betas=(opt.beta1, 0.999))
num_iter=500000
logging.basicConfig(filename='log/vaenet2.log', level=logging.INFO)
vae=VAE()

optimizer1=optim.Adam(vae.enco.parameters(),lr=opt.lr_vae)
optimizer2=optim.Adam(vae.deco.parameters(),lr=opt.lr_vae)
datalist=ld.getlist('list_attr_train1.txt')
iternow1=0
label = torch.FloatTensor(opt.batch_size)
label = label.cuda()
#checkpoint = torch.load('vae_iter_260000.pth.tar')
#vae.load_state_dict(checkpoint['VAE'])
vae = vae.cuda()
for iter1 in range(num_iter):
    vae.train()

    netD.zero_grad()
    imgpo,iternow1=ld.load_data('/ssd/randomcrop_resize_64/','list_attr_train1.txt',datalist,iternow1,opt.batch_size)
    imgpo_re,mu,logvar,mu1,logvar1,mask0,mask1=vae(imgpo)

    real = imgpo;
    label.resize_(opt.batch_size).fill_(real_label)
    labelv = Vb(label)
    output = netD(real);
    D_real = output.mean()
    #errD_real = criterion(output, labelv)
    #errD_real.backward()
    #D_x = output.data.mean()
    D_real.backward(mone)

    eps0 = Vb(mu.data.new(mu1.size()).normal_()) #gauss noise bs,256 mask
    eps1 = Vb(mu.data.new(mu.size()).normal_())  #gauss noise bs,128 texture
    fake,_=vae.deco(eps0,eps1)    # sample
    labelv = Vb(label.fill_(fake_label))

    output = netD(fake.detach())
    D_fake = output.mean()
    D_fake.backward(one)

    output_re=netD(imgpo_re.detach())
    D_re=output_re.mean()
    D_re.backward(one)

    gradient_penalty = calc_gradient_penalty(netD, real.data, fake.data)
    gradient_penalty.backward()
    
    gradient_penalty1 = calc_gradient_penalty(netD, real.data, imgpo_re.data)
    gradient_penalty1.backward()

    D_cost = D_fake +D_re- D_real + gradient_penalty+ gradient_penalty1
    Wasserstein_D = D_real - D_fake-D_re
    #errD_fake = criterion(output, labelv)
    #errD_fake.backward()
    #D_G_z1 = output.data.mean()
    #errD = errD_real + errD_fake
    optimizerD.step()

    vae.enco.zero_grad()
    #labelv = Vb(label.fill_(real_label))
    loss1 = loss_like(imgpo_re,imgpo,mask0,mask1)
    loss2= loss_prior(mu,logvar,mu1,logvar1)
    loss1.backward(retain_graph=True)
    loss2.backward(retain_graph=True)
    optimizer1.step()

    vae.deco.zero_grad()
    output = netD(fake)
    loss3 = output.mean()
    loss3.backward(mone)
    #loss3  = criterion(output, labelv)
    loss1=likeweight*loss1
    loss1.backward()
    optimizer2.step()

    if iter1%100==0:
        print(iter1)
        outinfo='vae: '+ str(iter1)+str(loss1)+str(loss2)+' '+str(loss3)
        logging.info(outinfo)
        print (outinfo)
        outinfo = 'd: '+str(iter1) + str(D_cost)
        logging.info(outinfo)
        print(outinfo)
    if iter1 % 200 == 0:
        vae.eval()
        saveim=fake.cpu().data
        tov.save_image(saveim,'./vae_gan2/fake'+str(iter1)+'.jpg')
        saveim=imgpo_re.cpu().data
        tov.save_image(saveim,'./vae_gan2/recon'+str(iter1)+'.jpg')
        saveim=real.cpu().data
        tov.save_image(saveim,'./vae_gan2/real'+str(iter1)+'.jpg')
    if iter1 %10000==0:
        # save model
        save_name = './vae_gan2/{}_iter_{}.pth.tar'.format('vae', iter1)
        torch.save({'VAE': vae.state_dict()}, save_name)
        logging.info('save model to {}'.format(save_name))
        save_name = './vae_gan2/{}_iter_{}.pth.tar'.format('gan', iter1)
        torch.save({'VAE': netD.state_dict()}, save_name)
        logging.info('save model to {}'.format(save_name))

