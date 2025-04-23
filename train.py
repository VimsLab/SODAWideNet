from DataHandling import *
from torch.nn import functional as F
from pytorch_msssim import ssim, SSIM
from SODAWideNet import *

def re_Dice_Loss(inputs, targets, cuda=False, balance=1.1):
	n, c, h, w = inputs.size()
	smooth=1
	inputs = torch.sigmoid(inputs)

	input_flat=inputs.view(-1)
	target_flat=targets.view(-1)

	intersecion=input_flat*target_flat
	unionsection=input_flat.pow(2).sum()+target_flat.pow(2).sum()+smooth
	loss=unionsection/(2*intersecion.sum()+smooth)
	loss=loss.mean()

	return loss

def _weighted_cross_entropy_loss(preds, edges, device, weight = 10):
	mask = (edges == 1.0).float()
	b, c, h, w = edges.shape
	num_pos = torch.sum(mask, dim=[1, 2, 3]).float()
	num_neg = c * h * w - num_pos
	weight = torch.zeros_like(edges)
	nx1 = num_neg / (num_pos + num_neg)
	nx2 = num_pos / (num_pos + num_neg)
	weight = torch.cat([torch.where(i == 1.0, j, k) for i, j, k in zip(edges, nx1, nx2)], dim = 0).unsqueeze(1)
	losses = F.binary_cross_entropy_with_logits(preds.float(),
				edges.float(),
				weight=weight,
				reduction='none')
	loss = torch.sum(losses) / b
	return loss

ssim_compute = SSIM(data_range = 1, size_average = True, channel = 1)

def structure_loss_salient(pred, mask):
	weit  = 1+5*torch.abs(F.max_pool2d(mask, kernel_size=31, stride=1, padding=15))
	wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce=False)
	wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

	pred  = torch.sigmoid(pred)
	inter = ((pred*mask)*weit).sum(dim=(2,3))
	union = ((pred+mask)*weit).sum(dim=(2,3))
	wiou  = 1-(inter+1)/(union-inter+1)

	mae = F.l1_loss(pred, mask, reduce=False)
	wmae = (mae*weit).sum(dim=(2,3))/weit.sum(dim=(2,3))

	ssim_val = 1 - ssim_compute(pred, mask)
	return (wbce + wiou + wmae).mean() + ssim_val

def structure_loss_contour(pred,target):

	bce_out = _weighted_cross_entropy_loss(pred,target,None)
	iou_out = re_Dice_Loss(pred, target)
	pred = torch.sigmoid(pred)
	ssim_val = 1 - ssim_compute(pred, target)

	loss = 0.001 * bce_out + iou_out + ssim_val

	return loss


def init_weights(net, init_type='normal', gain=0.02):
	def init_func(m):
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
			if init_type == 'normal' and classname.find('Conv1d') != -1:
				n = m.kernel_size[0] * m.in_channels
				init.normal_(m.weight.data, 0.0, math.sqrt(2.0 / n))
			elif init_type == 'normal' and classname.find('Conv') != -1:
				n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
				init.normal_(m.weight.data, 0.0, math.sqrt(2.0 / n))
			elif init_type == 'normal' and classname.find('Linear') != -1:
				init.normal_(m.weight.data, 0.0, 1.0)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=gain)
			else:
				raise NotImplementedError('initialization method [%s] is not implemented for [%s]' % (init_type, classname))
			if hasattr(m, 'bias') and m.bias is not None:
				init.constant_(m.bias.data, 0.0)
		elif classname.find('BatchNorm2d') != -1 or classname.find('BatchNorm1d') != -1:
			init.normal_(m.weight.data, 1.0, gain)
			init.constant_(m.bias.data, 0.0)

	print('initialize network with %s' % init_type)
	net.apply(init_func)

def training_deepx(epochs, eno, model, dataloader, cuda, criterion, optimizer, scheduler = None, f_name = 'model.pt', criterion2 = None, train_sampler = None):
	model.train()
	loss_before = 0.0
	fmean = 0.0
	mae = 1.0
	sig = nn.Sigmoid()
	for epoch in range(epochs):
		loss_end = 0.0
		count = 0
		iou = 0.0
		for i, data in enumerate(dataloader):
			model.zero_grad()
			images = data[0].to(device = cuda)
			saliency = data[1].to(device = cuda)
			contour = data[2].to(device = cuda)
			contours, saliency_maps = model(images)
			loss = 0.0

			for i in contours:
				loss += structure_loss_contour(i, contour)
			
			for i in saliency_maps:
				loss += structure_loss_salient(i, saliency)

			loss.backward()
			optimizer.step()
			loss_end += loss.mean().item()
			count += 1
		if scheduler is not None:
			scheduler.step()
		loss_before = loss_end/count
		print('[%d/%d]Loss: %.2f' % (epoch, epochs, loss_end/count), flush = True)
		if epoch < 30:
			continue
		
		torch.save({
			'epoch': epoch,
			'model_state_dict': model.module.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': loss_end/count
		}, f = f_name + str(epoch) + '.pt')
	return model

def main(train = False, lr = 0.001, epochs = 30, t = 25, f_name = 'checkpoints/model.pt', device_list = None, device = 0, batch = 0, sched = 1):
	cuda = torch.device("cuda:" + str(device) if torch.cuda.is_available() else "cpu")
	# model = SODAWideNet(3, 1, use_contour = True, deep_supervision = True, factorw = 1); init_weights(model)
  	model = SODAWideNet(3, 1, use_contour = True, deep_supervision = True, factorw = 2); init_weights(model)
	if device_list is not None:
		model = nn.DataParallel(model, device_ids = device_list)
	else:
		print(device, flush = True)
	model.to(cuda)
	print(cuda, flush = True)
	train_set = SODLoaderAugmentNew();print(len(train_set), flush = True);
	train_dataloader = DataLoader(train_set, batch_size = batch, shuffle = True, num_workers = 8)

	optimizer = torch.optim.Adam(model.parameters(), lr = lr)
	if sched:
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=t, gamma=0.1)
	else:
		scheduler = None
	criterion2 = None
	criterion = nn.BCEWithLogitsLoss()
	epoch = 0
	print('No Pre-Training', flush = True)
  	start = time.time()
	if train:
		model = training_deepx(abs(epochs - epoch), abs(epoch - t), model, train_dataloader, cuda, criterion, optimizer, scheduler, f_name = f_name, criterion2 = criterion2, train_sampler = train_sampler)
		end = time.time()
		print('Time taken for %d with a batch_size of %d is %.2f hours.' %(epochs, batch, (end - start) / (3600)), flush = True)

if __name__ == '__main__':
	seed_val = 60
	torch.manual_seed(seed_val)
	import random
	random.seed(seed_val)
	np.random.seed(seed_val)
	print("Seed value is %d" %(seed_val), flush = True)

	lr = float(sys.argv[1])
	n = int(sys.argv[5])
	b = int(sys.argv[6])
	if n > 1:
		main(train = True, lr = lr, epochs = int(sys.argv[2]), t = int(sys.argv[3]), f_name = 'checkpoints/' + str(sys.argv[4]), device_list = [i for i in range(n)], batch = n * b, sched = int(sys.argv[-1]))
	else:
		main(train = True, lr = lr, epochs = int(sys.argv[2]), t = int(sys.argv[3]), f_name = 'checkpoints/' + str(sys.argv[4]), device = n, batch = b, sched = int(sys.argv[-1]))
