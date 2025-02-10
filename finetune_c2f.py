import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import datetime
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]

def calculate_prototype(args, encoder, G_loader, num_cls, feat_dim):
    flip = transforms.RandomHorizontalFlip(p=1)
    encoder.eval()
    prototypes = torch.zeros(num_cls, feat_dim).to(args.device)
    with torch.no_grad():
        for batch, (img, label) in enumerate(G_loader):
            img, label = img.to(args.device), label.to(args.device)
            feat = 0.5 * (encoder(img) + encoder(flip(img)))
            for i in range(label.size(0)):
                prototypes[label[i]] += feat[i]
    prototypes = F.normalize(prototypes, dim=1)
    return prototypes

def masked_softmax(weights, dim=1):
    # Create a mask to mark elements that are not zero
    mask = weights != 0

    # softmax is computed for the elements of each row that are not zero
    masked_weights = torch.where(mask, weights, torch.tensor(float('-inf')).to(weights.device))
    softmax_weights = torch.softmax(masked_weights, dim=dim)

    # The softmax result is applied back to the original tensor, and the zero elements remain zero
    norm_weights = torch.where(mask, softmax_weights, torch.tensor(0.0).to(weights.device))
    return norm_weights

def weight_gallery_base(args, encoder, G_loader, num_cls, vgg_loader, num_cls_vgg, feat_dim):
    # Get the prototypes of the new class and the base class
    cur_protos = calculate_prototype(args, encoder, G_loader, num_cls, feat_dim)
    base_protos = calculate_prototype(args, encoder, vgg_loader, num_cls_vgg, feat_dim)

    weights = torch.mm(cur_protos, base_protos.T)

    # Find the top k largest values for each row and their indices
    top_values, top_indices = torch.topk(weights, args.top_k, dim=1)

    # Create an all-zero tensor with the same size as weights
    updated_weights = torch.zeros_like(weights)

    # Copy the top top_k largest values for each row into updated_weights
    for row in range(weights.size(0)):
        updated_weights[row, top_indices[row]] = weights[row, top_indices[row]]
    if torch.sum(updated_weights) == 0:
        raise ValueError("All values in updated_weights are zero, adjust your top_k or weights calculation.")
    # The tensor after updating the weights
    norm_weights = masked_softmax(updated_weights, dim=1)
    delta_protos = torch.matmul(norm_weights, base_protos)
    delta_protos = F.normalize(delta_protos, dim=1)

    updated_protos = (1 - args.base_weight) * cur_protos + args.base_weight * delta_protos
    return updated_protos


def fine_tune(args, trainloader,num_cls,train_loader_base,num_cls_base,encoder, classifier,
              optimizer, scheduler, verbose=True):
    CEloss = nn.CrossEntropyLoss()
    # Create a SummaryWriter object and set the log_dir based on the classifier's name
    log_dir = os.path.join('logs', classifier.__class__.__name__)
    writer = SummaryWriter(log_dir=log_dir)
    lr_values = []
    loss_values = []
    for epoch in range(args.num_epochs):
        prototypes = weight_gallery_base(args, encoder, trainloader, num_cls, train_loader_base, num_cls_base, 512)
        classifier.weight = nn.Parameter(prototypes.T)
        train_corr, train_tot = 0, 0
        for img, label in trainloader:
            img, label = img.to(args.device), label.to(args.device)
            with torch.cuda.amp.autocast():
                feat = encoder(img)
                logit, sim = classifier(feat, label)
                loss = CEloss(logit, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            corr = torch.argmax(sim, dim=1).eq(label).sum().item()
            train_corr += corr
            train_tot += label.size(0)
        train_acc = train_corr / train_tot * 100
        scheduler.step()
        lr = get_lr(optimizer)
        lr_values.append(lr)
        loss_values.append(loss.item())
        writer.add_scalar('Learning Rate_{}'.format(classifier.__class__.__name__), lr, epoch)
        writer.add_scalar('Loss_{}'.format(classifier.__class__.__name__), loss.item(), epoch)
        if verbose:
            print("epoch:{}, loss:{:.2f}, acc:{:.2f}%,  lr:{:.2e}".format(epoch, loss.item(),
                                                                          train_acc, get_lr(optimizer)))

        writer.close()
